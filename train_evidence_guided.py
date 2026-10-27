"""Three-stage training and inference for Evidence-Guided Selective Repair."""
import argparse
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from Datasets.evidence_guided import NpySegmentationDataset
from Models.evidence_guided_repair import EvidenceGuidedRepairNet
from utils.evidence_guided_losses import base_loss, outcome_loss, repair_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_checkpoint(path, device):
    # These are this trainer's own checkpoints; RNG states require full loading
    # on Torch versions that default to weights_only=True.
    options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(str(path), **options)


def smoke():
    torch.set_num_threads(2)
    set_seed(1234)
    model = EvidenceGuidedRepairNet().eval()
    image = torch.randn(1, 3, 65, 81)
    target = torch.randint(0, 2, (1, 65, 81))
    with torch.no_grad():
        outputs = model(image, return_details=True)
        losses = [base_loss(outputs, target)["total"], repair_loss(outputs, target), outcome_loss(outputs, target)]
    assert outputs["final_probabilities"].shape == (1, 2, 65, 81)
    assert all(torch.isfinite(value).item() for value in losses)
    assert torch.allclose(outputs["candidate_probabilities"], outputs["probabilities_0"], atol=1e-6)
    report = {"base_forward": "passed", "full_forward": "passed", "edge_regions": "passed",
              "zero_residual_identity": "passed", "shape": list(outputs["final_probabilities"].shape),
              "selected_regions": int(outputs["selected_mask"].sum()), "losses_finite": True,
              "synthetic_data_only": True}
    output = Path(__file__).resolve().parent / "implement-stage" / "smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def init_device(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if world_size > 1:
            device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    if args.batch_size < world_size or args.batch_size % world_size:
        raise ValueError("Global --batch-size must be divisible by WORLD_SIZE")
    return device, rank, world_size


def make_loader(args, split, rank=0, world_size=1):
    dataset = NpySegmentationDataset(args.root_path, args.data_folder, split, args.image_size,
                                     augment=(split == "train"))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 and split == "train" else None
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size // world_size if split == "train" else 1,
                      shuffle=split == "train" and sampler is None, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), generator=generator)


def global_weight(local_count, device, world_size):
    count = torch.tensor(float(local_count), device=device)
    if world_size > 1:
        dist.all_reduce(count)
    return world_size * float(local_count) / max(1.0, count.item()), count.item()


def train_epoch(model, loader, optimizer, scheduler, stage, epoch, device, world_size):
    model.train()
    total, count = 0.0, 0
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(sum((200, 100, 100)[:stage-1]) + epoch)
    for step, (image, target, _) in enumerate(loader):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, accept=False, return_details=True)
        if stage == 1:
            losses = base_loss(outputs, target)
            # Each term is averaged over images with its own valid supervision.
            sample_weight, _ = global_weight(image.size(0), device, world_size)
            nonconstant = target.flatten(1).min(1)[0] != target.flatten(1).max(1)[0]
            sdf_weight, _ = global_weight(int(nonconstant.sum()), device, world_size)
            loss = (losses["structure"] + losses["context"]) * sample_weight + losses["sdf"] * sdf_weight
            active_count = image.size(0)
        else:
            active_count = int(outputs["selected_mask"].any(1).sum())
            weight, global_count = global_weight(active_count, device, world_size)
            if global_count == 0:
                continue
            loss = (repair_loss(outputs, target) if stage == 2 else outcome_loss(outputs, target)) * weight
        # A rank can have no selected regions while another rank has supervision.
        # Touch all active parameters with zero so DDP still reduces every grad.
        zero = sum(parameter.reshape(-1)[0] * 0.0 for parameter in model.parameters() if parameter.requires_grad)
        loss = loss + zero
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss at stage {}, epoch {}, batch {}".format(stage, epoch, step))
        loss.backward()
        optimizer.step()
        scheduler.step(epoch - 1 + (step + 1) / max(1, len(loader)))
        total += float(loss.detach())
        count += 1
    if world_size > 1:
        stats = torch.tensor([total, count], dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        total, count = stats.tolist()
    if not count:
        raise RuntimeError("No samples had selected regions in this epoch; no checkpoint was saved. Check base uncertainty or image/patch sizes.")
    return total / count


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(model, optimizer, scheduler, stage, epoch, args, output_dir, rank, world_size, loader):
    rng = capture_rng()
    if world_size > 1:
        rng_states = [None for _ in range(world_size)]
        dist.all_gather_object(rng_states, rng)
    else:
        rng_states = [rng]
    if rank == 0:
        checkpoint = {"stage": stage, "epoch": epoch, "state_dict": unwrap(model).state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "config": vars(args), "rng_states": rng_states,
                      "loader_rng": loader.generator.get_state()}
        torch.save(checkpoint, str(output_dir / "stage{}_last.pt".format(stage)))


def run_stage(model, loader, stage, epochs, device, args, output_dir, rank=0, world_size=1, resume=None):
    model.set_stage(stage)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    start_epoch = 1
    # Reproduce sampler ordering and worker seeds at epoch-boundary resumes.
    if resume is not None and resume.get("loader_rng") is not None:
        loader.generator.set_state(resume["loader_rng"].cpu())
        rng_states = resume.get("rng_states", [])
        if rank < len(rng_states):
            restore_rng(rng_states[rank])
    if resume is not None and resume["stage"] == stage:
        start_epoch = resume["epoch"] + 1
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
    wrapped = model
    if world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                          find_unused_parameters=False)
    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(wrapped, loader, optimizer, scheduler, stage, epoch, device, world_size)
        save_checkpoint(wrapped, optimizer, scheduler, stage, epoch, args, output_dir, rank, world_size, loader)
        if rank == 0:
            record = {"stage": stage, "epoch": epoch, "epochs": epochs, "loss": loss,
                      "lr": optimizer.param_groups[0]["lr"]}
            print(json.dumps(record), flush=True)
            with (output_dir / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
    if world_size > 1:
        dist.barrier()


def evaluate(model, loader, device, prediction_dir=None):
    # Clear training-stage short circuits for the full inference pipeline.
    model._stage = None
    model.eval()
    dices, ious, selected, accepted = [], [], 0, 0
    if prediction_dir:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for image, target, names in loader:
            outputs = model(image.to(device), return_details=True)
            prediction = outputs["final_probabilities"].argmax(dim=1)
            truth = target.to(device)
            intersection = ((prediction == 1) & (truth == 1)).flatten(1).sum(1).float()
            denom = (prediction == 1).flatten(1).sum(1) + (truth == 1).flatten(1).sum(1)
            union = ((prediction == 1) | (truth == 1)).flatten(1).sum(1).float()
            dices.extend(torch.where(denom > 0, 2 * intersection / denom.clamp_min(1), torch.ones_like(intersection)).cpu().tolist())
            ious.extend(torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(intersection)).cpu().tolist())
            selected += int(outputs["selected_mask"].sum())
            accepted += int(outputs["accepted_mask"].sum())
            if prediction_dir:
                for name, mask in zip(names, prediction.cpu().numpy()):
                    np.save(str(prediction_dir / (Path(name).stem + ".npy")), mask.astype(np.uint8))
    if not dices:
        raise ValueError("Cannot evaluate an empty test split")
    result = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "samples": len(dices),
              "selected_regions": selected, "accepted_regions": accepted}
    print(json.dumps(result))
    return result


def main(args):
    if args.smoke:
        smoke()
        return
    if not args.root_path:
        raise ValueError("Supply --root-path containing image/ and label/ npy files")
    if (args.evaluate or args.stage in ("2", "3")) and not args.resume:
        raise ValueError("--evaluate and stages 2/3 require a trained --resume checkpoint")
    if args.eval_after and args.stage in ("1", "2"):
        raise ValueError("--eval-after requires completing stage 3")
    device, rank, world_size = init_device(args)
    set_seed(args.seed + rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = EvidenceGuidedRepairNet(patch_size=args.patch_size).to(device)
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        if checkpoint.get("config", {}).get("patch_size", args.patch_size) != args.patch_size:
            raise ValueError("Resume patch size differs from --patch-size")
        model.load_state_dict(checkpoint["state_dict"])
    try:
        if args.evaluate:
            if checkpoint["stage"] != 3:
                raise ValueError("Full router evaluation requires a stage-3 checkpoint")
            if rank == 0:
                result = evaluate(model, make_loader(args, "test"), device, args.save_predictions)
                (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
            return
        loader = make_loader(args, "train", rank, world_size)
        if not len(loader):
            raise ValueError("Training split is empty")
        epochs = {1: args.epochs1, 2: args.epochs2, 3: args.epochs3}
        requested = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
        if any(epochs[stage] < 1 for stage in requested):
            raise ValueError("Each requested stage must have at least one epoch")
        if checkpoint:
            if args.stage != "all" and int(args.stage) not in (checkpoint["stage"], checkpoint["stage"] + 1):
                raise ValueError("Resume must continue the same stage or the immediately following stage")
            requested = [stage for stage in requested if stage >= checkpoint["stage"]]
            if epochs[checkpoint["stage"]] < checkpoint["epoch"]:
                raise ValueError("Stage epoch limit is lower than the completed checkpoint epoch")
            if requested[0] > checkpoint["stage"] and checkpoint["epoch"] != epochs[checkpoint["stage"]]:
                raise ValueError("Previous stage is unfinished; resume its training or supply matching epoch limits")
        final_resume = args.resume if (checkpoint is not None and checkpoint["stage"] == 3
                                       and checkpoint["epoch"] == epochs[3]) else None
        for stage in requested:
            run_stage(model, loader, stage, epochs[stage], device, args, output_dir, rank, world_size, checkpoint)
            checkpoint = None
        if args.eval_after:
            if requested[-1] != 3:
                raise ValueError("--eval-after requires completing stage 3")
            if rank == 0:
                # Explicitly load the final epoch, rather than selecting on test metrics.
                final_path = Path(final_resume) if final_resume else output_dir / "stage3_last.pt"
                final = load_checkpoint(final_path, device)
                model.load_state_dict(final["state_dict"])
                result = evaluate(model, make_loader(args, "test"), device, args.save_predictions)
                (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--root-path", "--root_path", default="")
    parser.add_argument("--data-folder", "--val_folder", default="folder1")
    parser.add_argument("--output-dir", "--ckpt", default="./data/evidence_guided")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    parser.add_argument("--epochs1", type=int, default=200)
    parser.add_argument("--epochs2", type=int, default=100)
    parser.add_argument("--epochs3", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr_rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--save-predictions", default="")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
