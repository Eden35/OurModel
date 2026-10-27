"""Losses and target builders for the evidence-guided repair model."""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def prepare_binary_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]
    if target.dim() != 3:
        raise ValueError("Expected target with shape [B,H,W] or [B,1,H,W]")
    return (target > 0.5).long()


def soft_dice_loss(probabilities: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    target = prepare_binary_target(target).float()
    foreground = probabilities[:, 1]
    intersection = (foreground * target).flatten(1).sum(1)
    denominator = foreground.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def soft_iou_loss(probabilities: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    target = prepare_binary_target(target).float()
    foreground = probabilities[:, 1]
    intersection = (foreground * target).flatten(1).sum(1)
    union = (foreground + target - foreground * target).flatten(1).sum(1)
    return (1.0 - (intersection + smooth) / (union + smooth)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, dice: bool = True, iou: bool = False) -> torch.Tensor:
    target = prepare_binary_target(target)
    probabilities = F.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, target)
    if dice:
        loss = loss + soft_dice_loss(probabilities, target)
    if iou:
        loss = loss + soft_iou_loss(probabilities, target)
    return loss


def signed_distance_target(target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build normalized signed distances and a validity mask.

    Samples containing only foreground or only background have no foreground
    boundary and are marked invalid as specified by the paper.
    """
    target_binary = prepare_binary_target(target).detach().cpu().numpy().astype(np.uint8)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - only reached in minimal installs
        raise ImportError("SDF supervision requires scipy") from exc
    batch, height, width = target_binary.shape
    distances = np.zeros((batch, 1, height, width), dtype=np.float32)
    valid = np.zeros((batch,), dtype=np.bool_)
    diagonal = float((height * height + width * width) ** 0.5)
    for index, mask in enumerate(target_binary):
        if mask.min() == mask.max():
            continue
        valid[index] = True
        boundary = np.zeros_like(mask, dtype=np.bool_)
        vertical = mask[1:] != mask[:-1]
        horizontal = mask[:, 1:] != mask[:, :-1]
        boundary[1:] |= vertical
        boundary[:-1] |= vertical
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        raw = distance_transform_edt(~boundary).astype(np.float32)
        raw[mask.astype(bool)] *= -1.0
        distances[index, 0] = raw / diagonal
    return torch.from_numpy(distances).to(target.device), torch.from_numpy(valid).to(target.device)


def sdf_nll_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    distance_target, valid = signed_distance_target(target)
    if not valid.any():
        return outputs["sdf_mu"].sum() * 0.0
    mu = outputs["sdf_mu"][valid]
    sigma = outputs["sdf_sigma"][valid].clamp_min(1e-4)
    truth = distance_target[valid]
    variance = sigma.pow(2)
    return 0.5 * (((truth - mu).pow(2) / variance) + variance.log()).mean()


def base_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor, lambdas: Sequence[float] = (1., 1., 1., 1.)) -> Dict[str, torch.Tensor]:
    structure = lambdas[0] * soft_dice_loss(outputs["probabilities_s"], target) + lambdas[1] * F.cross_entropy(outputs["structure_logits"], prepare_binary_target(target))
    context = lambdas[2] * F.cross_entropy(outputs["context_logits"], prepare_binary_target(target)) + lambdas[3] * soft_iou_loss(outputs["probabilities_c"], target)
    sdf = sdf_nll_loss(outputs, target)
    return {"structure": structure, "context": context, "sdf": sdf, "total": structure + context + sdf}


def repair_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    valid_samples = outputs["selected_mask"].any(dim=1)
    if not valid_samples.any():
        return outputs["candidate_probabilities"].sum() * 0.0
    probabilities = outputs["candidate_probabilities"][valid_samples]
    truth = prepare_binary_target(target)[valid_samples]
    return F.cross_entropy(outputs["candidate_logits"][valid_samples], truth) + soft_dice_loss(probabilities, truth)


def outcome_targets(outputs: Dict[str, torch.Tensor], target: torch.Tensor):
    target = prepare_binary_target(target)
    initial = outputs["probabilities_0"].argmax(dim=1)
    candidate = outputs["candidate_probabilities"].argmax(dim=1)
    selected = outputs["selected_mask"]
    gains, harms, count = [], [], 0
    for batch_index in range(target.size(0)):
        for region_index, region in enumerate(outputs["regions"]):
            if not bool(selected[batch_index, region_index]):
                continue
            y0, x0, valid_h, valid_w = region
            gt = target[batch_index, y0:y0 + valid_h, x0:x0 + valid_w]
            initial_local = initial[batch_index, y0:y0 + valid_h, x0:x0 + valid_w]
            candidate_local = candidate[batch_index, y0:y0 + valid_h, x0:x0 + valid_w]
            gains.append(((initial_local != gt) & (candidate_local == gt)).float().mean())
            harms.append(((initial_local == gt) & (candidate_local != gt)).float().mean())
            count += 1
    if count == 0:
        return None
    return torch.stack(gains), torch.stack(harms)


def outcome_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    targets = outcome_targets(outputs, target)
    if targets is None:
        return outputs["outcome"].sum() * 0.0
    gains, harms = targets
    predictions = outputs["outcome"][outputs["selected_mask"]]
    errors = ((predictions[:, 0] - gains).pow(2) + (predictions[:, 1] - harms).pow(2))
    # First average over selected regions per image, then over images with
    # supervision. Images with many selected regions do not dominate a batch.
    image_ids = outputs["selected_mask"].nonzero(as_tuple=False)[:, 0]
    losses = [errors[image_ids == image_id].mean() for image_id in image_ids.unique()]
    return torch.stack(losses).mean()


__all__ = [
    "base_loss", "outcome_loss", "outcome_targets", "prepare_binary_target",
    "repair_loss", "sdf_nll_loss", "signed_distance_target", "soft_dice_loss", "soft_iou_loss",
]
