# Uncertainty Locates, Outcome Decides: Evidence-Guided Selective Repair

This repository now contains the network described in the supplied Methods section:

- dual structure/context branches with recurrent interaction and channel attention;
- probabilistic signed-distance supervision and entropy/JS descriptors;
- 16x16 Otsu-selected regions with self-excluding within-image reference attention;
- shared residual U-Net repair and a Gain–Harm acceptance router.

完整中文使用说明见 [论文网络更新说明.md](论文网络更新说明.md)。

## Environment

Install the minimal dependencies with `pip install -r requirements-evidence-guided.txt`. The original code used an older PyTorch stack. The paper-aligned implementation is compatible with PyTorch 1.10+ and CPU or CUDA. In the supplied Windows environment, use:

```powershell
$env:PATH = "C:\Users\11211\mjj\anaconda\envs\test\Library\bin;C:\Users\11211\mjj\anaconda\envs\test;" + $env:PATH
```

## Smoke test and unit checks

```powershell
python train_and_test_isic.py --smoke
python -m unittest discover -s tests -v
```

The smoke test uses synthetic tensors only and verifies shape, edge-region handling, finite losses, and zero residual identity.

## Dataset layout

The existing CNV format is supported:

```text
<root>/image/*.npy
<root>/label/*.npy
Datasets/folder1/folder1_{train,validation,test}.list
```

Images may be uint8 RGB, float RGB in [0,1], or grayscale. Labels must be binary 0/1 or 0/255.

## Three-stage training

The defaults follow the paper (200/100/100 epochs):

```powershell
python train_and_test_isic.py --root-path D:/CNV --data-folder folder1 --output-dir ./data/evidence_guided --eval-after
```

For a quick local run, lower the epoch counts, for example `--epochs1 1 --epochs2 1 --epochs3 1`. Stage 2 and 3 resume from the prior stage checkpoint:

```powershell
python train_evidence_guided.py --stage 2 --resume ./data/evidence_guided/stage1_last.pt --root-path D:/CNV
python train_evidence_guided.py --stage 3 --resume ./data/evidence_guided/stage2_last.pt --root-path D:/CNV --eval-after
```

Use `torchrun --nproc_per_node=8` for the paper's eight-GPU setup; `--batch-size` is the global batch size and must divide the world size. The final epoch checkpoint (`stage3_last.pt`) is used for evaluation.

## Legacy implementation

The pre-update SAUM network and its original augmentation pipeline are preserved in `train_and_test_isic_legacy.py` and `Models/SAUM_Net.py`.


## Inference

```python
import torch
from Models.evidence_guided_repair import EvidenceGuidedRepairNet
from train_evidence_guided import load_checkpoint

model = EvidenceGuidedRepairNet()
model.load_state_dict(load_checkpoint("./data/evidence_guided/stage3_last.pt", "cpu")["state_dict"])
model.eval()
with torch.no_grad():
    probabilities = model(normalized_rgb_batch)  # [B,2,H,W]
    mask = probabilities.argmax(1)
```

Ground-truth masks are required only for losses during training/evaluation, never by the inference forward path. Use `return_details=True` for initial/candidate/final probabilities, selected/accepted regions, outcome estimates and reference weights. `set_stage(1/2/3)` configures training; instantiate a fresh model for full inference or use the trainer's `evaluate` function.
