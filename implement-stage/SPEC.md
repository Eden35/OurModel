# Specification — Evidence-Guided Selective Repair

## Target
Add the paper's dual-branch uncertainty estimation, Otsu-guided regional residual repair, and Gain–Harm Router to the existing PyTorch segmentation repository.

## Inputs
RGB tensors `[B,3,H,W]`; optional binary masks `[B,H,W]` or `[B,1,H,W]` for training. Existing CNV `.npy` dataset remains supported.

## Outputs
`Models/evidence_guided_repair.py` exposes `EvidenceGuidedRepairNet`; inference returns final two-class probabilities and optional diagnostics. `train_evidence_guided.py` implements the three-stage training plan. `utils/evidence_guided_losses.py` provides losses.

## Success command
`python train_evidence_guided.py --smoke`

## Base commit
none (the supplied directory is not a Git worktree).

## Scope cuts
The implementation supports single-device CPU/CUDA and torchrun DDP. Full 400-epoch training, eight-GPU execution, and mixed precision were not run in this update. Foreground Dice/IoU reporting is provided; medical surface metrics remain in the legacy utilities. The architecture follows the paper's interfaces and dimensions while using compact native PyTorch blocks instead of importing a separate ResNet package.
