# Build Note — Evidence-Guided Selective Repair

| Rung | Feature | Acceptance check | Tier | Status |
|------|---------|------------------|------|--------|
| F0 | base forward spine | `python train_evidence_guided.py --smoke` | MUST | passed |
| F1 | SDF, regional pooling and exact Otsu | `python -m unittest discover -s tests -v` | MUST | passed |
| F2 | self-excluding attention, shared repair and full-image assembly | `python -m unittest discover -s tests -v` | MUST | passed |
| F3 | router, staged training, checkpoint/resume and default entry | `python train_and_test_isic.py --smoke` | MUST | passed |

## Run record

- System Python could not load Torch c10.dll; the pre-existing `test` conda environment ran PyTorch 1.10.2+cpu, NumPy 1.19.5, SciPy 1.5.4 after putting its Library/bin on PATH. No package installation or environment mutation was needed.
- F0: initial base forward [1,2,64,80], exit 0. No stubs in the base network.
- F1/F2: exact Otsu, constant/single-region handling, valid edge pooling/padding, SDF convention, self exclusion, zero residual identity and batching checks passed.
- F3: 13 behavioral unit tests passed. All stages backpropagated only through active parameters; frozen parameters and normalization buffers stayed unchanged.
- Synthetic integration: two 64x80 training images and one held-out synthetic image; each stage trained one epoch; checkpoints saved/reloaded; final inference and npy prediction writing passed. These tensors and numbers are smoke data, not experimental evidence.
- Synthetic resume: stage-1 checkpoint -> stages 2/3 exactly matched uninterrupted final state. A stage-2 epoch boundary resume exactly matched uninterrupted two-epoch stage-2 training.
- CPU/Gloo DDP: two processes, two iterations in each stage; passed with one rank lacking SDF supervision and later having no selected regions. All ranks participated in gradient reduction.
- The folder is not a Git worktree; no commits were created. Original files were archived before replacing README/default entry.

## Deferred

- Full 200/100/100-epoch real-data experiment and metric verification.
- Eight RTX 4090 / CUDA / NCCL runtime verification.
- External cross-model assumption sweep: SWEEP_UNAVAILABLE (no reviewer endpoint exposed).

## Live stubs

None in the training/inference implementation. Synthetic inputs exist only in explicitly named smoke/tests.

## Blockers

No implementation blockers. Real training needs a dataset root and suitable compute.
