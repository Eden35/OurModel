# Assumption Ledger — Evidence-Guided Selective Repair
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| A-001 | Whether to replace the legacy SAUM_Net or preserve it | Add the paper network alongside legacy code; route the default script to the new trainer | interface | default |
| A-002 | Exact image size and dataset split for validation | Reuse existing CNV dataset and accept arbitrary H/W; default training resize is 224x224; smoke test uses 65x81 | interface | default |
| A-003 | How to handle empty/all-equal Otsu scores | Select no regions and keep the initial prediction | semantic | default |
| A-004 | Whether incomplete edge regions contribute padded pixels | Replication-pad tensors but exclude padded positions from pooling, assembly, and losses | semantic | default |
| A-005 | How to expose training stages | Add `train_evidence_guided.py` with `--stage`, `--resume`, `--evaluate`, `--smoke` and an alias at the default entry | interface | default |
| A-007 | The paper does not specify an SDF raster boundary convention | Use pixels adjacent to the opposite class (4-connectivity, no artificial image-border boundary), set both sides to zero, and EDT to this boundary; negative foreground, positive background | semantic | default |
| A-008 | Augmentation scale range, noise amplitude and restart schedule are unspecified | Scale 0.8–1.2, zero-background Gaussian noise std=0.05 in [0,1] units before normalization; cosine restart T0=10, multiplier=2, minimum lr=1e-5; bilinear image and nearest label affine transforms with zero outside the image | semantic | default |
| A-009 | SDF loss weight and decoder feature widths are unspecified | Unit SDF loss weight; both final decoder features have 32 channels; BatchNorm; random initialization without downloaded weights | semantic | default |
| A-010 | Metric aggregation and empty masks are unspecified | Report per-image foreground Dice/IoU means; both-empty masks score 1; final epoch checkpoint supplies test inference | semantic | default |
| A-011 | Data storage precision and label encoding vary | New npy loader accepts uint8 images or floats in [0,1], gray images repeated to RGB; binary labels 0/1 or 0/255 only; use existing explicit split lists | interface | default |
| A-012 | Default entry point remains old code | Replace train_and_test_isic.py with a thin new-entry alias; keep its previous contents as train_and_test_isic_legacy.py | interface | default |
| A-013 | Training hardware may differ from the paper | Single-device CPU/CUDA and torchrun DDP; batch-size argument is global and must divide world size; default global batch=16 | interface | default |
| A-014 | How to resume safely across stage transitions | Full state checkpoints include stage, completed epoch, optimizer/scheduler/RNG/config; resume continues completed final-epoch stage into the next stage | interface | default |
| A-015 | Variance parameterization and numerical stability are unspecified | Predict variance as softplus(raw)+1e-6; sigma=sqrt(variance); clamp only for log/variance stability, no output activation on residuals | semantic | default |
| A-016 | ResNet stem, skip fusion, and normalization exactness are unspecified | 7x7 stride-2 stem plus stride-2 max pooling, concatenated decoder skips, one residual block per decoder scale, ordinary per-rank BatchNorm | semantic | default |
| A-006 | Availability of an external reviewer backend | Record the local static validation; no external sweep is available in this environment | interface | default |

## Notes

- A-001 keeps existing experiments reproducible while making the paper-aligned implementation directly runnable.
- A-003 follows the paper's explicit empty-candidate rule and avoids arbitrary repairs when an image has no relative uncertainty signal.
- A-004 ensures edge padding cannot change regional statistics or supervision targets.
