"""Backward-compatible default entry point for the paper-aligned trainer.

The previous SAUM implementation remains in ``train_and_test_isic_legacy.py``.
Use that file explicitly only to reproduce old experiments.
"""
from train_evidence_guided import build_parser, main

if __name__ == "__main__":
    main(build_parser().parse_args())
