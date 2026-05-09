#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config as cfg
from src.final_module import run_final_training


def _pick_best_family_from_benchmark() -> str:
    bench_csv = Path(cfg.RESULTS_DIR) / "benchmark" / "classification_benchmark_summary.csv"
    if not bench_csv.is_file():
        raise FileNotFoundError(
            f"Benchmark summary not found: {bench_csv}\n"
            "Run run_benchmark.py first or pass --family explicitly."
        )
    df = pd.read_csv(bench_csv)
    if df.empty or "model_family" not in df.columns:
        raise RuntimeError(f"Invalid benchmark summary file: {bench_csv}")
    # Benchmark file is already sorted by OOF ROC-AUC in benchmark.py
    fam = str(df.iloc[0]["model_family"]).strip()
    if not fam:
        raise RuntimeError("Failed to infer best model from benchmark summary.")
    return fam


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final single-model training (nested tuning ON).")
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Model family to train, e.g. extratrees. If omitted, auto-pick top benchmark model.",
    )
    args = parser.parse_args()

    family = args.family.strip() if args.family else _pick_best_family_from_benchmark()

    # Final stage: single model + nested tuning on scaffold split
    cfg.CLASSIFICATION_CANDIDATE_FAMILIES = [family]
    cfg.FINAL_CLASSIFIER_KIND = family
    cfg.TASK_TYPE = "classification"
    cfg.FINAL_CV_PROTOCOL = "scaffold"
    cfg.FINAL_CV_FOLDS = 5
    cfg.NESTED_TUNING_ENABLED = True

    print(
        f"[run_final_model] family={family}, scaffold 5-fold, nested_tuning=True",
        flush=True,
    )
    run_final_training("classification", family)


if __name__ == "__main__":
    main()

