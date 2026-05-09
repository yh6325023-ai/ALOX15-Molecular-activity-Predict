#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import config as cfg
import benchmark


def main() -> None:
    # Fixed 16-model benchmark set (removed unstable: qda, lgbm)
    cfg.CLASSIFICATION_CANDIDATE_FAMILIES = [
        "gaussiannb",
        "bernoullinb",
        "logreg",
        "ridgecls",
        "sgd",
        "passiveaggr",
        "knn",
        "svc",
        "linearsvc",
        "lda",
        "bagging",
        "adaboost",
        "gbr",
        "histgb",
        "randomforest",
        "extratrees",
    ]
    cfg.TASK_TYPE = "classification"
    cfg.FINAL_CV_PROTOCOL = "scaffold"
    cfg.FINAL_CV_FOLDS = 5

    # Benchmark stage: disable nested tuning for fair speed/consistency.
    cfg.NESTED_TUNING_ENABLED = False

    print(
        "[run_benchmark] 16-model benchmark, scaffold 5-fold, nested_tuning=False",
        flush=True,
    )
    benchmark.main()


if __name__ == "__main__":
    main()

