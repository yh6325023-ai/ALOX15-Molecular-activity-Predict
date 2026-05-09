#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Y-randomization (label permutation) for classification on train_dev.

Compares the real scaffold-aligned OOF ROC-AUC to a null distribution obtained
by repeatedly shuffling labels and refitting the same ExtraTrees + outer-fold
protocol. Reports an empirical p-value: p = (1 + #{null >= obs}) / (1 + Nperm).
With small Nperm the smallest reportable one-sided p is 1/(Nperm+1); default
Nperm=499 gives floor ~0.002 (use --n-perm 999 for ~0.001).

This is a standard QSAR safeguard against chance correlation; it does not
validate biological mechanism, but it strengthens the case that the model is
not a trivial artifact of fitting noise.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import config as cfg
from src.publication_figures import apply_journal_style, single_column_figsize
from src.final_module import build_shared_supervised_data


def _log(msg):
    print(msg, flush=True)


def oof_auc_extratrees(X, y, fold_td, present_folds, et_params, random_state):
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold in present_folds:
        tr_m = fold_td != int(fold)
        va_m = fold_td == int(fold)
        if len(np.unique(y[tr_m])) < 2 or len(np.unique(y[va_m])) < 2:
            return float("nan")
        clf = ExtraTreesClassifier(**et_params)
        clf.set_params(random_state=int(random_state) + int(fold))
        clf.fit(X[tr_m], y[tr_m])
        oof[va_m] = clf.predict_proba(X[va_m])[:, 1]
    if np.isnan(oof).any():
        return float("nan")
    return float(roc_auc_score(y, oof))


def main():
    parser = argparse.ArgumentParser(description="Y-randomization (permutation) OOF AUC null.")
    parser.add_argument(
        "--n-perm",
        type=int,
        default=499,
        help="Shuffled label draws (499 => min one-sided p≈0.002; use 999 for ~0.001).",
    )
    parser.add_argument("--n-estimators", type=int, default=400, help="Trees per fit (speed).")
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="Do not load IPCA pretrain artifacts (fit imputer/VT/scaler on train_dev only).",
    )
    args = parser.parse_args()

    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.TASK_TYPE = "classification"
    if bool(getattr(args, "no_pretrain", False)):
        cfg.FINAL_USE_PRETRAIN_ARTIFACTS = False
        _log("[y_rand] --no-pretrain: FINAL_USE_PRETRAIN_ARTIFACTS=False")
    _log("[y_rand] Loading supervised data …")
    prepared = build_shared_supervised_data(log=_log)
    X_all = prepared.X_all
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    fold_id = np.asarray(prepared.fold_id, dtype=int)

    X_td = X_all.iloc[train_dev_idx].values.astype(np.float64, copy=False)
    y_td = y[train_dev_idx]
    fold_td = fold_id[train_dev_idx]
    present_folds = sorted({int(f) for f in fold_td.tolist() if int(f) >= 0})

    et_params = dict(cfg.EXTRATREES_PARAMS)
    et_params["n_estimators"] = int(args.n_estimators)

    rs = int(cfg.RANDOM_STATE)
    real_auc = oof_auc_extratrees(X_td, y_td, fold_td, present_folds, et_params, rs)
    if np.isnan(real_auc):
        raise RuntimeError("Real OOF AUC is NaN — check folds / class balance.")

    rng = np.random.RandomState(rs)
    n_perm = int(args.n_perm)
    null_aucs = []
    for p in range(n_perm):
        y_perm = rng.permutation(y_td)
        a = oof_auc_extratrees(X_td, y_perm, fold_td, present_folds, et_params, rs + 1000 + p)
        null_aucs.append(a if not np.isnan(a) else 0.5)
        if (p + 1) % max(1, n_perm // 5) == 0:
            _log("[y_rand]  permutations %d / %d" % (p + 1, n_perm))

    null_aucs = np.asarray(null_aucs, dtype=np.float64)
    p_value = (float(np.sum(null_aucs >= real_auc)) + 1.0) / (float(n_perm) + 1.0)

    apply_journal_style()
    w, h = single_column_figsize(2.45)
    plt.figure(figsize=(w, h), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    plt.hist(null_aucs, bins=20, color="#9ecae1", edgecolor="white", density=True)
    plt.axvline(real_auc, color="#d62728", linewidth=1.6)
    plt.xlabel("AUROC")
    plt.ylabel("Density")
    plt.tight_layout()
    p_fig = out_dir / "y_randomization_oof_auc_null.png"
    plt.savefig(p_fig, bbox_inches="tight")
    plt.close()

    summary = {
        "task": "classification",
        "classification_threshold": float(cfg.CLASSIFICATION_THRESHOLD),
        "n_perm": n_perm,
        "n_estimators": int(args.n_estimators),
        "observed_oof_roc_auc": float(real_auc),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "null_auc_p95": float(np.percentile(null_aucs, 95)),
        "empirical_p_value_one_sided": float(p_value),
        "note": "p = (1 + #{null >= obs}) / (1 + Nperm). One-sided vs chance high AUC.",
    }
    with open(out_dir / "y_randomization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _log("[y_rand] Observed OOF AUC = %.4f" % real_auc)
    _log("[y_rand] Null mean AUC = %.4f  empirical p = %.4f" % (float(np.mean(null_aucs)), p_value))
    _log("[y_rand] Saved %s" % p_fig.name)


if __name__ == "__main__":
    main()
