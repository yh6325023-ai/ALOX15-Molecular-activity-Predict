#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute Hanley-McNeil ROC-AUC CIs and bootstrap PR-AUC CIs for OOF and test."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config as cfg
from src.ensemble import MeanEnsembleRegressor
from src.final_module import build_shared_supervised_data


def _predict_proba_pos(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def reconstruct_oof_scores(
    X: np.ndarray,
    fold_id: np.ndarray,
    train_dev_idx: np.ndarray,
    fold_models_dir: Path,
) -> np.ndarray:
    """Out-of-fold scores: each validation molecule scored by its held-out fold model."""
    oof = np.full(len(X), np.nan, dtype=np.float64)
    present_folds = sorted({int(f) for f in fold_id[train_dev_idx].tolist() if int(f) >= 0})
    for fold in present_folds:
        model_path = fold_models_dir / f"fold_{int(fold) + 1:02d}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing fold model: {model_path}")
        model = joblib.load(model_path)
        va_idx = np.where(fold_id == int(fold))[0]
        va_idx = va_idx[np.isin(va_idx, train_dev_idx)]
        if len(va_idx) == 0:
            continue
        oof[va_idx] = _predict_proba_pos(model, X[va_idx])
    td_oof = oof[train_dev_idx]
    if np.isnan(td_oof).any():
        raise RuntimeError("OOF reconstruction incomplete — check fold_id vs fold models.")
    return oof


def hanley_mcneil_ci(y: np.ndarray, score: np.ndarray, alpha: float = 0.05) -> dict:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("Need both classes for Hanley-McNeil CI.")
    auc = float(roc_auc_score(y, s))
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    num = (
        auc * (1.0 - auc)
        + (n1 - 1) * (q1 - auc * auc)
        + (n0 - 1) * (q2 - auc * auc)
    )
    se = float(np.sqrt(max(num, 0.0) / (n1 * n0)))
    z = 1.959963984540054  # 95% two-sided
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return {
        "auc": auc,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_pos": n1,
        "n_neg": n0,
        "method": "Hanley-McNeil",
    }


def bootstrap_pr_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    ap = float(average_precision_score(y, s))
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both classes for PR bootstrap CI.")
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=n_pos, replace=True)
        bn = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([bp, bn])
        boots.append(float(average_precision_score(y[idx], s[idx])))
    boots = np.asarray(boots, dtype=np.float64)
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "pr_auc": ap,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_boot": n_boot,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence_baseline": float(n_pos / (n_pos + n_neg)),
        "method": "stratified_bootstrap_percentile",
        "note": "Treats samples as independent; does not cluster by scaffold.",
    }


def main() -> None:
    cfg.TASK_TYPE = "classification"
    print("[ci] Loading data and ensemble …", flush=True)
    prepared = build_shared_supervised_data(log=print)
    y = (prepared.y_all_cont >= float(cfg.CLASSIFICATION_THRESHOLD)).astype(int).values
    train_dev_idx = np.asarray(prepared.train_dev_idx, dtype=int)
    test_idx = np.asarray(prepared.test_idx, dtype=int)

    ens_path = cfg.FINAL_ENSEMBLE_PATH.with_name(f"{cfg.FINAL_ENSEMBLE_PATH.stem}_cls.joblib")
    if not ens_path.is_file():
        raise FileNotFoundError(f"Missing ensemble: {ens_path}")
    ensemble: MeanEnsembleRegressor = joblib.load(ens_path)
    X = prepared.X_all.values.astype(np.float64, copy=False)
    fold_id = np.asarray(prepared.fold_id, dtype=int)

    oof_full = reconstruct_oof_scores(
        X,
        fold_id,
        train_dev_idx,
        cfg.FINAL_FOLD_MODELS_DIR / "cls",
    )
    s_oof = oof_full[train_dev_idx]
    s_te = ensemble.predict(X[test_idx])
    y_oof = y[train_dev_idx]
    y_te = y[test_idx]

    out_dir = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": str(getattr(cfg, "FINAL_CLASSIFIER_KIND", "extratrees")),
        "oof": {
            "n": int(len(y_oof)),
            "roc": hanley_mcneil_ci(y_oof, s_oof),
            "pr": bootstrap_pr_ci(y_oof, s_oof, seed=int(cfg.RANDOM_STATE)),
        },
        "test": {
            "n": int(len(y_te)),
            "roc": hanley_mcneil_ci(y_te, s_te),
            "pr": bootstrap_pr_ci(y_te, s_te, seed=int(cfg.RANDOM_STATE) + 1),
        },
    }

    pred_dir = cfg.RESULTS_DIR / "classification_candidate_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"{summary['model']}.npz",
        y_oof=y_oof.astype(np.float64),
        score_oof=s_oof.astype(np.float64),
        y_test=y_te.astype(np.float64),
        score_test=s_te.astype(np.float64),
    )

    out_path = out_dir / "auc_uncertainty_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[ci] OOF ROC-AUC = %.3f [%.3f, %.3f]" % (
        summary["oof"]["roc"]["auc"],
        summary["oof"]["roc"]["ci_low"],
        summary["oof"]["roc"]["ci_high"],
    ), flush=True)
    print("[ci] OOF PR-AUC  = %.3f [%.3f, %.3f] (baseline %.3f)" % (
        summary["oof"]["pr"]["pr_auc"],
        summary["oof"]["pr"]["ci_low"],
        summary["oof"]["pr"]["ci_high"],
        summary["oof"]["pr"]["prevalence_baseline"],
    ), flush=True)
    print("[ci] Test ROC-AUC = %.3f [%.3f, %.3f]" % (
        summary["test"]["roc"]["auc"],
        summary["test"]["roc"]["ci_low"],
        summary["test"]["roc"]["ci_high"],
    ), flush=True)
    print("[ci] Test PR-AUC  = %.3f [%.3f, %.3f] (baseline %.3f)" % (
        summary["test"]["pr"]["pr_auc"],
        summary["test"]["pr"]["ci_low"],
        summary["test"]["pr"]["ci_high"],
        summary["test"]["pr"]["prevalence_baseline"],
    ), flush=True)
    print("[ci] Saved %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
