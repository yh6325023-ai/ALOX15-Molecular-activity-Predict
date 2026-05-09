#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

import config as cfg


def _metrics_cls(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    yhat = (s >= float(threshold)).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bacc = 0.5 * (tpr + tnr)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {"f1": f1, "mcc": mcc, "bacc": bacc}


def _default_family() -> str:
    return str(getattr(cfg, "FINAL_CLASSIFIER_KIND", "extratrees"))


def main() -> None:
    p = argparse.ArgumentParser(description="Plot additional model-selection evidence figures.")
    p.add_argument("--family", default=_default_family(), help="Model family name used in benchmark outputs.")
    args = p.parse_args()

    fam = str(args.family)
    pred_npz = cfg.RESULTS_DIR / "classification_candidate_predictions" / f"{fam}.npz"
    if not pred_npz.is_file():
        raise FileNotFoundError(f"Missing prediction bundle: {pred_npz}")
    arr = np.load(pred_npz)
    y_oof = np.asarray(arr["y_oof"], dtype=np.int32)
    s_oof = np.asarray(arr["score_oof"], dtype=np.float64)
    y_te = np.asarray(arr["y_test"], dtype=np.int32)
    s_te = np.asarray(arr["score_test"], dtype=np.float64)

    fig_dir = cfg.FIGURES_FINAL_DIR / "selection_evidence"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Calibration curve (OOF vs Test)
    plt.figure(figsize=(6, 6), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    frac_oof, prob_oof = calibration_curve(y_oof, s_oof, n_bins=10, strategy="quantile")
    frac_te, prob_te = calibration_curve(y_te, s_te, n_bins=10, strategy="quantile")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7, label="Perfect calibration")
    plt.plot(prob_oof, frac_oof, marker="o", label="OOF")
    plt.plot(prob_te, frac_te, marker="o", label="Test")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"Calibration Curve — {fam}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{fam}_calibration_oof_test.png", bbox_inches="tight")
    plt.close()

    # 2) Threshold-vs-metrics (OOF)
    thresholds = np.linspace(0.01, 0.99, 99)
    f1 = []
    mcc = []
    bacc = []
    for t in thresholds:
        m = _metrics_cls(y_oof, s_oof, threshold=float(t))
        f1.append(m["f1"])
        mcc.append(m["mcc"])
        bacc.append(m["bacc"])
    f1 = np.asarray(f1, dtype=float)
    mcc = np.asarray(mcc, dtype=float)
    bacc = np.asarray(bacc, dtype=float)
    best_f1_idx = int(np.nanargmax(f1))
    best_mcc_idx = int(np.nanargmax(mcc))

    plt.figure(figsize=(7, 5), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    plt.plot(thresholds, f1, label="F1")
    plt.plot(thresholds, mcc, label="MCC")
    plt.plot(thresholds, bacc, label="Balanced Accuracy")
    plt.axvline(thresholds[best_f1_idx], color="k", linestyle="--", alpha=0.6, label=f"Best F1={thresholds[best_f1_idx]:.2f}")
    plt.axvline(thresholds[best_mcc_idx], color="gray", linestyle=":", alpha=0.8, label=f"Best MCC={thresholds[best_mcc_idx]:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(f"Threshold Sensitivity (OOF) — {fam}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{fam}_threshold_sensitivity_oof.png", bbox_inches="tight")
    plt.close()

    # 3) Fold stability from metadata (if present)
    meta_path = cfg.FINAL_TRAINING_METADATA_PATH.with_name(f"{cfg.FINAL_TRAINING_METADATA_PATH.stem}_cls.joblib")
    if meta_path.is_file():
        meta = joblib.load(meta_path)
        fm = meta.get("fold_metrics") or {}
        roc_by_fold = fm.get("roc_auc_by_fold") or []
        pr_by_fold = fm.get("pr_auc_by_fold") or []
        if roc_by_fold and pr_by_fold:
            x = np.arange(1, len(roc_by_fold) + 1)
            plt.figure(figsize=(6, 4), dpi=int(cfg.FIGURE_DPI), facecolor="white")
            plt.plot(x, roc_by_fold, marker="o", label="ROC-AUC")
            plt.plot(x, pr_by_fold, marker="o", label="PR-AUC")
            plt.xticks(x, [f"F{i}" for i in x])
            plt.ylim(0.0, 1.0)
            plt.xlabel("Fold")
            plt.ylabel("AUC")
            plt.title(f"Fold Stability — {fam}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"{fam}_fold_stability_auc.png", bbox_inches="tight")
            plt.close()

    # 4) Keep a compact ROC/PR curve snapshot for the selected family
    fpr_oof, tpr_oof, _ = roc_curve(y_oof, s_oof)
    fpr_te, tpr_te, _ = roc_curve(y_te, s_te)
    prc_oof, rec_oof, _ = precision_recall_curve(y_oof, s_oof)
    prc_te, rec_te, _ = precision_recall_curve(y_te, s_te)
    plt.figure(figsize=(12, 5), dpi=int(cfg.FIGURE_DPI), facecolor="white")
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fpr_oof, tpr_oof, label="OOF")
    ax1.plot(fpr_te, tpr_te, label="Test")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(rec_oof, prc_oof, label="OOF")
    ax2.plot(rec_te, prc_te, label="Test")
    ax2.set_title("PR")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{fam}_roc_pr_oof_test_snapshot.png", bbox_inches="tight")
    plt.close()

    print(f"Saved selection-evidence figures under: {fig_dir}")


if __name__ == "__main__":
    main()

