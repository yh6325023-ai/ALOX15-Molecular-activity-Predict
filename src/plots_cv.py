# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_cv_roc(
    fold_curves: List[Tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
    title: str,
    dpi: int = 300,
) -> None:
    """
    fold_curves: list of (fpr, tpr, fold_auc)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7), dpi=dpi, facecolor="white")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1)

    mean_fpr = np.linspace(0, 1, 500)
    tprs = []
    for i, (fpr, tpr, fold_auc) in enumerate(fold_curves, start=1):
        plt.plot(fpr, tpr, lw=1.6, alpha=0.85, label=f"Fold {i} (AUC = {fold_auc:.4f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    mean_auc = float(auc(mean_fpr, mean_tpr))
    plt.plot(mean_fpr, mean_tpr, color="brown", lw=2.4, label=f"Mean ROC (AUC = {mean_auc:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_cv_pr(
    fold_curves: List[Tuple[np.ndarray, np.ndarray, float]],
    baseline: float,
    out_path: Path,
    title: str,
    dpi: int = 300,
) -> None:
    """
    fold_curves: list of (recall, precision, fold_ap) where fold_ap = auc(recall, precision)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7), dpi=dpi, facecolor="white")
    plt.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, lw=1, label=f"Baseline = {baseline:.3f}")

    mean_rec = np.linspace(0, 1, 500)
    precs = []
    fold_aps = []
    for i, (rec, prec, fold_ap) in enumerate(fold_curves, start=1):
        plt.plot(rec, prec, lw=1.6, alpha=0.85, label=f"Fold {i} (AP = {fold_ap:.4f})")
        # Ensure recall is monotonic ascending before interpolation.
        # sklearn precision_recall_curve may return descending recall arrays.
        order = np.argsort(rec)
        rec_sorted = np.asarray(rec)[order]
        prec_sorted = np.asarray(prec)[order]
        precs.append(np.interp(mean_rec, rec_sorted, prec_sorted))
        fold_aps.append(float(fold_ap))

    mean_prec = np.mean(np.vstack(precs), axis=0)
    # Report mean AP as average over folds (more interpretable for CV),
    # while plotting the averaged PR curve for visual reference.
    mean_ap = float(np.mean(fold_aps)) if fold_aps else float("nan")
    plt.plot(
        mean_rec,
        mean_prec,
        color="brown",
        lw=2.0,
        linestyle="--",
        alpha=0.8,
        label=f"Mean PR (mean AP = {mean_ap:.4f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_fold_roc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(auc(fpr, tpr))


def compute_fold_pr(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    # For PR AUC, integrate precision over recall
    return rec, prec, float(auc(rec, prec))

