# -*- coding: utf-8 -*-
"""Overlay ROC + PR curves for multiple classification families (same y_true, different scores)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

# Default colors (similar to paper-style multi-curve figures)
DEFAULT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")

# Pretty names for legend
_DISPLAY_NAME = {
    "extratrees": "ExtraTrees",
    "gbr": "GBR",
    "lgbm": "LGBM",
    "randomforest": "RandomForest",
    "histgb": "HistGB",
    "bagging": "Bagging",
}


def _display_name(family: str) -> str:
    return _DISPLAY_NAME.get(family, family)


def save_classification_overlay_curves(
    pred_dir: Path,
    *,
    families: list[str],
    out_path: Path,
    split: str = "test",
    dpi: int = 300,
) -> Path | None:
    """
    Load one .npz per family from `pred_dir` / f"{family}.npz" with keys:
      y_test, score_test, y_oof, score_oof
    Choose split='test' or 'oof' for plotting.

    Saves one figure: left = ROC, right = PR (like reference panels J/K).
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        return None

    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for fam in families:
        fp = pred_dir / f"{fam}.npz"
        if not fp.is_file():
            continue
        data = np.load(fp, allow_pickle=False)
        if split == "test":
            y = data["y_test"]
            s = data["score_test"]
        else:
            y = data["y_oof"]
            s = data["score_oof"]
        y = np.asarray(y).reshape(-1)
        s = np.asarray(s).reshape(-1)
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        curves.append((str(fam), y, s))

    if len(curves) < 1:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=dpi, facecolor="white")

    # --- ROC ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, lw=1)
    for i, (fam, y, s) in enumerate(curves):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        label = f"{_display_name(fam)} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, lw=2.0, color=color, label=label, alpha=0.95)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    # --- PR ---
    ax = axes[1]
    for i, (fam, y, s) in enumerate(curves):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        prec, rec, _ = precision_recall_curve(y, s)
        ap = float(average_precision_score(y, s))
        yhat = (s >= 0.5).astype(int)
        f1 = float(f1_score(y, yhat, zero_division=0))
        label = f"{_display_name(fam)} (AP={ap:.3f}, F1={f1:.3f})"
        ax.plot(rec, prec, lw=2.0, color=color, label=label, alpha=0.95)
    pos_rate = float(np.mean(curves[0][1]))  # same y for all splits
    ax.axhline(y=pos_rate, color="k", linestyle="--", alpha=0.45, lw=1, label=f"Baseline = {pos_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    split_tag = "scaffold test" if split == "test" else "OOF (train_dev)"
    fig.suptitle(f"Classification candidates — {split_tag}", fontsize=12, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
