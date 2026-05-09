# -*- coding: utf-8 -*-
"""Bar charts comparing classification candidate families (OOF vs holdout test)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_classification_comparison_figure(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 300,
) -> Path | None:
    """
    Save a single figure with ROC-AUC and PR-AUC (OOF + scaffold test) per model_family.
    Returns the path if written, else None when there is nothing to plot.
    """
    if summary_df.empty or "task_type" not in summary_df.columns:
        return None
    cls = summary_df[summary_df["task_type"] == "classification"].copy()
    if cls.empty or len(cls) < 1:
        return None
    need = {"model_family", "roc_auc_oof", "roc_auc_test", "pr_auc_oof", "pr_auc_test"}
    if not need.issubset(cls.columns):
        return None

    cls = cls.dropna(subset=["model_family"])
    cls = cls.sort_values("model_family")
    families = cls["model_family"].astype(str).tolist()
    if not families:
        return None

    x = np.arange(len(families), dtype=float)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=dpi, facecolor="white")

    roc_oof = cls["roc_auc_oof"].astype(float).values
    roc_te = cls["roc_auc_test"].astype(float).values
    pr_oof = cls["pr_auc_oof"].astype(float).values
    pr_te = cls["pr_auc_test"].astype(float).values

    ax0 = axes[0]
    ax0.bar(x - w / 2, roc_oof, width=w, label="OOF (train_dev)", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax0.bar(x + w / 2, roc_te, width=w, label="Scaffold test", color="#DD8452", edgecolor="white", linewidth=0.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels(families, rotation=25, ha="right")
    ax0.set_ylabel("ROC-AUC")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_title("Classification candidates — ROC-AUC")
    ax0.legend(frameon=False, fontsize=9)
    ax0.grid(axis="y", linestyle=":", alpha=0.4)

    ax1 = axes[1]
    ax1.bar(x - w / 2, pr_oof, width=w, label="OOF (train_dev)", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.bar(x + w / 2, pr_te, width=w, label="Scaffold test", color="#DD8452", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=25, ha="right")
    ax1.set_ylabel("PR-AUC (average precision)")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Classification candidates — PR-AUC")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    best_idx = int(np.nanargmax(roc_te)) if len(roc_te) else 0
    fig.suptitle(
        f"Best by scaffold-test ROC-AUC: {families[best_idx]} ({roc_te[best_idx]:.4f})",
        fontsize=11,
        y=1.02,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
