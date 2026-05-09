# -*- coding: utf-8 -*-
"""Train ALOX15 classifier (classification-only pipeline)."""
from pathlib import Path

import argparse

import pandas as pd

import config as cfg
from src.final_module import build_shared_supervised_data, run_final_training
from src.plot_classification_comparison import save_classification_comparison_figure
from src.plot_classification_overlay_curves import save_classification_overlay_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALOX15 classification models.")
    parser.add_argument(
        "--task",
        choices=("classification",),
        default="classification",
        help="Classification-only pipeline.",
    )
    args = parser.parse_args()
    task_type = args.task

    # Classification candidates: follow original "compare classifiers" step.
    cls_candidates = list(cfg.CLASSIFICATION_CANDIDATE_FAMILIES)

    summaries = []

    def _flatten_summary(s: dict) -> dict:
        # Make CSV inspection/manual selection easier (avoid nested dict columns).
        out = {
            "task_type": s.get("task_type"),
            "model_family": s.get("model_family"),
            "seconds": s.get("seconds"),
        }
        if s.get("task_type") == "classification":
            m_oof = s.get("metrics_oof") or {}
            m_te = s.get("metrics_test") or {}
            out.update(
                {
                    "roc_auc_oof": m_oof.get("roc_auc"),
                    "pr_auc_oof": m_oof.get("pr_auc"),
                    "roc_auc_test": m_te.get("roc_auc"),
                    "pr_auc_test": m_te.get("pr_auc"),
                }
            )
        return out

    # One shared feature/split build; run classification families only.
    shared = build_shared_supervised_data(log=print)
    for fam in cls_candidates:
        summaries.append(run_final_training("classification", fam, prepared_data=shared))

    out = pd.DataFrame([_flatten_summary(s) for s in summaries])
    out_path = Path(cfg.RESULTS_DIR) / "strict_scaffold_nested_candidates_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved candidate comparison table: {out_path}")

    fig_path = save_classification_comparison_figure(
        out,
        Path(cfg.FIGURES_FINAL_DIR) / "classification_family_comparison.png",
        dpi=int(cfg.FIGURE_DPI),
    )
    if fig_path is not None:
        print(f"Saved classification family comparison figure: {fig_path}")
    pred_dir = Path(cfg.RESULTS_DIR) / "classification_candidate_predictions"
    fams = [str(x) for x in cls_candidates]
    p_test = save_classification_overlay_curves(
        pred_dir,
        families=fams,
        out_path=Path(cfg.FIGURES_FINAL_DIR) / "classification_overlay_test_roc_pr.png",
        split="test",
        dpi=int(cfg.FIGURE_DPI),
    )
    if p_test is not None:
        print(f"Saved classification overlay (test): {p_test}")
    p_oof = save_classification_overlay_curves(
        pred_dir,
        families=fams,
        out_path=Path(cfg.FIGURES_FINAL_DIR) / "classification_overlay_oof_roc_pr.png",
        split="oof",
        dpi=int(cfg.FIGURE_DPI),
    )
    if p_oof is not None:
        print(f"Saved classification overlay (OOF): {p_oof}")
