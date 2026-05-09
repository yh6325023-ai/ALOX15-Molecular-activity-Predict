# -*- coding: utf-8 -*-
"""Classification benchmark (grouped split + nested tuning on outer-train only)."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
from src.final_module import build_shared_supervised_data, run_final_training
from src.plot_classification_comparison import save_classification_comparison_figure
from src.plot_classification_overlay_curves import save_classification_overlay_curves


def main() -> None:
    benchmark_dir = Path(cfg.RESULTS_DIR) / "benchmark"
    benchmark_fig_dir = benchmark_dir / "figures"
    benchmark_pred_dir = benchmark_dir / "classification_candidate_predictions"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_fig_dir.mkdir(parents=True, exist_ok=True)
    benchmark_pred_dir.mkdir(parents=True, exist_ok=True)

    families = list(cfg.CLASSIFICATION_CANDIDATE_FAMILIES)
    shared = build_shared_supervised_data(log=print)
    summaries = []
    failed = []
    for fam in families:
        try:
            summaries.append(run_final_training("classification", fam, prepared_data=shared))
            src_npz = Path(cfg.RESULTS_DIR) / "classification_candidate_predictions" / f"{fam}.npz"
            dst_npz = benchmark_pred_dir / f"{fam}.npz"
            if src_npz.is_file():
                shutil.copy2(src_npz, dst_npz)
        except Exception as exc:  # pragma: no cover
            failed.append((fam, str(exc)))
            print(f"[benchmark] model failed and will be skipped: {fam} -> {exc}")

    rows = []
    for s in summaries:
        m_oof = s.get("metrics_oof") or {}
        m_te = s.get("metrics_test") or {}
        rows.append(
            {
                "task_type": s.get("task_type"),
                "model_family": s.get("model_family"),
                "seconds": s.get("seconds"),
                "roc_auc_oof": m_oof.get("roc_auc"),
                "pr_auc_oof": m_oof.get("pr_auc"),
                "roc_auc_test": m_te.get("roc_auc"),
                "pr_auc_test": m_te.get("pr_auc"),
                "f1_oof": m_oof.get("f1"),
                "bacc_oof": m_oof.get("balanced_acc"),
                "mcc_oof": m_oof.get("mcc"),
                "precision_oof": m_oof.get("precision"),
                "recall_oof": m_oof.get("recall"),
                "accuracy_oof": m_oof.get("accuracy"),
                "f1_test": m_te.get("f1"),
                "bacc_test": m_te.get("balanced_acc"),
                "mcc_test": m_te.get("mcc"),
                "precision_test": m_te.get("precision"),
                "recall_test": m_te.get("recall"),
                "accuracy_test": m_te.get("accuracy"),
            }
        )
    out = pd.DataFrame(rows)
    # Rank by OOF metrics (more stable than a single small holdout test split).
    # Tie-breaker uses OOF PR-AUC, then test metrics for reference.
    out = out.sort_values(
        ["roc_auc_oof", "pr_auc_oof", "roc_auc_test", "pr_auc_test"],
        ascending=False,
    ).reset_index(drop=True)
    out_path = benchmark_dir / "classification_benchmark_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved classification benchmark table: {out_path}")
    if failed:
        print(f"[benchmark] skipped {len(failed)} failed model(s); not writing failed-model CSV by design.")

    # Top-30 ranking figure (auto-truncated if fewer than 30 models)
    top_n = min(30, len(out))
    if top_n > 0:
        top = out.iloc[:top_n].copy()
        top = top.iloc[::-1]  # best at bottom for horizontal bar readability
        plt.figure(figsize=(10, max(6, 0.35 * top_n)), dpi=int(cfg.FIGURE_DPI), facecolor="white")
        plt.barh(top["model_family"], top["roc_auc_oof"], alpha=0.85)
        plt.xlabel("OOF ROC-AUC")
        plt.title(f"Top {top_n} Classification Models by OOF ROC-AUC")
        plt.tight_layout()
        top_fig = benchmark_fig_dir / "classification_top30_roc_auc_oof.png"
        top_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(top_fig, bbox_inches="tight")
        plt.close()
        print(f"Saved Top-{top_n} ROC-AUC ranking figure: {top_fig}")

        # Side-by-side OOF/Test ROC-AUC bars (each panel sorted high -> low).
        fig_h = max(6, 0.35 * top_n)
        top_oof = out.sort_values("roc_auc_oof", ascending=False).head(top_n).iloc[::-1]
        top_test = out.sort_values("roc_auc_test", ascending=False).head(top_n).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_h), dpi=int(cfg.FIGURE_DPI), facecolor="white")
        axes[0].barh(top_oof["model_family"], top_oof["roc_auc_oof"], alpha=0.85, color="#4C78A8")
        axes[0].set_xlabel("OOF ROC-AUC")
        axes[0].set_title(f"Top {top_n} — OOF ROC-AUC")
        axes[0].set_xlim(0.0, 1.0)

        axes[1].barh(top_test["model_family"], top_test["roc_auc_test"], alpha=0.85, color="#F58518")
        axes[1].set_xlabel("Test ROC-AUC")
        axes[1].set_title(f"Top {top_n} — Test ROC-AUC")
        axes[1].set_xlim(0.0, 1.0)

        plt.tight_layout()
        dual_roc_fig = benchmark_fig_dir / "classification_top30_roc_auc_oof_vs_test.png"
        dual_roc_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dual_roc_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved OOF/Test ROC-AUC comparison figure: {dual_roc_fig}")

        # Side-by-side OOF/Test PR-AUC bars (each panel sorted high -> low).
        top_oof_pr = out.sort_values("pr_auc_oof", ascending=False).head(top_n).iloc[::-1]
        top_test_pr = out.sort_values("pr_auc_test", ascending=False).head(top_n).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_h), dpi=int(cfg.FIGURE_DPI), facecolor="white")
        axes[0].barh(top_oof_pr["model_family"], top_oof_pr["pr_auc_oof"], alpha=0.85, color="#54A24B")
        axes[0].set_xlabel("OOF PR-AUC (AP)")
        axes[0].set_title(f"Top {top_n} — OOF PR-AUC")
        axes[0].set_xlim(0.0, 1.0)

        axes[1].barh(top_test_pr["model_family"], top_test_pr["pr_auc_test"], alpha=0.85, color="#E45756")
        axes[1].set_xlabel("Test PR-AUC (AP)")
        axes[1].set_title(f"Top {top_n} — Test PR-AUC")
        axes[1].set_xlim(0.0, 1.0)

        plt.tight_layout()
        dual_pr_fig = benchmark_fig_dir / "classification_top30_pr_auc_oof_vs_test.png"
        dual_pr_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dual_pr_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved OOF/Test PR-AUC comparison figure: {dual_pr_fig}")

        # Generalization gap (OOF - Test); closer to 0 is usually more stable.
        gap = top.copy()
        gap["roc_auc_gap_oof_minus_test"] = gap["roc_auc_oof"] - gap["roc_auc_test"]
        gap["pr_auc_gap_oof_minus_test"] = gap["pr_auc_oof"] - gap["pr_auc_test"]
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_h), dpi=int(cfg.FIGURE_DPI), facecolor="white")
        axes[0].barh(gap["model_family"], gap["roc_auc_gap_oof_minus_test"], color="#72B7B2", alpha=0.9)
        axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)
        axes[0].set_xlabel("OOF - Test ROC-AUC")
        axes[0].set_title("Generalization Gap (ROC-AUC)")

        axes[1].barh(gap["model_family"], gap["pr_auc_gap_oof_minus_test"], color="#B279A2", alpha=0.9)
        axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("OOF - Test PR-AUC")
        axes[1].set_title("Generalization Gap (PR-AUC)")

        plt.tight_layout()
        gap_fig = benchmark_fig_dir / "classification_top30_generalization_gap.png"
        gap_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(gap_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved generalization gap figure: {gap_fig}")

    fig_path = save_classification_comparison_figure(
        out, benchmark_fig_dir / "classification_family_comparison.png", dpi=int(cfg.FIGURE_DPI)
    )
    if fig_path is not None:
        print(f"Saved classification family comparison figure: {fig_path}")

    pred_dir = benchmark_pred_dir
    p_test = save_classification_overlay_curves(
        pred_dir=pred_dir,
        families=families,
        out_path=benchmark_fig_dir / "classification_overlay_test_roc_pr.png",
        split="test",
        dpi=int(cfg.FIGURE_DPI),
    )
    if p_test is not None:
        print(f"Saved classification overlay (test): {p_test}")
    p_oof = save_classification_overlay_curves(
        pred_dir=pred_dir,
        families=families,
        out_path=benchmark_fig_dir / "classification_overlay_oof_roc_pr.png",
        split="oof",
        dpi=int(cfg.FIGURE_DPI),
    )
    if p_oof is not None:
        print(f"Saved classification overlay (OOF): {p_oof}")


if __name__ == "__main__":
    main()
