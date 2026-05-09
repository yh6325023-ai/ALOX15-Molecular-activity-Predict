#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove stray / obsolete files under results/figures/feature_evidence/.

Keeps only the reproducible reviewer-evidence bundle (figures + tables + JSON).
Safe to re-run; does not touch other result directories.
"""

from __future__ import annotations

from pathlib import Path

import config as cfg

KEEP = frozenset(
    {
        # Figures (publication + SI)
        "feature_importance_fold_rank_correlation.png",
        "bootstrap_importance_rank_correlation.png",
        "y_randomization_oof_auc_null.png",
        "partial_dependence_top_features.png",
        "feature_importance_cumulative_gini.png",
        "shap_summary_beeswarm.png",
        "shap_fold_meanabs_rank_correlation.png",
        "shap_dependence_top_features.png",
        "ipca_supervised_top_overlap.png",
        "feature_ablation_scaffold_cv_val_rocauc.png",
        # Tables / machine-readable summaries
        "feature_importance_top_summary.csv",
        "feature_importance_stability.json",
        "bootstrap_top_feature_frequency.csv",
        "bootstrap_rank_stability.json",
        "y_randomization_summary.json",
        "ipca_loading_top_features.csv",
        "ipca_supervised_top_overlap.csv",
        "ipca_supervised_overlap.json",
        "shap_top_mean_abs_summary.csv",
        "shap_stability.json",
        "shap_dose_response_spearman_top20.csv",
        "shap_quantitative_dose_response.json",
        "permutation_importance_top_summary.csv",
        "TableS3_permutation_importance_top_summary.csv",
        "permutation_vs_gini.json",
        "TableS3_permutation_vs_gini.json",
        "ablation_top_features_oof.json",
        "TableS4_shap_quantitative_dose_response.json",
    }
)

# Any top-k from analyze_shap_quantitative_dose_response.py --top-k
_TABLE_S4_SPEARMAN_PREFIX = "TableS4_shap_dose_response_spearman_top"


def main():
    d = cfg.RESULTS_DIR / "figures" / "feature_evidence"
    if not d.is_dir():
        print("Nothing to prune (missing %s)" % d)
        return
    removed = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        if p.name in KEEP:
            continue
        if p.name.startswith(_TABLE_S4_SPEARMAN_PREFIX) and p.suffix.lower() == ".csv":
            continue
        try:
            p.unlink()
            removed.append(p.name)
        except OSError as e:
            print("Could not remove %s: %s" % (p, e))
    if removed:
        print("Removed %d file(s):" % len(removed))
        for n in removed:
            print("  ", n)
    else:
        print("No extra files under %s (already clean)." % d)


if __name__ == "__main__":
    main()
