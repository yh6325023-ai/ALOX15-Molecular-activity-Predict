ALOX15 — reviewer-facing feature-evidence scripts (integrated layout)

Layout in this repository:
  Reviewer *.py scripts -> repository root (same directory as train.py, run_benchmark.py, …)
  publication_figures.py -> src/publication_figures.py

Prerequisite: pip install "shap>=0.43" (also listed in full repo requirements.txt).

Entry point: python run_reviewer_evidence_bundle.py
Outputs go to results/figures/feature_evidence/ (create locally; see full project .gitignore policy).

Scripts:
run_reviewer_evidence_bundle.py, analyze_feature_evidence.py, analyze_bootstrap_rank_stability.py, analyze_y_randomization.py, analyze_shap_trees.py, analyze_shap_quantitative_dose_response.py, analyze_ipca_supervised_overlap.py, analyze_permutation_and_ablation.py, prune_feature_evidence.py
src/publication_figures.py
