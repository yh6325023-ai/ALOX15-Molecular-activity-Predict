#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the full computational package for reviewer-facing feature-evidence:

  1) analyze_feature_evidence.py   — CV consistency + PDP
  2) analyze_shap_trees.py         — TreeSHAP (requires shap)
  3) analyze_bootstrap_rank_stability.py
  4) analyze_y_randomization.py
  5) analyze_ipca_supervised_overlap.py
  6) analyze_permutation_and_ablation.py — permutation ΔAUC + Top-K ablation OOF

All write under results/figures/feature_evidence/ (except logs).
"""

import subprocess
import sys
from pathlib import Path

import config as cfg

ROOT = Path(__file__).resolve().parent


def _run(script_name, extra_args=None):
    path = ROOT / script_name
    extra_args = list(extra_args or [])
    print("\n=== Running %s %s ===" % (script_name, extra_args), flush=True)
    subprocess.check_call([sys.executable, str(path)] + extra_args)


def main():
    pretrain_ok = cfg.PRETRAIN_PREPROCESSOR_PATH.is_file()
    if not pretrain_ok:
        print(
            "\n[note] Missing %s — using --no-pretrain for feature pipelines; skipping IPCA overlap."
            % cfg.PRETRAIN_PREPROCESSOR_PATH,
            flush=True,
        )
    no_pt = ["--no-pretrain"] if not pretrain_ok else []

    _run("analyze_feature_evidence.py", no_pt)
    _run("analyze_bootstrap_rank_stability.py", no_pt)
    _run("analyze_y_randomization.py", no_pt)
    if pretrain_ok:
        _run("analyze_ipca_supervised_overlap.py", [])
    else:
        print("\n[skip] analyze_ipca_supervised_overlap.py — no pretrain preprocessor", flush=True)

    try:
        import shap  # noqa: F401
        _run("analyze_shap_trees.py", no_pt)
        _run("analyze_shap_quantitative_dose_response.py", no_pt)
    except ImportError:
        print(
            "\n[skip] analyze_shap_trees.py — install shap: pip install \"shap>=0.43\"",
            flush=True,
        )
    _run("analyze_permutation_and_ablation.py", no_pt)
    print("\n[done] Reviewer evidence bundle finished.", flush=True)


if __name__ == "__main__":
    main()
