# -*- coding: utf-8 -*-
"""Shared feature matrix for benchmarks (Morgan/MACCS/PaDEL + optional pretrain repr)."""
from __future__ import annotations

import joblib
import pandas as pd

import config as cfg
from src.dataio import aggregate_duplicates, load_activity_table
from src.features import build_feature_matrix


def load_benchmark_xy(
    *,
    log=print,
) -> tuple[pd.DataFrame, pd.Series]:
    df = load_activity_table(cfg.MAIN_DATA_CSV, cfg.SMILES_COL, cfg.TARGET_COL)
    if cfg.DROP_DUPLICATE_SMILES:
        df = aggregate_duplicates(df, cfg.SMILES_COL, cfg.TARGET_COL, cfg.DUPLICATE_AGG)

    master_cols = None
    pre = None
    if cfg.BENCHMARK_USE_PRETRAIN_REPR and cfg.PRETRAIN_PREPROCESSOR_PATH.is_file():
        pre = joblib.load(cfg.PRETRAIN_PREPROCESSOR_PATH)
        master_cols = pre["master_columns"]

    X_raw = build_feature_matrix(
        df[cfg.SMILES_COL],
        morgan_radius=cfg.MORGAN_RADIUS,
        morgan_n_bits=cfg.MORGAN_N_BITS,
        master_columns=master_cols,
        log=log,
    )
    y = df.loc[X_raw.index, cfg.TARGET_COL].astype(float)

    if pre is None:
        return X_raw, y

    X_imp = pre["imputer"].transform(X_raw)
    X_sel = pre["variance_selector"].transform(X_imp)
    X_scaled = pre["scaler"].transform(X_sel)
    X_repr = pre["projector"].transform(X_scaled)
    X_repr_df = pd.DataFrame(
        X_repr,
        index=X_raw.index,
        columns=[f"repr_{i}" for i in range(X_repr.shape[1])],
    )
    X_sel_df = pd.DataFrame(X_sel, index=X_raw.index, columns=list(pre["selected_columns"]))
    return pd.concat([X_sel_df, X_repr_df], axis=1), y
