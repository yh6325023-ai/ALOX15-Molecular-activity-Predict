# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as cfg
from src.dataio import load_smi_unlabeled
from src.features import build_feature_matrix


def _log(msg: str) -> None:
    print(msg, flush=True)


def _identity_features(X):
    """Passthrough for imputer/variance steps; must be a top-level function for joblib pickle."""
    return X


def _find_pretrain_smi_path() -> Path:
    if cfg.PRETRAIN_SMI_PATH is not None:
        p = Path(cfg.PRETRAIN_SMI_PATH)
        if p.is_file():
            return p
        raise FileNotFoundError(f"cfg.PRETRAIN_SMI_PATH not found: {p}")

    candidates: list[Path] = []
    for d in cfg.PRETRAIN_SMI_SEARCH_DIRS:
        if d.is_dir():
            candidates.extend(list(d.glob("*.smi")))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "No pretrain `.smi` file found. Put your SMILES+PIC50 file under:\n"
            f"- {cfg.PRETRAIN_SMI_SEARCH_DIRS}\n"
            "or set cfg.PRETRAIN_SMI_PATH explicitly."
        )
    _log(f"Auto-selected pretrain file: {candidates[0]}")
    return candidates[0]


def run_pretraining() -> None:
    t0 = time.time()
    sns.set_style("white")

    pretrain_path = _find_pretrain_smi_path()
    _log(f"Loading unlabeled pretrain SMILES from: {pretrain_path}")
    df_pre = load_smi_unlabeled(
        pretrain_path,
        smiles_col=cfg.SMILES_COL,
        max_rows=cfg.PRETRAIN_MAX_ROWS,
        sampling_seed=cfg.PRETRAIN_SAMPLING_SEED,
    )
    df_pre = df_pre.drop_duplicates(subset=[cfg.SMILES_COL]).reset_index(drop=True)
    _log(f"Pretrain SMILES loaded: {len(df_pre)}")

    # 1) Stream-featurize unlabeled molecules in chunks to avoid OOM on 3M rows
    _log("Featurizing unlabeled pretrain SMILES (Morgan + MACCS) in streaming mode …")
    stream_batch_size = int(getattr(cfg, "PRETRAIN_STREAM_BATCH_SIZE", 20000))
    if stream_batch_size <= 0:
        raise ValueError("PRETRAIN_STREAM_BATCH_SIZE must be > 0")

    master_cols: list[str] | None = None
    n_valid = 0

    # 2) Preprocess fit (streaming): imputer -> variance filter -> scale
    # For very large pretraining with Morgan+MACCS, no NaNs are expected.
    # Keep identity imputer/selector for compatibility with downstream modules.
    imputer = FunctionTransformer(_identity_features, validate=False)
    selector = FunctionTransformer(_identity_features, validate=False)
    scaler = StandardScaler(with_mean=True, with_std=True)

    smiles_series = df_pre[cfg.SMILES_COL]
    n_total = len(smiles_series)
    n_batches = (n_total + stream_batch_size - 1) // stream_batch_size

    for b in range(n_batches):
        start = b * stream_batch_size
        end = min((b + 1) * stream_batch_size, n_total)
        X_chunk = build_feature_matrix(
            smiles_series.iloc[start:end],
            morgan_radius=cfg.MORGAN_RADIUS,
            morgan_n_bits=cfg.MORGAN_N_BITS,
            master_columns=master_cols,
            log=_log,
        )
        if X_chunk.empty:
            continue
        if master_cols is None:
            master_cols = list(X_chunk.columns)

        X_np = X_chunk.values.astype(np.float32, copy=False)
        scaler.partial_fit(X_np)
        n_valid += X_np.shape[0]
        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            _log(f"Scaler pass progress: batch {b + 1}/{n_batches}, valid rows={n_valid}")

    if n_valid == 0 or master_cols is None:
        raise RuntimeError("No valid molecules could be featurized in pretraining.")

    selected_cols = list(master_cols)
    _log(f"Featurized pretrain rows: {n_valid}; cols(before filter): {len(master_cols)}")

    # 3) Unsupervised representation model (IncrementalPCA) fitted in streaming pass
    repr_dim = max(2, min(int(cfg.PRETRAIN_REPR_DIM), len(selected_cols)))
    _log(f"Fitting unsupervised representation (IncrementalPCA, dim={repr_dim}) …")
    ipca = IncrementalPCA(n_components=repr_dim, batch_size=int(cfg.PRETRAIN_IPCA_BATCH_SIZE))

    n_ipca_rows = 0
    for b in range(n_batches):
        start = b * stream_batch_size
        end = min((b + 1) * stream_batch_size, n_total)
        X_chunk = build_feature_matrix(
            smiles_series.iloc[start:end],
            morgan_radius=cfg.MORGAN_RADIUS,
            morgan_n_bits=cfg.MORGAN_N_BITS,
            master_columns=master_cols,
            log=_log,
        )
        if X_chunk.empty:
            continue

        X_np = X_chunk.values.astype(np.float32, copy=False)
        X_scaled_batch = scaler.transform(X_np)
        ipca.partial_fit(X_scaled_batch)
        n_ipca_rows += X_scaled_batch.shape[0]
        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            _log(f"IPCA pass progress: batch {b + 1}/{n_batches}, valid rows={n_ipca_rows}")

    if n_ipca_rows == 0:
        raise RuntimeError("IPCA received zero valid rows.")

    # 4) Save artifacts (unsupervised transfer module)
    preprocessor = {
        "imputer": imputer,
        "variance_selector": selector,
        "scaler": scaler,
        "projector": ipca,
        "master_columns": master_cols,
        "selected_columns": selected_cols,
        "repr_dim": repr_dim,
        "morgan_radius": cfg.MORGAN_RADIUS,
        "morgan_n_bits": cfg.MORGAN_N_BITS,
    }

    joblib.dump(preprocessor, cfg.PRETRAIN_PREPROCESSOR_PATH)
    joblib.dump(ipca, cfg.PRETRAIN_REPR_MODEL_PATH)
    joblib.dump(
        {
            "n_unlabeled": int(n_valid),
            "n_features_raw": int(len(master_cols)),
            "n_features_selected": int(len(selected_cols)),
            "repr_dim": int(repr_dim),
            "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
            "seconds": float(time.time() - t0),
        },
        cfg.PRETRAIN_TRAINING_METADATA_PATH,
    )

    # 5) Save figures
    fig_dir = cfg.FIGURES_PRETRAIN_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Explained variance curve
    plt.figure(figsize=(6, 4), dpi=cfg.FIGURE_DPI, facecolor="white")
    csum = np.cumsum(ipca.explained_variance_ratio_)
    plt.plot(np.arange(1, len(csum) + 1), csum, color="#3b6fb6", linewidth=2)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Unsupervised pretrain representation quality")
    plt.tight_layout()
    plt.savefig(fig_dir / "pretrain_ipca_explained_variance.png", bbox_inches="tight")
    plt.close()

    # 2D embedding preview (first two components) on a manageable sample
    n_preview = min(n_total, 10000)
    preview_smiles = smiles_series.iloc[:n_preview]
    X_preview = build_feature_matrix(
        preview_smiles,
        morgan_radius=cfg.MORGAN_RADIUS,
        morgan_n_bits=cfg.MORGAN_N_BITS,
        master_columns=master_cols,
        log=_log,
    )
    if X_preview.empty:
        raise RuntimeError("Could not build preview embedding: no valid preview rows.")
    X_preview_scaled = scaler.transform(X_preview.values.astype(np.float32, copy=False))
    X_emb = ipca.transform(X_preview_scaled)
    n_show = min(len(X_emb), 10000)
    idx = np.random.RandomState(cfg.RANDOM_STATE).choice(len(X_emb), size=n_show, replace=False)
    plt.figure(figsize=(6, 6), dpi=cfg.FIGURE_DPI, facecolor="white")
    plt.scatter(X_emb[idx, 0], X_emb[idx, 1], s=6, alpha=0.35, c="#1f77b4", edgecolors="none")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Unlabeled chemical-space embedding (sample)")
    plt.tight_layout()
    plt.savefig(fig_dir / "pretrain_embedding_preview.png", bbox_inches="tight")
    plt.close()

    _log(f"Pretraining artifacts saved under: {cfg.MODELS_DIR}")
    _log(
        "Unsupervised pretraining done. Main training can now use projector features for fine-tuning."
    )
    _log(f"Pretraining finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_pretraining()

