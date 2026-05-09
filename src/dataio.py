# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np


def load_smi_unlabeled(path, smiles_col: str = "Smiles", max_rows: int | None = None, sampling_seed: int = 42) -> pd.DataFrame:
    """Load unlabeled `.smi` and return one-column DataFrame with SMILES."""
    import random

    rng = random.Random(sampling_seed)
    rows = []
    seen = 0

    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith("#"):
                        continue
                    line = line.replace(",", " ")
                    toks = line.split()
                    if len(toks) < 1:
                        continue
                    smi = toks[0].strip()
                    if not smi:
                        continue
                    seen += 1
                    if max_rows is None:
                        rows.append(smi)
                    else:
                        if len(rows) < max_rows:
                            rows.append(smi)
                        else:
                            j = rng.randint(0, seen - 1)
                            if j < max_rows:
                                rows[j] = smi
            break
        except FileNotFoundError:
            raise
        except Exception:
            continue

    df = pd.DataFrame({smiles_col: rows})
    if df.empty:
        raise ValueError(f"No valid SMILES rows found in file: {path}")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col].astype(bool)].reset_index(drop=True)
    return df


def load_smi_file(path, smiles_col: str = "Smiles", target_col: str = "PIC50") -> pd.DataFrame:
    """
    Load `.smi` / whitespace separated file containing SMILES and a numeric target.

    Supported formats (per line):
    - `SMILES PIC50`
    - `SMILES ... PIC50` (PIC50 assumed to be the last numeric token)

    Comment lines starting with `#` and empty lines are ignored.
    """
    rows = []
    # Try utf-8 first, then fallback to gb18030 (for some Chinese exports)
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith("#"):
                        continue
                    # normalize delimiters: commas -> whitespace
                    line = line.replace(",", " ")
                    toks = line.split()
                    if len(toks) < 2:
                        continue
                    smi = toks[0]
                    # target is the last numeric token in the line
                    y = None
                    for t in reversed(toks[1:]):
                        try:
                            y = float(t)
                            break
                        except ValueError:
                            continue
                    if y is None:
                        continue
                    rows.append((smi, y))
            break
        except FileNotFoundError:
            raise
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=[smiles_col, target_col])
    if df.empty:
        raise ValueError(f"No valid rows found in SMILES file: {path}")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col].astype(bool)]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df


def normalize_df(df: pd.DataFrame, smiles_col: str, target_col: str) -> pd.DataFrame:
    """Map common column names to canonical ``Smiles`` / ``PIC50``."""
    colmap = {c.lower().strip(): c for c in df.columns}
    out = df.copy()
    if smiles_col not in out.columns:
        for key in ("smiles", "canonical_smiles", "structure_smiles"):
            if key in colmap:
                src = colmap[key]
                out = out.rename(columns={src: smiles_col})
                break
    if target_col not in out.columns:
        for key in ("pic50", "pic50_value", "pchembl_value", "value"):
            if key in colmap:
                src = colmap[key]
                out = out.rename(columns={src: target_col})
                break
    return out


def load_activity_table(path, smiles_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_df(df, smiles_col, target_col)
    missing = [c for c in (smiles_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")
    df = df[[smiles_col, target_col]].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col].astype(bool)]
    df = df.dropna(subset=[target_col])
    return df.reset_index(drop=True)


def aggregate_duplicates(df: pd.DataFrame, smiles_col: str, target_col: str, how: str) -> pd.DataFrame:
    if how == "mean":
        return df.groupby(smiles_col, as_index=False)[target_col].mean()
    if how == "first":
        return df.drop_duplicates(subset=[smiles_col], keep="first")
    raise ValueError(f"Unknown duplicate aggregation: {how}")
