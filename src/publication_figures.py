# -*- coding: utf-8 -*-
"""Journal-style matplotlib defaults and figure sizes for submission-quality plots."""

from __future__ import annotations

import matplotlib as mpl

import config as cfg


def apply_journal_style() -> None:
    """
    Sans-serif, 7–8 pt axis text (typical ACS / Nature / Springer SI single-column).
    Call once per script before creating figures.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": int(cfg.FIGURE_DPI),
            "savefig.dpi": int(cfg.FIGURE_DPI),
            "font.size": int(cfg.FIGURE_FONT_PT),
            "axes.labelsize": int(cfg.FIGURE_FONT_PT),
            "axes.titlesize": int(cfg.FIGURE_FONT_PT),
            "xtick.labelsize": int(cfg.FIGURE_TICK_PT),
            "ytick.labelsize": int(cfg.FIGURE_TICK_PT),
            "legend.fontsize": int(cfg.FIGURE_TICK_PT),
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def single_column_figsize(height_in: float | None = None) -> tuple[float, float]:
    h = float(height_in) if height_in is not None else float(cfg.FIGURE_HEIGHT_SINGLE_PANEL_IN)
    return (float(cfg.FIGURE_WIDTH_SINGLE_COL_IN), h)


def double_column_figsize(height_in: float) -> tuple[float, float]:
    return (float(cfg.FIGURE_WIDTH_DOUBLE_COL_IN), float(height_in))
