"""
Scatter-графики OI vs AS и ED vs EPS для отчёта: читаемые подписи, квадранты, jitter, экспорт SVG/PNG.
Использует matplotlib и adjustText для разнесения подписей. Расчёты не меняются — только визуализация.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Параметры визуализации
SHOW_TOP_N = 10
JITTER_STRENGTH = 0.01
RANDOM_SEED = 42
FIG_HEIGHT = 6
FIG_WIDTH_PER_POINT = 0.6
MIN_FIG_WIDTH = 8
ALPHA = 0.75
MARKER_SIZE = 80


def _build_indices_df(derived_indices: Dict[str, Any]) -> pd.DataFrame:
    """Строит DataFrame ethnos, OI, AS, ED, EPS, frequency из derived_indices."""
    oi = derived_indices.get("OI") or {}
    raw_oi = oi.get("raw_OI") or oi
    as_ = derived_indices.get("AS") or {}
    ed = derived_indices.get("ED") or {}
    eps = derived_indices.get("EPS") or {}
    mentions = derived_indices.get("mentions_per_ethnos") or {}
    ethnos_set = set()
    for k in (raw_oi, as_, ed, eps, mentions):
        if isinstance(k, dict):
            ethnos_set.update(k.keys())
    rows = []
    for eth in ethnos_set:
        oi_val = raw_oi.get(eth)
        as_val = as_.get(eth)
        ed_val = ed.get(eth)
        eps_val = eps.get(eth)
        freq = mentions.get(eth) or 0
        if oi_val is None and as_val is None and ed_val is None and eps_val is None:
            continue
        rows.append({
            "ethnos": eth,
            "OI": float(oi_val) if oi_val is not None else np.nan,
            "AS": float(as_val) if as_val is not None else np.nan,
            "ED": float(ed_val) if ed_val is not None else np.nan,
            "EPS": float(eps_val) if eps_val is not None else np.nan,
            "frequency": int(freq),
        })
    return pd.DataFrame(rows)


def _labels_to_show_oi_as(df: pd.DataFrame) -> pd.Series:
    """Маска: подписывать top SHOW_TOP_N по frequency + те, у кого |AS| > 0.5 или OI > 0.6."""
    show = pd.Series(False, index=df.index)
    if df.empty:
        return show
    # Top N по частоте
    top_n = df.nlargest(SHOW_TOP_N, "frequency").index
    show.loc[top_n] = True
    # Заметные по AS или OI
    notable = (df["AS"].abs() > 0.5) | (df["OI"] > 0.6)
    show |= notable
    return show


def _labels_to_show_ed_eps(df: pd.DataFrame) -> pd.Series:
    """Маска: подписывать top SHOW_TOP_N по frequency + те, у кого |EPS| > 0.5 или ED > 0.6."""
    show = pd.Series(False, index=df.index)
    if df.empty:
        return show
    top_n = df.nlargest(SHOW_TOP_N, "frequency").index
    show.loc[top_n] = True
    notable = (df["EPS"].abs() > 0.5) | (df["ED"] > 0.6)
    show |= notable
    return show


def _plot_oi_vs_as(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Строит OI vs AS с jitter, квадрантами, разнесёнными подписями. Возвращает пути к PNG и SVG."""
    df = df.dropna(subset=["OI", "AS"]).copy()
    if df.empty:
        return None, None
    df = df.sort_values("frequency", ascending=True)  # менее значимые рисуем первыми
    rng = np.random.default_rng(RANDOM_SEED)
    df["OI_j"] = df["OI"] + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, len(df))
    df["AS_j"] = df["AS"] + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, len(df))
    show_label = _labels_to_show_oi_as(df)

    fig_width = max(MIN_FIG_WIDTH, len(df) * FIG_WIDTH_PER_POINT)
    fig, ax = plt.subplots(figsize=(fig_width, FIG_HEIGHT))
    ax.scatter(df["OI_j"], df["AS_j"], alpha=ALPHA, s=MARKER_SIZE, color="#4a6fa5", edgecolors="gray", linewidths=0.5)
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0.5, color="lightgray", linestyle="--", lw=1)
    ax.margins(0.15)
    ax.set_xlabel("OI (Orientalization Index)")
    ax.set_ylabel("AS (Agency Score)")
    ax.set_title("OI vs Agency Score\nQuadrants: high OI / low AS (bottom-right) vs high OI / high AS (top-right)")

    texts = []
    for i, row in df.loc[show_label].iterrows():
        t = ax.text(row["OI"], row["AS"], row["ethnos"], fontsize=9)
        texts.append(t)
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand_points=(1.2, 1.2),
                expand_text=(1.2, 1.2),
            )
        except ImportError:
            for t in texts:
                t.set_position((t.get_position()[0], t.get_position()[1]))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path_png = output_dir / "oi_vs_as.png"
    path_svg = output_dir / "oi_vs_as.svg"
    fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close()
    return path_png, path_svg


def _plot_ed_vs_eps(
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Строит ED vs EPS с jitter, квадрантами, разнесёнными подписями."""
    df = df.dropna(subset=["ED", "EPS"]).copy()
    if df.empty:
        return None, None
    df = df.sort_values("frequency", ascending=True)
    rng = np.random.default_rng(RANDOM_SEED)
    df["ED_j"] = df["ED"] + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, len(df))
    df["EPS_j"] = df["EPS"] + rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, len(df))
    show_label = _labels_to_show_ed_eps(df)

    fig_width = max(MIN_FIG_WIDTH, len(df) * FIG_WIDTH_PER_POINT)
    fig, ax = plt.subplots(figsize=(fig_width, FIG_HEIGHT))
    ax.scatter(df["ED_j"], df["EPS_j"], alpha=ALPHA, s=MARKER_SIZE, color="#6b8e23", edgecolors="gray", linewidths=0.5)
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="lightgray", linestyle="--", lw=1)
    ax.margins(0.15)
    ax.set_xlabel("ED (Essentialization Degree)")
    ax.set_ylabel("EPS (Evaluative Polarity Score)")
    ax.set_title("ED vs EPS\nQuadrants: high ED / negative EPS vs high ED / positive EPS")

    texts = []
    for i, row in df.loc[show_label].iterrows():
        t = ax.text(row["ED"], row["EPS"], row["ethnos"], fontsize=9)
        texts.append(t)
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand_points=(1.2, 1.2),
                expand_text=(1.2, 1.2),
            )
        except ImportError:
            for t in texts:
                t.set_position((t.get_position()[0], t.get_position()[1]))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path_png = output_dir / "ed_vs_eps.png"
    path_svg = output_dir / "ed_vs_eps.svg"
    fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close()
    return path_png, path_svg


def build_indices_scatter_plots(
    derived_indices: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """
    Строит OI vs AS и ED vs EPS, сохраняет PNG и SVG в output_dir/figures/.
    Возвращает словарь путей для отчёта: oi_vs_as_png, oi_vs_as_svg, ed_vs_eps_png, ed_vs_eps_svg.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    result = {"oi_vs_as_png": None, "oi_vs_as_svg": None, "ed_vs_eps_png": None, "ed_vs_eps_svg": None}

    if not derived_indices:
        return result
    df = _build_indices_df(derived_indices)
    if df.empty:
        return result

    png1, svg1 = _plot_oi_vs_as(df, figures_dir)
    if png1:
        result["oi_vs_as_png"] = "figures/" + png1.name
        result["oi_vs_as_svg"] = "figures/" + svg1.name
    png2, svg2 = _plot_ed_vs_eps(df, figures_dir)
    if png2:
        result["ed_vs_eps_png"] = "figures/" + png2.name
        result["ed_vs_eps_svg"] = "figures/" + svg2.name
    return result
