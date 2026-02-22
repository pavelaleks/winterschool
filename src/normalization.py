"""
Нормировка по документам: не только counts, но и rates per 10k sentences,
среднее/медиана по документам, опционально bootstrap CI.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

SENTENCES_PER_10K = 10_000


def counts_by_doc(
    df: pd.DataFrame,
    group_col: str,
    doc_col: str = "file_name",
    sent_count_col: str = "doc_sent_count",
) -> pd.DataFrame:
    """
    Считает количество упоминаний по (doc, group_col).
    df должен содержать doc_col и sent_count_col.
    Возвращает DataFrame с колонками doc_col, group_col, count, doc_sent_count.
    """
    out = (
        df.groupby([doc_col, group_col], dropna=False)
        .size()
        .reset_index(name="count")
    )
    doc_sents = df.groupby(doc_col)[sent_count_col].first().reset_index()
    out = out.merge(doc_sents, on=doc_col, how="left")
    return out


def rates_per_10k(
    counts_df: pd.DataFrame,
    count_col: str = "count",
    sent_col: str = "doc_sent_count",
) -> pd.Series:
    """rate = count / doc_sent_count * 10000 (per 10k sentences)."""
    if sent_col not in counts_df.columns or counts_df[sent_col].eq(0).all():
        return counts_df[count_col] * 0.0
    return counts_df[count_col] / counts_df[sent_col].clip(lower=1) * SENTENCES_PER_10K


def aggregate_rates(
    df: pd.DataFrame,
    group_col: str,
    doc_col: str = "file_name",
    doc_sent_count_col: str = "doc_sent_count",
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """
    По каждому значению group_col: counts, mean rate, median rate по документам,
    опционально bootstrap 95% CI для среднего.
    Возвращает: {group_value: {count, mean_rate, median_rate, ci_low, ci_high}}.
    """
    by_doc = counts_by_doc(df, group_col, doc_col=doc_col, sent_count_col=doc_sent_count_col)
    by_doc["rate_per_10k"] = rates_per_10k(by_doc, count_col="count", sent_col=doc_sent_count_col)

    result = {}
    for g in by_doc[group_col].dropna().unique():
        sub = by_doc[by_doc[group_col] == g]
        rates = sub["rate_per_10k"].values
        total_count = sub["count"].sum()
        mean_rate = float(np.mean(rates)) if len(rates) else 0.0
        median_rate = float(np.median(rates)) if len(rates) else 0.0
        ci_low, ci_high = None, None
        if n_bootstrap > 0 and len(rates) >= 2:
            rng = np.random.default_rng(42)
            boot = rng.choice(rates, size=(n_bootstrap, len(rates)), replace=True)
            boot_means = boot.mean(axis=1)
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
        result[str(g)] = {
            "count": int(total_count),
            "mean_rate_per_10k": round(mean_rate, 4),
            "median_rate_per_10k": round(median_rate, 4),
            "ci_low": round(ci_low, 4) if ci_low is not None else None,
            "ci_high": round(ci_high, 4) if ci_high is not None else None,
        }
    return result


def normalized_stats_ethnos(
    df: pd.DataFrame,
    doc_col: str = "file_name",
    doc_sent_count_col: str = "doc_sent_count",
) -> Dict[str, Dict[str, Any]]:
    """Агрегат по ethnos_norm: сырые counts + нормированные rates."""
    if "ethnos_norm" not in df.columns:
        return {}
    return aggregate_rates(
        df,
        group_col="ethnos_norm",
        doc_col=doc_col,
        doc_sent_count_col=doc_sent_count_col,
    )


def normalized_stats_R(
    df: pd.DataFrame,
    doc_col: str = "file_name",
    doc_sent_count_col: str = "doc_sent_count",
) -> Dict[str, Dict[str, Any]]:
    """Агрегат по R (тип репрезентации)."""
    if "R" not in df.columns:
        return {}
    return aggregate_rates(
        df,
        group_col="R",
        doc_col=doc_col,
        doc_sent_count_col=doc_sent_count_col,
    )


def normalized_stats_O(
    df: pd.DataFrame,
    doc_col: str = "file_name",
    doc_sent_count_col: str = "doc_sent_count",
) -> Dict[str, Dict[str, Any]]:
    """Агрегат по O_situation."""
    if "O_situation" not in df.columns:
        return {}
    return aggregate_rates(
        df,
        group_col="O_situation",
        doc_col=doc_col,
        doc_sent_count_col=doc_sent_count_col,
    )
