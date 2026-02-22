"""
Производные индексы ориентализации для DH-дашборда.
Сохраняет output/derived_indices.json.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "output" / "derived_indices.json"
BOOTSTRAP_N = 500


def compute_oi(
    piro_records: List[Dict],
    norm_rates_per_ethnos: Optional[Dict[str, Dict[str, Any]]] = None,
    bootstrap_n: int = BOOTSTRAP_N,
) -> Dict[str, Any]:
    """
    Orientalization Index по этносам: OI = (negative + exotic) / total_mentions.
    raw_OI, normalized_OI (если передан norm_rates), confidence_interval (bootstrap).
    """
    by_ethnos: Dict[str, Dict[str, int]] = defaultdict(lambda: {"negative": 0, "exotic": 0, "positive": 0, "neutral": 0, "uncertain": 0, "total": 0})
    for r in piro_records:
        eth = r.get("P") or r.get("ethnos_norm") or ""
        if not eth:
            continue
        by_ethnos[eth]["total"] += 1
        rr = (r.get("R") or "neutral").lower()
        if rr in by_ethnos[eth]:
            by_ethnos[eth][rr] += 1
        else:
            by_ethnos[eth]["neutral"] += 1

    raw_oi = {}
    for eth, c in by_ethnos.items():
        t = c["total"]
        if t == 0:
            continue
        neg_exo = c.get("negative", 0) + c.get("exotic", 0)
        raw_oi[eth] = round(neg_exo / t, 4)

    # Bootstrap CI: по списку 0/1 (1 = negative или exotic) сэмплируем с возвратом
    ethnos_list = list(by_ethnos.keys())
    ci = {}
    rng = np.random.default_rng(42)
    for eth in ethnos_list:
        t = by_ethnos[eth]["total"]
        if t < 2:
            ci[eth] = {"low": raw_oi.get(eth, 0), "high": raw_oi.get(eth, 0)}
            continue
        n_neg_exo = by_ethnos[eth].get("negative", 0) + by_ethnos[eth].get("exotic", 0)
        arr = np.array([1] * n_neg_exo + [0] * (t - n_neg_exo), dtype=float)
        vals = []
        for _ in range(bootstrap_n):
            idx = rng.integers(0, t, size=t)
            vals.append(float(arr[idx].mean()))
        ci[eth] = {"low": round(float(np.percentile(vals, 2.5)), 4), "high": round(float(np.percentile(vals, 97.5)), 4)}

    normalized_oi = {}
    if norm_rates_per_ethnos:
        for eth, norm in norm_rates_per_ethnos.items():
            # Нормированный OI: доля negative+exotic в нормированных частотах (относительно суммы всех R по этносу)
            # Упрощённо: оставляем raw_OI как основной; normalized можно взять равным raw при отсутствии по-документной разбивки
            normalized_oi[eth] = raw_oi.get(eth)

    return {
        "raw_OI": raw_oi,
        "normalized_OI": normalized_oi or raw_oi,
        "confidence_interval": ci,
        "by_ethnos_counts": {k: dict(v) for k, v in by_ethnos.items()},
    }


def compute_as(interaction_edges: List[Dict]) -> Dict[str, float]:
    """
    Agency Score по узлам interaction network: AS = (outgoing - incoming) / (outgoing + incoming).
    По узлу: out_degree - in_degree, нормализовано на сумму рёбер узла.
    """
    out_d: Dict[str, int] = defaultdict(int)
    in_d: Dict[str, int] = defaultdict(int)
    for e in interaction_edges or []:
        s, d = e.get("src"), e.get("dst")
        cnt = e.get("count", 1)
        if s:
            out_d[s] += cnt
        if d:
            in_d[d] += cnt
    as_scores = {}
    all_nodes = set(out_d.keys()) | set(in_d.keys())
    for node in all_nodes:
        o, i = out_d.get(node, 0), in_d.get(node, 0)
        total = o + i
        if total == 0:
            as_scores[node] = 0.0
        else:
            as_scores[node] = round((o - i) / total, 4)
    return as_scores


def compute_ed(
    essentialization_table: Dict[str, int],
    mentions_per_ethnos: Dict[str, int],
) -> Dict[str, float]:
    """
    Essentialization Density: ED = essentializing_sentences / total_sentences_about_ethnos.
    """
    ed = {}
    for eth in set(essentialization_table.keys()) | set(mentions_per_ethnos.keys()):
        ess = essentialization_table.get(eth, 0)
        total = mentions_per_ethnos.get(eth, 0)
        if total == 0:
            ed[eth] = 0.0
        else:
            ed[eth] = round(ess / total, 4)
    return ed


def compute_eps(piro_records: List[Dict]) -> Dict[str, float]:
    """
    Evaluative Polarity Score: EPS = (negative - positive) / total_mentions по этносу.
    """
    by_ethnos: Dict[str, Dict[str, int]] = defaultdict(lambda: {"negative": 0, "positive": 0, "total": 0})
    for r in piro_records:
        eth = r.get("P") or r.get("ethnos_norm") or ""
        if not eth:
            continue
        by_ethnos[eth]["total"] += 1
        rr = (r.get("R") or "").lower()
        if rr == "negative":
            by_ethnos[eth]["negative"] += 1
        elif rr == "positive":
            by_ethnos[eth]["positive"] += 1
    eps = {}
    for eth, c in by_ethnos.items():
        t = c["total"]
        if t == 0:
            continue
        eps[eth] = round((c["negative"] - c["positive"]) / t, 4)
    return eps


def run_derived_indices(
    piro_clean: List[Dict],
    norm_ethnos: Optional[Dict[str, Dict[str, Any]]] = None,
    essentialization_table: Optional[Dict[str, int]] = None,
    interaction_edges: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Считает все индексы, сохраняет output/derived_indices.json.
    Возвращает полный словарь индексов.
    """
    output_path = output_path or OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    essentialization_table = essentialization_table or {}
    interaction_edges = interaction_edges or []

    mentions_per_ethnos = dict(Counter(r.get("P") or r.get("ethnos_norm") for r in piro_clean if (r.get("P") or r.get("ethnos_norm"))))

    oi = compute_oi(piro_clean, norm_rates_per_ethnos=norm_ethnos)
    as_scores = compute_as(interaction_edges)
    ed = compute_ed(essentialization_table, mentions_per_ethnos)
    eps = compute_eps(piro_clean)

    out = {
        "OI": {"raw_OI": oi["raw_OI"], "normalized_OI": oi.get("normalized_OI") or oi["raw_OI"], "confidence_interval": oi["confidence_interval"]},
        "AS": as_scores,
        "ED": ed,
        "EPS": eps,
        "mentions_per_ethnos": mentions_per_ethnos,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out
