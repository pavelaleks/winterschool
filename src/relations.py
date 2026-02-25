"""
Два типа сетей: (1) co-mention — совместные упоминания в контексте ±2 предложения, без самопетель, raw + Jaccard;
(2) interaction — направленная сеть по глаголам взаимодействия (subject -> verb -> object).
Interaction строится только из «нормальных» предложений (не индекс/колонтитул).
"""

import re
import math
import random
import yaml
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd

from .ethnonym_extractor import load_ethnonyms, build_ethnonym_patterns

# Фильтр качества предложения для interaction: исключаем индексные/короткие строки
MIN_SENTENCE_TOKENS = 5
MIN_LETTER_RATIO = 0.5
INDEX_LIKE_RE = re.compile(
    r"^(?:[A-Z][A-Z\s]+\s+\d+)\s*$|^[\s\.,\d\-]+$|(?:,\s*\d+\s*){3,}",
    re.IGNORECASE,
)


def is_normal_sentence_for_interaction(sent: Dict, text: Optional[str] = None) -> bool:
    """
    Предложение считается пригодным для interaction, если:
    - не менее MIN_SENTENCE_TOKENS токенов (или слов);
    - доля букв в тексте >= MIN_LETTER_RATIO;
    - не совпадает с индексным паттерном (номера страниц, "THE WEDDING 17" и т.п.).
    """
    t = text if text is not None else (sent.get("text") or "")
    if not t or not t.strip():
        return False
    t = t.strip()
    if INDEX_LIKE_RE.match(t) or INDEX_LIKE_RE.search(t):
        return False
    letters = sum(1 for c in t if c.isalpha())
    if letters < MIN_LETTER_RATIO * len(t):
        return False
    tokens = sent.get("token_objects") or sent.get("tokens", [])
    if tokens:
        n = len(tokens)
    else:
        n = len(t.split())
    return n >= MIN_SENTENCE_TOKENS

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
INTERACTION_VERBS_PATH = RESOURCES_DIR / "interaction_verbs.yml"


def _load_interaction_verbs() -> Dict[str, List[str]]:
    """Загружает interaction_verbs.yml: verb_group -> [lemma/form, ...]."""
    if not INTERACTION_VERBS_PATH.exists():
        return {}
    with open(INTERACTION_VERBS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_ethnonyms_in_text(
    text: str,
    patterns: List[Tuple[str, Any]],
) -> List[Tuple[int, int, str]]:
    """Возвращает список (start, end, canonical_ethnonym)."""
    result = []
    for canonical, pattern in patterns:
        for m in pattern.finditer(text):
            result.append((m.start(), m.end(), canonical))
    return sorted(result, key=lambda x: x[0])


def cooccurrence_context_pairs(
    doc_list: List[Dict],
    patterns: List[tuple],
    window: int = 2,
) -> Tuple[List[Tuple[str, str]], Dict[str, Set[int]]]:
    """
    Co-occurrence в контексте ±window предложений. Только разные этносы (без самопетель).
    Возвращает (список пар (eth1, eth2), eth1 < eth2 лексикографически),
    и dict context_id -> set of (doc_idx, sent_start) для Jaccard.
    """
    pairs = []
    context_to_ethnos = defaultdict(set)
    context_id = 0
    for doc_idx, doc in enumerate(doc_list):
        sents = doc.get("sentences", [])
        for i, sent in enumerate(sents):
            start = max(0, i - window)
            end = min(len(sents), i + window + 1)
            ctx_sents = sents[start:end]
            context_text = " ".join(s.get("text", "") for s in ctx_sents)
            found = get_ethnonyms_in_text(context_text, patterns)
            uniq = sorted(set(f[2] for f in found))
            cid = context_id
            context_id += 1
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    pairs.append((uniq[a], uniq[b]))
                    context_to_ethnos[cid].add((uniq[a], uniq[b]))
    return pairs, dict(context_to_ethnos)


def build_comention_raw(
    doc_list: List[Dict],
    patterns: List[tuple],
    window: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Матрица co-mention: только разные этносы, веса = количество контекстов."""
    pairs, _ = cooccurrence_context_pairs(doc_list, patterns, window)
    matrix = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        matrix[a][b] += 1
        matrix[b][a] += 1
    return {k: dict(v) for k, v in matrix.items()}


def build_comention_jaccard(
    doc_list: List[Dict],
    patterns: List[tuple],
    window: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Jaccard: для каждой пары (a,b) — |contexts containing both| / |contexts containing a or b|."""
    ethnos_contexts, _ = _collect_ethnos_contexts(doc_list, patterns, window)
    matrix = defaultdict(lambda: defaultdict(float))
    ethnos_list = list(ethnos_contexts.keys())
    for i in range(len(ethnos_list)):
        for j in range(i + 1, len(ethnos_list)):
            a, b = ethnos_list[i], ethnos_list[j]
            u = ethnos_contexts[a] | ethnos_contexts[b]
            inter = ethnos_contexts[a] & ethnos_contexts[b]
            if u:
                jacc = len(inter) / len(u)
                matrix[a][b] = round(jacc, 4)
                matrix[b][a] = round(jacc, 4)
    return {k: dict(v) for k, v in matrix.items()}


def _collect_ethnos_contexts(
    doc_list: List[Dict],
    patterns: List[tuple],
    window: int = 2,
) -> Tuple[Dict[str, Set[Tuple[int, int]]], int]:
    """Собирает контексты по этносам для ассоциативных метрик co-mention."""
    ethnos_contexts: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    total_contexts = 0
    for doc_idx, doc in enumerate(doc_list):
        sents = doc.get("sentences", [])
        for i in range(len(sents)):
            start = max(0, i - window)
            end = min(len(sents), i + window + 1)
            context_text = " ".join(sents[j].get("text", "") for j in range(start, end))
            found = get_ethnonyms_in_text(context_text, patterns)
            ctx_id = (doc_idx, i)
            total_contexts += 1
            for _, _, eth in found:
                ethnos_contexts[eth].add(ctx_id)
    return dict(ethnos_contexts), total_contexts


def build_comention_npmi(
    doc_list: List[Dict],
    patterns: List[tuple],
    window: int = 2,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    NPMI для пары этносов:
      PMI(a,b)=log2(P(a,b)/(P(a)P(b)))
      NPMI=PMI/(-log2(P(a,b)))
    Возвращает матрицу npmi и stats (total_contexts, context_count_per_ethnos).
    """
    ethnos_contexts, total_contexts = _collect_ethnos_contexts(doc_list, patterns, window)
    matrix = defaultdict(lambda: defaultdict(float))
    ethnos_list = list(ethnos_contexts.keys())
    if total_contexts <= 0:
        return {}, {"total_contexts": 0, "context_count_per_ethnos": {}}
    for i in range(len(ethnos_list)):
        for j in range(i + 1, len(ethnos_list)):
            a, b = ethnos_list[i], ethnos_list[j]
            inter = ethnos_contexts[a] & ethnos_contexts[b]
            if not inter:
                continue
            p_ab = len(inter) / total_contexts
            p_a = len(ethnos_contexts[a]) / total_contexts
            p_b = len(ethnos_contexts[b]) / total_contexts
            if p_ab <= 0 or p_a <= 0 or p_b <= 0:
                continue
            pmi = math.log2(p_ab / (p_a * p_b))
            denom = -math.log2(p_ab)
            npmi = pmi / denom if denom > 0 else 0.0
            matrix[a][b] = round(float(npmi), 4)
            matrix[b][a] = round(float(npmi), 4)
    stats = {
        "total_contexts": int(total_contexts),
        "context_count_per_ethnos": {k: len(v) for k, v in ethnos_contexts.items()},
    }
    return {k: dict(v) for k, v in matrix.items()}, stats


def build_robust_interaction_edges(
    interaction_edges: List[Dict],
    min_count: int = 2,
    allowed_confidence: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Фильтрует interaction edges для «robust» слоя аналитики."""
    allowed = allowed_confidence or {"high", "medium"}
    out = []
    for e in interaction_edges or []:
        cnt = int(e.get("count", 0) or 0)
        conf = str(e.get("confidence", "")).lower()
        if cnt >= min_count and conf in allowed:
            out.append(e)
    out.sort(key=lambda x: int(x.get("count", 0) or 0), reverse=True)
    return out


def _network_metrics_from_edges(
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Вычисляет базовые сетевые метрики узлов и графа для directed interaction сети.
    Возвращает (node_metrics, graph_metrics).
    """
    try:
        import networkx as nx
    except ImportError:
        return [], {}

    G = nx.DiGraph()
    for e in edges or []:
        src = e.get("src")
        dst = e.get("dst")
        w = float(e.get("count", 0) or 0)
        if not src or not dst or src == dst or w <= 0:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += w
        else:
            G.add_edge(src, dst, weight=w)

    if G.number_of_nodes() == 0:
        return [], {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "reciprocity": 0.0,
            "avg_weighted_degree": 0.0,
        }

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    total_deg = {n: float(in_deg.get(n, 0.0) + out_deg.get(n, 0.0)) for n in G.nodes()}
    try:
        bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        bet = {n: 0.0 for n in G.nodes()}
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    node_metrics = []
    for n in G.nodes():
        node_metrics.append({
            "node": n,
            "in_degree_w": round(float(in_deg.get(n, 0.0)), 4),
            "out_degree_w": round(float(out_deg.get(n, 0.0)), 4),
            "degree_w": round(float(total_deg.get(n, 0.0)), 4),
            "agency_balance": round(float(out_deg.get(n, 0.0) - in_deg.get(n, 0.0)), 4),
            "betweenness": round(float(bet.get(n, 0.0)), 6),
            "pagerank": round(float(pr.get(n, 0.0)), 6),
        })
    node_metrics.sort(key=lambda x: x["degree_w"], reverse=True)

    try:
        reciprocity = nx.reciprocity(G)
        reciprocity = float(reciprocity) if reciprocity is not None else 0.0
    except Exception:
        reciprocity = 0.0

    graph_metrics = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "density": round(float(nx.density(G)), 6),
        "reciprocity": round(reciprocity, 6),
        "avg_weighted_degree": round(float(sum(total_deg.values()) / max(len(total_deg), 1)), 6),
    }
    return node_metrics, graph_metrics


def _edge_key(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(edge.get("src") or ""),
        str(edge.get("dst") or ""),
        str(edge.get("type") or "unknown"),
    )


def compute_interaction_bootstrap_stability(
    doc_list: List[Dict],
    patterns: List[tuple],
    base_robust_edges: List[Dict[str, Any]],
    bootstrap_n: int = 50,
    top_k: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Bootstrap устойчивость top-K robust edges по документам.
    stability = доля bootstrap-сэмплов, где ребро присутствует.
    """
    if not doc_list or not base_robust_edges:
        return []
    base_top = sorted(base_robust_edges, key=lambda x: int(x.get("count", 0) or 0), reverse=True)[:top_k]
    targets = {_edge_key(e): e for e in base_top}
    hit_counts = {k: 0 for k in targets.keys()}
    rng = random.Random(seed)
    n_docs = len(doc_list)
    if n_docs == 0:
        return []

    for _ in range(max(1, int(bootstrap_n))):
        sample_docs = [doc_list[rng.randrange(0, n_docs)] for _ in range(n_docs)]
        _mx, sample_edges = build_interaction_network(sample_docs, patterns)
        sample_robust = build_robust_interaction_edges(sample_edges, min_count=2, allowed_confidence={"high", "medium"})
        present = {_edge_key(e) for e in sample_robust}
        for k in hit_counts.keys():
            if k in present:
                hit_counts[k] += 1

    out = []
    for k, base in targets.items():
        st = hit_counts[k] / max(1, int(bootstrap_n))
        out.append({
            "src": base.get("src"),
            "dst": base.get("dst"),
            "type": base.get("type"),
            "count": int(base.get("count", 0) or 0),
            "confidence": base.get("confidence", ""),
            "stability": round(float(st), 4),
            "bootstrap_n": int(bootstrap_n),
        })
    out.sort(key=lambda x: (x["stability"], x["count"]), reverse=True)
    return out


def compute_comention_window_sensitivity(
    doc_list: List[Dict],
    patterns: List[tuple],
    windows: Optional[List[int]] = None,
    top_k: int = 30,
) -> Dict[str, Any]:
    """
    Sensitivity по окнам co-mention: ±1 / ±2 / ±3.
    Считает число рёбер, top-K overlap c базовым окном (2), top pairs.
    """
    ws = windows or [1, 2, 3]
    by_window: Dict[int, Dict[str, Any]] = {}
    for w in ws:
        raw = build_comention_raw(doc_list, patterns, window=w)
        pairs = []
        for a, targets in (raw or {}).items():
            for b, cnt in targets.items():
                if a < b and cnt > 0:
                    pairs.append((a, b, int(cnt)))
        pairs.sort(key=lambda x: x[2], reverse=True)
        by_window[w] = {
            "n_edges": len(pairs),
            "top_pairs": [{"a": a, "b": b, "count": c} for a, b, c in pairs[:top_k]],
            "top_pair_keys": {(a, b) for a, b, _c in pairs[:top_k]},
        }
    base = by_window.get(2, by_window.get(ws[0], {"top_pair_keys": set()}))
    base_keys = base.get("top_pair_keys", set())
    summary = []
    for w in ws:
        keys = by_window[w].get("top_pair_keys", set())
        overlap = len(base_keys & keys)
        denom = max(1, min(len(base_keys), len(keys)))
        summary.append({
            "window": int(w),
            "n_edges": int(by_window[w].get("n_edges", 0)),
            "topk_overlap_with_w2": int(overlap),
            "topk_overlap_ratio_with_w2": round(float(overlap / denom), 4),
        })
    serializable = {
        "summary": summary,
        "top_pairs_by_window": {
            str(w): by_window[w]["top_pairs"] for w in ws
        },
    }
    return serializable


def find_interaction_verbs_and_arguments(
    sentence: Dict,
    ethnonym_positions: List[Tuple[int, int, str]],
    verb_groups: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Ищет глаголы из verb_groups в предложении, через deps определяет subject/object.
    Возвращает список {verb, verb_group, subject_ethnonym, object_ethnonym, confidence}.
    confidence:
      - high: и subject, и object извлечены напрямую по deps;
      - medium: извлечена одна роль, вторая восстановлена из остальных этнонимов предложения;
      - low: эвристическое сопоставление без явных subject/object.
    """
    tokens = sentence.get("token_objects", [])
    if not tokens:
        return []
    text_lower = (sentence.get("text") or "").lower()
    eths_in_sent = [eth for _s, _e, eth in sorted(ethnonym_positions, key=lambda x: x[0])]
    token_to_eth = {}
    for t in tokens:
        tt = t.get("text", "").lower()
        for _s, _e, eth in ethnonym_positions:
            e = eth.lower()
            if tt == e or tt == e + "s" or tt == e.rstrip("s") or e.startswith(tt) or tt.startswith(e):
                token_to_eth[t["i"]] = eth
                break
        else:
            token_to_eth[t["i"]] = None
    # Построить lemma -> verb_group
    lemma_to_group = {}
    for group, lemmas in verb_groups.items():
        for L in lemmas:
            lemma_to_group[L.lower()] = group
    interactions = []
    for t in tokens:
        lemma = t.get("lemma", "").lower()
        text = t.get("text", "").lower()
        group = lemma_to_group.get(lemma) or lemma_to_group.get(text)
        if not group:
            continue
        subj_eth = None
        obj_eth = None
        has_subj_dep = False
        has_obj_dep = False
        for t2 in tokens:
            if t2.get("head_i") == t["i"]:
                dep = t2.get("dep", "")
                if dep in ("nsubj", "nsubjpass"):
                    has_subj_dep = True
                    subj_eth = token_to_eth.get(t2["i"])
                if dep in ("dobj", "pobj", "obj", "iobj"):
                    has_obj_dep = True
                    obj_eth = token_to_eth.get(t2["i"])
        confidence = "high"
        if subj_eth and obj_eth:
            confidence = "high"
        elif subj_eth and len(eths_in_sent) >= 2:
            obj_candidates = [e for e in eths_in_sent if e != subj_eth]
            if obj_candidates:
                obj_eth = obj_candidates[0]
                confidence = "medium"
        elif obj_eth and len(eths_in_sent) >= 2:
            subj_candidates = [e for e in eths_in_sent if e != obj_eth]
            if subj_candidates:
                subj_eth = subj_candidates[0]
                confidence = "medium"
        elif len(eths_in_sent) >= 2 and not has_subj_dep and not has_obj_dep:
            # Крайний эвристический fallback: сохраняем, но помечаем как low.
            subj_eth, obj_eth = eths_in_sent[0], eths_in_sent[1]
            confidence = "low"
        if subj_eth or obj_eth:
            interactions.append({
                "verb": text,
                "verb_group": group,
                "subject_ethnonym": subj_eth,
                "object_ethnonym": obj_eth,
                "confidence": confidence,
            })
    return interactions


def build_interaction_network(
    doc_list: List[Dict],
    patterns: List[tuple],
) -> Tuple[Dict[str, Dict[str, int]], List[Dict]]:
    """
    Направленная сеть по глаголам взаимодействия.
    Возвращает (matrix[src][dst]=count, list of edge records с type, count, example_ids).
    """
    verb_groups = _load_interaction_verbs()
    if not verb_groups:
        return {}, []
    matrix = defaultdict(lambda: defaultdict(int))
    edge_records = defaultdict(lambda: {
        "count": 0,
        "examples": [],
        "confidence_counts": {"high": 0, "medium": 0, "low": 0},
    })
    for doc in doc_list:
        for sent in doc.get("sentences", []):
            text = sent.get("text", "")
            if not is_normal_sentence_for_interaction(sent, text):
                continue
            found = get_ethnonyms_in_text(text, patterns)
            if not found:
                continue
            interactions = find_interaction_verbs_and_arguments(sent, found, verb_groups)
            for rec in interactions:
                subj = rec.get("subject_ethnonym")
                obj = rec.get("object_ethnonym")
                if not subj or not obj or subj == obj:
                    continue
                key = (subj, obj, rec.get("verb_group", "unknown"))
                matrix[subj][obj] += 1
                edge_records[key]["count"] += 1
                conf = rec.get("confidence", "low")
                if conf not in edge_records[key]["confidence_counts"]:
                    conf = "low"
                edge_records[key]["confidence_counts"][conf] += 1
                quote = (text or "").strip()[:300]
                if quote and len(edge_records[key]["examples"]) < 3:
                    if quote not in edge_records[key]["examples"]:
                        edge_records[key]["examples"].append(quote)
    return {k: dict(v) for k, v in matrix.items()}, [
        {
            "src": a,
            "dst": b,
            "type": t,
            "count": r["count"],
            "confidence": (
                "high" if r["confidence_counts"]["high"] >= max(r["confidence_counts"]["medium"], r["confidence_counts"]["low"])
                else ("medium" if r["confidence_counts"]["medium"] >= r["confidence_counts"]["low"] else "low")
            ),
            "confidence_counts": r["confidence_counts"],
            "examples": r["examples"][:3],
        }
        for (a, b, t), r in edge_records.items()
    ]


def save_network_png(
    matrix: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str,
    directed: bool = False,
) -> None:
    """Строит граф и сохраняет PNG. directed=False для co-mention, True для interaction."""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for src, targets in matrix.items():
        for tgt, w in targets.items():
            if src != tgt and w > 0:
                G.add_edge(src, tgt, weight=w)
    if G.number_of_nodes() == 0:
        return
    plt.figure(figsize=(10, 8))
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    pos = nx.spring_layout(G, k=1.5, seed=42)
    edges = list(G.edges())
    weights = [G[u][v].get("weight", 1) for u, v in edges]
    max_w = max(weights) if weights else 1
    nx.draw_networkx_nodes(G, pos, node_color="lightsteelblue", edgecolors="gray", node_size=800)
    # Для направленного графа — фиксированная толщина, чтобы рисовались стрелки (FancyArrowPatches)
    if directed:
        nx.draw_networkx_edges(
            G, pos, width=2, alpha=0.6, edge_color="gray",
            arrows=True, arrowstyle="-|>", arrowsize=15,
        )
    else:
        nx.draw_networkx_edges(
            G, pos,
            width=[2 + 2 * (w / max_w) for w in weights],
            alpha=0.6, edge_color="gray",
        )
    nx.draw_networkx_labels(G, pos, font_size=9)
    edge_labels = {(u, v): f"{G[u][v].get('weight', 1):.2g}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def get_relation_stats(
    doc_list: List[Dict],
) -> Dict[str, Any]:
    """Совместимость со старым API: co-occurrence и матрица (co-mention raw + interaction)."""
    patterns = build_ethnonym_patterns(load_ethnonyms())
    pairs_raw, _ = cooccurrence_context_pairs(doc_list, patterns, window=2)
    comention_raw = build_comention_raw(doc_list, patterns, window=2)
    interaction_matrix, interaction_edges = build_interaction_network(doc_list, patterns)
    return {
        "cooccurrence_context": pairs_raw,
        "comention_raw": comention_raw,
        "interaction_matrix": interaction_matrix,
        "interaction_edges": interaction_edges,
    }


def build_graph_and_save_png(
    matrix: Dict[str, Dict[str, int]],
    output_path: str,
) -> None:
    """Совместимость: сохраняет один граф (направленный по матрице)."""
    save_network_png(
        matrix,
        Path(output_path),
        title="Сеть межэтнических взаимодействий",
        directed=True,
    )


def run_relations_pipeline(
    doc_list: List[Dict],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Строит обе сети, сохраняет PNG и interaction_edges.csv.
    Возвращает dict с comention_raw, comention_jaccard, comention_npmi,
    interaction_matrix, interaction_edges, robust_interaction_edges, network metrics.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = build_ethnonym_patterns(load_ethnonyms())

    comention_raw = build_comention_raw(doc_list, patterns, window=2)
    save_network_png(
        comention_raw,
        output_dir / "network_comention_raw.png",
        title="Co-mention (raw count)",
        directed=False,
    )

    comention_jaccard = build_comention_jaccard(doc_list, patterns, window=2)
    save_network_png(
        comention_jaccard,
        output_dir / "network_comention_jaccard.png",
        title="Co-mention (Jaccard)",
        directed=False,
    )
    comention_npmi, comention_stats = build_comention_npmi(doc_list, patterns, window=2)
    comention_window_sensitivity = compute_comention_window_sensitivity(
        doc_list, patterns, windows=[1, 2, 3], top_k=30
    )

    interaction_matrix, interaction_edges = build_interaction_network(doc_list, patterns)
    robust_interaction_edges = build_robust_interaction_edges(
        interaction_edges,
        min_count=2,
        allowed_confidence={"high", "medium"},
    )
    node_metrics_all, graph_metrics_all = _network_metrics_from_edges(interaction_edges)
    node_metrics_robust, graph_metrics_robust = _network_metrics_from_edges(robust_interaction_edges)
    bootstrap_stability = compute_interaction_bootstrap_stability(
        doc_list, patterns, robust_interaction_edges, bootstrap_n=50, top_k=20, seed=42
    )
    save_network_png(
        interaction_matrix,
        output_dir / "network_interaction_directed.png",
        title="Interaction (directed)",
        directed=True,
    )

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if interaction_edges:
        edges_df = pd.DataFrame([
            {
                "src": r["src"],
                "dst": r["dst"],
                "type": r["type"],
                "count": r["count"],
                "confidence": r.get("confidence", ""),
                "confidence_high": (r.get("confidence_counts") or {}).get("high", 0),
                "confidence_medium": (r.get("confidence_counts") or {}).get("medium", 0),
                "confidence_low": (r.get("confidence_counts") or {}).get("low", 0),
                "example_1": r.get("examples", [])[0] if len(r.get("examples", [])) > 0 else "",
                "example_2": r.get("examples", [])[1] if len(r.get("examples", [])) > 1 else "",
                "example_3": r.get("examples", [])[2] if len(r.get("examples", [])) > 2 else "",
            }
            for r in interaction_edges
        ])
        edges_df.to_csv(tables_dir / "interaction_edges.csv", index=False, encoding="utf-8")
        robust_df = pd.DataFrame([
            {
                "src": r["src"],
                "dst": r["dst"],
                "type": r["type"],
                "count": r["count"],
                "confidence": r.get("confidence", ""),
                "confidence_high": (r.get("confidence_counts") or {}).get("high", 0),
                "confidence_medium": (r.get("confidence_counts") or {}).get("medium", 0),
                "confidence_low": (r.get("confidence_counts") or {}).get("low", 0),
                "example_1": r.get("examples", [])[0] if len(r.get("examples", [])) > 0 else "",
            }
            for r in robust_interaction_edges
        ])
        robust_df.to_csv(tables_dir / "interaction_edges_robust.csv", index=False, encoding="utf-8")
        npmi_rows = []
        for a, targets in (comention_npmi or {}).items():
            for b, score in targets.items():
                if a < b:
                    npmi_rows.append({"ethnos_a": a, "ethnos_b": b, "npmi": score})
        if npmi_rows:
            npmi_df = pd.DataFrame(npmi_rows).sort_values("npmi", ascending=False)
            npmi_df.to_csv(tables_dir / "comention_npmi.csv", index=False, encoding="utf-8")
        if bootstrap_stability:
            pd.DataFrame(bootstrap_stability).to_csv(
                tables_dir / "interaction_edge_stability.csv", index=False, encoding="utf-8"
            )
        if node_metrics_all:
            pd.DataFrame(node_metrics_all).to_csv(tables_dir / "interaction_node_metrics.csv", index=False, encoding="utf-8")
        if node_metrics_robust:
            pd.DataFrame(node_metrics_robust).to_csv(tables_dir / "interaction_node_metrics_robust.csv", index=False, encoding="utf-8")
    try:
        import json
        (output_dir / "derived").mkdir(parents=True, exist_ok=True)
        metrics_json = {
            "interaction_all": graph_metrics_all,
            "interaction_robust": graph_metrics_robust,
        }
        (output_dir / "derived" / "network_graph_metrics.json").write_text(
            json.dumps(metrics_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "derived" / "comention_window_sensitivity.json").write_text(
            json.dumps(comention_window_sensitivity, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "derived" / "comention_stats.json").write_text(
            json.dumps(comention_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "derived" / "interaction_edge_stability.json").write_text(
            json.dumps(bootstrap_stability, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return {
        "comention_raw": comention_raw,
        "comention_jaccard": comention_jaccard,
        "comention_npmi": comention_npmi,
        "comention_stats": comention_stats,
        "interaction_matrix": interaction_matrix,
        "interaction_edges": interaction_edges,
        "robust_interaction_edges": robust_interaction_edges,
        "interaction_node_metrics": node_metrics_all,
        "interaction_node_metrics_robust": node_metrics_robust,
        "interaction_graph_metrics": graph_metrics_all,
        "interaction_graph_metrics_robust": graph_metrics_robust,
        "comention_window_sensitivity": comention_window_sensitivity,
        "interaction_edge_stability": bootstrap_stability,
    }
