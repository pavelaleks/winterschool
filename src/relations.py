"""
Два типа сетей: (1) co-mention — совместные упоминания в контексте ±2 предложения, без самопетель, raw + Jaccard;
(2) interaction — направленная сеть по глаголам взаимодействия (subject -> verb -> object).
Interaction строится только из «нормальных» предложений (не индекс/колонтитул).
"""

import re
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
    # Контексты, в которых встречается каждый этнос (id контекста = (doc_idx, sent_start))
    ethnos_contexts = defaultdict(set)
    for doc_idx, doc in enumerate(doc_list):
        sents = doc.get("sentences", [])
        for i in range(len(sents)):
            start = max(0, i - window)
            end = min(len(sents), i + window + 1)
            context_text = " ".join(sents[j].get("text", "") for j in range(start, end))
            found = get_ethnonyms_in_text(context_text, patterns)
            ctx_id = (doc_idx, i)
            for _, _, eth in found:
                ethnos_contexts[eth].add(ctx_id)
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


def find_interaction_verbs_and_arguments(
    sentence: Dict,
    ethnonym_positions: List[Tuple[int, int, str]],
    verb_groups: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Ищет глаголы из verb_groups в предложении, через deps определяет subject/object.
    Возвращает список {verb, verb_group, subject_ethnonym, object_ethnonym}.
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
        for t2 in tokens:
            if t2.get("head_i") == t["i"]:
                dep = t2.get("dep", "")
                if dep in ("nsubj", "nsubjpass"):
                    subj_eth = token_to_eth.get(t2["i"])
                if dep in ("dobj", "pobj", "obj"):
                    obj_eth = token_to_eth.get(t2["i"])
        if subj_eth is None and obj_eth is None and len(eths_in_sent) >= 2:
            subj_eth, obj_eth = eths_in_sent[0], eths_in_sent[1]
        elif subj_eth is None and eths_in_sent:
            subj_eth = eths_in_sent[0]
        elif obj_eth is None and eths_in_sent:
            obj_eth = eths_in_sent[0]
        if subj_eth or obj_eth:
            interactions.append({
                "verb": text,
                "verb_group": group,
                "subject_ethnonym": subj_eth,
                "object_ethnonym": obj_eth,
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
    edge_records = defaultdict(lambda: {"count": 0, "examples": []})
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
                quote = (text or "").strip()[:300]
                if quote and len(edge_records[key]["examples"]) < 3:
                    if quote not in edge_records[key]["examples"]:
                        edge_records[key]["examples"].append(quote)
    return {k: dict(v) for k, v in matrix.items()}, [
        {"src": a, "dst": b, "type": t, "count": r["count"], "examples": r["examples"][:3]}
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
    Возвращает dict с comention_raw, comention_jaccard, interaction_matrix, interaction_edges.
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

    interaction_matrix, interaction_edges = build_interaction_network(doc_list, patterns)
    save_network_png(
        interaction_matrix,
        output_dir / "network_interaction_directed.png",
        title="Interaction (directed)",
        directed=True,
    )

    if interaction_edges:
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        edges_df = pd.DataFrame([
            {
                "src": r["src"],
                "dst": r["dst"],
                "type": r["type"],
                "count": r["count"],
                "example_1": r.get("examples", [])[0] if len(r.get("examples", [])) > 0 else "",
                "example_2": r.get("examples", [])[1] if len(r.get("examples", [])) > 1 else "",
                "example_3": r.get("examples", [])[2] if len(r.get("examples", [])) > 2 else "",
            }
            for r in interaction_edges
        ])
        edges_df.to_csv(tables_dir / "interaction_edges.csv", index=False, encoding="utf-8")

    return {
        "comention_raw": comention_raw,
        "comention_jaccard": comention_jaccard,
        "interaction_matrix": interaction_matrix,
        "interaction_edges": interaction_edges,
    }
