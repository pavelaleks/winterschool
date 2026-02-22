"""
Визуализации: частотность этносов, heatmap репрезентация/ситуация, wordcloud, таблица эссенциализации, сеть.
Все подписи на русском, белый фон, 300 dpi, сдержанная палитра.
"""

from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Академическая сдержанная палитра
COLORS = ["#4a6fa5", "#7d8b9e", "#c4a77d", "#8b7355", "#6b5b4f", "#4a7c59", "#9b8b6e", "#5c6b7a", "#8b6914", "#6b8e8e"]
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10


def _get_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "output"


def plot_ethnonym_frequency(piro_records: List[Dict], output_dir: Path = None) -> str:
    """Частотность упоминаний этносов. Сохраняет в output/."""
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["P"] for r in piro_records)
    if not counts:
        return ""
    names = list(counts.keys())
    values = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, values, color=COLORS[: len(names)])
    ax.set_xlabel("Количество упоминаний")
    ax.set_ylabel("Этнос")
    ax.set_title("Частотность упоминаний этносов")
    fig.tight_layout()
    path = output_dir / "частотность_этносов.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def plot_heatmap_representation(
    piro_records: List[Dict],
    output_dir: Path = None,
) -> str:
    """Heatmap: этнос × тип репрезентации (R)."""
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: defaultdict(int))
    for r in piro_records:
        counts[r["P"]][r["R"]] += 1
    if not counts:
        return ""
    ethnonyms = sorted(counts.keys())
    rep_types = ["negative", "exotic", "positive", "neutral"]
    rep_ru = {"negative": "негативная", "exotic": "экзотизация", "positive": "позитивная", "neutral": "нейтральная"}
    M = np.array([[counts[e].get(r, 0) for r in rep_types] for e in ethnonyms])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(rep_types)))
    ax.set_xticklabels([rep_ru.get(r, r) for r in rep_types])
    ax.set_yticks(range(len(ethnonyms)))
    ax.set_yticklabels(ethnonyms)
    ax.set_xlabel("Тип репрезентации")
    ax.set_ylabel("Этнос")
    ax.set_title("Этнос × тип репрезентации")
    for i in range(len(ethnonyms)):
        for j in range(len(rep_types)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Количество")
    fig.tight_layout()
    path = output_dir / "heatmap_репрезентация.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def plot_heatmap_situation(
    piro_records: List[Dict],
    output_dir: Path = None,
) -> str:
    """Heatmap: этнос × тип ситуации (O_situation)."""
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: defaultdict(int))
    for r in piro_records:
        counts[r["P"]][r["O_situation"]] += 1
    if not counts:
        return ""
    ethnonyms = sorted(counts.keys())
    situations = sorted(set().union(*(set(counts[e].keys()) for e in counts)))
    M = np.array([[counts[e].get(s, 0) for s in situations] for e in ethnonyms])
    fig, ax = plt.subplots(figsize=(max(8, len(situations) * 0.8), 6))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(situations)))
    ax.set_xticklabels(situations, rotation=45, ha="right")
    ax.set_yticks(range(len(ethnonyms)))
    ax.set_yticklabels(ethnonyms)
    ax.set_xlabel("Тип ситуации")
    ax.set_ylabel("Этнос")
    ax.set_title("Этнос × тип ситуации")
    for i in range(len(ethnonyms)):
        for j in range(len(situations)):
            if M[i, j] > 0:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Количество")
    fig.tight_layout()
    path = output_dir / "heatmap_ситуация.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def plot_wordclouds(
    piro_records: List[Dict],
    output_dir: Path = None,
) -> List[str]:
    """Wordcloud: общий, негативный, экзотизирующий. Возвращает пути к файлам."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return []
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    all_text = " ".join(r.get("context", "") for r in piro_records)
    neg_text = " ".join(r.get("context", "") for r in piro_records if r.get("R") == "negative")
    exo_text = " ".join(r.get("context", "") for r in piro_records if r.get("R") == "exotic")
    for name, text, fname in [
        ("Общий контекст", all_text, "wordcloud_общий.png"),
        ("Негативная репрезентация", neg_text, "wordcloud_негативный.png"),
        ("Экзотизирующий контекст", exo_text, "wordcloud_экзотизация.png"),
    ]:
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(name)
        ax.axis("off")
        path = output_dir / fname
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        paths.append(str(path))
    return paths


def plot_essentialization_table(
    essentialization_table: Dict[str, int],
    output_dir: Path = None,
) -> str:
    """Таблица-картинка: эссенциализирующие конструкции по этносам."""
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not essentialization_table:
        return ""
    names = sorted(essentialization_table.keys())
    values = [essentialization_table[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color=COLORS[: len(names)])
    ax.set_ylabel("Количество эссенциализирующих конструкций")
    ax.set_xlabel("Этнос")
    ax.set_title("Эссенциализирующие конструкции по этносам")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    path = output_dir / "эссенциализация_по_этносам.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def plot_network(
    matrix: Dict[str, Dict[str, int]],
    output_dir: Path = None,
) -> str:
    """Сеть межэтнических взаимодействий. Файл: Сеть межэтнических взаимодействий.png"""
    try:
        import networkx as nx
    except ImportError:
        return ""
    output_dir = output_dir or _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    G = nx.DiGraph()
    for src, targets in matrix.items():
        for tgt, w in targets.items():
            if w > 0:
                G.add_edge(src, tgt, weight=w)
    if G.number_of_nodes() == 0:
        return ""
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42)
    edges = list(G.edges())
    weights = [G[u][v].get("weight", 1) for u, v in edges]
    max_w = max(weights) if weights else 1
    nx.draw_networkx_nodes(G, pos, node_color="lightsteelblue", edgecolors="gray", node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[2 + 2 * (w / max_w) for w in weights], alpha=0.6, edge_color="gray", arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    edge_labels = {(u, v): str(G[u][v].get("weight", 1)) for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    ax.set_title("Сеть межэтнических взаимодействий")
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / "Сеть межэтнических взаимодействий.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(path)


def run_all_visualizations(
    piro_records: List[Dict],
    essentialization_table: Dict[str, int],
    interaction_matrix: Dict[str, Dict[str, int]],
    output_dir: Path = None,
) -> Dict[str, str]:
    """Запускает все визуализации. Возвращает словарь {название: путь}."""
    output_dir = output_dir or _get_output_dir()
    out = {}
    p = plot_ethnonym_frequency(piro_records, output_dir)
    if p:
        out["частотность_этносов"] = p
    p = plot_heatmap_representation(piro_records, output_dir)
    if p:
        out["heatmap_репрезентация"] = p
    p = plot_heatmap_situation(piro_records, output_dir)
    if p:
        out["heatmap_ситуация"] = p
    for path in plot_wordclouds(piro_records, output_dir):
        out[Path(path).name] = path
    p = plot_essentialization_table(essentialization_table, output_dir)
    if p:
        out["эссенциализация"] = p
    p = plot_network(interaction_matrix, output_dir)
    if p:
        out["сеть_взаимодействий"] = p
    return out
