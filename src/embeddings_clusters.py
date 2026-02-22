"""
Embedding + кластеризация для валидации (не замена разметки).
Кеш: output/embeddings_cache.parquet.
HDBSCAN или KMeans(k=8), UMAP 2D, картинки: clusters, by R, by O_situation.
Метрики: silhouette, purity по R/O, доля шума. output/cluster_validation.json, output/cluster_profiles.md.
Запуск только по флагу --run-embeddings.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


def get_embeddings(
    context_texts: List[str],
    cache_path: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Загружает или вычисляет эмбеддинги, кеширует в parquet."""
    cache_path = cache_path or Path(__file__).resolve().parent.parent / "output" / "embeddings_cache.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Установите sentence-transformers: pip install sentence-transformers")

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            dim_cols = [c for c in cached.columns if c.startswith("dim_")]
            if dim_cols and len(cached) == len(context_texts):
                return cached[dim_cols].values.astype(np.float32)
        except Exception:
            pass

    model = SentenceTransformer(model_name)
    arr = model.encode(context_texts, show_progress_bar=len(context_texts) > 100)
    arr = np.asarray(arr, dtype=np.float32)
    df_cache = pd.DataFrame(arr, columns=[f"dim_{i}" for i in range(arr.shape[1])])
    try:
        df_cache.to_parquet(cache_path, index=False)
    except Exception:
        pass
    return arr


def cluster_embeddings(
    X: np.ndarray,
    method: str = "hdbscan",
    k: int = 8,
) -> np.ndarray:
    """Метки кластеров. HDBSCAN: шум = -1."""
    if method == "hdbscan":
        try:
            import hdbscan
            labels = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(X)
            return labels
        except ImportError:
            method = "kmeans"
    if method == "kmeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        return labels
    return np.zeros(len(X), dtype=int)


def run_umap(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, random_state=42).fit_transform(X)


def plot_umap(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def purity(labels: np.ndarray, true_labels: List[str]) -> float:
    """Средняя чистота кластеров по true_labels."""
    from collections import Counter
    true_ser = pd.Series(true_labels)
    total = 0
    for u in np.unique(labels):
        if u < 0:
            continue
        mask = labels == u
        c = Counter(true_ser[mask].dropna())
        if c:
            total += c.most_common(1)[0][1]
    n_valid = (labels >= 0).sum()
    return total / n_valid if n_valid else 0.0


def run_embeddings_pipeline(
    clean_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    n_sample: Optional[int] = 5000,
    k: int = 8,
) -> Dict[str, Any]:
    """
    Вход: clean_df с context_text, R, O_situation.
    Считает эмбеддинги, кластеризует, UMAP, картинки, метрики.
    Возвращает cluster_validation dict и пути к картинкам.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "embeddings_cache.parquet"

    texts = clean_df["context_text"].fillna("").astype(str).tolist()
    if n_sample and len(texts) > n_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=n_sample, replace=False)
        texts = [texts[i] for i in idx]
        clean_sub = clean_df.iloc[idx]
    else:
        clean_sub = clean_df

    X = get_embeddings(texts, cache_path=cache_path)
    labels = cluster_embeddings(X, method="hdbscan", k=k)
    X_2d = run_umap(X)

    R_vals = clean_sub["R"].fillna("").astype(str).tolist()
    O_vals = clean_sub["O_situation"].fillna("").astype(str).tolist()

    plot_umap(X_2d, labels, "Clusters (HDBSCAN/KMeans)", output_dir / "umap_clusters.png")
    R_numeric = pd.Series(R_vals).astype("category").cat.codes.values
    plot_umap(X_2d, R_numeric, "Colored by R", output_dir / "umap_by_R.png")
    O_numeric = pd.Series(O_vals).astype("category").cat.codes.values
    plot_umap(X_2d, O_numeric, "Colored by O_situation", output_dir / "umap_by_O.png")

    try:
        from sklearn.metrics import silhouette_score
        mask = labels >= 0
        sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0.0
    except Exception:
        sil = None
    noise_ratio = (labels < 0).mean() if labels.min() < 0 else 0.0
    pur_R = purity(labels, R_vals)
    pur_O = purity(labels, O_vals)

    validation = {
        "silhouette": sil,
        "purity_R": round(pur_R, 4),
        "purity_O_situation": round(pur_O, 4),
        "noise_ratio": round(noise_ratio, 4),
        "n_points": int(len(labels)),
    }
    with open(output_dir / "cluster_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    with open(output_dir / "cluster_profiles.md", "w", encoding="utf-8") as f:
        f.write("# Cluster profiles\n")
        for u in np.unique(labels):
            if u < 0:
                f.write(f"\n## Noise (label {u})\n")
            else:
                f.write(f"\n## Cluster {u}\n")
            mask = labels == u
            sub = clean_sub.iloc[np.where(mask)[0]]
            f.write(f"Size: {mask.sum()}\n")
            if "sentence_text" in sub.columns:
                for _, row in sub.head(3).iterrows():
                    f.write(f"- {row.get('sentence_text', '')[:200]}...\n")

    umap_xy = X_2d.tolist()
    labels_list = labels.tolist()
    return {
        "validation": validation,
        "labels": labels_list,
        "umap_xy": umap_xy,
        "n_points": len(labels_list),
    }
