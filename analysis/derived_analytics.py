"""
Второй аналитический слой: профили этносов, индексы, тесты, корреляции, кластеры.
Читает данные из pipeline (piro_clean, essentialization_table, interaction_edges) и пишет
артефакты в output/derived/ и output/tables/. Не изменяет corpus.db и не ломает пайплайн.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DERIVED = PROJECT_ROOT / "output" / "derived"
OUTPUT_TABLES = PROJECT_ROOT / "output" / "tables"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "derived_analytics.log"
NORMALIZATION_PER = 10_000  # предложений
BOOTSTRAP_N = 500
TOP_N_ETHNOS = 10


def _log(msg: str, log_path: Optional[Path] = None) -> None:
    logging.info(msg)
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} {msg}\n")
        except Exception:
            pass


def build_mentions_df_from_piro(
    piro_clean: List[Dict],
    clean_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Строит DataFrame «упоминаний» из piro_clean (один ряд = одно упоминание).
    Поля: ethnos, R, O, is_essentializing, is_noise, sentence_text, context_text, file_name, sent_idx.
    Если clean_df передан — дополняем doc_sent_count для нормализации.
    """
    if not piro_clean:
        return pd.DataFrame()
    rows = []
    for r in piro_clean:
        rows.append({
            "ethnos": r.get("P") or r.get("ethnos_norm") or "",
            "R": r.get("R") or "",
            "O": r.get("O_situation") or "",
            "is_essentializing": bool(r.get("is_essentializing", False)),
            "is_noise": bool(r.get("is_noise", False)),
            "sentence_text": r.get("sentence_text") or r.get("context_text") or "",
            "context_text": r.get("context_text") or r.get("sentence_text") or "",
            "file_name": r.get("file_name") or "",
            "sent_idx": r.get("sent_idx"),
        })
    df = pd.DataFrame(rows)
    if clean_df is not None and "doc_sent_count" in clean_df.columns:
        # merge doc_sent_count by file_name if needed for norm base
        pass
    return df


def load_mentions_from_db(conn) -> pd.DataFrame:
    """
    Загружает DataFrame упоминаний из источника данных.
    conn: путь к pipeline.db (Path/str) или словарь pipeline_data (результат load_pipeline).
    В этом проекте R/O/essentialization хранятся в pipeline.db (артефакты piro_clean), не в corpus.db.
    Возвращает DataFrame с колонками ethnos, R, O, is_essentializing, is_noise, ...; отсутствующие поля — NaN.
    """
    if isinstance(conn, (Path, str)):
        path = Path(conn)
        if not path.exists():
            return pd.DataFrame()
        try:
            from src.pipeline_db import load_pipeline
            data = load_pipeline(path)
            if not data:
                return pd.DataFrame()
            return build_mentions_df_from_piro(data.get("piro_clean") or [], data.get("clean_df"))
        except Exception:
            return pd.DataFrame()
    if isinstance(conn, dict):
        return build_mentions_df_from_piro(conn.get("piro_clean") or [], conn.get("clean_df"))
    return pd.DataFrame()


def load_data_for_profiles(
    pipeline_data: Optional[Dict[str, Any]] = None,
    pipeline_db_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Загружает данные для профилей: либо из переданного pipeline_data, либо из pipeline.db.
    Возвращает (mentions_df, extras): extras = essentialization_table, interaction_edges, corpus (для n_sents).
    """
    if pipeline_data:
        piro = pipeline_data.get("piro_clean") or []
        clean_df = pipeline_data.get("clean_df")
        corpus = pipeline_data.get("corpus") or []
        essentialization_table = pipeline_data.get("essentialization_table") or {}
        interaction_edges = pipeline_data.get("interaction_edges") or []
    else:
        pipeline_db_path = pipeline_db_path or (PROJECT_ROOT / "output" / "pipeline.db")
        if not pipeline_db_path.exists():
            return pd.DataFrame(), {}
        try:
            from src.pipeline_db import load_pipeline
            pipeline_data = load_pipeline(pipeline_db_path)
            if not pipeline_data:
                return pd.DataFrame(), {}
            return load_data_for_profiles(pipeline_data=pipeline_data, pipeline_db_path=None)
        except Exception:
            return pd.DataFrame(), {}

    df = build_mentions_df_from_piro(piro, clean_df)
    n_sents = sum(len(d.get("sentences", [])) for d in corpus) if corpus else 0
    extras = {
        "essentialization_table": pipeline_data.get("essentialization_table") or {},
        "interaction_edges": pipeline_data.get("interaction_edges") or [],
        "n_sents": n_sents,
        "corpus": corpus,
    }
    return df, extras


def compute_normalization_base(df: pd.DataFrame, n_sents: int = 0) -> Dict[str, Any]:
    """
    Выбор базы нормализации: per 10k sentences. Возвращает dict для run_config.
    """
    return {
        "normalization": "per_10k_sentences",
        "normalization_per": NORMALIZATION_PER,
        "n_sents_total": n_sents,
    }


def compute_ethnic_profiles(
    df: pd.DataFrame,
    essentialization_table: Dict[str, int],
    interaction_edges: List[Dict],
    n_sents: int,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Профили по этносам: mentions_raw, mentions_norm, OI, EPS, ED, AS, shares, deltas.
    """
    if df.empty:
        return pd.DataFrame()

    # Sanity: топ-этнос с mentions_raw < 5
    ethnos_counts = df["ethnos"].value_counts()
    if not ethnos_counts.empty and ethnos_counts.iloc[0] < 5:
        _log(f"Предупреждение: топ-этнос имеет mentions_raw = {ethnos_counts.iloc[0]} (< 5)", log_path)

    total_mentions = len(df)
    n_sents_base = n_sents or 1
    norm_factor = NORMALIZATION_PER / n_sents_base

    # AS по узлам (out - in) / (out + in)
    out_d: Dict[str, int] = defaultdict(int)
    in_d: Dict[str, int] = defaultdict(int)
    for e in interaction_edges:
        s, d = e.get("src"), e.get("dst")
        cnt = e.get("count", 1)
        if s:
            out_d[s] += cnt
        if d:
            in_d[d] += cnt
    total_edges = sum(out_d.values()) + sum(in_d.values()) or 1
    as_scores = {}
    for node in set(out_d.keys()) | set(in_d.keys()):
        o, i = out_d.get(node, 0), in_d.get(node, 0)
        tot = o + i
        as_scores[node] = (o - i) / tot if tot else 0.0

    profiles = []
    for ethnos, grp in df.groupby("ethnos"):
        if not ethnos:
            continue
        n = len(grp)
        if n == 0:
            continue
        # OI = (negative + exotic) / total
        r_counts = grp["R"].value_counts()
        neg = int(r_counts.get("negative", 0))
        exo = int(r_counts.get("exotic", 0))
        pos = int(r_counts.get("positive", 0))
        unc = int(r_counts.get("uncertain", 0))
        oi = (neg + exo) / n if n else 0.0
        eps = (neg - pos) / n if n else 0.0
        uncertain_R_share = unc / n if n else 0.0
        o_counts = grp["O"].value_counts()
        unknown_o = int(o_counts.get("unknown", 0)) + int(o_counts.get("mixed", 0))
        unknown_O_share = unknown_o / n if n else 0.0
        ess_count = essentialization_table.get(ethnos, 0)
        ed = ess_count / n if n else 0.0
        as_val = as_scores.get(ethnos, np.nan)
        if np.isnan(as_val) and total_edges:
            as_val = 0.0
        noise_n = grp["is_noise"].sum()
        noise_share = noise_n / n if n else 0.0

        profiles.append({
            "ethnos": ethnos,
            "mentions_raw": n,
            "mentions_norm": round(n * norm_factor, 4),
            "OI": round(oi, 4),
            "EPS": round(eps, 4),
            "ED": round(ed, 4),
            "AS": round(as_val, 4) if not np.isnan(as_val) else None,
            "uncertain_R_share": round(uncertain_R_share, 4),
            "unknown_O_share": round(unknown_O_share, 4),
            "noise_share": round(noise_share, 4),
        })

    pro_df = pd.DataFrame(profiles)
    if pro_df.empty:
        return pro_df

    # Sanity: много NaN в профилях
    nan_share = pro_df[["OI", "ED", "EPS", "AS"]].isna().sum().sum() / (4 * len(pro_df))
    if nan_share > 0.5:
        _log(f"Предупреждение: высокая доля NaN в профилях ({nan_share:.2%})", log_path)

    # Weighted mean for deltas (weight = mentions_raw)
    for col in ["OI", "EPS", "ED", "AS"]:
        if col not in pro_df.columns:
            continue
        vals = pro_df[col].dropna()
        if vals.empty:
            continue
        w = pro_df.loc[vals.index, "mentions_raw"]
        mean_val = (vals * w).sum() / w.sum() if w.sum() else vals.mean()
        pro_df[f"{col}_delta"] = pro_df[col] - mean_val
        pro_df[f"{col}_delta"] = pro_df[f"{col}_delta"].round(4)

    return pro_df


def compute_stats_tests(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Chi-square Ethnos×R, Ethnos×O, Cramér's V, bootstrap 95% CI для OI/EPS/ED по топ-N этносам.
    """
    out: Dict[str, Any] = {"chi2_R": None, "chi2_O": None, "bootstrap_CI": {}}
    if df.empty or profiles.empty:
        return out

    # Chi-square Ethnos × R
    try:
        ct_r = pd.crosstab(df["ethnos"], df["R"])
        if ct_r.size > 0:
            from scipy.stats import chi2_contingency
            chi2, p, dof, _ = chi2_contingency(ct_r)
            n = ct_r.sum().sum()
            min_dim = min(ct_r.shape) - 1
            cramer = np.sqrt(chi2 / (n * min_dim)) if n and min_dim > 0 else 0.0
            out["chi2_R"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": round(float(cramer), 4)}
    except Exception as e:
        _log(f"Chi2 R: {e}", log_path)

    # Chi-square Ethnos × O
    try:
        ct_o = pd.crosstab(df["ethnos"], df["O"])
        if ct_o.size > 0:
            from scipy.stats import chi2_contingency
            chi2, p, dof, _ = chi2_contingency(ct_o)
            n = ct_o.sum().sum()
            min_dim = min(ct_o.shape) - 1
            cramer = np.sqrt(chi2 / (n * min_dim)) if n and min_dim > 0 else 0.0
            out["chi2_O"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": round(float(cramer), 4)}
    except Exception as e:
        _log(f"Chi2 O: {e}", log_path)

    # Bootstrap 95% CI for OI, EPS, ED — по всем этносам (включая малые выборки)
    rng = np.random.default_rng(42)
    for eth in profiles["ethnos"].tolist():
        sub = df[df["ethnos"] == eth]
        n = len(sub)
        if n == 0:
            continue
        if n < 2:
            # Точечная оценка для n=1 (интервал не имеет смысла)
            oi_val = (sub["R"].isin(["negative", "exotic"])).astype(float).mean()
            eps_val = sub["R"].apply(lambda r: 1.0 if r == "negative" else (-1.0 if r == "positive" else 0.0)).mean()
            ed_val = sub["is_essentializing"].astype(float).mean()
            out["bootstrap_CI"][eth] = {
                "OI": {"low": round(oi_val, 4), "high": round(oi_val, 4)},
                "EPS": {"low": round(eps_val, 4), "high": round(eps_val, 4)},
                "ED": {"low": round(ed_val, 4), "high": round(ed_val, 4)},
            }
            continue
        ci_oi = _bootstrap_ci(sub["R"].apply(lambda r: 1.0 if r in ("negative", "exotic") else 0.0), n, rng)
        ci_eps = _bootstrap_ci(sub["R"].apply(lambda r: 1.0 if r == "negative" else (-1.0 if r == "positive" else 0.0)), n, rng)
        ci_ed = _bootstrap_ci(sub["is_essentializing"].astype(float), n, rng)
        out["bootstrap_CI"][eth] = {"OI": ci_oi, "EPS": ci_eps, "ED": ci_ed}
    return out


def _bootstrap_ci(series: pd.Series, n: int, rng: np.random.Generator) -> Dict[str, float]:
    arr = series.values
    vals = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(arr[idx])))
    return {"low": round(float(np.percentile(vals, 2.5)), 4), "high": round(float(np.percentile(vals, 97.5)), 4)}


def compute_correlations(profiles: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman для OI, ED, EPS, AS, mentions_norm."""
    num_cols = ["OI", "ED", "EPS", "AS", "mentions_norm"]
    available = [c for c in num_cols if c in profiles.columns and profiles[c].notna().any()]
    if len(available) < 2:
        return pd.DataFrame()
    sub = profiles[available].dropna(how="all")
    if len(sub) < 3:
        return pd.DataFrame()
    from scipy.stats import pearsonr, spearmanr
    rows = []
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            x, y = sub[a], sub[b]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            xv, yv = x[valid], y[valid]
            if xv.std() == 0 or yv.std() == 0:
                rows.append({"var1": a, "var2": b, "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None})
                continue
            r_p, p_p = pearsonr(xv, yv)
            r_s, p_s = spearmanr(xv, yv)
            rows.append({"var1": a, "var2": b, "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 4), "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 4)})
    return pd.DataFrame(rows)


def cluster_ethnos(profiles: pd.DataFrame, log_path: Optional[Path] = None) -> pd.DataFrame:
    """KMeans k=2..5, выбрать по silhouette. Сохранить ethnos, cluster_id."""
    feat_cols = ["OI", "ED", "EPS", "AS"]
    available = [c for c in feat_cols if c in profiles.columns]
    if len(available) < 2:
        return pd.DataFrame(columns=["ethnos", "cluster_id"])
    X = profiles[["ethnos"] + available].dropna(subset=available)
    if len(X) < 3:
        return pd.DataFrame(columns=["ethnos", "cluster_id"])
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X_mat = X[available].values
    best_k, best_sil, best_labels = 2, -1.0, None
    for k in range(2, 6):
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X_mat)
        sil = silhouette_score(X_mat, lab)
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, lab
    out = pd.DataFrame({"ethnos": X["ethnos"].values, "cluster_id": best_labels})
    _log(f"Кластеры: k={best_k}, silhouette={best_sil:.4f}", log_path)
    return out


def save_all_outputs(
    profiles: pd.DataFrame,
    tests: Dict[str, Any],
    correlations: pd.DataFrame,
    clusters: pd.DataFrame,
    derived_indices: Dict[str, Any],
    run_config: Dict[str, Any],
    out_derived: Path,
    out_tables: Path,
    log_path: Optional[Path] = None,
) -> None:
    """Сохраняет все артефакты в output/derived/ и output/tables/."""
    out_derived.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    if not profiles.empty:
        profiles.to_csv(out_tables / "ethnic_profiles.csv", index=False, encoding="utf-8")
        try:
            profiles.to_excel(out_tables / "ethnic_profiles.xlsx", index=False, engine="openpyxl")
        except Exception as e:
            _log(f"Excel ethnic_profiles: {e}", log_path)
    (out_derived / "derived_indices.json").write_text(json.dumps(derived_indices, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_derived / "stats_tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")
    if not correlations.empty:
        correlations.to_csv(out_derived / "correlations.csv", index=False, encoding="utf-8")
    if not clusters.empty:
        clusters.to_csv(out_derived / "ethnos_clusters.csv", index=False, encoding="utf-8")
    run_config["timestamp"] = datetime.now().isoformat()
    (out_derived / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("Сохранили: ethnic_profiles.csv/xlsx, derived_indices.json, stats_tests.json, correlations.csv, ethnos_clusters.csv, run_config.json", log_path)


def run_derived_analytics(
    pipeline_data: Optional[Dict[str, Any]] = None,
    pipeline_db_path: Optional[Path] = None,
    output_derived: Optional[Path] = None,
    output_tables: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Главная точка входа: загружает данные, считает профили, тесты, корреляции, кластеры, сохраняет артефакты.
    Не меняет corpus.db. Возвращает dict с profiles_df, tests, correlations_df, clusters_df, derived_indices, run_config.
    """
    log_path = log_path or LOG_PATH
    out_derived = output_derived or OUTPUT_DERIVED
    out_tables = output_tables or OUTPUT_TABLES
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _log("Derived analytics: старт", log_path)

    df, extras = load_data_for_profiles(pipeline_data=pipeline_data, pipeline_db_path=pipeline_db_path)
    if df.empty:
        _log("Нет данных для профилей (пустой piro_clean или pipeline).", log_path)
        return {}

    n_sents = extras.get("n_sents") or 0
    essentialization_table = extras.get("essentialization_table") or {}
    interaction_edges = extras.get("interaction_edges") or []

    run_config = compute_normalization_base(df, n_sents)
    run_config["n_mentions"] = len(df)
    run_config["n_ethnos"] = df["ethnos"].nunique()

    profiles = compute_ethnic_profiles(df, essentialization_table, interaction_edges, n_sents, log_path)
    if profiles.empty:
        _log("Профили пусты.", log_path)
        return {}

    # OI/ED/EPS/AS в формате derived_indices для отчёта
    derived_indices = {
        "OI": {"raw_OI": dict(zip(profiles["ethnos"], profiles["OI"])), "normalized_OI": {}, "confidence_interval": {}},
        "ED": dict(zip(profiles["ethnos"], profiles["ED"])),
        "EPS": dict(zip(profiles["ethnos"], profiles["EPS"])),
        "AS": dict(zip(profiles["ethnos"], profiles["AS"].fillna(0))),
        "mentions_per_ethnos": dict(zip(profiles["ethnos"], profiles["mentions_raw"])),
        "formulas": {"OI": "(negative+exotic)/total", "EPS": "(negative-positive)/total", "ED": "essentializing/total", "AS": "(out-in)/(out+in)"},
        "normalization_base": run_config,
    }
    # Bootstrap CI для derived_indices
    tests = compute_stats_tests(df, profiles, log_path)
    for eth, ci in (tests.get("bootstrap_CI") or {}).items():
        derived_indices["OI"]["confidence_interval"][eth] = ci.get("OI", {})

    correlations = compute_correlations(profiles)
    clusters = cluster_ethnos(profiles, log_path)

    save_all_outputs(profiles, tests, correlations, clusters, derived_indices, run_config, out_derived, out_tables, log_path)
    _log("Derived analytics: готово", log_path)
    return {
        "profiles_df": profiles,
        "tests": tests,
        "correlations_df": correlations,
        "clusters_df": clusters,
        "derived_indices": derived_indices,
        "run_config": run_config,
    }
