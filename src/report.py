"""
Итоговый PDF-отчёт research-grade: корпус и очистка, метрики и нормировка,
R/O с confidence и unknown, эссенциализация с примерами, keyness, две сети, evidence, ограничения.
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FONTS_DIR = _PROJECT_ROOT / "resources" / "fonts"
_DEJAVU_LOCAL = _FONTS_DIR / "DejaVuSans.ttf"
_DEJAVU_URL = "https://github.com/prawnpdf/prawn-manual_builder/raw/master/data/fonts/DejaVuSans.ttf"
CYRILLIC_FONT_PATHS = [
    _DEJAVU_LOCAL,
    Path("resources/fonts/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
]


def _ensure_dejavu_downloaded() -> None:
    if _DEJAVU_LOCAL.exists():
        return
    try:
        _FONTS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_DEJAVU_URL, _DEJAVU_LOCAL)
    except Exception:
        pass


def _register_cyrillic_font() -> str:
    _ensure_dejavu_downloaded()
    for path in CYRILLIC_FONT_PATHS:
        p = Path(path)
        if p.exists():
            try:
                pdfmetrics.registerFont(TTFont("CyrillicFont", str(p)))
                return "CyrillicFont"
            except Exception:
                continue
    return "Helvetica"


def _escape(s: str) -> str:
    if not s:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return str(x)[:200]
    return str(x)


def build_report(
    corpus_description: str,
    key_frequencies: Dict[str, int],
    dominant_epithets: List[tuple],
    situation_distribution: Dict[str, int],
    representation_distribution: Dict[str, int],
    essentialization_level: Dict[str, int],
    limitations: List[str],
    image_paths: Dict[str, str],
    output_path: Optional[Path] = None,
    noise_stats: Optional[Dict[str, Any]] = None,
    normalized_rates_note: Optional[str] = None,
    share_unknown_uncertain: Optional[Dict[str, float]] = None,
    keyness_top_words: Optional[Dict[str, List[str]]] = None,
    essentialization_examples: Optional[List[Dict]] = None,
    evidence_examples: Optional[List[Dict]] = None,
    cluster_validation_note: Optional[str] = None,
) -> str:
    output_path = output_path or _PROJECT_ROOT / "output" / "final_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    font_name = _register_cyrillic_font()
    normal = styles["Normal"]
    heading = styles["Heading1"]
    h2 = styles.get("Heading2", heading)
    if font_name != "Helvetica":
        for st in (normal, heading, h2):
            st.fontName = font_name
    story = []

    story.append(Paragraph(_escape("Итоговый отчёт: цифровой анализ травелогов о Сибири"), heading))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("1. Корпус и очистка"), h2))
    story.append(Paragraph(_escape(_safe_str(corpus_description)), normal))
    if noise_stats:
        story.append(Paragraph(_escape(f"Удалено как header/footer: {noise_stats.get('removed', 0)}. Топ повторений: output/noise_top_repeats.csv."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("2. Метрики (counts и нормированные rates)"), h2))
    if key_frequencies:
        rows = [[_escape("Этнос"), _escape("Упоминаний")]] + [[_escape(k), str(v)] for k, v in sorted(key_frequencies.items(), key=lambda x: -x[1])]
        t = Table(rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    if normalized_rates_note:
        story.append(Paragraph(_escape(normalized_rates_note), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("3. Репрезентации R (с confidence и доля uncertain)"), h2))
    if representation_distribution:
        rows = [[_escape("Тип"), _escape("Количество")]] + [[_escape(k), str(v)] for k, v in sorted(representation_distribution.items(), key=lambda x: -x[1])]
        t = Table(rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    if share_unknown_uncertain:
        story.append(Paragraph(_escape(f"Доля uncertain: {share_unknown_uncertain.get('uncertain', 0):.1%}. Доля unknown по O: {share_unknown_uncertain.get('unknown_O', 0):.1%}."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("4. Ситуации O (доля unknown/mixed)"), h2))
    if situation_distribution:
        rows = [[_escape("Тип ситуации"), _escape("Количество")]] + [[_escape(k), str(v)] for k, v in sorted(situation_distribution.items(), key=lambda x: -x[1])]
        t = Table(rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("5. Эссенциализация (частоты и примеры)"), h2))
    if essentialization_level:
        rows = [[_escape("Этнос"), _escape("Конструкций")]] + [[_escape(k), str(v)] for k, v in sorted(essentialization_level.items(), key=lambda x: -x[1])]
        t = Table(rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
    for ex in (essentialization_examples or [])[:5]:
        story.append(Paragraph(_escape("Пример: " + _safe_str(ex.get("sentence_text", ex.get("span", "")))[:200]), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("6. Keyness (слова отличия negative/exotic vs neutral)"), h2))
    if keyness_top_words:
        for label, words in list(keyness_top_words.items())[:3]:
            story.append(Paragraph(_escape(label + ": " + ", ".join(words[:15])), normal))
    story.append(Paragraph(_escape("Таблицы: output/keyness_*.csv. Визуализации: keyness_*_top20.png."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("7. Сети: co-mention (без самопетель) и interaction (направленная)"), h2))
    story.append(Paragraph(_escape("Co-mention: network_comention_raw.png, network_comention_jaccard.png. Interaction: network_interaction_directed.png, interaction_edges.csv."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("8. Evidence pack (примеры для валидации)"), h2))
    for ex in (evidence_examples or [])[:5]:
        story.append(Paragraph(_escape(_safe_str(ex.get("sentence_text", ""))[:250]), normal))
    story.append(Paragraph(_escape("Полный набор: output/evidence_pack.xlsx."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("9. Embedding-валидация"), h2))
    if cluster_validation_note:
        story.append(Paragraph(_escape(cluster_validation_note), normal))
    else:
        story.append(Paragraph(_escape("Запуск с --run-embeddings: cluster_validation.json, cluster_profiles.md, umap_*.png."), normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(_escape("10. Ограничения и next steps"), h2))
    for lim in limitations:
        story.append(Paragraph("• " + _escape(_safe_str(lim)), normal))
    story.append(Spacer(1, 0.5 * cm))

    for desc, img_path in (image_paths or {}).items():
        if not img_path or not Path(img_path).exists():
            continue
        try:
            story.append(Paragraph(_escape(desc), h2))
            try:
                reader = ImageReader(img_path)
                iw, ih = reader.getSize()
                if iw and ih:
                    scale = min(14 * cm / iw, 8 * cm / ih)
                    w, h = iw * scale, ih * scale
                else:
                    w, h = 14 * cm, 8 * cm
                story.append(Image(img_path, width=w, height=h))
            except Exception:
                story.append(Image(img_path, width=14 * cm, height=8 * cm))
            story.append(Spacer(1, 0.5 * cm))
        except Exception:
            story.append(Paragraph(_escape(f"{desc} (изображение не загружено)"), normal))

    doc.build(story)
    return str(output_path)


def build_report_from_pipeline(
    doc_list: List[Dict],
    piro_records: List[Dict],
    essentialization_table: Dict[str, int],
    interaction_matrix: Dict[str, Dict[str, int]],
    llm_analysis: Dict[str, Any],
    image_paths: Dict[str, str],
    output_path: Optional[Path] = None,
    raw_df_len: int = 0,
    clean_df_len: int = 0,
    share_uncertain: float = 0,
    share_unknown_o: float = 0,
    keyness_top: Optional[Dict[str, List[str]]] = None,
    cluster_validation: Optional[Dict] = None,
) -> str:
    num_docs = len(doc_list)
    num_sents = sum(len(d.get("sentences", [])) for d in doc_list)
    corpus_description = f"Документов: {num_docs}. Предложений: {num_sents}. Упоминаний (raw): {raw_df_len or len(piro_records)}. После очистки: {clean_df_len or len(piro_records)}."
    key_frequencies = dict(Counter(r.get("P") for r in piro_records))
    epithet_counts = Counter()
    for r in piro_records:
        for e in r.get("epithets", []):
            epithet_counts[e] += 1
    dominant_epithets = epithet_counts.most_common(30)
    situation_distribution = dict(Counter(r.get("O_situation") for r in piro_records))
    representation_distribution = dict(Counter(r.get("R") for r in piro_records))
    limitations = llm_analysis.get("limitations", [])
    if not limitations:
        limitations = [
            "Корпус может быть малым для надёжных выводов.",
            "Классификация R основана на лексиконе.",
            "Эссенциализация по ограниченному набору конструкций.",
        ]
    noise_stats = None
    if raw_df_len and clean_df_len:
        noise_stats = {"removed": raw_df_len - clean_df_len}
    normalized_note = "Нормированные rates per 10k предложений: output и таблицы нормализации."
    share_unknown_uncertain = {"uncertain": share_uncertain, "unknown_O": share_unknown_o}
    ess_examples = [r for r in piro_records if r.get("is_essentializing")][:10]
    ess_examples = [{"sentence_text": r.get("sentence_text"), "span": r.get("essentialization_span")} for r in ess_examples]
    evidence_examples = [{"sentence_text": r.get("sentence_text")} for r in piro_records if r.get("R") in ("negative", "exotic")][:10]
    cluster_note = None
    if cluster_validation:
        cluster_note = f"Silhouette: {cluster_validation.get('silhouette')}. Purity R: {cluster_validation.get('purity_R')}. Purity O: {cluster_validation.get('purity_O_situation')}. Noise: {cluster_validation.get('noise_ratio')}."
    return build_report(
        corpus_description=corpus_description,
        key_frequencies=key_frequencies,
        dominant_epithets=dominant_epithets,
        situation_distribution=situation_distribution,
        representation_distribution=representation_distribution,
        essentialization_level=essentialization_table,
        limitations=limitations,
        image_paths=image_paths,
        output_path=output_path,
        noise_stats=noise_stats,
        normalized_rates_note=normalized_note,
        share_unknown_uncertain=share_unknown_uncertain,
        keyness_top_words=keyness_top,
        essentialization_examples=ess_examples,
        evidence_examples=evidence_examples,
        cluster_validation_note=cluster_note,
    )
