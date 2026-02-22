"""
Загрузка базы знаний (domain briefing) для подстановки в системный промпт.
Позволяет получать интерпретацию в рамках ориентализма, травелогов, постколониальных
и новой имперской истории за счёт явной теоретической рамки в промпте.
"""

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "resources" / "knowledge"
DOMAIN_BRIEFING_PATH = KNOWLEDGE_DIR / "domain_briefing.txt"

# Плейсхолдер в system_prompt.txt: сюда подставляется содержимое domain_briefing
PLACEHOLDER_THEORY = "{{DOMAIN_KNOWLEDGE}}"


def load_domain_briefing(path: Optional[Path] = None) -> str:
    """
    Загружает текст теоретического брифинга из resources/knowledge/domain_briefing.txt.
    Возвращает пустую строку, если файл отсутствует.
    """
    p = path or DOMAIN_BRIEFING_PATH
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def build_system_prompt_with_knowledge(
    base_system_prompt: str,
    domain_briefing: Optional[str] = None,
    placeholder: str = PLACEHOLDER_THEORY,
) -> str:
    """
    Подставляет базу знаний в базовый системный промпт.
    Если в base_system_prompt есть placeholder ({{DOMAIN_KNOWLEDGE}}), он заменяется на domain_briefing.
    Если domain_briefing не передан, загружается из DOMAIN_BRIEFING_PATH.
    Если placeholder отсутствует в base_system_prompt, то блок знаний добавляется перед правилами работы
    (после первого абзаца или после строки «Правила работы:»).
    """
    briefing = domain_briefing if domain_briefing is not None else load_domain_briefing()
    if placeholder in base_system_prompt:
        return base_system_prompt.replace(placeholder, briefing if briefing else "(Теоретическая рамка не загружена. Интерпретируй данные в контексте ориентализма и репрезентации «других» в травелогах.)")
    if not briefing:
        return base_system_prompt
    # Вставить блок знаний после первого абзаца
    lines = base_system_prompt.split("\n")
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Правила работы:") or (i > 0 and lines[i - 1].strip() == "" and line.strip()):
            insert_at = i
            break
        if i >= 2:
            insert_at = i + 1
            break
    head = "\n".join(lines[:insert_at])
    tail = "\n".join(lines[insert_at:])
    return f"{head}\n\n{briefing}\n\n{tail}"


def get_system_prompt(use_knowledge: bool = True, base_prompt: Optional[str] = None) -> str:
    """
    Возвращает системный промпт для LLM: базовый (из system_prompt.txt) с опциональной подстановкой базы знаний.
    use_knowledge=True — загрузить domain_briefing и вставить в промпт (через placeholder или перед правилами).
    """
    from .deepseek_client import load_system_prompt as load_base
    base = base_prompt or load_base()
    if not use_knowledge:
        return base
    return build_system_prompt_with_knowledge(base)
