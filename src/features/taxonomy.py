"""Clasificación núcleo / anillo / fuera. Ver §5.1 del spec."""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils.config import load_skills, load_taxonomy


@lru_cache(maxsize=1)
def _patrones() -> tuple[re.Pattern[str], re.Pattern[str], int, frozenset[str]]:
    tax = load_taxonomy()
    ia = re.compile("|".join(tax["nucleo"]["terminos_ia"]), re.I)
    rol = re.compile("|".join(tax["nucleo"]["terminos_rol"]), re.I)
    return ia, rol, tax["anillo"]["min_skills"], frozenset(load_skills()["ia_aplicada"])


def clasificar(title: str, *, skills: list[str]) -> str:
    ia, rol, min_skills, skills_ia = _patrones()
    titulo = title or ""

    if ia.search(titulo) and rol.search(titulo):
        return "nucleo"

    if len([s for s in skills if s in skills_ia]) >= min_skills:
        return "anillo"

    return "fuera"
