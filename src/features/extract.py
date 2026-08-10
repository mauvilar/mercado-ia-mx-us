"""Extracción de seniority y skills desde el texto de la vacante.

Se lee la descripción y no sólo el título porque el título miente (§1.4 del spec).
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils.config import load_skills

_SENIORITY_TITULO = [
    ("principal", r"\bprincipal\b"),
    ("staff", r"\bstaff\b"),
    ("lead", r"\b(lead|l[íi]der|head of)\b"),
    ("sr", r"\b(senior|sr\.?|s[ée]nior)\b"),
    ("jr", r"\b(junior|jr\.?|entry[- ]level|trainee|becari[oa])\b"),
    ("intern", r"\b(intern|internship|pasant[íi]a|practicante)\b"),
    ("mid", r"\b(mid[- ]level|semi[- ]?senior|intermedio)\b"),
]
_ANIOS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|años?)", re.I)


@lru_cache(maxsize=1)
def _lexico() -> list[tuple[str, re.Pattern[str]]]:
    lexico = load_skills()
    salida: list[tuple[str, re.Pattern[str]]] = []
    for grupo in ("ia_aplicada", "soporte"):
        for skill, alias in lexico[grupo].items():
            # Cada palabra admite plural simple (-s/-es): las descripciones reales
            # dicen "modelos de lenguaje" o "bases de datos vectoriales", no la
            # forma singular exacta que vive en el YAML.
            patron = "|".join(
                r"\b"
                + r"\s+".join(rf"{re.escape(palabra)}(?:es|s)?" for palabra in a.split())
                + r"\b"
                for a in alias
            )
            salida.append((skill, re.compile(patron, re.I)))
    return salida


def extraer_skills(texto: str) -> list[str]:
    if not texto:
        return []
    encontradas = {skill for skill, patron in _lexico() if patron.search(texto)}
    return sorted(encontradas)


def extraer_seniority(title: str, description: str) -> str:
    titulo = title or ""
    for nivel, patron in _SENIORITY_TITULO:
        if re.search(patron, titulo, re.I):
            return nivel

    descripcion = description or ""
    for nivel, patron in _SENIORITY_TITULO:
        if re.search(patron, descripcion, re.I):
            return nivel

    match = _ANIOS.search(descripcion)
    if match:
        anios = int(match.group(1))
        if anios <= 2:
            return "jr"
        if anios <= 5:
            return "mid"
        return "sr"

    return "unknown"
