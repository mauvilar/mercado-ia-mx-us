"""Traduce respuestas crudas de cada fuente al esquema canónico.

Aísla el formato de la fuente: cuando Adzuna cambie un campo, sólo se toca este archivo.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any

from src.data.schema import CANONICAL_COLUMNS
from src.utils.config import load_cities

MONEDA_POR_PAIS = {"MX": "MXN", "US": "USD"}
# Asunción pendiente de confirmar con la sonda del día 0 (aún no corre: no hay
# API key de Adzuna en esta máquina): que Adzuna anualiza los importes en ambos países.
PERIODO_ADZUNA = "año"

_PALABRAS_ES = {" de ", " para ", " con ", " en ", " y ", "experiencia", "conocimientos"}
_REMOTO = re.compile(r"\b(remote|remoto|home office|teletrabajo|trabajo a distancia)\b", re.I)


def _canonizar_ciudad(country: str, area: list[str], display: str) -> tuple[str | None, str | None]:
    """Devuelve (ciudad_canónica, metro). None si la ubicación no es una de las zonas objetivo."""
    candidatos = [*area, display]
    texto = " | ".join(str(c) for c in candidatos)
    for entrada in load_cities().get(country, []):
        nombres = [entrada["canonical"], *entrada["aliases"]]
        for nombre in nombres:
            if re.search(rf"\b{re.escape(nombre)}\b", texto, re.I):
                return entrada["canonical"], entrada["metro"]
    return None, None


def _detectar_idioma(texto: str) -> str:
    bajo = f" {texto.lower()} "
    return "es" if sum(p in bajo for p in _PALABRAS_ES) >= 2 else "en"


def _a_fecha(valor: str | None) -> date | None:
    if not valor:
        return None
    try:
        return datetime.fromisoformat(valor.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def map_adzuna_row(raw: dict[str, Any], *, country: str, snapshot_date: date) -> dict[str, Any]:
    source = f"adzuna_{country.lower()}"
    posting_id = hashlib.sha1(f"{source}:{raw.get('id')}".encode()).hexdigest()

    area = raw.get("location", {}).get("area", []) or []
    display = raw.get("location", {}).get("display_name", "") or ""
    ciudad, metro = _canonizar_ciudad(country, area, display)

    titulo = raw.get("title", "") or ""
    descripcion = raw.get("description", "") or ""
    texto = f"{titulo} {display} {descripcion}"

    salary_min = raw.get("salary_min")
    salary_max = raw.get("salary_max")
    # Adzuna manda el flag como string "0"/"1". Tratarlo como bool de Python
    # convertiría "0" en True y contaminaría todo el análisis.
    predicho = str(raw.get("salary_is_predicted", "0")) == "1"
    tiene_salario = salary_min is not None

    fila: dict[str, Any] = dict.fromkeys(CANONICAL_COLUMNS)
    fila.update(
        posting_id=posting_id,
        snapshot_date=snapshot_date,
        source=source,
        title_raw=titulo,
        title_norm=titulo.strip().lower(),
        company=(raw.get("company") or {}).get("display_name"),
        category=(raw.get("category") or {}).get("label"),
        country=country,
        state=area[1] if len(area) > 1 else None,
        city=ciudad,
        metro=metro,
        is_remote=bool(_REMOTO.search(texto)),
        posted_date=_a_fecha(raw.get("created")),
        salary_min_raw=salary_min,
        salary_max_raw=salary_max,
        salary_currency=MONEDA_POR_PAIS[country],
        salary_period=PERIODO_ADZUNA if tiene_salario else None,
        salary_is_predicted=predicho,
        salary_observed=bool(tiene_salario and not predicho),
        description_text=descripcion,
        description_lang=_detectar_idioma(descripcion),
        url=raw.get("redirect_url"),
    )
    return fila
