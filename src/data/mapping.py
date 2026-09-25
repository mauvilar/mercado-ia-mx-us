"""Traduce respuestas crudas de cada fuente al esquema canónico.

Aísla el formato de la fuente: cuando Adzuna cambie un campo, sólo se toca este archivo.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any

from src.data.salario_texto import parsear
from src.data.schema import CANONICAL_COLUMNS
from src.utils.config import load_cities, moneda_de

# Confirmado en la sonda del día 0: Adzuna anualiza los importes. La moneda ya no se
# resuelve aquí sino en config/countries.yml, porque el corpus dejó de ser bimonetario.
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


# Códigos de periodicidad de USAJOBS. "WC" es "Without Compensation": una vacante sin
# sueldo, que no es lo mismo que una vacante que no lo publica, pero para el conjunto
# observado da igual — no hay número de mercado que aprender de ella.
PERIODO_USAJOBS = {
    "PA": "año",
    "PM": "mes",
    "BW": "quincena",
    "PW": "semana",
    "PH": "hora",
}


def map_usajobs_row(item: dict[str, Any], *, snapshot_date: date) -> dict[str, Any]:
    """Un SearchResultItem de USAJOBS al esquema canónico.

    Todo lo que llega aquí trae salario real: el rango del grado de pago es obligatorio
    en la convocatoria federal. Por eso `salary_is_predicted` es siempre False, y no por
    omisión sino porque la fuente no modela nada.
    """
    d = item.get("MatchedObjectDescriptor", {}) or {}
    posting_id = hashlib.sha1(f"usajobs:{item.get('MatchedObjectId')}".encode()).hexdigest()

    ubicaciones = d.get("PositionLocation", []) or []
    primera = ubicaciones[0] if ubicaciones else {}
    ciudad_cruda = str(primera.get("CityName", "") or "")
    estado = primera.get("CountrySubDivisionCode")
    ciudad, metro = _canonizar_ciudad("US", [], ciudad_cruda)

    titulo = d.get("PositionTitle", "") or ""
    detalles = (d.get("UserArea", {}) or {}).get("Details", {}) or {}
    descripcion = str(detalles.get("JobSummary", "") or d.get("QualificationSummary", "") or "")
    texto = f"{titulo} {ciudad_cruda} {descripcion}"

    remuneracion = (d.get("PositionRemuneration", []) or [{}])[0]
    periodo = PERIODO_USAJOBS.get(str(remuneracion.get("RateIntervalCode", "")).strip())
    minimo = _a_float(remuneracion.get("MinimumRange"))
    maximo = _a_float(remuneracion.get("MaximumRange"))
    tiene_salario = bool(periodo and minimo and minimo > 0)

    categorias = d.get("JobCategory", []) or []

    fila: dict[str, Any] = dict.fromkeys(CANONICAL_COLUMNS)
    fila.update(
        posting_id=posting_id,
        snapshot_date=snapshot_date,
        source="usajobs",
        title_raw=titulo,
        title_norm=titulo.strip().lower(),
        company=d.get("OrganizationName"),
        category=categorias[0].get("Name") if categorias else None,
        country="US",
        state=estado,
        city=ciudad or (ciudad_cruda or None),
        metro=metro,
        is_remote=bool(_REMOTO.search(texto)),
        posted_date=_a_fecha(d.get("PublicationStartDate")),
        salary_min_raw=minimo if tiene_salario else None,
        salary_max_raw=maximo if tiene_salario else None,
        salary_currency="USD",
        salary_period=periodo if tiene_salario else None,
        salary_is_predicted=False,
        salary_observed=tiene_salario,
        description_text=descripcion,
        description_lang="en",
        url=d.get("PositionURI"),
    )
    return fila


def map_jooble_row(raw: dict[str, Any], *, country: str, snapshot_date: date) -> dict[str, Any]:
    """Una vacante de Jooble al esquema canónico, con la compuerta puesta.

    Jooble no separa el salario que publicó el empleador del que estimó el portal: los
    dos llegan en la misma cadena de texto. `salario_texto.parsear` decide, y su criterio
    es pesimista — cualquier marca de estimación, o una periodicidad que no se pueda
    leer, apaga `salary_observed` y enciende `salary_is_predicted`.
    """
    source = f"jooble_{country.lower()}"
    posting_id = hashlib.sha1(f"{source}:{raw.get('id')}".encode()).hexdigest()

    titulo = raw.get("title", "") or ""
    ubicacion = raw.get("location", "") or ""
    descripcion = raw.get("snippet", "") or ""
    texto = f"{titulo} {ubicacion} {descripcion}"

    ciudad, metro = _canonizar_ciudad(country, [], ubicacion)
    salario = parsear(raw.get("salary"), moneda_por_defecto=moneda_de(country))

    fila: dict[str, Any] = dict.fromkeys(CANONICAL_COLUMNS)
    fila.update(
        posting_id=posting_id,
        snapshot_date=snapshot_date,
        source=source,
        title_raw=titulo,
        title_norm=titulo.strip().lower(),
        company=raw.get("company") or None,
        category=raw.get("type") or None,
        country=country,
        state=None,
        city=ciudad,
        metro=metro,
        is_remote=bool(_REMOTO.search(texto)),
        posted_date=_a_fecha(raw.get("updated")),
        salary_min_raw=salario.minimo,
        salary_max_raw=salario.maximo,
        salary_currency=salario.moneda or moneda_de(country),
        salary_period=salario.periodo if salario.observado else None,
        # Todo lo que trae importe pero no pasa la compuerta se marca como predicho:
        # es un número que existe pero que no podemos atribuir al empleador.
        salary_is_predicted=bool(salario.minimo is not None and not salario.observado),
        salary_observed=salario.observado,
        description_text=descripcion,
        description_lang=_detectar_idioma(descripcion),
        url=raw.get("link"),
    )
    return fila


def _a_float(valor: Any) -> float | None:
    if valor is None or valor == "":
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
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
        salary_currency=moneda_de(country),
        salary_period=PERIODO_ADZUNA if tiene_salario else None,
        salary_is_predicted=predicho,
        salary_observed=bool(tiene_salario and not predicho),
        description_text=descripcion,
        description_lang=_detectar_idioma(descripcion),
        url=raw.get("redirect_url"),
    )
    return fila
