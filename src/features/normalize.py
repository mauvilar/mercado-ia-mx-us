"""Normalización salarial: periodicidad → moneda → poder adquisitivo. Funciones puras."""

from __future__ import annotations

import json
from typing import Any

import requests

HORAS_LABORALES_ANUALES = 2080
MULTIPLICADOR = {"año": 1, "mes": 12, "hora": HORAS_LABORALES_ANUALES}


def a_anual(monto: float | None, periodo: str) -> float | None:
    if monto is None:
        return None
    if periodo not in MULTIPLICADOR:
        raise ValueError(f"periodo desconocido: {periodo!r}. Válidos: {sorted(MULTIPLICADOR)}")
    return monto * MULTIPLICADOR[periodo]


def a_usd(monto: float | None, moneda: str, *, fx_usd_mxn: float) -> float | None:
    if monto is None:
        return None
    if moneda == "USD":
        return monto
    if moneda == "MXN":
        return monto / fx_usd_mxn
    raise ValueError(f"moneda desconocida: {moneda!r}")


def a_ppp(monto_local: float | None, pais: str, factores: dict[str, float]) -> float | None:
    """Convierte un monto en moneda local a dólares internacionales.

    PA.NUS.PPP del Banco Mundial se expresa como 'unidades de moneda local por
    dólar internacional', así que la conversión es una división del monto local.
    """
    if monto_local is None:
        return None
    if pais not in factores:
        raise ValueError(f"sin factor PPP para {pais!r}")
    return monto_local / factores[pais]


def obtener_fx_usd_mxn() -> float:
    """Tipo de cambio del día vía Frankfurter (BCE, sin llave)."""
    r = requests.get(
        "https://api.frankfurter.app/latest", params={"from": "USD", "to": "MXN"}, timeout=20
    )
    r.raise_for_status()
    return float(r.json()["rates"]["MXN"])


def obtener_factores_ppp() -> dict[str, float]:
    """PA.NUS.PPP del Banco Mundial, año más reciente disponible, para MEX y USA."""
    params: dict[str, Any] = {"format": "json", "date": "2022:2026", "per_page": 100}
    r = requests.get(
        "https://api.worldbank.org/v2/country/MEX;USA/indicator/PA.NUS.PPP",
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    registros: list[dict[str, Any]] = json.loads(r.text)[1]
    factores: dict[str, float] = {}
    for reg in sorted(registros, key=lambda x: x["date"], reverse=True):
        pais = {"MX": "MX", "US": "US"}.get(reg["country"]["id"])
        if pais and reg["value"] is not None and pais not in factores:
            factores[pais] = float(reg["value"])
    factores.setdefault("US", 1.0)
    return factores
