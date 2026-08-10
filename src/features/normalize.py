"""Normalización salarial: periodicidad → moneda → poder adquisitivo.

Las funciones de conversión son puras; las dos que consultan APIs externas reintentan
y caen a un caché en disco, porque el Banco Mundial es intermitente.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import requests

from src.utils.config import DATA_DIR

log = logging.getLogger(__name__)

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


CACHE = DATA_DIR / "interim" / "tasas.json"
INTENTOS = 3


def _leer_cache(clave: str) -> Any:
    if not CACHE.exists():
        return None
    guardado = json.loads(CACHE.read_text(encoding="utf-8"))
    return guardado.get(clave)


def _escribir_cache(clave: str, valor: Any) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    guardado = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    guardado[clave] = valor
    guardado[f"{clave}_actualizado"] = date.today().isoformat()
    CACHE.write_text(json.dumps(guardado, indent=2), encoding="utf-8")


def _con_reintentos(fn: Callable[[], Any], clave: str, etiqueta: str) -> Any:
    """Reintenta, y si la API sigue caída cae al último valor cacheado.

    El Banco Mundial es intermitente — se le vieron timeouts de 30 s seguidos de
    respuestas en 0.6 s. Sin esto, cualquier lunes que ande lento tumba la corrida
    semanal entera del GitHub Action.
    """
    for intento in range(INTENTOS):
        try:
            valor = fn()
            _escribir_cache(clave, valor)
            return valor
        except Exception as exc:  # noqa: BLE001
            log.warning("%s falló (intento %s/%s): %s", etiqueta, intento + 1, INTENTOS, exc)

    cacheado = _leer_cache(clave)
    if cacheado is None:
        raise RuntimeError(
            f"{etiqueta} no responde y no hay valor cacheado en {CACHE}. "
            "Corre de nuevo cuando la API esté disponible."
        )
    log.warning("%s no responde: se usa el valor cacheado %s", etiqueta, cacheado)
    return cacheado


def obtener_fx_usd_mxn() -> float:
    """Tipo de cambio del día vía Frankfurter (BCE, sin llave)."""

    def _pedir() -> float:
        r = requests.get(
            "https://api.frankfurter.app/latest", params={"from": "USD", "to": "MXN"}, timeout=20
        )
        r.raise_for_status()
        return float(r.json()["rates"]["MXN"])

    return float(_con_reintentos(_pedir, "fx_usd_mxn", "Frankfurter"))


def obtener_factores_ppp() -> dict[str, float]:
    """PA.NUS.PPP del Banco Mundial, año más reciente disponible, para MEX y USA."""

    def _pedir() -> dict[str, float]:
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
        if "MX" not in factores:
            raise RuntimeError("el Banco Mundial respondió sin factor PPP para México")
        return factores

    return dict(_con_reintentos(_pedir, "ppp", "Banco Mundial"))
