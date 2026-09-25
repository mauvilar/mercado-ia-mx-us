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

from src.utils.config import DATA_DIR, load_countries, monedas

log = logging.getLogger(__name__)

HORAS_LABORALES_ANUALES = 2080
MULTIPLICADOR = {
    "año": 1,
    "mes": 12,
    "quincena": 26,
    "semana": 52,
    "hora": HORAS_LABORALES_ANUALES,
}


def a_anual(monto: float | None, periodo: str) -> float | None:
    if monto is None:
        return None
    if periodo not in MULTIPLICADOR:
        raise ValueError(f"periodo desconocido: {periodo!r}. Válidos: {sorted(MULTIPLICADOR)}")
    return monto * MULTIPLICADOR[periodo]


def a_usd(monto: float | None, moneda: str, tasas: dict[str, float]) -> float | None:
    """Convierte a dólares nominales. `tasas` va en unidades de moneda por USD.

    Recibe el diccionario completo en vez de un único tipo de cambio porque el corpus
    dejó de ser bimonetario: los países de calibración traen euros, rupias, reales y
    zlotys. Una moneda sin tasa revienta a propósito — convertirla con un factor
    inventado sería peor que no tener la fila.
    """
    if monto is None:
        return None
    if moneda == "USD":
        return monto
    if moneda not in tasas:
        raise ValueError(f"sin tipo de cambio para {moneda!r}. Disponibles: {sorted(tasas)}")
    return monto / tasas[moneda]


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
    """Tipo de cambio USD/MXN del día. Atajo para quien sólo necesita México."""
    return obtener_tasas()["MXN"]


def obtener_tasas() -> dict[str, float]:
    """Unidades de moneda local por dólar, vía Frankfurter (BCE, sin llave).

    Una sola llamada trae todas las monedas del registro de países. Frankfurter
    publica las referencias del BCE, así que cubre las monedas de la lista actual;
    si un país nuevo trae una moneda fuera de esa cobertura, la respuesta llega sin
    esa clave y `a_usd` reventará al encontrarla, que es el comportamiento correcto.
    """
    objetivo = sorted(m for m in monedas() if m != "USD")

    def _pedir() -> dict[str, float]:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": ",".join(objetivo)},
            timeout=20,
        )
        r.raise_for_status()
        tasas = {k: float(v) for k, v in r.json()["rates"].items()}
        faltantes = set(objetivo) - set(tasas)
        if faltantes:
            log.warning("Frankfurter no devolvió tasa para %s", ", ".join(sorted(faltantes)))
        if "MXN" not in tasas:
            raise RuntimeError("Frankfurter respondió sin tipo de cambio para el peso mexicano")
        return tasas

    return dict(_con_reintentos(_pedir, "tasas", "Frankfurter"))


def obtener_factores_ppp() -> dict[str, float]:
    """PA.NUS.PPP del Banco Mundial, año más reciente disponible, por país del registro.

    México y Estados Unidos son obligatorios porque sostienen el análisis publicado.
    Un país de calibración sin factor se registra y se omite: perderlo degrada la
    calibración del modelo, pero no invalida ningún hallazgo.
    """
    registro = load_countries()
    iso3 = {str(v["wb"]): pais for pais, v in registro.items()}

    def _pedir() -> dict[str, float]:
        params: dict[str, Any] = {"format": "json", "date": "2020:2026", "per_page": 500}
        r = requests.get(
            f"https://api.worldbank.org/v2/country/{';'.join(sorted(iso3))}/indicator/PA.NUS.PPP",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        registros: list[dict[str, Any]] = json.loads(r.text)[1]
        factores: dict[str, float] = {}
        for reg in sorted(registros, key=lambda x: x["date"], reverse=True):
            pais = iso3.get(str(reg.get("countryiso3code") or "")) or reg["country"]["id"]
            if pais in registro and reg["value"] is not None and pais not in factores:
                factores[pais] = float(reg["value"])
        factores.setdefault("US", 1.0)
        if "MX" not in factores:
            raise RuntimeError("el Banco Mundial respondió sin factor PPP para México")
        sin_factor = sorted(set(registro) - set(factores))
        if sin_factor:
            log.warning("Sin factor PPP, quedan fuera del corpus: %s", ", ".join(sin_factor))
        return factores

    return dict(_con_reintentos(_pedir, "ppp", "Banco Mundial"))
