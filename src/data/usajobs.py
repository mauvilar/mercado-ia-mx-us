"""Cliente HTTP de la API de USAJOBS. No conoce el esquema canónico — eso vive en mapping.py.

USAJOBS es el portal oficial del gobierno federal estadounidense. Entra al proyecto por
una razón concreta: **todas** sus vacantes traen salario, porque la ley obliga a publicar
el rango del grado de pago. Contra el 20 % de cobertura salarial de Adzuna en EE.UU., aquí
es 100 %, y ninguno de esos números es modelado.

El sesgo es evidente y hay que declararlo: sólo hay empleo federal, con su propia escala de
pago (General Schedule) que no se mueve como el mercado privado. Sirve para dar volumen y
estructura al lado estadounidense del modelo, no para estimar lo que paga la industria.

La llave es gratuita y se pide en developer.usajobs.gov. La API exige mandar el correo con
el que te registraste en el User-Agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

BASE = "https://data.usajobs.gov/api/Search"
MAX_REINTENTOS = 3
PAUSA_ENTRE_PAGINAS = 1.0

log = logging.getLogger(__name__)


class UsajobsError(RuntimeError):
    """USAJOBS no respondió algo usable."""


class UsajobsClient:
    def __init__(self, email: str, api_key: str, results_per_page: int = 250) -> None:
        self.email = email
        self.api_key = api_key
        self.results_per_page = results_per_page

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Host": "data.usajobs.gov",
            "User-Agent": self.email,
            "Authorization-Key": self.api_key,
        }

    def search(self, *, keyword: str, page: int, location: str = "") -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "Keyword": keyword,
            "ResultsPerPage": self.results_per_page,
            "Page": page,
        }
        if location:
            params["LocationName"] = location

        espera = 2
        for intento in range(MAX_REINTENTOS):
            resp = requests.get(BASE, params=params, headers=self._headers, timeout=30)
            if resp.status_code == 429:
                if intento == MAX_REINTENTOS - 1:
                    break
                log.warning("USAJOBS 429, esperando %ss (intento %s)", espera, intento + 1)
                time.sleep(espera)
                espera *= 2
                continue
            resp.raise_for_status()
            cuerpo = resp.json().get("SearchResult", {})
            return list(cuerpo.get("SearchResultItems", []))
        raise UsajobsError(f"USAJOBS devolvió 429 tras {MAX_REINTENTOS} intentos ({keyword})")

    def search_all(self, *, keyword: str, location: str = "", max_pages: int = 4) -> list[Any]:
        acumulado: list[Any] = []
        for page in range(1, max_pages + 1):
            filas = self.search(keyword=keyword, page=page, location=location)
            if not filas:
                break
            acumulado.extend(filas)
            time.sleep(PAUSA_ENTRE_PAGINAS)
        return acumulado
