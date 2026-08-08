"""Cliente HTTP de la API de Adzuna. No conoce el esquema canónico — eso vive en mapping.py."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

BASE = "https://api.adzuna.com/v1/api/jobs"
MAX_REINTENTOS = 3
PAUSA_ENTRE_PAGINAS = 1.5

log = logging.getLogger(__name__)


class AdzunaError(RuntimeError):
    """Adzuna no respondió algo usable."""


class AdzunaClient:
    def __init__(self, app_id: str, app_key: str, results_per_page: int = 50) -> None:
        self.app_id = app_id
        self.app_key = app_key
        self.results_per_page = results_per_page

    def search(
        self, country: str, *, what: str, where: str, page: int, max_days_old: int = 60
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": self.results_per_page,
            "what": what,
            "where": where,
            "max_days_old": max_days_old,
            "content-type": "application/json",
        }
        espera = 2
        for intento in range(MAX_REINTENTOS):
            resp = requests.get(f"{BASE}/{country}/search/{page}", params=params, timeout=30)
            if resp.status_code == 429:
                if intento == MAX_REINTENTOS - 1:
                    break
                log.warning("Adzuna 429, esperando %ss (intento %s)", espera, intento + 1)
                time.sleep(espera)
                espera *= 2
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        raise AdzunaError(f"Adzuna devolvió 429 tras {MAX_REINTENTOS} intentos ({country}/{where})")

    def search_all(
        self, country: str, *, what: str, where: str, max_pages: int = 5
    ) -> list[dict[str, Any]]:
        acumulado: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            filas = self.search(country, what=what, where=where, page=page)
            if not filas:
                break
            acumulado.extend(filas)
            time.sleep(PAUSA_ENTRE_PAGINAS)
        return acumulado
