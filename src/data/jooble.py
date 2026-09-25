"""Cliente HTTP de la API de Jooble. No conoce el esquema canónico — eso vive en mapping.py.

Jooble es un agregador con cobertura mexicana real, que es justo lo que le falta al
corpus: el hold-out del notebook 04 tiene 4 filas y el criterio del proyecto pide 30.

Entra con una advertencia grande. Jooble devuelve el salario como una cadena de texto
libre en la que conviven importes publicados por el empleador y estimaciones del propio
portal, sin ningún campo que los distinga. Por eso todo lo que sale de aquí pasa por
`src/data/salario_texto.py`, que ante cualquier duda marca la fila como predicha, y por
el candado de `schema.validate_frame`, que impide que una fila predicha llegue al
conjunto observado.

La llave es gratuita y se pide en jooble.org/api/about.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

BASE = "https://jooble.org/api"
MAX_REINTENTOS = 3
PAUSA_ENTRE_PAGINAS = 1.0

log = logging.getLogger(__name__)


class JoobleError(RuntimeError):
    """Jooble no respondió algo usable."""


class JoobleClient:
    def __init__(self, api_key: str, results_per_page: int = 100) -> None:
        self.api_key = api_key
        self.results_per_page = results_per_page

    def search(self, *, keywords: str, location: str, page: int) -> list[dict[str, Any]]:
        cuerpo = {
            "keywords": keywords,
            "location": location,
            "page": str(page),
            "ResultOnPage": self.results_per_page,
        }
        espera = 2
        for intento in range(MAX_REINTENTOS):
            resp = requests.post(f"{BASE}/{self.api_key}", json=cuerpo, timeout=30)
            if resp.status_code == 429:
                if intento == MAX_REINTENTOS - 1:
                    break
                log.warning("Jooble 429, esperando %ss (intento %s)", espera, intento + 1)
                time.sleep(espera)
                espera *= 2
                continue
            resp.raise_for_status()
            return list(resp.json().get("jobs", []))
        raise JoobleError(f"Jooble devolvió 429 tras {MAX_REINTENTOS} intentos ({keywords})")

    def search_all(self, *, keywords: str, location: str, max_pages: int = 3) -> list[Any]:
        acumulado: list[Any] = []
        for page in range(1, max_pages + 1):
            filas = self.search(keywords=keywords, location=location, page=page)
            if not filas:
                break
            acumulado.extend(filas)
            time.sleep(PAUSA_ENTRE_PAGINAS)
        return acumulado
