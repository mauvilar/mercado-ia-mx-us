"""Única puerta a la configuración del proyecto."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def load_cities() -> dict[str, Any]:
    return _load_yaml("cities.yml")


@cache
def load_skills() -> dict[str, Any]:
    return _load_yaml("skills.yml")


@cache
def load_taxonomy() -> dict[str, Any]:
    return _load_yaml("taxonomy.yml")


@cache
def load_countries() -> dict[str, Any]:
    return _load_yaml("countries.yml")


def paises(*, rol: str | None = None) -> list[str]:
    """Códigos de país del registro, opcionalmente filtrados por rol.

    `rol="analisis"` devuelve los países cuyas cifras se publican (MX, US).
    `rol="calibracion"` devuelve los que sólo existen para dar varianza a la
    columna `country` durante el entrenamiento. Sin rol, devuelve todos.
    """
    reg = load_countries()
    return sorted(c for c, v in reg.items() if rol is None or v["rol"] == rol)


def moneda_de(pais: str) -> str:
    reg = load_countries()
    if pais not in reg:
        raise KeyError(f"país {pais!r} no está en config/countries.yml")
    return str(reg[pais]["currency"])


def monedas() -> list[str]:
    return sorted({str(v["currency"]) for v in load_countries().values()})


def adzuna_credentials() -> tuple[str, str]:
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError(
            "Faltan ADZUNA_APP_ID / ADZUNA_APP_KEY. "
            "Sácalas gratis en developer.adzuna.com y ponlas en .env"
        )
    return app_id, app_key


def usajobs_credentials() -> tuple[str, str] | None:
    """(email, api_key) de data.usajobs.gov, o None si no están configuradas.

    Devuelve None en vez de reventar: USAJOBS es una fuente opcional y una corrida
    sin su llave debe seguir recolectando Adzuna.
    """
    email = os.getenv("USAJOBS_EMAIL", "")
    api_key = os.getenv("USAJOBS_API_KEY", "")
    return (email, api_key) if email and api_key else None


def jooble_key() -> str | None:
    """Llave de la API de Jooble, o None si no está configurada."""
    return os.getenv("JOOBLE_API_KEY", "") or None
