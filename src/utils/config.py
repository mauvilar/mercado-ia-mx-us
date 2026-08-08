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


def adzuna_credentials() -> tuple[str, str]:
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError(
            "Faltan ADZUNA_APP_ID / ADZUNA_APP_KEY. "
            "Sácalas gratis en developer.adzuna.com y ponlas en .env"
        )
    return app_id, app_key
