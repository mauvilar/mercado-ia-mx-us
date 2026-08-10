"""CLI de una corrida de recolección. Escribe un snapshot inmutable + su manifest."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adzuna import AdzunaClient
from src.data.mapping import map_adzuna_row
from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.utils.config import DATA_DIR, adzuna_credentials, load_cities
from src.utils.logging import setup_logging

CONSULTAS = [
    "AI engineer",
    "machine learning engineer",
    "LLM engineer",
    "MLOps engineer",
    "data scientist",
    "NLP engineer",
    "computer vision engineer",
    "inteligencia artificial",
    "aprendizaje automático",
]

log = logging.getLogger(__name__)


def _zonas() -> list[tuple[str, str]]:
    """(country, where) para cada ciudad objetivo, más el remoto nacional."""
    pares: list[tuple[str, str]] = []
    for country, entradas in load_cities().items():
        for e in entradas:
            pares.append((country, e["canonical"]))
    pares += [("MX", "remote"), ("US", "remote")]
    return pares


def ejecutar_corrida(
    *, destino: Path, snapshot_date: date, app_id: str, app_key: str
) -> dict[str, Any]:
    carpeta = destino / snapshot_date.isoformat()
    carpeta.mkdir(parents=True, exist_ok=True)
    parquet = carpeta / "adzuna.parquet"

    client = AdzunaClient(app_id, app_key)
    filas: list[dict[str, Any]] = []
    fallidas: list[dict[str, str]] = []

    for country, donde in _zonas():
        for que in CONSULTAS:
            try:
                crudas = client.search_all(country.lower(), what=que, where=donde, max_pages=5)
            except Exception as exc:  # noqa: BLE001 - un hueco no debe tumbar la corrida
                log.warning("Consulta fallida %s/%s/%s: %s", country, donde, que, exc)
                fallidas.append(
                    {"country": country, "where": donde, "what": que, "error": str(exc)}
                )
                continue
            for cruda in crudas:
                filas.append(map_adzuna_row(cruda, country=country, snapshot_date=snapshot_date))

    df = pd.DataFrame(filas, columns=CANONICAL_COLUMNS)
    if len(df):
        validate_frame(df)

    total_consultas = len(_zonas()) * len(CONSULTAS)
    manifest = {
        "snapshot_date": snapshot_date.isoformat(),
        "filas": int(len(df)),
        "filas_con_salario_observado": int(df["salary_observed"].fillna(False).sum())
        if len(df)
        else 0,
        "consultas_totales": total_consultas,
        "consultas_fallidas": fallidas,
        "tasa_exito": round(1 - len(fallidas) / total_consultas, 4) if total_consultas else 0.0,
    }

    if parquet.exists():
        log.warning("El snapshot %s ya existe, no se sobrescribe.", parquet)
    else:
        df.to_parquet(parquet, index=False)

    (carpeta / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(
        "Snapshot %s: %s filas, %s observadas",
        snapshot_date,
        manifest["filas"],
        manifest["filas_con_salario_observado"],
    )
    return manifest


def main() -> None:
    setup_logging()
    app_id, app_key = adzuna_credentials()
    ejecutar_corrida(
        destino=DATA_DIR / "raw", snapshot_date=date.today(), app_id=app_id, app_key=app_key
    )


if __name__ == "__main__":
    main()
