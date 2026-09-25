"""CLI de una corrida de recolección. Escribe un snapshot inmutable + su manifest."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adzuna import AdzunaClient
from src.data.jooble import JoobleClient
from src.data.mapping import map_adzuna_row, map_jooble_row, map_usajobs_row
from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.data.usajobs import UsajobsClient
from src.utils.config import (
    DATA_DIR,
    adzuna_credentials,
    jooble_key,
    load_cities,
    load_countries,
    paises,
    usajobs_credentials,
)
from src.utils.logging import setup_logging
from src.utils.redaccion import describir_error

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

# Malla de los países de calibración: una búsqueda nacional, tres consultas y dos
# páginas. Su trabajo es dar varianza a la columna `country`, no describir su mercado,
# así que no necesitan la malla ciudad por ciudad. La diferencia importa en la cuota:
# la malla completa cuesta ~540 llamadas por corrida y el plan Trial de Adzuna no la
# aguanta multiplicada por diez países.
CONSULTAS_CALIBRACION = ["AI engineer", "machine learning engineer", "data scientist"]
PAGINAS_ANALISIS = 5
PAGINAS_CALIBRACION = 2

# USAJOBS indexa por palabra clave sobre convocatorias federales, donde los títulos son
# más formales que en el sector privado: no hay "LLM engineer" pero sí "data scientist"
# y "computer scientist" con trabajo de IA en la descripción.
CONSULTAS_USAJOBS = [
    "artificial intelligence",
    "machine learning",
    "data scientist",
    "computer scientist",
]

log = logging.getLogger(__name__)


def _plan() -> list[tuple[str, str, str, int]]:
    """(country, where, what, max_pages) de toda la corrida.

    Los países de análisis se recorren ciudad por ciudad más el remoto nacional; los
    de calibración, con una sola búsqueda nacional acotada.
    """
    tareas: list[tuple[str, str, str, int]] = []

    ciudades = load_cities()
    for country in paises(rol="analisis"):
        zonas = [e["canonical"] for e in ciudades.get(country, [])] + ["remote"]
        for donde in zonas:
            for que in CONSULTAS:
                tareas.append((country, donde, que, PAGINAS_ANALISIS))

    registro = load_countries()
    for country in paises(rol="calibracion"):
        donde = str(registro[country].get("where") or "")
        for que in CONSULTAS_CALIBRACION:
            tareas.append((country, donde, que, PAGINAS_CALIBRACION))

    return tareas


def ejecutar_corrida(
    *,
    destino: Path,
    snapshot_date: date,
    app_id: str,
    app_key: str,
    solo_roles: tuple[str, ...] = ("analisis", "calibracion"),
) -> dict[str, Any]:
    """Una corrida completa. `solo_roles` permite recolectar por partes.

    Recolectar sólo un rol sirve para no gastar la cuota entera de golpe: la
    calibración se puede levantar una vez y no volver a tocarse en semanas, mientras
    que México y Estados Unidos sí conviene refrescarlos en cada corrida.
    """
    carpeta = destino / snapshot_date.isoformat()
    carpeta.mkdir(parents=True, exist_ok=True)
    parquet = carpeta / "adzuna.parquet"

    registro = load_countries()
    tareas = [t for t in _plan() if registro[t[0]]["rol"] in solo_roles]
    log.info(
        "Plan: %s consultas sobre %s países (roles: %s)",
        len(tareas),
        len({t[0] for t in tareas}),
        ", ".join(solo_roles),
    )

    client = AdzunaClient(app_id, app_key)
    filas: list[dict[str, Any]] = []
    fallidas: list[dict[str, str]] = []

    for country, donde, que, paginas in tareas:
        try:
            crudas = client.search_all(
                registro[country]["adzuna"], what=que, where=donde, max_pages=paginas
            )
        except Exception as exc:  # noqa: BLE001 - un hueco no debe tumbar la corrida
            # str(exc) de requests trae la URL con app_id y app_key en el query string,
            # y el manifest se versiona en un repo público: se guarda ya redactado.
            error = describir_error(exc, app_id, app_key)
            log.warning("Consulta fallida %s/%s/%s: %s", country, donde, que, error)
            fallidas.append({"country": country, "where": donde, "what": que, "error": error})
            continue
        for cruda in crudas:
            filas.append(map_adzuna_row(cruda, country=country, snapshot_date=snapshot_date))

    df = pd.DataFrame(filas, columns=CANONICAL_COLUMNS)
    if len(df):
        validate_frame(df)

    total_consultas = len(tareas)
    por_pais: dict[str, dict[str, int]] = {}
    if len(df):
        obs = df["salary_observed"].fillna(False).astype(bool)
        for pais, grupo in df.groupby("country"):
            por_pais[str(pais)] = {
                "filas": int(len(grupo)),
                "con_salario_observado": int(obs.loc[grupo.index].sum()),
            }

    manifest = {
        "snapshot_date": snapshot_date.isoformat(),
        "filas": int(len(df)),
        "filas_con_salario_observado": int(df["salary_observed"].fillna(False).sum())
        if len(df)
        else 0,
        "roles_recolectados": list(solo_roles),
        "por_pais": por_pais,
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


def recolectar_usajobs(*, destino: Path, snapshot_date: date) -> dict[str, Any]:
    """Vacantes federales estadounidenses. Opcional: sin llave, no pasa nada.

    Se escribe en su propio parquet dentro del mismo snapshot. Separar por fuente deja
    borrar o rehacer una sin tocar las demás, y hace visible en disco de dónde salió cada
    fila antes de que el build las junte.
    """
    credenciales = usajobs_credentials()
    if credenciales is None:
        log.info("Sin USAJOBS_EMAIL / USAJOBS_API_KEY: se omite USAJOBS.")
        return {"filas": 0, "omitida": True}

    email, api_key = credenciales
    client = UsajobsClient(email, api_key)
    filas: list[dict[str, Any]] = []
    fallidas: list[dict[str, str]] = []

    for que in CONSULTAS_USAJOBS:
        try:
            crudas = client.search_all(keyword=que, max_pages=4)
        except Exception as exc:  # noqa: BLE001
            error = describir_error(exc, email, api_key)
            log.warning("USAJOBS falló en %r: %s", que, error)
            fallidas.append({"what": que, "error": error})
            continue
        filas.extend(map_usajobs_row(c, snapshot_date=snapshot_date) for c in crudas)

    return _escribir(
        filas, fallidas, destino=destino, snapshot_date=snapshot_date, nombre="usajobs"
    )


def recolectar_jooble(*, destino: Path, snapshot_date: date) -> dict[str, Any]:
    """Agregador mexicano. Opcional, y con la compuerta de salario modelado puesta."""
    key = jooble_key()
    if key is None:
        log.info("Sin JOOBLE_API_KEY: se omite Jooble.")
        return {"filas": 0, "omitida": True}

    client = JoobleClient(key)
    filas: list[dict[str, Any]] = []
    fallidas: list[dict[str, str]] = []

    ciudades = [e["canonical"] for e in load_cities().get("MX", [])]
    for donde in ciudades:
        for que in CONSULTAS_CALIBRACION:
            try:
                crudas = client.search_all(keywords=que, location=donde, max_pages=3)
            except Exception as exc:  # noqa: BLE001
                # La llave de Jooble va en la ruta de la URL, que el error cita entera.
                error = describir_error(exc, key)
                log.warning("Jooble falló en %s/%s: %s", donde, que, error)
                fallidas.append({"where": donde, "what": que, "error": error})
                continue
            filas.extend(
                map_jooble_row(c, country="MX", snapshot_date=snapshot_date) for c in crudas
            )

    manifest = _escribir(
        filas, fallidas, destino=destino, snapshot_date=snapshot_date, nombre="jooble"
    )
    if filas:
        modelados = sum(1 for f in filas if f["salary_is_predicted"])
        log.info(
            "Jooble: %s filas, %s con salario que la compuerta marcó como no atribuible "
            "al empleador",
            len(filas),
            modelados,
        )
        manifest["descartados_por_la_compuerta"] = modelados
    return manifest


def _escribir(
    filas: list[dict[str, Any]],
    fallidas: list[dict[str, str]],
    *,
    destino: Path,
    snapshot_date: date,
    nombre: str,
) -> dict[str, Any]:
    carpeta = destino / snapshot_date.isoformat()
    carpeta.mkdir(parents=True, exist_ok=True)
    parquet = carpeta / f"{nombre}.parquet"

    df = pd.DataFrame(filas, columns=CANONICAL_COLUMNS)
    if len(df):
        validate_frame(df)

    if parquet.exists():
        log.warning("El snapshot %s ya existe, no se sobrescribe.", parquet)
    elif len(df):
        df.to_parquet(parquet, index=False)

    manifest = {
        "fuente": nombre,
        "filas": int(len(df)),
        "filas_con_salario_observado": int(df["salary_observed"].fillna(False).sum())
        if len(df)
        else 0,
        "consultas_fallidas": fallidas,
    }
    (carpeta / f"manifest_{nombre}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    log.info(
        "%s: %s filas, %s observadas",
        nombre,
        manifest["filas"],
        manifest["filas_con_salario_observado"],
    )
    return manifest


def main() -> None:
    """`python -m src.data.collect [analisis|calibracion|todo|fuentes]`.

    `fuentes` recolecta sólo las que no son Adzuna, para poder levantarlas sin gastar
    cuota de Adzuna. Por defecto corre todo.
    """
    setup_logging()
    arg = sys.argv[1] if len(sys.argv) > 1 else "todo"
    hoy = date.today()
    raiz = DATA_DIR / "raw"

    if arg != "fuentes":
        roles = ("analisis", "calibracion") if arg == "todo" else (arg,)
        app_id, app_key = adzuna_credentials()
        ejecutar_corrida(
            destino=raiz,
            snapshot_date=hoy,
            app_id=app_id,
            app_key=app_key,
            solo_roles=roles,
        )

    if arg in ("todo", "fuentes"):
        recolectar_usajobs(destino=raiz, snapshot_date=hoy)
        recolectar_jooble(destino=raiz, snapshot_date=hoy)


if __name__ == "__main__":
    main()
