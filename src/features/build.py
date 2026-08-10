"""CLI: data/raw/* → data/processed/vacantes.parquet"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.kaggle_sources import map_mannacharya_frame
from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.features.dedupe import deduplicar
from src.features.extract import extraer_seniority, extraer_skills
from src.features.normalize import a_anual, a_ppp, a_usd, obtener_factores_ppp, obtener_fx_usd_mxn
from src.features.taxonomy import clasificar
from src.utils.config import DATA_DIR
from src.utils.logging import setup_logging

MULTINACIONALES = {
    "mastercard",
    "amazon",
    "google",
    "microsoft",
    "ibm",
    "oracle",
    "sap",
    "accenture",
    "deloitte",
    "pwc",
    "kpmg",
    "ey",
    "nvidia",
    "intel",
    "meta",
    "apple",
    "salesforce",
    "hp",
    "dell",
    "cisco",
    "bosch",
    "siemens",
    "ge",
    "bbva",
    "santander",
    "citi",
}

# Señales de que un remoto publicado en México es en realidad para un equipo de EE.UU.
_ARBITRAJE = re.compile(
    r"\b(usd|us\$|d[óo]lares|u\.?s\.?[- ]based|united states|ee\.?\s?uu|latam|nearshore)\b", re.I
)
_GLOBAL = re.compile(r"\b(worldwide|anywhere in the world|globally|fully distributed)\b", re.I)

log = logging.getLogger(__name__)


def _es_multinacional(nombre: object) -> bool:
    texto = str(nombre or "").lower()
    return any(m in texto for m in MULTINACIONALES)


def _remote_scope(fila: pd.Series) -> str:
    """Alcance del remoto. Sostiene la pregunta 6 del spec (arbitraje MX → EE.UU.)."""
    if not bool(fila.get("is_remote")):
        return "local"
    texto = f"{fila.get('title_raw') or ''} {fila.get('description_text') or ''}"
    if _GLOBAL.search(texto):
        return "global"
    if fila.get("country") == "MX" and _ARBITRAJE.search(texto):
        return "us_desde_mx"
    return "nacional"


def _punto_medio(fila: pd.Series) -> float | None:
    lo, hi = fila["salary_min_raw"], fila["salary_max_raw"]
    if pd.isna(lo):
        return None
    if pd.isna(hi):
        return float(lo)
    return (float(lo) + float(hi)) / 2


def _cargar_snapshots(raiz_datos: Path) -> list[pd.DataFrame]:
    """Sólo los snapshots que escribió el colector: data/raw/<fecha>/adzuna.parquet.

    Deliberadamente NO se hace rglob("*.parquet") sobre data/raw/: ahí abajo también
    viven las descargas de Kaggle, y el dataset descartado por sintético trae su propio
    .parquet de 51,932 filas. Un glob recursivo lo metería al análisis — justo el dato
    que este proyecto existe para rechazar. Peor aún, sin snapshots de Adzuna sería el
    único parquet encontrado y el pipeline "funcionaría" con datos inventados.
    """
    return [pd.read_parquet(p) for p in sorted(raiz_datos.glob("raw/*/adzuna.parquet"))]


def _cargar_mannacharya(raiz_datos: Path, snapshot_date: date) -> list[pd.DataFrame]:
    """Dataset real de Kaggle que aporta profundidad de EE.UU. (§4 del spec). Opcional."""
    csv = raiz_datos / "raw" / "kaggle" / "kaggle_mannacharya" / "aijobs_dataset.csv"
    if not csv.exists():
        log.info("Sin %s: se construye sólo con snapshots de Adzuna.", csv.name)
        return []
    crudo = pd.read_csv(csv, low_memory=False)
    log.info("Cargadas %s filas de kaggle_mannacharya", len(crudo))
    return [map_mannacharya_frame(crudo, snapshot_date=snapshot_date)]


def construir(raiz_datos: Path) -> pd.DataFrame:
    marcos = _cargar_snapshots(raiz_datos)
    if not marcos:
        raise FileNotFoundError(f"No hay snapshots en {raiz_datos / 'raw'}. Corre `make collect`.")
    n_snapshots = len(marcos)
    marcos += _cargar_mannacharya(raiz_datos, date.today())

    df = pd.concat(marcos, ignore_index=True)
    log.info("Cargadas %s filas de %s snapshots", len(df), n_snapshots)

    # §3 del spec: el alcance son México y Estados Unidos. Fuera de eso hay vacantes en
    # euros y libras que a_usd no convierte a propósito, porque no queremos compararlas.
    # OJO: parte de lo que se cae aquí son filas estadounidenses cuyo país no mapeó —
    # kaggle_mannacharya trae "Santa Clara", "CA", "NY" en su columna de país. Es pérdida
    # conocida y está medida en el log; mejorar ese mapeo recuperaría cientos de filas.
    en_alcance = df["country"].isin(["MX", "US"])
    con_salario = df["salary_observed"].fillna(False).astype(bool)
    perdidas, perdidas_con_salario = (
        int((~en_alcance).sum()),
        int((~en_alcance & con_salario).sum()),
    )
    if perdidas:
        log.info(
            "Descartadas %s filas fuera de alcance (país != MX/US); %s de ellas traían salario",
            perdidas,
            perdidas_con_salario,
        )
    df = df[en_alcance].reset_index(drop=True)

    # --- extracción sobre el texto ---
    texto = df["title_raw"].fillna("") + " " + df["description_text"].fillna("")
    df["skills"] = texto.map(extraer_skills)
    df["seniority"] = [
        extraer_seniority(t, d)
        for t, d in zip(df["title_raw"].fillna(""), df["description_text"].fillna(""), strict=True)
    ]
    df["tier"] = [
        clasificar(t, skills=s)
        for t, s in zip(df["title_raw"].fillna(""), df["skills"], strict=True)
    ]

    # --- normalización salarial: sólo sobre lo observado ---
    fx = obtener_fx_usd_mxn()
    ppp = obtener_factores_ppp()
    log.info("FX USD/MXN=%.2f  PPP=%s", fx, ppp)

    observado = df["salary_observed"].fillna(False).astype(bool)
    # pandas-stubs no tiene overload para un callable que devuelve `float | None`
    # (sólo `float | NAType`): es un hueco de los stubs, no un error real.
    medio = df.apply(_punto_medio, axis=1).where(observado)  # type: ignore[call-overload]
    df["salary_annual_local"] = [
        a_anual(m, p) if pd.notna(m) else None
        for m, p in zip(medio, df["salary_period"].fillna("año"), strict=True)
    ]
    df["salary_annual_usd"] = [
        a_usd(m, c, fx_usd_mxn=fx) if m is not None else None
        for m, c in zip(df["salary_annual_local"], df["salary_currency"].fillna("USD"), strict=True)
    ]
    df["salary_annual_usd_ppp"] = [
        a_ppp(m, p, ppp) if m is not None and pd.notna(p) else None
        for m, p in zip(df["salary_annual_local"], df["country"], strict=True)
    ]

    # --- atributos de empresa y de modalidad ---
    df["company_is_multinational"] = df["company"].map(_es_multinacional)
    df["remote_scope"] = df.apply(_remote_scope, axis=1)

    df = deduplicar(df)
    df["company_posting_count"] = df.groupby("company")["posting_id"].transform("count")

    df = df[CANONICAL_COLUMNS]
    validate_frame(df)
    return df


def main() -> None:
    setup_logging()
    df = construir(DATA_DIR)
    salida = DATA_DIR / "processed" / "vacantes.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida, index=False)
    df.to_csv(DATA_DIR / "processed" / "vacantes.csv", index=False)

    obs = df["salary_observed"].fillna(False).sum()
    log.info("✅ %s filas → %s (%s con salario observado)", len(df), salida, obs)
    print("\nSalarios observados por ciudad:")
    print(
        df[df["salary_observed"].fillna(False)]
        .groupby(["country", "city"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
