"""CLI: data/raw/* → data/processed/vacantes.parquet"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.kaggle_sources import map_mannacharya_frame
from src.data.oflc import cargar as cargar_oflc
from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.features.dedupe import deduplicar
from src.features.extract import extraer_seniority, extraer_skills
from src.features.normalize import a_anual, a_ppp, a_usd, obtener_factores_ppp, obtener_tasas
from src.features.taxonomy import clasificar
from src.utils.config import DATA_DIR, paises
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


def solo_analisis(df: pd.DataFrame) -> pd.DataFrame:
    """El corpus recortado a los países cuyas cifras se publican (MX, US).

    Existe para que abrir la recolección a países de calibración no cambie ni una cifra
    de los notebooks 01 a 03 por accidente. La regla del proyecto es que sólo México y
    Estados Unidos son objeto de estudio; el resto entra al parquet porque el modelo de
    imputación necesita que `country` tenga varianza, y para nada más. Todo análisis
    descriptivo debe empezar por esta función.
    """
    return df[df["country"].isin(paises(rol="analisis"))].reset_index(drop=True)


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


# Fuentes que el colector escribe dentro de cada snapshot fechado. La lista es explícita
# a propósito: ver el comentario de _cargar_snapshots.
PARQUETS_DE_SNAPSHOT = ("adzuna.parquet", "usajobs.parquet", "jooble.parquet")


def _cargar_snapshots(raiz_datos: Path) -> tuple[list[pd.DataFrame], int]:
    """Sólo los parquets que escribió el colector: data/raw/<fecha>/<fuente>.parquet.

    Deliberadamente NO se hace rglob("*.parquet") sobre data/raw/: ahí abajo también
    viven las descargas de Kaggle, y el dataset descartado por sintético trae su propio
    .parquet de 51,932 filas. Un glob recursivo lo metería al análisis — justo el dato
    que este proyecto existe para rechazar. Peor aún, sin snapshots de Adzuna sería el
    único parquet encontrado y el pipeline "funcionaría" con datos inventados.

    Por eso los nombres de archivo se enumeran uno por uno en PARQUETS_DE_SNAPSHOT, y
    dar de alta una fuente nueva obliga a añadirla ahí de forma consciente.

    Devuelve (marcos, n_snapshots_de_adzuna): Adzuna se cuenta aparte porque es la única
    fuente obligatoria — sin ella no hay corrida que valga.
    """
    marcos: list[pd.DataFrame] = []
    n_adzuna = 0
    for nombre in PARQUETS_DE_SNAPSHOT:
        rutas = sorted(raiz_datos.glob(f"raw/*/{nombre}"))
        if nombre == "adzuna.parquet":
            n_adzuna = len(rutas)
        for ruta in rutas:
            marco = pd.read_parquet(ruta)
            log.info("Cargadas %s filas de %s", len(marco), ruta.relative_to(raiz_datos))
            marcos.append(marco)
    return marcos, n_adzuna


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
    marcos, n_snapshots = _cargar_snapshots(raiz_datos)
    if not n_snapshots:
        raise FileNotFoundError(f"No hay snapshots en {raiz_datos / 'raw'}. Corre `make collect`.")
    marcos += _cargar_mannacharya(raiz_datos, date.today())
    marcos += cargar_oflc(raiz_datos / "raw" / "oflc", snapshot_date=date.today())

    df = pd.concat(marcos, ignore_index=True)
    log.info(
        "Cargadas %s filas de %s snapshots de Adzuna y las fuentes anexas", len(df), n_snapshots
    )

    # El alcance ya no es sólo México y Estados Unidos: son los países del registro,
    # que incluye a los de calibración (config/countries.yml). Lo que cae aquí es lo
    # que no está registrado en absoluto.
    # OJO: parte de lo que se cae son filas estadounidenses cuyo país no mapeó —
    # kaggle_mannacharya trae "Santa Clara", "CA", "NY" en su columna de país. Es pérdida
    # conocida y está medida en el log; mejorar ese mapeo recuperaría cientos de filas.
    en_alcance = df["country"].isin(paises())
    con_salario = df["salary_observed"].fillna(False).astype(bool)
    perdidas, perdidas_con_salario = (
        int((~en_alcance).sum()),
        int((~en_alcance & con_salario).sum()),
    )
    if perdidas:
        log.info(
            "Descartadas %s filas de países fuera del registro; %s de ellas traían salario",
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
    fx = obtener_tasas()
    ppp = obtener_factores_ppp()
    log.info("FX=%s  PPP=%s", fx, ppp)

    # Un país sin factor PPP no se puede comparar con nada, así que sale del corpus
    # antes de normalizar. Si el que faltara fuera México, obtener_factores_ppp() ya
    # habría reventado; aquí sólo pueden caerse países de calibración.
    sin_ppp = ~df["country"].isin(ppp)
    if sin_ppp.any():
        log.warning(
            "Descartadas %s filas de países sin factor PPP: %s",
            int(sin_ppp.sum()),
            ", ".join(sorted(df.loc[sin_ppp, "country"].dropna().unique())),
        )
        df = df[~sin_ppp].reset_index(drop=True)

    observado = df["salary_observed"].fillna(False).astype(bool)
    # pandas-stubs no tiene overload para un callable que devuelve `float | None`
    # (sólo `float | NAType`): es un hueco de los stubs, no un error real.
    medio = df.apply(_punto_medio, axis=1).where(observado)  # type: ignore[call-overload]
    df["salary_annual_local"] = [
        a_anual(m, p) if pd.notna(m) else None
        for m, p in zip(medio, df["salary_period"].fillna("año"), strict=True)
    ]
    df["salary_annual_usd"] = [
        a_usd(m, c, fx) if m is not None else None
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

    analisis = solo_analisis(df)
    log.info(
        "De esas, %s filas son de países de análisis (%s) y %s de calibración",
        len(analisis),
        ", ".join(paises(rol="analisis")),
        len(df) - len(analisis),
    )

    print("\nSalarios observados por ciudad (sólo países de análisis):")
    print(
        analisis[analisis["salary_observed"].fillna(False)]
        .groupby(["country", "city"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )
    print("\nSalarios observados por país (corpus completo):")
    print(
        df[df["salary_observed"].fillna(False)]
        .groupby("country")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
