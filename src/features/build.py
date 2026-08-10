"""CLI: data/raw/* → data/processed/vacantes.parquet"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

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


def construir(raiz_datos: Path) -> pd.DataFrame:
    parquets = sorted((raiz_datos / "raw").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No hay snapshots en {raiz_datos / 'raw'}. Corre `make collect`.")

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    log.info("Cargadas %s filas de %s snapshots", len(df), len(parquets))

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
