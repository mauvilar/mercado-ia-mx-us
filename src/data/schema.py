"""Esquema canónico. Única fuente de verdad de las columnas del dataset."""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "posting_id",
    "snapshot_date",
    "source",
    "title_raw",
    "title_norm",
    "company",
    "company_posting_count",
    "company_is_multinational",
    "category",
    "country",
    "state",
    "city",
    "metro",
    "is_remote",
    "remote_scope",
    "posted_date",
    "salary_min_raw",
    "salary_max_raw",
    "salary_currency",
    "salary_period",
    "salary_is_predicted",
    "salary_observed",
    "salary_annual_local",
    "salary_annual_usd",
    "salary_annual_usd_ppp",
    "seniority",
    "tier",
    "skills",
    "description_text",
    "description_lang",
    "url",
]

SOURCES = {"adzuna_mx", "adzuna_us", "kaggle_mannacharya", "aijobs_net", "indeed"}
COUNTRIES = {"MX", "US"}
CURRENCIES = {"MXN", "USD"}
PERIODS = {"hora", "mes", "año"}
SENIORITIES = {"intern", "jr", "mid", "sr", "lead", "staff", "principal", "unknown"}
TIERS = {"nucleo", "anillo", "fuera"}
REMOTE_SCOPES = {"local", "nacional", "us_desde_mx", "global"}

_ENUMS = {
    "source": SOURCES,
    "country": COUNTRIES,
    "salary_currency": CURRENCIES,
    "salary_period": PERIODS,
    "seniority": SENIORITIES,
    "tier": TIERS,
    "remote_scope": REMOTE_SCOPES,
}


class SchemaError(ValueError):
    """El frame no cumple el esquema canónico."""


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def validate_frame(df: pd.DataFrame) -> None:
    faltantes = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if faltantes:
        raise SchemaError(f"Faltan columnas: {', '.join(faltantes)}")

    for col, permitidos in _ENUMS.items():
        valores = set(df[col].dropna().unique()) - permitidos
        if valores:
            raise SchemaError(f"Valores inválidos en {col}: {sorted(valores)}")

    # Candado de integridad (§4.1 del spec): un salario predicho por Adzuna
    # jamás puede contar como observado.
    contaminadas = df[
        df["salary_is_predicted"].fillna(False).astype(bool)
        & df["salary_observed"].fillna(False).astype(bool)
    ]
    if len(contaminadas):
        raise SchemaError(
            f"{len(contaminadas)} filas con salario predicho marcadas como observadas. "
            "Ver §4.1 del spec: Adzuna modela sueldos y no pueden entrar al conjunto observado."
        )
