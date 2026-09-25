"""Esquema canónico. Única fuente de verdad de las columnas del dataset."""

from __future__ import annotations

import pandas as pd

from src.utils.config import monedas, paises

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

# Fuentes fijas más una por país de Adzuna: el registro de países manda, para que
# dar de alta un país nuevo no obligue a tocar esta lista a mano.
SOURCES = {
    "kaggle_mannacharya",
    "aijobs_net",
    "indeed",
    "usajobs",
    "oflc_lca",
    "jooble_mx",
} | {f"adzuna_{p.lower()}" for p in paises()}

# Fuentes cuyo campo de salario es, total o parcialmente, una estimación del propio
# agregador y no un número que el empleador haya publicado. Ver FUENTES_MODELADAS
# más abajo: para estas fuentes el candado de integridad es más estricto.
FUENTES_MODELADAS = {"jooble_mx"}

COUNTRIES = set(paises())
CURRENCIES = set(monedas())
# "semana" y "quincena" existen por los archivos de divulgación del DOL, que declaran
# la tarifa ofrecida en cualquiera de estas cinco unidades.
PERIODS = {"hora", "semana", "quincena", "mes", "año"}
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

    predicho = df["salary_is_predicted"].fillna(False).astype(bool)
    observado = df["salary_observed"].fillna(False).astype(bool)

    # Candado de integridad (§4.1 del spec): un salario predicho por Adzuna
    # jamás puede contar como observado.
    contaminadas = df[predicho & observado]
    if len(contaminadas):
        raise SchemaError(
            f"{len(contaminadas)} filas con salario predicho marcadas como observadas. "
            "Ver §4.1 del spec: Adzuna modela sueldos y no pueden entrar al conjunto observado."
        )

    # Segundo candado, para las fuentes que abrimos después de Adzuna. Adzuna al menos
    # marca cuál sueldo modeló; agregadores como Jooble mezclan importes publicados por
    # el empleador con estimaciones propias sin distinguirlos en un campo. Para esas
    # fuentes la regla es más dura: sólo cuenta como observado lo que el mapeo pudo
    # verificar explícitamente, y eso exige haber pasado por la compuerta que apaga
    # salary_observed ante cualquier señal de estimación. Una fila modelada de una de
    # estas fuentes en el conjunto observado invalidaría el hold-out, que es la pieza
    # sobre la que descansa todo el veredicto de publicación del notebook 04.
    de_fuente_modelada = df["source"].isin(FUENTES_MODELADAS)
    sospechosas = df[de_fuente_modelada & observado & predicho]
    if len(sospechosas):
        raise SchemaError(
            f"{len(sospechosas)} filas de fuentes con salario modelado "
            f"({', '.join(sorted(FUENTES_MODELADAS))}) entraron como observadas pese a estar "
            "marcadas como estimadas. La compuerta de src/data/salario_texto.py no corrió."
        )
