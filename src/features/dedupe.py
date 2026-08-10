"""Colapso de vacantes duplicadas entre agregadores."""

from __future__ import annotations

import re

import pandas as pd


def _llave(fila: pd.Series) -> str:
    titulo = re.sub(r"[^a-z0-9]+", "", str(fila.get("title_norm", "")).lower())
    empresa = re.sub(r"[^a-z0-9]+", "", str(fila.get("company", "")).lower())
    ciudad = re.sub(r"[^a-z0-9]+", "", str(fila.get("city", "")).lower())
    return f"{titulo}|{empresa}|{ciudad}"


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (título, empresa, ciudad).

    Ante duplicados se prefiere la fila con salario observado: es la que aporta
    información al análisis.
    """
    if df.empty:
        return df

    trabajo = df.copy()
    trabajo["_llave"] = trabajo.apply(_llave, axis=1)
    trabajo["_prioridad"] = trabajo["salary_observed"].fillna(False).astype(int)

    trabajo = trabajo.sort_values("_prioridad", ascending=False, kind="stable")
    salida = trabajo.drop_duplicates(subset="_llave", keep="first")
    return salida.drop(columns=["_llave", "_prioridad"]).reset_index(drop=True)
