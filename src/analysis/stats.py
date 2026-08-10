"""Estadística descriptiva del proyecto.

Mediana en vez de media porque los salarios tienen cola derecha larga, y regla
dura de n >= 30 para no publicar números que no se sostienen (§6 del spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_N = 30


def mediana_ci(
    valores: np.ndarray | pd.Series, *, n_boot: int = 10_000, alpha: float = 0.10, seed: int = 42
) -> tuple[float, float, float]:
    """Mediana con intervalo de confianza bootstrap. Devuelve (mediana, inferior, superior)."""
    datos = np.asarray(pd.Series(valores).dropna(), dtype=float)
    if datos.size == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    muestras = rng.choice(datos, size=(n_boot, datos.size), replace=True)
    medianas = np.median(muestras, axis=1)
    lo, hi = np.quantile(medianas, [alpha / 2, 1 - alpha / 2])
    return (float(np.median(datos)), float(lo), float(hi))


def resumir_por(
    df: pd.DataFrame,
    grupos: list[str],
    *,
    valor: str = "salary_annual_usd_ppp",
    min_n: int = MIN_N,
    n_boot: int = 10_000,
) -> pd.DataFrame:
    """Un renglón por grupo con n, mediana, IC, IQR y la bandera de suficiencia."""
    filas: list[dict[str, object]] = []
    for llaves, sub in df.groupby(grupos, dropna=False):
        llaves = llaves if isinstance(llaves, tuple) else (llaves,)
        datos = sub[valor].dropna()
        n = int(datos.size)
        suficiente = n >= min_n

        if suficiente:
            med, lo, hi = mediana_ci(datos, n_boot=n_boot)
            q1, q3 = float(datos.quantile(0.25)), float(datos.quantile(0.75))
        else:
            med = lo = hi = q1 = q3 = np.nan

        filas.append(
            {
                **dict(zip(grupos, llaves, strict=True)),
                "n": n,
                "suficiente": suficiente,
                "mediana": med,
                "ic_inf": lo,
                "ic_sup": hi,
                "q1": q1,
                "q3": q3,
            }
        )
    resumen = pd.DataFrame(filas).sort_values("n", ascending=False).reset_index(drop=True)
    # dtype=object explícito: pandas infiere bool nativo de numpy para una columna
    # de puros bool de Python, y ese numpy.bool_ falla comparaciones `is True`/
    # `is False` en los tests (mismo defecto que en src/data/kaggle_sources.py).
    resumen["suficiente"] = resumen["suficiente"].astype(object)
    return resumen
