"""Pruebas de hipótesis con reporte legible, al estilo de los notebooks de TripleTen."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MIN_N = 30


def _limpio(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(pd.Series(x).dropna(), dtype=float)


def mann_whitney(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    *,
    etiqueta_a: str,
    etiqueta_b: str,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """H0: las dos muestras vienen de la misma distribución. No asume normalidad."""
    xa, xb = _limpio(a), _limpio(b)
    if xa.size < MIN_N or xb.size < MIN_N:
        return {
            "n_a": int(xa.size),
            "n_b": int(xb.size),
            "p_valor": None,
            "rechaza_h0": None,
            "conclusion": (
                f"Muestra insuficiente para comparar {etiqueta_a} (n={xa.size}) "
                f"con {etiqueta_b} (n={xb.size}). Se requiere n>={MIN_N} en ambos."
            ),
        }

    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    rechaza = bool(p < alpha)
    mayor, menor = (
        (etiqueta_a, etiqueta_b) if np.median(xa) > np.median(xb) else (etiqueta_b, etiqueta_a)
    )
    conclusion = (
        (
            f"Se rechaza H0 (p={p:.2e}): la mediana de {mayor} es significativamente "
            f"mayor que la de {menor}."
        )
        if rechaza
        else (
            f"No se rechaza H0 (p={p:.3f}): no hay evidencia de diferencia entre "
            f"{etiqueta_a} y {etiqueta_b}."
        )
    )
    return {
        "n_a": int(xa.size),
        "n_b": int(xb.size),
        "u": float(u),
        "p_valor": float(p),
        "rechaza_h0": rechaza,
        "mediana_a": float(np.median(xa)),
        "mediana_b": float(np.median(xb)),
        "conclusion": conclusion,
    }


def welch_log(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    *,
    etiqueta_a: str,
    etiqueta_b: str,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Welch sobre log-salarios: compara razones y no diferencias absolutas."""
    xa, xb = _limpio(a), _limpio(b)
    xa, xb = xa[xa > 0], xb[xb > 0]
    if xa.size < MIN_N or xb.size < MIN_N:
        return {
            "n_a": int(xa.size),
            "n_b": int(xb.size),
            "p_valor": None,
            "rechaza_h0": None,
            "razon_medianas": float("nan"),
            "conclusion": (
                f"Muestra insuficiente ({etiqueta_a} n={xa.size}, {etiqueta_b} n={xb.size})."
            ),
        }

    t, p = stats.ttest_ind(np.log(xa), np.log(xb), equal_var=False)
    razon = float(np.median(xa) / np.median(xb))
    rechaza = bool(p < alpha)
    return {
        "n_a": int(xa.size),
        "n_b": int(xb.size),
        "t": float(t),
        "p_valor": float(p),
        "rechaza_h0": rechaza,
        "razon_medianas": razon,
        "conclusion": (
            f"{'Se rechaza' if rechaza else 'No se rechaza'} H0 (p={p:.2e}). "
            f"La mediana de {etiqueta_a} es {razon:.2f}x la de {etiqueta_b}."
        ),
    }
