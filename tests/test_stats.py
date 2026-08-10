import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import mediana_ci, resumir_por


def test_mediana_ci_devuelve_mediana_y_cotas():
    datos = np.arange(1, 101, dtype=float)
    med, lo, hi = mediana_ci(datos, n_boot=2000, seed=42)
    assert med == pytest.approx(50.5, abs=1.0)
    assert lo < med < hi


def test_mediana_ci_es_reproducible_con_semilla():
    datos = np.random.default_rng(0).normal(100, 20, 200)
    assert mediana_ci(datos, n_boot=1000, seed=7) == mediana_ci(datos, n_boot=1000, seed=7)


def test_mediana_ci_con_muestra_vacia_devuelve_nan():
    med, lo, hi = mediana_ci(np.array([]), n_boot=100, seed=1)
    assert np.isnan(med) and np.isnan(lo) and np.isnan(hi)


def test_resumir_por_marca_los_grupos_con_n_insuficiente():
    """Regla dura del spec §6: n < 30 no se grafica, se marca."""
    df = pd.DataFrame(
        {
            "city": ["CDMX"] * 40 + ["Querétaro"] * 12,
            "salary_annual_usd_ppp": list(np.linspace(50000, 90000, 40))
            + list(np.linspace(40000, 60000, 12)),
        }
    )
    out = resumir_por(df, ["city"], valor="salary_annual_usd_ppp", min_n=30)

    cdmx = out[out["city"] == "CDMX"].iloc[0]
    qro = out[out["city"] == "Querétaro"].iloc[0]
    assert cdmx["suficiente"] is True
    assert qro["suficiente"] is False
    assert qro["n"] == 12
    assert pd.isna(qro["mediana"])  # no se reporta un número que no se sostiene


def test_resumir_por_ignora_los_nulos_al_contar():
    df = pd.DataFrame(
        {
            "city": ["CDMX"] * 50,
            "salary_annual_usd_ppp": [60000.0] * 20 + [None] * 30,
        }
    )
    out = resumir_por(df, ["city"], valor="salary_annual_usd_ppp", min_n=30)
    assert out.iloc[0]["n"] == 20
    assert out.iloc[0]["suficiente"] is False
