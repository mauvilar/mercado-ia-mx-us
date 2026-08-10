from datetime import date

import pandas as pd

from src.data.kaggle_sources import map_mannacharya_frame, verificar_no_sintetico
from src.data.schema import CANONICAL_COLUMNS


def test_detecta_dataset_sintetico_por_uniformidad():
    """El chequeo que descartó m0sm71: 12 países con ~4,300 filas cada uno
    y 100% de salarios presentes. Ver §1.2 del spec."""
    sintetico = pd.DataFrame(
        {"country": ["A"] * 4300 + ["B"] * 4290 + ["C"] * 4310, "salary": range(12900)}
    )
    veredicto = verificar_no_sintetico(sintetico, col_pais="country", col_salario="salary")
    assert veredicto["sospechoso"] is True
    assert "uniforme" in veredicto["razon"]


def test_acepta_un_dataset_real():
    real = pd.DataFrame(
        {
            "country": ["US"] * 2000 + ["UK"] * 94 + ["MX"] * 12,
            "salary": [1.0] * 400 + [None] * 1706,
        }
    )
    veredicto = verificar_no_sintetico(real, col_pais="country", col_salario="salary")
    assert veredicto["sospechoso"] is False


def test_mapea_mannacharya_al_esquema_canonico():
    crudo = pd.DataFrame(
        [
            {
                "id": "x1",
                "title": "Machine Learning Engineer",
                "company": "OpenAI",
                "location": "San Francisco, California, United States",
                "city": "San Francisco",
                "country": "United States",
                "salary_min": 200000.0,
                "salary_max": 300000.0,
                "salary_currency": "USD",
                "salary_period": "year",
                "tags": "python, pytorch",
                "description_text": "We need PyTorch and LLM experience",
                "posted_date": "2026-05-10",
            }
        ]
    )
    out = map_mannacharya_frame(crudo, snapshot_date=date(2026, 8, 7))
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.loc[0, "source"] == "kaggle_mannacharya"
    assert out.loc[0, "country"] == "US"
    assert out.loc[0, "salary_observed"] is True
    # Este dataset no trae salarios modelados, así que nunca es predicho.
    assert out.loc[0, "salary_is_predicted"] is False
