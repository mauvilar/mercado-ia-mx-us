import base64
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data import kaggle_sources
from src.data.kaggle_sources import descargar, map_mannacharya_frame, verificar_no_sintetico
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


def test_un_curl_fallido_no_repite_la_cabecera_de_autorizacion(tmp_path, monkeypatch):
    """CalledProcessError cita el comando entero, cabecera Basic incluida, y main() lo
    imprime en el log de Actions de un repo público. GitHub enmascara el secreto tal
    cual, no su base64: el mensaje no debe llevar ni uno ni otro."""
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text(
        json.dumps({"username": "alguien", "key": "llave-kaggle"})
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def curl_sin_red(cmd, **_):
        raise subprocess.CalledProcessError(6, cmd)

    monkeypatch.setattr(kaggle_sources.subprocess, "run", curl_sin_red)

    with pytest.raises(RuntimeError) as info:
        descargar("dueno/dataset", tmp_path / "destino")

    mensaje = str(info.value)
    basic = base64.b64encode(b"alguien:llave-kaggle").decode()
    assert "llave-kaggle" not in mensaje
    assert basic not in mensaje
    assert "6" in mensaje and "dueno/dataset" in mensaje
    assert info.value.__suppress_context__
