from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.schema import CANONICAL_COLUMNS
from src.features.build import construir


def _snapshot(tmp_path, filas):
    carpeta = tmp_path / "raw" / "2026-08-07"
    carpeta.mkdir(parents=True)
    pd.DataFrame(filas, columns=CANONICAL_COLUMNS).to_parquet(carpeta / "adzuna.parquet")
    return tmp_path


def _cruda(**kw):
    base = dict.fromkeys(CANONICAL_COLUMNS)
    base.update(
        posting_id="a",
        snapshot_date=date(2026, 8, 7),
        source="adzuna_mx",
        title_raw="Senior Software Engineer",
        title_norm="senior software engineer",
        company="Acme",
        country="MX",
        city="Ciudad de México",
        metro="Valle de México",
        salary_min_raw=600000.0,
        salary_max_raw=800000.0,
        salary_currency="MXN",
        salary_period="año",
        salary_is_predicted=False,
        salary_observed=True,
        description_text="Necesitamos RAG y PyTorch para agentes",
        description_lang="es",
    )
    base.update(kw)
    return base


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_construye_el_dataset_procesado(_fx, _ppp, tmp_path):
    raiz = _snapshot(tmp_path, [_cruda()])
    df = construir(raiz)

    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["tier"] == "anillo"  # título genérico + 2 skills de IA
    assert set(["rag", "pytorch"]) <= set(fila["skills"])
    assert fila["seniority"] == "sr"
    assert fila["salary_annual_local"] == 700000.0  # punto medio de min y max
    assert fila["salary_annual_usd"] == pytest.approx(38888.9, rel=1e-3)
    assert fila["salary_annual_usd_ppp"] == pytest.approx(70000.0)
    assert fila["remote_scope"] == "local"


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_remote_scope_detecta_el_arbitraje_hacia_estados_unidos(_fx, _ppp, tmp_path):
    """Pregunta 6 del spec: un remoto desde México que paga en dólares o menciona
    un equipo en EE.UU. es la señal de arbitraje."""
    raiz = _snapshot(
        tmp_path,
        [
            _cruda(
                posting_id="a",
                is_remote=True,
                description_text="Remoto desde México para un equipo en United States, pago en USD",
            ),
            _cruda(
                posting_id="b",
                title_norm="ml engineer",
                is_remote=True,
                description_text="Trabajo remoto desde cualquier parte de la república",
            ),
            _cruda(posting_id="c", title_norm="ai engineer", is_remote=False),
        ],
    )
    df = construir(raiz).set_index("posting_id")
    assert df.loc["a", "remote_scope"] == "us_desde_mx"
    assert df.loc["b", "remote_scope"] == "nacional"
    assert df.loc["c", "remote_scope"] == "local"


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_calcula_el_conteo_de_vacantes_por_empresa(_fx, _ppp, tmp_path):
    raiz = _snapshot(
        tmp_path,
        [
            _cruda(posting_id="a", title_norm="ai engineer"),
            _cruda(posting_id="b", title_norm="ml engineer"),
            _cruda(posting_id="c", company="Globex", title_norm="ai engineer"),
        ],
    )
    df = construir(raiz)
    acme = df[df["company"] == "Acme"]
    assert (acme["company_posting_count"] == 2).all()


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_el_salario_predicho_nunca_llega_a_las_columnas_normalizadas(_fx, _ppp, tmp_path):
    raiz = _snapshot(
        tmp_path,
        [_cruda(salary_is_predicted=True, salary_observed=False)],
    )
    df = construir(raiz)
    assert pd.isna(df.iloc[0]["salary_annual_local"])


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_nunca_ingiere_el_parquet_del_dataset_descartado(_fx, _ppp, tmp_path):
    """El dataset que el proyecto descarta por sintético trae su propio .parquet bajo
    data/raw/kaggle/_descartado/. Un glob recursivo lo metería al análisis — y sin
    snapshots de Adzuna sería el único encontrado, así que el pipeline "funcionaría"
    con 51,932 filas inventadas."""
    raiz = _snapshot(tmp_path, [_cruda()])
    descartado = tmp_path / "raw" / "kaggle" / "_descartado"
    descartado.mkdir(parents=True)
    pd.DataFrame({"Job Title": ["AI Engineer"] * 5, "Salary Range": ["100k"] * 5}).to_parquet(
        descartado / "ai_jobs_dataset_2026.parquet"
    )

    df = construir(raiz)

    assert len(df) == 1, "se coló el parquet del dataset sintético"
    assert "Job Title" not in df.columns


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_tasas", return_value={"MXN": 18.0})
def test_revienta_si_solo_existe_el_parquet_descartado(_fx, _ppp, tmp_path):
    """Sin snapshots reales debe fallar ruidoso, nunca construir con lo sintético."""
    descartado = tmp_path / "raw" / "kaggle" / "_descartado"
    descartado.mkdir(parents=True)
    pd.DataFrame({"Job Title": ["AI Engineer"] * 5}).to_parquet(descartado / "x.parquet")

    with pytest.raises(FileNotFoundError, match="No hay snapshots"):
        construir(tmp_path)
