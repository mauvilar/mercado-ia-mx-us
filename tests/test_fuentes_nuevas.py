"""Mapeos de las fuentes que se abrieron después de Adzuna.

USAJOBS y el OFLC traen salario real al 100 %: su prueba es que no lo pierdan.
Jooble trae salario ambiguo: su prueba es que la compuerta lo detenga.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.mapping import map_jooble_row, map_usajobs_row
from src.data.oflc import OflcError, map_oflc_frame
from src.data.schema import CANONICAL_COLUMNS, SchemaError, validate_frame

FIXTURE = Path(__file__).parent / "fixtures" / "usajobs.json"


def _usajobs():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ── USAJOBS ──────────────────────────────────────────────────────────────────


def test_usajobs_mapea_todas_las_columnas_canonicas():
    out = map_usajobs_row(_usajobs(), snapshot_date=date(2026, 8, 12))
    assert set(out) == set(CANONICAL_COLUMNS)


def test_usajobs_trae_salario_real_nunca_predicho():
    out = map_usajobs_row(_usajobs(), snapshot_date=date(2026, 8, 12))
    assert out["salary_is_predicted"] is False
    assert out["salary_observed"] is True
    assert out["salary_min_raw"] == 117962.0
    assert out["salary_max_raw"] == 153354.0
    assert out["salary_period"] == "año"
    assert out["salary_currency"] == "USD"


def test_usajobs_canoniza_la_ciudad_con_el_mismo_mapa_que_adzuna():
    out = map_usajobs_row(_usajobs(), snapshot_date=date(2026, 8, 12))
    assert out["city"] == "Seattle"
    assert out["metro"] == "Seattle"


def test_usajobs_por_hora_conserva_la_periodicidad():
    crudo = _usajobs()
    crudo["MatchedObjectDescriptor"]["PositionRemuneration"] = [
        {"MinimumRange": "45.0", "MaximumRange": "58.0", "RateIntervalCode": "PH"}
    ]
    out = map_usajobs_row(crudo, snapshot_date=date(2026, 8, 12))
    assert out["salary_period"] == "hora"
    assert out["salary_observed"] is True


def test_usajobs_sin_compensacion_no_cuenta_como_observado():
    """Código WC: una vacante sin sueldo no es una vacante que no lo publica."""
    crudo = _usajobs()
    crudo["MatchedObjectDescriptor"]["PositionRemuneration"] = [
        {"MinimumRange": "0.0", "MaximumRange": "0.0", "RateIntervalCode": "WC"}
    ]
    out = map_usajobs_row(crudo, snapshot_date=date(2026, 8, 12))
    assert out["salary_observed"] is False
    assert out["salary_period"] is None


# ── OFLC ─────────────────────────────────────────────────────────────────────


def _oflc_crudo(**kw):
    base = {
        "CASE_NUMBER": ["I-200-26001-123456", "I-200-26001-123457"],
        "CASE_STATUS": ["Certified", "Denied"],
        "JOB_TITLE": ["Machine Learning Engineer", "Data Scientist"],
        "SOC_TITLE": ["Software Developers", "Data Scientists"],
        "EMPLOYER_NAME": ["Acme Corp", "Globex"],
        "WORKSITE_CITY": ["Mountain View", "Austin"],
        "WORKSITE_STATE": ["CA", "TX"],
        "WAGE_RATE_OF_PAY_FROM": [185000.0, 150000.0],
        "WAGE_RATE_OF_PAY_TO": [220000.0, 170000.0],
        "WAGE_UNIT_OF_PAY": ["Year", "Year"],
        "RECEIVED_DATE": ["2026-01-15", "2026-01-16"],
    }
    base.update(kw)
    return pd.DataFrame(base)


def test_oflc_solo_deja_pasar_casos_certificados():
    out = map_oflc_frame(_oflc_crudo(), snapshot_date=date(2026, 8, 12))
    assert len(out) == 1
    assert out.iloc[0]["title_raw"] == "Machine Learning Engineer"


def test_oflc_el_salario_ofrecido_es_observado_y_no_predicho():
    out = map_oflc_frame(_oflc_crudo(), snapshot_date=date(2026, 8, 12))
    fila = out.iloc[0]
    assert fila["salary_observed"] is True
    assert fila["salary_is_predicted"] is False
    assert fila["salary_min_raw"] == 185000.0
    assert fila["salary_period"] == "año"


def test_oflc_canoniza_el_metro():
    out = map_oflc_frame(_oflc_crudo(), snapshot_date=date(2026, 8, 12))
    assert out.iloc[0]["metro"] == "SF Bay Area"


def test_oflc_unidad_de_pago_desconocida_no_cuenta_como_observado():
    """Sin periodicidad legible no se puede anualizar, y anualizar mal es peor."""
    crudo = _oflc_crudo(WAGE_UNIT_OF_PAY=["Fortnight", "Year"])
    out = map_oflc_frame(crudo, snapshot_date=date(2026, 8, 12))
    assert out.iloc[0]["salary_observed"] is False


def test_oflc_acepta_los_alias_de_otros_anos_fiscales():
    crudo = _oflc_crudo().rename(
        columns={
            "WAGE_RATE_OF_PAY_FROM": "LCA_CASE_WAGE_RATE_FROM",
            "WAGE_UNIT_OF_PAY": "LCA_CASE_WAGE_RATE_UNIT",
        }
    )
    out = map_oflc_frame(crudo, snapshot_date=date(2026, 8, 12))
    assert out.iloc[0]["salary_observed"] is True


def test_oflc_sin_columnas_de_salario_revienta_con_un_mensaje_util():
    crudo = _oflc_crudo().drop(columns=["WAGE_RATE_OF_PAY_FROM", "WAGE_UNIT_OF_PAY"])
    with pytest.raises(OflcError, match="salario_desde"):
        map_oflc_frame(crudo, snapshot_date=date(2026, 8, 12))


# ── Jooble y la compuerta ────────────────────────────────────────────────────


def _jooble(**kw):
    base = {
        "id": "abc123",
        "title": "Ingeniero de Machine Learning",
        "location": "Ciudad de México",
        "snippet": "Buscamos experiencia en LLM y RAG para agentes de IA",
        "salary": "$45,000 - $60,000 al mes",
        "company": "Acme",
        "type": "Tiempo completo",
        "link": "https://jooble.org/x",
        "updated": "2026-08-01T00:00:00",
    }
    base.update(kw)
    return base


def test_jooble_salario_publicado_pasa():
    out = map_jooble_row(_jooble(), country="MX", snapshot_date=date(2026, 8, 12))
    assert out["salary_observed"] is True
    assert out["salary_is_predicted"] is False
    assert out["salary_min_raw"] == 45000
    assert out["salary_period"] == "mes"
    assert out["salary_currency"] == "MXN"


def test_jooble_salario_estimado_queda_marcado_como_predicho():
    out = map_jooble_row(
        _jooble(salary="Salario estimado: $50,000 al mes"),
        country="MX",
        snapshot_date=date(2026, 8, 12),
    )
    assert out["salary_observed"] is False
    assert out["salary_is_predicted"] is True


def test_jooble_sin_salario_no_inventa_nada():
    out = map_jooble_row(_jooble(salary=""), country="MX", snapshot_date=date(2026, 8, 12))
    assert out["salary_observed"] is False
    assert out["salary_is_predicted"] is False
    assert out["salary_min_raw"] is None


def test_el_candado_del_esquema_atrapa_una_fila_modelada_marcada_como_observada():
    """Si alguien saltara la compuerta a mano, validate_frame tiene que detenerlo."""
    fila = map_jooble_row(
        _jooble(salary="Estimado $50,000 al mes"), country="MX", snapshot_date=date(2026, 8, 12)
    )
    fila["salary_observed"] = True  # manipulación deliberada
    df = pd.DataFrame([fila], columns=CANONICAL_COLUMNS)
    with pytest.raises(SchemaError, match="predicho"):
        validate_frame(df)
