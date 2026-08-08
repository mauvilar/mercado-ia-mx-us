import json
from datetime import date
from pathlib import Path

from src.data.mapping import map_adzuna_row
from src.data.schema import CANONICAL_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "adzuna_mx.json"


def _fila():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_mapea_todas_las_columnas_canonicas():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert set(out) == set(CANONICAL_COLUMNS)


def test_posting_id_es_estable_y_depende_de_la_fuente():
    a = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    b = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 9, 1))
    assert a["posting_id"] == b["posting_id"]  # no depende de la fecha del snapshot
    assert len(a["posting_id"]) == 40  # sha1 hex


def test_salario_publicado_y_no_predicho_cuenta_como_observado():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_is_predicted"] is False
    assert out["salary_observed"] is True
    assert out["salary_min_raw"] == 780000.0
    assert out["salary_currency"] == "MXN"


def test_flag_predicho_llega_como_string_uno():
    """Adzuna manda "1"/"0" como string, no como booleano. Si esto se rompe,
    los salarios modelados se cuelan al análisis (§4.1 del spec)."""
    fila = _fila()
    fila["salary_is_predicted"] = "1"
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_is_predicted"] is True
    assert out["salary_observed"] is False


def test_sin_salario_no_es_observado():
    fila = _fila()
    del fila["salary_min"]
    del fila["salary_max"]
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_observed"] is False
    assert out["salary_min_raw"] is None


def test_ciudad_se_canoniza_desde_la_alcaldia():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["city"] == "Ciudad de México"
    assert out["metro"] == "Valle de México"


def test_source_refleja_el_pais():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["source"] == "adzuna_mx"


def test_detecta_remoto_en_el_titulo():
    fila = _fila()
    fila["title"] = "Remote AI Engineer"
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["is_remote"] is True


def test_detecta_idioma_de_la_descripcion():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["description_lang"] == "es"
