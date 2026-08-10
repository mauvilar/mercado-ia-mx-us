import pandas as pd

from src.features.dedupe import deduplicar


def _fila(**kw):
    base = {
        "posting_id": "a",
        "title_norm": "ai engineer",
        "company": "Acme",
        "city": "Ciudad de México",
        "source": "adzuna_mx",
        "salary_observed": False,
        "salary_annual_local": None,
        "snapshot_date": pd.Timestamp("2026-08-07").date(),
    }
    base.update(kw)
    return base


def test_colapsa_la_misma_vacante_de_dos_fuentes():
    df = pd.DataFrame([_fila(posting_id="a"), _fila(posting_id="b", source="kaggle_mannacharya")])
    assert len(deduplicar(df)) == 1


def test_conserva_la_fila_con_salario_observado():
    df = pd.DataFrame(
        [
            _fila(posting_id="a", salary_observed=False),
            _fila(posting_id="b", salary_observed=True, salary_annual_local=600000.0),
        ]
    )
    out = deduplicar(df)
    assert len(out) == 1
    assert out.iloc[0]["salary_observed"]


def test_no_colapsa_vacantes_de_empresas_distintas():
    df = pd.DataFrame([_fila(company="Acme"), _fila(posting_id="b", company="Globex")])
    assert len(deduplicar(df)) == 2


def test_no_colapsa_la_misma_empresa_en_ciudades_distintas():
    df = pd.DataFrame([_fila(city="Ciudad de México"), _fila(posting_id="b", city="Querétaro")])
    assert len(deduplicar(df)) == 2


def test_es_idempotente():
    df = pd.DataFrame([_fila(posting_id="a"), _fila(posting_id="b")])
    una = deduplicar(df)
    dos = deduplicar(una)
    assert len(una) == len(dos) == 1
