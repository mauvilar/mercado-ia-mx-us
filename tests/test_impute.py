import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import baseline_pais_seniority, evaluar_holdout_mx, mdape
from src.models.impute import FEATURES, entrenar, particionar


def _corpus(n_us=400, n_mx=60, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for pais, n, base in [("US", n_us, 150_000), ("MX", n_mx, 55_000)]:
        for _ in range(n):
            sen = rng.choice(["jr", "mid", "sr"])
            mult = {"jr": 0.7, "mid": 1.0, "sr": 1.4}[sen]
            filas.append(
                {
                    "country": pais,
                    "seniority": sen,
                    "tier": "nucleo",
                    "metro": "SF Bay Area" if pais == "US" else "Valle de México",
                    "category": "IT Jobs",
                    "title_norm": "ai engineer",
                    "is_remote": False,
                    "remote_scope": "local",
                    "company_posting_count": 3,
                    "company_is_multinational": True,
                    "skills": ["llm", "pytorch"],
                    "salary_annual_usd_ppp": base * mult * rng.normal(1, 0.12),
                    "salary_observed": True,
                }
            )
    return pd.DataFrame(filas)


def test_particionar_saca_todo_mexico_del_entrenamiento():
    """El hold-out del spec §7: el modelo no puede ver ni una fila mexicana."""
    train, holdout = particionar(_corpus())
    assert (train["country"] != "MX").all()
    assert (holdout["country"] == "MX").all()
    assert len(holdout) == 60


def test_particionar_solo_usa_salarios_observados():
    corpus = _corpus()
    corpus.loc[:9, "salary_observed"] = False
    train, holdout = particionar(corpus)
    assert len(train) + len(holdout) == len(corpus) - 10


def test_entrenar_devuelve_un_modelo_que_predice():
    train, _ = particionar(_corpus())
    modelo = entrenar(train)
    pred = modelo.predict(train[FEATURES].head(5))
    assert len(pred) == 5
    assert np.all(pred > 0)


def test_mdape_calcula_la_mediana_del_error_porcentual():
    real = np.array([100.0, 100.0, 100.0])
    pred = np.array([110.0, 90.0, 150.0])
    assert mdape(real, pred) == pytest.approx(0.10)


def test_baseline_usa_la_mediana_por_pais_y_seniority():
    corpus = _corpus()
    train, holdout = particionar(corpus)
    pred = baseline_pais_seniority(train, holdout)
    assert len(pred) == len(holdout)
    assert np.all(pred > 0)


def test_evaluar_holdout_emite_un_veredicto_de_publicacion():
    train, holdout = particionar(_corpus())
    modelo = entrenar(train)
    res = evaluar_holdout_mx(modelo, train, holdout, umbral_mdape=0.35)
    assert set(res) >= {"mdape_modelo", "mdape_baseline", "supera_baseline", "publicable"}
    assert isinstance(res["publicable"], bool)


def test_no_es_publicable_si_no_supera_el_umbral():
    train, holdout = particionar(_corpus())
    modelo = entrenar(train)
    res = evaluar_holdout_mx(modelo, train, holdout, umbral_mdape=0.0001)
    assert res["publicable"] is False
