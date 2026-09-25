"""Anclas oficiales: nivel salarial de cada país medido fuera del corpus."""

import pandas as pd
import pytest

from src.analysis.anclas import (
    Ancla,
    AnclaError,
    ancla_imss,
    ancla_oews,
    contrastar,
    niveles_ppp,
    posicion_relativa,
)


def test_imss_pondera_por_asegurados_y_no_promedia_promedios():
    """Un patrón de 3 trabajadores no puede pesar lo mismo que uno de 3,000."""
    crudo = pd.DataFrame(
        {
            "masa_sal_ta": [3000.0, 300000.0],  # masa salarial diaria
            "ta": [3, 1000],  # asegurados
        }
    )
    ancla = ancla_imss(crudo)
    # (3000 + 300000) / (3 + 1000) = 302.09 diarios
    assert ancla.salario_anual_local == pytest.approx(302.09 * 365, rel=1e-3)
    assert ancla.moneda == "MXN"
    assert ancla.pais == "MX"


def test_oews_toma_la_mediana_de_las_ocupaciones_de_interes():
    crudo = pd.DataFrame(
        {
            "OCC_CODE": ["15-2051", "15-1252", "29-1141"],
            "A_MEDIAN": [110000, 130000, 81000],
        }
    )
    ancla = ancla_oews(crudo)
    assert ancla.salario_anual_local == pytest.approx(120000)  # enfermería queda fuera
    assert ancla.pais == "US"


def test_oews_sin_los_codigos_soc_revienta_en_vez_de_devolver_cualquier_cosa():
    crudo = pd.DataFrame({"OCC_CODE": ["29-1141"], "A_MEDIAN": [81000]})
    with pytest.raises(AnclaError, match="SOC"):
        ancla_oews(crudo)


def test_el_ancla_se_convierte_a_dolares_internacionales():
    ancla = Ancla("MX", "IMSS", 180000.0, "MXN", "prueba", "")
    assert ancla.a_ppp({"MX": 10.0}) == pytest.approx(18000.0)


def test_posicion_relativa_normaliza_por_el_nivel_de_cada_pais():
    """El objetivo que sí puede transferir: cuántas veces su propio país paga la vacante."""
    salarios = pd.Series([200000.0, 40000.0])
    paises = pd.Series(["US", "MX"])
    niveles = {"US": 50000.0, "MX": 10000.0}
    rel = posicion_relativa(salarios, paises, niveles)
    assert rel.tolist() == [4.0, 4.0]


def test_posicion_relativa_sin_ancla_devuelve_nulo_y_no_presta_el_denominador():
    salarios = pd.Series([200000.0, 40000.0])
    paises = pd.Series(["US", "BR"])
    rel = posicion_relativa(salarios, paises, {"US": 50000.0})
    assert rel.iloc[0] == 4.0
    assert pd.isna(rel.iloc[1])


def test_contrastar_compara_el_corpus_contra_el_ancla():
    df = pd.DataFrame(
        {
            "country": ["US", "US", "MX"],
            "salary_observed": [True, True, True],
            "salary_annual_usd_ppp": [200000.0, 220000.0, 20000.0],
        }
    )
    anclas = {
        "US": Ancla("US", "BLS", 120000.0, "USD", "prueba", ""),
        "MX": Ancla("MX", "IMSS", 100000.0, "MXN", "prueba", ""),
    }
    out = contrastar(df, anclas, {"US": 1.0, "MX": 10.0}).set_index("country")
    assert out.loc["US", "n"] == 2
    assert out.loc["US", "veces_el_ancla"] == pytest.approx(210000 / 120000)
    assert out.loc["MX", "veces_el_ancla"] == pytest.approx(20000 / 10000)


def test_niveles_ppp_ignora_un_pais_sin_factor():
    anclas = {"MX": Ancla("MX", "IMSS", 100000.0, "MXN", "p", "")}
    assert niveles_ppp(anclas, {"US": 1.0}) == {}
