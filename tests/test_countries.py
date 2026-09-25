"""El registro de países y la línea entre análisis y calibración.

Abrir el corpus a más países fue el arreglo de la causa que el notebook 04 documentó:
`country` era una constante en el entrenamiento. Estas pruebas cuidan que ese arreglo no
se lleve por delante la regla que sí importa — que sólo México y Estados Unidos son
objeto de estudio.
"""

import pandas as pd
import pytest

from src.data.schema import COUNTRIES, CURRENCIES, SOURCES
from src.features.build import solo_analisis
from src.utils.config import load_countries, moneda_de, monedas, paises


def test_mexico_y_estados_unidos_son_los_unicos_paises_de_analisis():
    assert paises(rol="analisis") == ["MX", "US"]


def test_hay_paises_de_calibracion_y_ninguno_es_de_analisis():
    calibracion = paises(rol="calibracion")
    assert len(calibracion) >= 5
    assert not set(calibracion) & set(paises(rol="analisis"))


def test_el_registro_cubre_el_rango_de_niveles_de_precio():
    """La calibración sirve si hay países más baratos y más caros que EE.UU.

    Con puros países ricos, el modelo seguiría sin ver nada por debajo del nivel
    estadounidense, que es justo lo que le faltaba para no errar 958 % contra México.
    """
    calibracion = set(paises(rol="calibracion"))
    assert calibracion & {"IN", "ZA", "BR", "PL"}, "faltan mercados por debajo de EE.UU."


def test_cada_pais_declara_moneda_codigo_de_adzuna_y_banco_mundial():
    for pais, datos in load_countries().items():
        assert datos["currency"], f"{pais} sin moneda"
        assert datos["adzuna"], f"{pais} sin código de Adzuna"
        assert datos["wb"], f"{pais} sin código del Banco Mundial"
        assert datos["rol"] in {"analisis", "calibracion"}


def test_el_esquema_se_deriva_del_registro():
    """Dar de alta un país no debe obligar a tocar el esquema a mano."""
    assert COUNTRIES == set(paises())
    assert CURRENCIES == set(monedas())
    for pais in paises():
        assert f"adzuna_{pais.lower()}" in SOURCES


def test_moneda_de_un_pais_no_registrado_revienta():
    with pytest.raises(KeyError, match="countries.yml"):
        moneda_de("JP")


def test_solo_analisis_deja_fuera_a_los_paises_de_calibracion():
    df = pd.DataFrame({"country": ["MX", "US", "BR", "IN", "US"], "x": range(5)})
    recortado = solo_analisis(df)
    assert set(recortado["country"]) == {"MX", "US"}
    assert len(recortado) == 3
