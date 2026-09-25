import pytest

from src.features.normalize import a_anual, a_ppp, a_usd


def test_mensual_a_anual():
    assert a_anual(45000, "mes") == 540000


def test_anual_se_queda_igual():
    assert a_anual(540000, "año") == 540000


def test_por_hora_a_anual_con_2080_horas():
    assert a_anual(50, "hora") == 104000


def test_periodo_desconocido_revienta():
    """ "quincena" dejó de servir de ejemplo cuando entraron los archivos del DOL, que
    declaran la tarifa ofrecida en cinco unidades y ésa es una de ellas."""
    with pytest.raises(ValueError, match="periodo"):
        a_anual(100, "bimestre")


TASAS = {"MXN": 18.0, "EUR": 0.92, "INR": 83.0}


def test_mxn_a_usd():
    assert a_usd(540000, "MXN", TASAS) == pytest.approx(30000.0)


def test_usd_se_queda_igual():
    assert a_usd(30000, "USD", TASAS) == 30000


def test_convierte_las_monedas_de_los_paises_de_calibracion():
    """El corpus dejó de ser bimonetario cuando entraron los países de calibración."""
    assert a_usd(92000, "EUR", TASAS) == pytest.approx(100000.0)
    assert a_usd(8300000, "INR", TASAS) == pytest.approx(100000.0)


def test_moneda_sin_tasa_revienta_en_vez_de_inventar():
    """Convertir con un factor inventado es peor que perder la fila."""
    with pytest.raises(ValueError, match="sin tipo de cambio"):
        a_usd(1000, "JPY", TASAS)


def test_ppp_divide_el_monto_local_por_el_factor_del_pais():
    """PA.NUS.PPP del Banco Mundial es 'moneda local por dólar internacional'.
    El ajuste se hace sobre el monto LOCAL, no sobre el ya convertido a USD."""
    factores = {"MX": 10.0, "US": 1.0}
    assert a_ppp(540000, "MX", factores) == pytest.approx(54000.0)
    assert a_ppp(150000, "US", factores) == pytest.approx(150000.0)


def test_ppp_hace_visible_que_el_tipo_de_cambio_subestima_mexico():
    """A 18 MXN/USD un sueldo de 540k MXN son 30k USD nominales,
    pero 54k dólares internacionales en poder adquisitivo."""
    factores = {"MX": 10.0, "US": 1.0}
    assert a_usd(540000, "MXN", TASAS) < a_ppp(540000, "MX", factores)


def test_none_se_propaga_sin_reventar():
    assert a_anual(None, "mes") is None
    assert a_usd(None, "MXN", TASAS) is None
    assert a_ppp(None, "MX", {"MX": 10.0}) is None


def test_periodos_del_dol_se_anualizan():
    """Los archivos de divulgación del DOL declaran la tarifa en cinco unidades."""
    assert a_anual(2000, "semana") == 104000
    assert a_anual(4000, "quincena") == 104000
