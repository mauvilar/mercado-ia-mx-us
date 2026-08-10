import pytest

from src.features.normalize import a_anual, a_ppp, a_usd


def test_mensual_a_anual():
    assert a_anual(45000, "mes") == 540000


def test_anual_se_queda_igual():
    assert a_anual(540000, "año") == 540000


def test_por_hora_a_anual_con_2080_horas():
    assert a_anual(50, "hora") == 104000


def test_periodo_desconocido_revienta():
    with pytest.raises(ValueError, match="periodo"):
        a_anual(100, "quincena")


def test_mxn_a_usd():
    assert a_usd(540000, "MXN", fx_usd_mxn=18.0) == pytest.approx(30000.0)


def test_usd_se_queda_igual():
    assert a_usd(30000, "USD", fx_usd_mxn=18.0) == 30000


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
    assert a_usd(540000, "MXN", fx_usd_mxn=18.0) < a_ppp(540000, "MX", factores)


def test_none_se_propaga_sin_reventar():
    assert a_anual(None, "mes") is None
    assert a_usd(None, "MXN", fx_usd_mxn=18.0) is None
    assert a_ppp(None, "MX", {"MX": 10.0}) is None
