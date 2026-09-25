"""La compuerta que decide si un salario en texto libre cuenta como publicado.

Es la pieza que sostiene la integridad del corpus cuando entra un agregador que no
distingue entre lo que publicó el empleador y lo que estimó el portal. Si estas pruebas
se rompen, el hold-out del notebook 04 deja de significar lo que dice.
"""

import pytest

from src.data.salario_texto import parsear

# ── Lo que sí es un salario publicado ────────────────────────────────────────


def test_rango_mensual_en_pesos():
    s = parsear("$25,000 - $35,000 al mes", moneda_por_defecto="MXN")
    assert s.minimo == 25000
    assert s.maximo == 35000
    assert s.periodo == "mes"
    assert s.moneda == "MXN"
    assert s.observado is True


def test_importe_unico_sin_rango():
    s = parsear("MXN 60000 mensual")
    assert s.minimo == 60000
    assert s.maximo is None
    assert s.moneda == "MXN"
    assert s.observado is True


def test_anual_en_ingles():
    s = parsear("$120,000 - $150,000 per year")
    assert s.periodo == "año"
    assert s.minimo == 120000
    assert s.observado is True


def test_por_hora():
    s = parsear("$45.50 per hour", moneda_por_defecto="USD")
    assert s.periodo == "hora"
    assert s.minimo == pytest.approx(45.5)
    assert s.observado is True


def test_sufijo_k():
    s = parsear("120k - 150k a year")
    assert s.minimo == 120000
    assert s.maximo == 150000
    assert s.observado is True


def test_separador_de_miles_con_punto():
    """En español el punto separa miles: 40.000 son cuarenta mil, no cuarenta."""
    s = parsear("40.000 pesos mensuales")
    assert s.minimo == 40000
    assert s.observado is True


# ── Lo que la compuerta tiene que apagar ─────────────────────────────────────


@pytest.mark.parametrize(
    "texto",
    [
        "Estimado: $80,000 al año",
        "Salario estimado $45,000 mensual",
        "Estimated $100,000 a year",
        "Aproximadamente $50,000 al mes",
        "~$90,000 per year",
        "Promedio de $70,000 anuales",
    ],
)
def test_cualquier_marca_de_estimacion_apaga_lo_observado(texto):
    s = parsear(texto, moneda_por_defecto="MXN")
    assert s.es_estimado is True
    assert s.observado is False, f"se coló una estimación: {texto!r}"
    # El importe se conserva: sirve para marcar la fila como predicha, no para usarla.
    assert s.minimo is not None


def test_sin_periodicidad_no_es_observado():
    """Sin unidad no hay forma de anualizar, y suponerla sería inventar."""
    s = parsear("$50,000", moneda_por_defecto="MXN")
    assert s.periodo is None
    assert s.observado is False


def test_cadena_vacia_no_revienta():
    for entrada in (None, "", "   ", "Salario a convenir", "Competitivo"):
        s = parsear(entrada)
        assert s.observado is False
        assert s.minimo is None


def test_numeros_que_no_son_salarios_no_pasan_el_piso():
    """Un número suelto con periodicidad pegada no debe volverse un sueldo."""
    s = parsear("2 vacantes al mes", moneda_por_defecto="MXN")
    assert s.observado is False


def test_techo_absurdo_no_pasa():
    s = parsear("$999,999,999,999 al mes", moneda_por_defecto="MXN")
    assert s.observado is False


def test_moneda_explicita_gana_a_la_por_defecto():
    s = parsear("USD 8,000 al mes", moneda_por_defecto="MXN")
    assert s.moneda == "USD"


def test_moneda_por_defecto_cuando_solo_hay_signo_de_pesos():
    """En México "$" es ambiguo. Se resuelve con la moneda del país, y se documenta."""
    s = parsear("$30,000 mensuales", moneda_por_defecto="MXN")
    assert s.moneda == "MXN"
