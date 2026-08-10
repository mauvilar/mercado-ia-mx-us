import numpy as np

from src.analysis.hypothesis import mann_whitney, welch_log


def test_mann_whitney_detecta_diferencia_real():
    rng = np.random.default_rng(0)
    a = rng.normal(100_000, 10_000, 200)
    b = rng.normal(60_000, 10_000, 200)
    out = mann_whitney(a, b, etiqueta_a="SF", etiqueta_b="CDMX")
    assert out["p_valor"] < 0.01
    assert out["rechaza_h0"] is True
    assert "SF" in out["conclusion"] and "CDMX" in out["conclusion"]


def test_mann_whitney_no_inventa_diferencia_donde_no_hay():
    rng = np.random.default_rng(1)
    a = rng.normal(80_000, 10_000, 200)
    b = rng.normal(80_000, 10_000, 200)
    out = mann_whitney(a, b, etiqueta_a="A", etiqueta_b="B")
    assert out["rechaza_h0"] is False


def test_mann_whitney_se_niega_con_muestra_insuficiente():
    out = mann_whitney(np.array([1.0, 2.0]), np.arange(100.0), etiqueta_a="A", etiqueta_b="B")
    assert out["rechaza_h0"] is None
    assert "insuficiente" in out["conclusion"].lower()


def test_welch_log_trabaja_sobre_logaritmos():
    """Sobre log-salarios la prueba compara razones, no diferencias absolutas,
    que es lo correcto para comparar dos mercados de escalas distintas."""
    rng = np.random.default_rng(2)
    a = np.exp(rng.normal(np.log(100_000), 0.3, 300))
    b = np.exp(rng.normal(np.log(50_000), 0.3, 300))
    out = welch_log(a, b, etiqueta_a="US", etiqueta_b="MX")
    assert out["p_valor"] < 0.01
    assert out["razon_medianas"] > 1.5
