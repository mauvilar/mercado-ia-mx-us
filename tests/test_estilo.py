"""El estilo de las gráficas respeta la marca y no rompe nada fuera de un notebook."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.viz import estilo  # noqa: E402

RETIRADOS = {"#021B1A", "#030303", "#F1F7F6", "#707D7D"}


def test_la_paleta_no_usa_colores_retirados_ni_blanco_puro():
    valores = {v.upper() for v in estilo.PALETA.values()}
    assert not valores & RETIRADOS
    assert "#FFFFFF" not in valores
    assert estilo.PALETA["papel"] == "#050706"
    assert estilo.PALETA["tinta"] == "#F3F1EC"
    assert estilo.PALETA["verde"] == "#00DF81"


def test_el_signo_de_pesos_no_activa_el_modo_matematico():
    assert estilo.escapar("de $10 a $20") == r"de \$10 a \$20"
    # Escapar dos veces no duplica la barra.
    assert estilo.escapar(estilo.escapar("$5")) == r"\$5"


def test_colores_resalta_solo_lo_pedido():
    salida = estilo.colores(["a", "b", "c"], "b")
    assert salida == [estilo.PALETA["neutro"], estilo.PALETA["verde"], estilo.PALETA["neutro"]]


def test_componer_titula_la_figura_y_no_la_devuelve():
    """Si devolviera la figura, IPython la mostraría dos veces al final de la celda."""
    estilo.aplicar_estilo(retina=False)
    fig, ax = plt.subplots()
    ax.barh(["x", "y"], [1, 2])
    assert estilo.componer(fig, "Un título largo " * 8, "Subtítulo", "Fuente: prueba") is None
    textos = [t.get_text() for t in fig.texts]
    assert len(textos) == 3
    assert "\n" in textos[0], "un título más ancho que la figura se parte en líneas"
    plt.close(fig)


@pytest.mark.skipif(
    not all(
        any((d / f).exists() for d in estilo.DIRECTORIOS_DE_FUENTE)
        for f in estilo.ARCHIVOS_DE_FUENTE
    ),
    reason="las fuentes de la marca no están instaladas en esta máquina",
)
def test_cada_rol_resuelve_a_su_archivo_estatico():
    """Outfit variable se dibuja en peso 100: el subtítulo tiene que caer en el Regular."""
    familias = estilo.aplicar_estilo(retina=False)
    assert familias == {"titulo": "Archivo", "subtitulo": "Outfit", "mono": "IBM Plex Mono"}
    for familia, peso, archivo in estilo.ROLES.values():
        assert estilo._archivo_resuelto(familia, peso) == archivo
