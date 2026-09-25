"""Estilo Editorial v3 de NyxAI para las gráficas de los notebooks.

Fondo rich-black, tinta crema y el verde como único acento: la serie o la barra que
dice el hallazgo va en verde y todo lo demás en neutros. La tipografía va por rol:
Archivo para el título, Outfit para el subtítulo e IBM Plex Mono para ticks, leyendas,
etiquetas y pie.

Los colores salen del bloque :root de tokens.css del portafolio. Tenue, leve y los dos
filetes son los crema translúcidos de ese bloque (.70, .52, .12 y .22) ya aplanados
sobre el fondo, porque matplotlib no compone transparencias como el navegador.

Uso en un notebook:

    from src.viz.estilo import PALETA, aplicar_estilo, componer
    aplicar_estilo()
    fig, ax = plt.subplots()
    ...
    componer(fig, "Título con el hallazgo", "Qué se mide, en qué unidades", "Fuente: ...")
"""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager as fm
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text
from matplotlib.ticker import FuncFormatter, MaxNLocator

PALETA = {
    "papel": "#050706",  # --nyx-night-900, el fondo canónico
    "noche": "#141917",  # --nyx-night-700
    "tinta": "#F3F1EC",  # --nyx-on-dark: crema, nunca blanco puro
    "tenue": "#ACABA7",  # --nyx-on-dark-muted (.70) aplanado
    "leve": "#81817E",  # --nyx-on-dark-faint (.52) aplanado
    "filete": "#222321",  # --nyx-line-dark (.12) aplanado
    "filete_fuerte": "#393A38",  # --nyx-line-dark-2 (.22) aplanado
    "neutro": "#646561",
    "neutro_claro": "#9A9A96",
    "verde": "#00DF81",  # --nyx-green, el único acento
    "verde_700": "#009C5B",  # --nyx-green-700
    "verde_800": "#0A6E45",  # --nyx-green-800
    "verde_900": "#0C3A28",  # --nyx-green-900
}

# Magnitud: de la noche al verde, sin arcoíris.
RAMPA_VERDE = LinearSegmentedColormap.from_list(
    "nyx_verde",
    [
        PALETA["noche"],
        PALETA["verde_900"],
        PALETA["verde_800"],
        PALETA["verde_700"],
        PALETA["verde"],
    ],
)

# Sólo archivos estáticos. Los variables (-VF.ttf) quedan registrados con el peso de su
# instancia por defecto, y el de Outfit sale en 100: pedir 400 debe caer en el Regular.
ARCHIVOS_DE_FUENTE = (
    "NyxArchivo-Bold.ttf",
    "NyxArchivo-ExtraBold.ttf",
    "NyxOutfit-Regular.ttf",
    "NyxIBMPlexMono-Regular.ttf",
    "NyxIBMPlexMono-Medium.ttf",
)
DIRECTORIOS_DE_FUENTE = (
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path.home() / ".local" / "share" / "fonts",
)

# Rol -> (familia, peso, archivo estático que debe resolver). Si el archivo no está, el
# rol cae en DejaVu Sans y aplicar_estilo() lo avisa: mejor un aviso que una gráfica
# que parece de marca y no lo es.
ROLES = {
    "titulo": ("Archivo", 800, "NyxArchivo-ExtraBold.ttf"),
    "subtitulo": ("Outfit", 400, "NyxOutfit-Regular.ttf"),
    "mono": ("IBM Plex Mono", 400, "NyxIBMPlexMono-Regular.ttf"),
}
RESERVA = "DejaVu Sans"

TAMANO = {"titulo": 17.0, "subtitulo": 12.0, "pie": 9.0, "texto": 10.0, "etiqueta": 9.5}

# Márgenes de componer(), en pulgadas.
_MARGEN = 0.2
_ENTRE_TITULO_Y_SUBTITULO = 0.1
_ANTES_DE_LOS_EJES = 0.25
_ANTES_DEL_PIE = 0.22

_familias: dict[str, str] = {rol: RESERVA for rol in ROLES}


def registrar_fuentes() -> list[str]:
    """Registra en matplotlib los archivos estáticos que encuentre. Devuelve sus nombres."""
    encontrados = []
    for nombre in ARCHIVOS_DE_FUENTE:
        for carpeta in DIRECTORIOS_DE_FUENTE:
            ruta = carpeta / nombre
            if ruta.exists():
                fm.fontManager.addfont(str(ruta))
                encontrados.append(nombre)
                break
    return encontrados


def _archivo_resuelto(familia: str, peso: int) -> str:
    """Nombre del archivo que matplotlib usaría para esa familia y peso ('' si no hay)."""
    try:
        ruta = fm.findfont(
            fm.FontProperties(family=familia, weight=peso), fallback_to_default=False
        )
    except ValueError:
        return ""
    return Path(ruta).name


def familias() -> dict[str, str]:
    """Familia en uso para cada rol, tal como la dejó aplicar_estilo()."""
    return dict(_familias)


def aplicar_estilo(*, retina: bool = True) -> dict[str, str]:
    """Registra las fuentes, fija los rcParams y devuelve la familia de cada rol."""
    registrar_fuentes()
    for rol, (familia, peso, archivo) in ROLES.items():
        if _archivo_resuelto(familia, peso) == archivo:
            _familias[rol] = familia
        else:
            _familias[rol] = RESERVA
            warnings.warn(
                f"No se encontró {archivo}: el {rol} de las gráficas sale en {RESERVA}.",
                stacklevel=2,
            )

    if "nyx_verde" not in mpl.colormaps:
        mpl.colormaps.register(RAMPA_VERDE)

    mono = _familias["mono"]
    p = PALETA
    mpl.rcParams.update(
        {
            "font.family": [mono],
            "font.weight": 400,
            "font.size": TAMANO["texto"],
            # Si algún texto activara mathtext, que tampoco caiga en DejaVu.
            "mathtext.fontset": "custom",
            "mathtext.rm": mono,
            "mathtext.it": mono,
            "mathtext.bf": f"{mono}:medium",
            "mathtext.sf": mono,
            "mathtext.tt": mono,
            "mathtext.cal": mono,
            "text.color": p["tinta"],
            "figure.facecolor": p["papel"],
            "figure.edgecolor": p["papel"],
            "figure.figsize": (11, 6),
            "figure.dpi": 100,
            "savefig.facecolor": p["papel"],
            "savefig.edgecolor": p["papel"],
            "savefig.dpi": 200,
            "savefig.pad_inches": 0.25,
            "axes.facecolor": p["papel"],
            "axes.edgecolor": p["filete_fuerte"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": p["tenue"],
            "axes.labelsize": TAMANO["texto"],
            "axes.labelpad": 8,
            "axes.titlesize": 10.5,
            "axes.titleweight": 500,
            "axes.titlecolor": p["tinta"],
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=[p["verde"], p["neutro_claro"], p["neutro"]]),
            "grid.color": p["filete"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": p["filete_fuerte"],
            "ytick.color": p["filete_fuerte"],
            "xtick.labelcolor": p["tenue"],
            "ytick.labelcolor": p["tenue"],
            "xtick.labelsize": TAMANO["texto"],
            "ytick.labelsize": TAMANO["texto"],
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.pad": 5,
            "ytick.major.pad": 5,
            "legend.frameon": False,
            "legend.fontsize": TAMANO["texto"],
            "legend.title_fontsize": TAMANO["texto"],
            "legend.labelcolor": p["tenue"],
            "legend.handlelength": 1.1,
            "legend.handleheight": 0.9,
            "legend.borderaxespad": 0.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "patch.linewidth": 0,
            "scatter.edgecolors": p["papel"],
            "errorbar.capsize": 0,
            "image.cmap": "nyx_verde",
        }
    )
    if retina:
        _retina()
    return familias()


def _retina() -> None:
    """PNG al doble de resolución en el notebook. Fuera de IPython no hace nada.

    Se fija también en la configuración del backend: `%matplotlib inline` vuelve a
    aplicar esa configuración y, si sólo se llamara a set_matplotlib_formats, un
    `%matplotlib inline` posterior regresaría las gráficas a PNG simple.
    """
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import InlineBackend, set_matplotlib_formats
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return
    InlineBackend.instance(parent=shell).figure_formats = {"retina"}
    set_matplotlib_formats("retina")


def escapar(texto: str) -> str:
    """Escapa el signo de pesos: dos "$" en un texto activan el modo matemático."""
    return texto.replace(r"\$", "$").replace("$", r"\$")


# Grupos que no deben partirse al envolver: "EE." al final de una línea y "UU." al
# principio de la siguiente se leen mal.
_SIN_CORTE = ("EE. UU.",)

_ENVOLVER = textwrap.TextWrapper(break_long_words=False, break_on_hyphens=False)


def _sin_corte(texto: str) -> str:
    for grupo in _SIN_CORTE:
        texto = texto.replace(grupo, grupo.replace(" ", "\u00a0"))
    return texto


def _envolver(
    texto: Text, ancho_max: float, dpi: float, render: RendererBase, *, parejo: bool
) -> None:
    """Parte el texto en líneas hasta que quepa en `ancho_max` pulgadas.

    Con `parejo`, usa el menor número de líneas que cabe y las reparte para que la
    última no quede con una palabra suelta. Sin `parejo`, llena cada línea (pies).
    """
    original = texto.get_text()

    def ancho() -> float:
        return float(texto.get_window_extent(render).width) / dpi

    def poner(columnas: int) -> None:
        _ENVOLVER.width = columnas
        texto.set_text("\n".join(_ENVOLVER.wrap(original)))

    if ancho() <= ancho_max:
        return
    if not parejo:
        columnas = int(len(original) * ancho_max / ancho())
        while columnas > 20:
            poner(columnas)
            if ancho() <= ancho_max:
                return
            columnas -= 2
        return
    for lineas in range(2, 7):
        # El ancho de columna más angosto que todavía da `lineas` líneas o menos.
        bajo, alto = 1, len(original)
        while bajo < alto:
            medio = (bajo + alto) // 2
            _ENVOLVER.width = medio
            if len(_ENVOLVER.wrap(original)) <= lineas:
                alto = medio
            else:
                bajo = medio + 1
        poner(bajo)
        if ancho() <= ancho_max:
            return


def componer(
    fig: Figure,
    titulo: str,
    subtitulo: str | None = None,
    fuente: str | None = None,
) -> None:
    """Título, subtítulo y pie editoriales; acomoda los ejes en el espacio restante.

    Reemplaza a plt.tight_layout() y se llama al final, con la gráfica ya dibujada. Los
    tres textos se alinean con el borde izquierdo de lo que ocupan los ejes (etiquetas
    incluidas) y se parten en líneas cuando no caben a lo ancho.

    No devuelve la figura a propósito: como última línea de una celda, IPython la
    mostraría dos veces (el valor de la celda y la figura abierta).
    """
    ancho_fig, alto_fig = fig.get_size_inches()
    dpi = fig.dpi
    render = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    comunes: dict[str, Any] = {"ha": "left", "va": "top", "linespacing": 1.2}

    t = fig.text(
        0,
        1,
        escapar(_sin_corte(titulo)),
        fontfamily=_familias["titulo"],
        fontweight=ROLES["titulo"][1],
        fontsize=TAMANO["titulo"],
        color=PALETA["tinta"],
        **comunes,
    )
    s = (
        fig.text(
            0,
            1,
            escapar(_sin_corte(subtitulo)),
            fontfamily=_familias["subtitulo"],
            fontweight=ROLES["subtitulo"][1],
            fontsize=TAMANO["subtitulo"],
            color=PALETA["tenue"],
            **comunes,
        )
        if subtitulo
        else None
    )
    p = (
        fig.text(
            0,
            0,
            escapar(_sin_corte(fuente)),
            fontfamily=_familias["mono"],
            fontsize=TAMANO["pie"],
            color=PALETA["leve"],
            ha="left",
            va="bottom",
            linespacing=1.3,
        )
        if fuente
        else None
    )

    disponible = ancho_fig - 2 * _MARGEN
    _envolver(t, disponible, dpi, render, parejo=True)
    if s is not None:
        _envolver(s, disponible, dpi, render, parejo=True)
    if p is not None:
        _envolver(p, disponible, dpi, render, parejo=False)

    def alto(texto: Text) -> float:
        return float(texto.get_window_extent(render).height) / dpi

    y_sub = _MARGEN + alto(t) + _ENTRE_TITULO_Y_SUBTITULO
    arriba = (y_sub + alto(s) if s else _MARGEN + alto(t)) + _ANTES_DE_LOS_EJES
    abajo = _MARGEN + (alto(p) + _ANTES_DEL_PIE if p else 0)
    fig.tight_layout(rect=(0, abajo / alto_fig, 1, 1 - arriba / alto_fig))

    ejes = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    cajas = [ax.get_tightbbox(render) for ax in ejes]
    x0 = min((c.x0 for c in cajas if c is not None), default=_MARGEN * dpi)
    x = max(float(x0) / fig.bbox.width, 0.0)
    t.set_position((x, 1 - _MARGEN / alto_fig))
    if s is not None:
        s.set_position((x, 1 - y_sub / alto_fig))
    if p is not None:
        p.set_position((x, _MARGEN / alto_fig))


def colores(
    categorias: Sequence[Any],
    destacadas: Collection[Any] | str,
    *,
    resto: str = PALETA["neutro"],
) -> list[str]:
    """Verde para las categorías destacadas y un neutro para las demás."""
    marca = {destacadas} if isinstance(destacadas, str) else set(destacadas)
    return [PALETA["verde"] if c in marca else resto for c in categorias]


def miles(
    ax: Axes, eje: Literal["x", "y"] = "x", *, sufijo: str = "", nbins: int | None = None
) -> None:
    """Ticks con separador de miles (y signo menos tipográfico).

    `nbins` limita cuántos ticks caben, para paneles angostos donde se enciman.
    """
    eje_ = ax.xaxis if eje == "x" else ax.yaxis
    eje_.set_major_formatter(
        FuncFormatter(lambda v, _pos: f"{v:,.0f}{sufijo}".replace("-", "\u2212"))
    )
    if nbins is not None:
        eje_.set_major_locator(MaxNLocator(nbins))


def porcentaje(ax: Axes, eje: Literal["x", "y"] = "y", *, nbins: int | None = None) -> None:
    """Ticks en porcentaje sin decimales, para ejes que ya vienen en 0 a 100."""
    miles(ax, eje, sufijo="%", nbins=nbins)


def rejilla(ax: Axes, eje: Literal["x", "y", "both"] = "x") -> None:
    """Rejilla de filete, sólida y por debajo de las marcas."""
    ax.grid(axis=eje, color=PALETA["filete"], linewidth=0.8)
    ax.set_axisbelow(True)


def etiquetar(
    ax: Axes,
    barras: BarContainer,
    textos: Sequence[str],
    *,
    destacar: Collection[int] = (),
    padding: float = 4,
) -> list[Annotation]:
    """Etiqueta cada barra: tenue para todas, tinta y medium para las destacadas."""
    marcas = ax.bar_label(
        barras,
        labels=[escapar(t) for t in textos],
        padding=padding,
        color=PALETA["tenue"],
        fontsize=TAMANO["etiqueta"],
    )
    for i, marca in enumerate(marcas):
        if i in destacar:
            marca.set_color(PALETA["tinta"])
            marca.set_fontweight(500)
    return marcas
