"""Compuerta para fuentes que publican el salario como texto libre.

Adzuna al menos declara cuál sueldo modeló, en un campo aparte. Los agregadores que
siguen no lo hacen: Jooble, Careerjet y los envoltorios de Google for Jobs devuelven una
cadena como "$25,000 - $35,000 al mes" sin decir si ese número lo puso el empleador o lo
calculó el propio portal. Mezclados en la misma columna, un salario estimado entra al
conjunto observado sin que nadie lo note, y el hold-out mexicano del notebook 04 deja de
significar lo que dice que significa.

La regla de este módulo es pesimista a propósito: **ante la duda, no es observado.**
Un salario perdido cuesta una fila; un salario modelado colado cuesta el argumento entero
del proyecto. Por eso una cadena sin periodicidad legible, o con cualquier marca de
estimación, se marca como predicha y el candado de `schema.validate_frame` hace el resto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Marcas de que el número es un cálculo del portal y no una cifra del empleador.
# "promedio" y "average" entran porque un promedio de mercado no es la oferta de nadie.
_ESTIMACION = re.compile(
    r"(estimad|estimat|aproximad|approx\b|aprox\b|promedio|average|~|≈|"
    r"salary\s+estimate|sueldo\s+estimado|salario\s+estimado)",
    re.I,
)

# El orden importa: se evalúan de la unidad más corta a la más larga, para que
# "quincenal" no lo capture antes el patrón de "semana".
# Nota sobre el inglés: "a year" y "an hour" son tan comunes como "per year" en las
# vacantes reales, y omitirlos dejaba sin periodicidad a la mitad de las cadenas — y sin
# periodicidad la compuerta descarta la fila, así que el descuido costaba datos buenos.
_PERIODOS = [
    (
        r"\bpor\s+hora\b|\bp/?h\b|\bhourly\b|\bper\s+hour\b|\ban?\s+hour\b|/\s*h(ora|our)?\b",
        "hora",
    ),
    (
        r"\bquincenal\b|\bpor\s+quincena\b|\bbi[-\s_]?weekly\b|\bcada\s+15\s+d[íi]as\b",
        "quincena",
    ),
    (
        r"\bsemanal\b|\bpor\s+semana\b|\bweekly\b|\bper\s+week\b|\ba\s+week\b|/\s*sem(ana)?\b",
        "semana",
    ),
    (
        r"\bmensual(es)?\b|\bal\s+mes\b|\bpor\s+mes\b|\bmonthly\b|\bper\s+month\b"
        r"|\ba\s+month\b|/\s*mes\b|\bp/?m\b",
        "mes",
    ),
    (
        r"\banual(es)?\b|\bal\s+a[ñn]o\b|\bpor\s+a[ñn]o\b|\byearly\b|\bannual(ly)?\b"
        r"|\bper\s+year\b|\ba\s+year\b|/\s*(a[ñn]o|yr)\b",
        "año",
    ),
]
_PERIODOS_COMPILADOS = [(re.compile(p, re.I), nombre) for p, nombre in _PERIODOS]

_MONEDAS = [
    (re.compile(r"\b(usd|us\$|d[óo]lares|dollars?)\b", re.I), "USD"),
    (re.compile(r"\b(mxn|mx\$|pesos?|m\.?n\.?)\b", re.I), "MXN"),
    (re.compile(r"\b(eur|euros?)\b|€", re.I), "EUR"),
]

# Un importe: 40000 · 40,000 · 40.000 · 40k · 1.2M
#
# El grupo del número NO admite espacios a propósito. Admitirlos para leer "40 000" hacía
# que el patrón se tragara el espacio anterior al guion de un rango y luego retrocediera
# hasta un número truncado: "25,000 - 35,000" podía salir como 25,00. Un separador de
# miles con espacio se pierde (queda por debajo del piso de cordura y la fila se descarta),
# que es el error correcto de los dos posibles.
_IMPORTE = re.compile(r"(\d[\d.,]*)\s*(k\b|m\b|mil\b|millones?\b)?", re.I)

_MULTIPLICADOR_SUFIJO = {"k": 1_000, "mil": 1_000, "m": 1_000_000, "millon": 1_000_000}

# Debajo de esto, un "salario" anual no es un salario: es un número suelto que se coló
# (un código postal, un número de vacantes, un año). Por encima del techo pasa lo mismo
# al revés. Los límites son deliberadamente anchos: sólo atrapan basura evidente.
_PISO_ANUAL = 12_000
_TECHO_ANUAL = 100_000_000


@dataclass(frozen=True)
class SalarioTexto:
    """Lo que se pudo leer de una cadena de salario. `observado` es el veredicto."""

    minimo: float | None
    maximo: float | None
    periodo: str | None
    moneda: str | None
    es_estimado: bool
    crudo: str

    @property
    def observado(self) -> bool:
        """Sólo cuenta como publicado por el empleador si nada huele a estimación.

        Exige importe positivo y periodicidad legible: sin periodicidad no hay forma
        de anualizar, y anualizar con un supuesto sería exactamente el tipo de invento
        que este proyecto se niega a hacer.
        """
        return bool(
            not self.es_estimado
            and self.periodo is not None
            and self.minimo is not None
            and self.minimo > 0
        )


def _a_numero(texto: str, sufijo: str | None) -> float | None:
    limpio = texto.strip().replace(" ", "")
    if not limpio:
        return None

    # "40,000" y "40.000" son ambos cuarenta mil; "40,000.50" tiene decimales. La regla:
    # el último separador es decimal sólo si le siguen exactamente 1 o 2 dígitos y hay
    # un único separador de ese tipo.
    if re.fullmatch(r"\d+([.,]\d{1,2})", limpio):
        limpio = limpio.replace(",", ".")
    else:
        limpio = limpio.replace(",", "").replace(".", "")

    try:
        valor = float(limpio)
    except ValueError:
        return None

    if sufijo:
        clave = sufijo.lower().rstrip("es").replace("millon", "millon")
        valor *= _MULTIPLICADOR_SUFIJO.get(clave, _MULTIPLICADOR_SUFIJO.get(sufijo.lower(), 1))
    return valor


def parsear(texto: str | None, *, moneda_por_defecto: str | None = None) -> SalarioTexto:
    """Lee una cadena de salario. Nunca revienta: lo ilegible sale como no observado."""
    crudo = (texto or "").strip()
    if not crudo:
        return SalarioTexto(None, None, None, None, False, "")

    es_estimado = bool(_ESTIMACION.search(crudo))

    periodo = None
    for patron, nombre in _PERIODOS_COMPILADOS:
        if patron.search(crudo):
            periodo = nombre
            break

    moneda = moneda_por_defecto
    for patron, nombre in _MONEDAS:
        if patron.search(crudo):
            moneda = nombre
            break

    importes: list[float] = []
    for m in _IMPORTE.finditer(crudo):
        valor = _a_numero(m.group(1), m.group(2))
        if valor is not None and valor > 0:
            importes.append(valor)

    if not importes:
        return SalarioTexto(None, None, periodo, moneda, es_estimado, crudo)

    minimo = min(importes)
    # Un solo importe no es un rango de cero de ancho: es un importe sin techo declarado.
    maximo: float | None = max(importes) if len(set(importes)) > 1 else None

    # Guardarraíl de cordura sobre el equivalente anual. Sin esto, "vacante 2026" o
    # "hasta 20 plazas" se convierten en salarios.
    if periodo:
        factor = {"año": 1, "mes": 12, "quincena": 26, "semana": 52, "hora": 2080}[periodo]
        anual = minimo * factor
        if not (_PISO_ANUAL <= anual <= _TECHO_ANUAL):
            return SalarioTexto(None, None, periodo, moneda, es_estimado, crudo)

    return SalarioTexto(minimo, maximo, periodo, moneda, es_estimado, crudo)
