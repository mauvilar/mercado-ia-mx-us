"""Anclas salariales oficiales: el nivel de cada país medido fuera del corpus.

El notebook 04 documenta por qué el modelo erró 958 % contra México: `country` fue una
constante durante el entrenamiento, así que nunca hubo con qué aprender que existen
mercados más baratos. Los países de calibración atacan eso desde dentro del corpus. Este
módulo lo ataca desde fuera, con estadística oficial.

La idea es separar dos cosas que hoy viven pegadas en una sola cifra:

  · **Nivel nacional** — cuánto vale, en dólares internacionales, un salario típico en
    cada país. NO transfiere entre países, y no hace falta que lo haga: se toma del IMSS
    y del INEGI para México, y del BLS para Estados Unidos.

  · **Posición relativa** — cuántas veces el salario típico de su país paga una vacante.
    Que un ingeniero de ML senior gane 3.2 veces el salario medio de su país SÍ es una
    regularidad que puede transferir, y se puede validar contra muchos países en vez de
    contra cuatro filas mexicanas.

Advertencia de método, que condiciona todo uso de este módulo: estas fuentes miden
**salarios pagados a trabajadores**, no salarios publicados en vacantes. Son unidades de
observación distintas y no son intercambiables. Por eso las anclas no entran al hold-out
ni sustituyen una observación mexicana: sirven para poner en escala, para validar de forma
independiente las medianas del corpus, y para dar un denominador a la posición relativa.

Ninguna de las dos se descarga sola: se dejan en `data/raw/anclas/` y este módulo lee lo
que encuentre. Sin archivo no hay ancla, y sin ancla las funciones devuelven None en vez
de inventar un número.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CARPETA_ANCLAS = "anclas"


@dataclass(frozen=True)
class Ancla:
    """El nivel salarial de un país según una fuente oficial, ya anualizado."""

    pais: str
    fuente: str
    salario_anual_local: float
    moneda: str
    unidad: str
    nota: str

    def a_ppp(self, factores: dict[str, float]) -> float:
        """El ancla en dólares internacionales, comparable entre países."""
        if self.pais not in factores:
            raise KeyError(f"sin factor PPP para {self.pais!r}")
        return self.salario_anual_local / factores[self.pais]


# ── México ───────────────────────────────────────────────────────────────────
# El IMSS publica el salario base de cotización de los puestos registrados. Es el dato
# de salario más limpio que existe para México: no es una encuesta ni una estimación,
# es la base sobre la que se cotiza, por patrón y por municipio.
#
# Techo conocido: el SBC está topado en 25 UMA, así que la cola alta de la distribución
# queda comprimida. Para una vacante de IA en CDMX, que muy probablemente está sobre ese
# tope, el ancla del IMSS es un piso, no una media. Se usa como orden de magnitud.
COLUMNAS_IMSS = {
    "salario": ["masa_sal_ta", "salario_base", "SALARIO_BASE", "masa_sal"],
    "asegurados": ["ta", "asegurados", "TA", "no_trabajadores"],
    "estado": ["cve_entidad", "entidad", "CVE_ENTIDAD"],
}

# ── Estados Unidos ───────────────────────────────────────────────────────────
# El BLS publica la Occupational Employment and Wage Statistics (OEWS) por ocupación y
# área metropolitana. Para comparar contra este corpus interesa el código SOC de las
# ocupaciones de datos y software, no el total de la economía.
SOC_INTERES = {
    "15-2051",  # Data Scientists
    "15-1252",  # Software Developers
    "15-1221",  # Computer and Information Research Scientists
    "15-1211",  # Computer Systems Analysts
}

COLUMNAS_OEWS = {
    "soc": ["OCC_CODE", "occ_code"],
    "titulo": ["OCC_TITLE", "occ_title"],
    "area": ["AREA_TITLE", "area_title"],
    "media_anual": ["A_MEAN", "a_mean"],
    "mediana_anual": ["A_MEDIAN", "a_median"],
}


class AnclaError(ValueError):
    """El archivo del ancla no trae las columnas que se necesitan."""


def _columna(df: pd.DataFrame, alias: list[str], *, requerida: bool = True) -> str | None:
    presentes = {c.upper().strip(): c for c in df.columns}
    for a in alias:
        if a.upper() in presentes:
            return presentes[a.upper()]
    if requerida:
        raise AnclaError(
            f"El archivo no trae ninguna de las columnas {alias}. "
            f"Encontradas: {sorted(df.columns)[:20]}"
        )
    return None


def ancla_imss(crudo: pd.DataFrame) -> Ancla:
    """Salario diario promedio del IMSS, anualizado a 365 días.

    El SBC se declara por día, que es la unidad con la que trabaja la seguridad social
    mexicana, y se anualiza multiplicando por 365 y no por 250: cotiza todos los días
    del año, no sólo los laborables.
    """
    col_salario = _columna(crudo, COLUMNAS_IMSS["salario"])
    col_asegurados = _columna(crudo, COLUMNAS_IMSS["asegurados"], requerida=False)

    masa = pd.to_numeric(crudo[col_salario], errors="coerce")
    if col_asegurados:
        # El archivo trae masa salarial y número de asegurados por registro patronal:
        # el promedio correcto es la razón entre las dos sumas, no el promedio de los
        # promedios, que le daría el mismo peso a un patrón de 3 y a uno de 3,000.
        trabajadores = pd.to_numeric(crudo[col_asegurados], errors="coerce")
        valido = masa.notna() & trabajadores.notna() & (trabajadores > 0)
        diario = float(masa[valido].sum() / trabajadores[valido].sum())
    else:
        diario = float(masa.dropna().mean())

    return Ancla(
        pais="MX",
        fuente="IMSS · salario base de cotización",
        salario_anual_local=diario * 365,
        moneda="MXN",
        unidad="promedio de todos los asegurados",
        nota=(
            "Todas las ocupaciones, no sólo IA. Topado en 25 UMA, así que comprime la "
            "cola alta: para una vacante de IA funciona como piso, no como media."
        ),
    )


def ancla_oews(crudo: pd.DataFrame, *, soc: set[str] | None = None) -> Ancla:
    """Mediana anual del BLS para las ocupaciones de datos y software."""
    col_soc = _columna(crudo, COLUMNAS_OEWS["soc"])
    col_mediana = _columna(crudo, COLUMNAS_OEWS["mediana_anual"], requerida=False) or _columna(
        crudo, COLUMNAS_OEWS["media_anual"]
    )

    objetivo = soc or SOC_INTERES
    filtrado = crudo[crudo[col_soc].astype(str).str.strip().isin(objetivo)]
    if filtrado.empty:
        raise AnclaError(
            f"Ninguna fila con los códigos SOC {sorted(objetivo)}. "
            "Revisa si el archivo es el nacional por ocupación."
        )

    valores = pd.to_numeric(filtrado[col_mediana], errors="coerce").dropna()
    if valores.empty:
        raise AnclaError("Las filas del SOC objetivo no traen salario numérico legible.")

    return Ancla(
        pais="US",
        fuente="BLS · OEWS",
        salario_anual_local=float(valores.median()),
        moneda="USD",
        unidad=f"mediana de los SOC {sorted(objetivo)}",
        nota="Salarios pagados, no publicados en vacantes: sirve de contraste, no de hold-out.",
    )


def cargar_anclas(carpeta: Path) -> dict[str, Ancla]:
    """Lee data/raw/anclas/. Sin archivo no hay ancla, y eso se reporta, no se rellena."""
    anclas: dict[str, Ancla] = {}
    if not carpeta.exists():
        log.info("Sin carpeta de anclas en %s: el análisis corre sin referencia externa.", carpeta)
        return anclas

    lectores = {"imss": ancla_imss, "oews": ancla_oews}
    for prefijo, lector in lectores.items():
        archivos = sorted(carpeta.glob(f"{prefijo}*.csv"))
        archivos += sorted(carpeta.glob(f"{prefijo}*.xlsx"))
        if not archivos:
            log.info("Sin archivo de ancla %s en %s", prefijo, carpeta)
            continue
        archivo = archivos[-1]
        crudo = (
            pd.read_csv(archivo, low_memory=False)
            if archivo.suffix == ".csv"
            else pd.read_excel(archivo)
        )
        try:
            ancla = lector(crudo)
        except AnclaError as exc:
            log.warning("Ancla %s ilegible (%s): %s", prefijo, archivo.name, exc)
            continue
        anclas[ancla.pais] = ancla
        log.info("Ancla %s: %s %s anuales", ancla.pais, ancla.salario_anual_local, ancla.moneda)

    return anclas


def niveles_ppp(anclas: dict[str, Ancla], factores_ppp: dict[str, float]) -> dict[str, float]:
    """Nivel salarial de cada país en dólares internacionales, listo para comparar."""
    return {
        pais: ancla.a_ppp(factores_ppp) for pais, ancla in anclas.items() if pais in factores_ppp
    }


def posicion_relativa(
    salarios_ppp: pd.Series, paises: pd.Series, niveles: dict[str, float]
) -> pd.Series:
    """Cuántas veces el nivel de su propio país paga cada vacante.

    Es el objetivo que sí puede transferir entre mercados. Una vacante sin ancla para su
    país sale como nulo: dividir entre un nivel prestado sería peor que no medir.
    """
    denominador = paises.map(niveles)
    return pd.Series(
        [
            float(s) / float(d) if pd.notna(s) and pd.notna(d) and d > 0 else None
            for s, d in zip(salarios_ppp, denominador, strict=True)
        ],
        index=salarios_ppp.index,
        dtype="float64",
    )


def contrastar(
    df: pd.DataFrame, anclas: dict[str, Ancla], factores_ppp: dict[str, float]
) -> pd.DataFrame:
    """Mediana del corpus contra el ancla oficial, país por país.

    Sirve como validación independiente: si el corpus dice que una vacante de IA en
    Estados Unidos paga cuatro veces el ancla del BLS, eso es creíble; si dijera cuarenta,
    hay algo roto en la normalización mucho antes que en el modelo.
    """
    niveles = niveles_ppp(anclas, factores_ppp)
    filas = []
    observado = df["salary_observed"].fillna(False).astype(bool)
    for pais, grupo in df[observado].groupby("country"):
        nivel = niveles.get(str(pais))
        mediana = float(grupo["salary_annual_usd_ppp"].median())
        filas.append(
            {
                "country": pais,
                "n": int(len(grupo)),
                "mediana_corpus_ppp": mediana,
                "ancla_ppp": nivel,
                "veces_el_ancla": mediana / nivel if nivel else None,
                "fuente_ancla": anclas[str(pais)].fuente if str(pais) in anclas else None,
            }
        )
    return pd.DataFrame(filas)
