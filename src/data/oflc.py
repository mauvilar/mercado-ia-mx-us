"""Datos de divulgación del OFLC (Departamento del Trabajo de EE.UU.): archivos LCA.

Cada solicitud de visa H-1B obliga al patrón a declarar el puesto, la sede de trabajo y
**el salario que ofrece**, y el DOL publica esos registros cada trimestre. Es la fuente
salarial estadounidense más honesta que existe en abierto: no es una estimación de un
agregador ni una encuesta, es lo que una empresa se comprometió por escrito a pagar, con
consecuencias legales si miente.

Sesgo, que hay que declarar: sólo aparecen patrones que patrocinan visas. Sobrerrepresenta
consultoras de TI y grandes tecnológicas, e infrarrepresenta startups y empresas pequeñas.
Sirve para dar volumen y estructura salarial al lado estadounidense del modelo, no como
retrato del mercado completo.

No se descarga sola a propósito: los archivos pesan cientos de megas y cambian de nombre
cada trimestre. Se bajan a mano desde el portal de datos de divulgación del OFLC y se
dejan en `data/raw/oflc/`; este módulo lee lo que encuentre ahí.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.mapping import _canonizar_ciudad
from src.data.schema import CANONICAL_COLUMNS

log = logging.getLogger(__name__)

# El DOL renombra columnas entre años fiscales. Cada entrada lista los alias conocidos,
# de más reciente a más viejo, y el cargador toma el primero que exista en el archivo.
COLUMNAS = {
    "case": ["CASE_NUMBER", "CASE_NO"],
    "estado_caso": ["CASE_STATUS"],
    "titulo": ["JOB_TITLE"],
    "soc": ["SOC_TITLE", "SOC_NAME"],
    "empleador": ["EMPLOYER_NAME"],
    "ciudad": ["WORKSITE_CITY", "WORKSITE_CITY_1", "EMPLOYER_CITY"],
    "estado": ["WORKSITE_STATE", "WORKSITE_STATE_1", "EMPLOYER_STATE"],
    "salario_desde": [
        "WAGE_RATE_OF_PAY_FROM",
        "WAGE_RATE_OF_PAY_FROM_1",
        "LCA_CASE_WAGE_RATE_FROM",
    ],
    "salario_hasta": ["WAGE_RATE_OF_PAY_TO", "WAGE_RATE_OF_PAY_TO_1", "LCA_CASE_WAGE_RATE_TO"],
    "unidad": ["WAGE_UNIT_OF_PAY", "WAGE_UNIT_OF_PAY_1", "LCA_CASE_WAGE_RATE_UNIT"],
    "fecha": ["RECEIVED_DATE", "DECISION_DATE"],
}

PERIODOS = {
    "year": "año",
    "yr": "año",
    "month": "mes",
    "mth": "mes",
    "bi-weekly": "quincena",
    "biweekly": "quincena",
    "bi_weekly": "quincena",
    "week": "semana",
    "wk": "semana",
    "hour": "hora",
    "hr": "hora",
}

# Sólo interesan los casos que el DOL certificó: un caso denegado o retirado no
# representa un salario que alguien fuera a pagar.
ESTADOS_VALIDOS = {"CERTIFIED", "CERTIFIED-WITHDRAWN", "CERTIFIED - WITHDRAWN"}


class OflcError(ValueError):
    """El archivo del OFLC no trae las columnas que el mapeo necesita."""


def _resolver(crudo: pd.DataFrame) -> dict[str, str]:
    """Traduce los alias de cada año fiscal a los nombres que usa el mapeo."""
    presentes = {c.upper().strip(): c for c in crudo.columns}
    resuelto: dict[str, str] = {}
    for logico, alias in COLUMNAS.items():
        for a in alias:
            if a in presentes:
                resuelto[logico] = presentes[a]
                break
    faltan = [k for k in ("titulo", "salario_desde", "unidad") if k not in resuelto]
    if faltan:
        raise OflcError(
            f"El archivo del OFLC no trae columnas para {faltan}. "
            f"Columnas encontradas: {sorted(presentes)[:25]}... "
            "Revisa si el DOL cambió el formato y añade el alias en COLUMNAS."
        )
    return resuelto


def map_oflc_frame(crudo: pd.DataFrame, *, snapshot_date: date) -> pd.DataFrame:
    col = _resolver(crudo)

    if "estado_caso" in col:
        antes = len(crudo)
        estados = crudo[col["estado_caso"]].astype(str).str.upper().str.strip()
        crudo = crudo[estados.isin(ESTADOS_VALIDOS)]
        log.info("OFLC: %s de %s casos certificados", len(crudo), antes)

    crudo = crudo.reset_index(drop=True)
    out = pd.DataFrame(columns=CANONICAL_COLUMNS, index=crudo.index)

    identificador = (
        crudo[col["case"]].astype(str)
        if "case" in col
        else pd.Series(crudo.index.astype(str), index=crudo.index)
    )
    out["posting_id"] = identificador.map(
        lambda i: hashlib.sha1(f"oflc_lca:{i}".encode()).hexdigest()
    )
    out["snapshot_date"] = snapshot_date
    out["source"] = "oflc_lca"
    out["title_raw"] = crudo[col["titulo"]]
    out["title_norm"] = crudo[col["titulo"]].astype(str).str.strip().str.lower()
    out["company"] = crudo[col["empleador"]] if "empleador" in col else None
    out["category"] = crudo[col["soc"]] if "soc" in col else None
    out["country"] = "US"
    out["state"] = crudo[col["estado"]] if "estado" in col else None

    ciudad_cruda = (
        crudo[col["ciudad"]].fillna("").astype(str)
        if "ciudad" in col
        else pd.Series("", index=crudo.index)
    )
    canon = ciudad_cruda.map(lambda c: _canonizar_ciudad("US", [], c))
    out["city"] = [x[0] or (c or None) for x, c in zip(canon, ciudad_cruda, strict=True)]
    out["metro"] = [x[1] for x in canon]

    # Una LCA no describe modalidad de trabajo: declara una sede física. Marcarla como
    # remota sería inventar, así que se deja en False y remote_scope caerá en "local".
    out["is_remote"] = False
    out["description_text"] = ""
    out["description_lang"] = "en"
    out["posted_date"] = (
        pd.to_datetime(crudo[col["fecha"]], errors="coerce").dt.date if "fecha" in col else None
    )

    desde = pd.to_numeric(crudo[col["salario_desde"]], errors="coerce")
    hasta = (
        pd.to_numeric(crudo[col["salario_hasta"]], errors="coerce")
        if "salario_hasta" in col
        else pd.Series(pd.NA, index=crudo.index)
    )
    unidad = crudo[col["unidad"]].astype(str).str.strip().str.lower().map(PERIODOS)

    out["salary_min_raw"] = desde
    out["salary_max_raw"] = hasta
    out["salary_currency"] = "USD"
    out["salary_period"] = unidad
    out["salary_is_predicted"] = pd.Series(False, index=out.index, dtype=object)
    # El salario ofrecido es un compromiso legal, no una estimación. Lo único que puede
    # invalidarlo aquí es que venga vacío, en cero, o con una unidad que no sepamos leer:
    # sin periodicidad no hay forma de anualizarlo y anualizarlo mal sería peor.
    out["salary_observed"] = (desde.notna() & (desde > 0) & unidad.notna()).astype(object)

    sin_unidad = int((desde.notna() & (desde > 0) & unidad.isna()).sum())
    if sin_unidad:
        log.warning(
            "OFLC: %s filas con salario pero unidad de pago no reconocida, quedan sin observar",
            sin_unidad,
        )

    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def cargar(carpeta: Path, *, snapshot_date: date) -> list[pd.DataFrame]:
    """Lee todo lo que haya en data/raw/oflc/. Opcional: si no hay nada, no pasa nada."""
    if not carpeta.exists():
        return []
    archivos = sorted([*carpeta.glob("*.csv"), *carpeta.glob("*.xlsx"), *carpeta.glob("*.parquet")])
    if not archivos:
        log.info("Sin archivos del OFLC en %s: el corpus se construye sin ellos.", carpeta)
        return []

    marcos = []
    for archivo in archivos:
        if archivo.suffix == ".csv":
            crudo = pd.read_csv(archivo, low_memory=False)
        elif archivo.suffix == ".parquet":
            crudo = pd.read_parquet(archivo)
        else:
            crudo = pd.read_excel(archivo)
        log.info("OFLC: %s filas de %s", len(crudo), archivo.name)
        marcos.append(map_oflc_frame(crudo, snapshot_date=snapshot_date))
    return marcos
