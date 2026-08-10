"""Descarga y validación de los dos datasets reales de Kaggle.

Incluye el detector de datasets sintéticos que descartó m0sm71 (§1.2 del spec).
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS
from src.utils.config import DATA_DIR

REFS = {
    "kaggle_mannacharya": "mannacharya/ai-job-listings-bi-weekly-updated",
    "aijobs_net": "aijobs/global-salaries-in-ai-ml-data-science",
    # Se descarga a propósito aunque NO se use en el análisis: el notebook 01 lo
    # abre para mostrar por qué se descartó (§1.2 del spec). Vive bajo _descartado/
    # para que nadie lo confunda con una fuente válida.
    "_descartado": "m0sm71/ai-jobs-dataset-2026",
}


def verificar_no_sintetico(df: pd.DataFrame, *, col_pais: str, col_salario: str) -> dict[str, Any]:
    """Un dataset real de vacantes es desigual por país y tiene salarios ausentes.

    Uniformidad + 100% de salarios presentes = generado.
    """
    conteos = df[col_pais].value_counts()
    if len(conteos) < 3:
        return {"sospechoso": False, "razon": "muy pocos países para juzgar"}

    cv = conteos.std() / conteos.mean()  # coeficiente de variación
    llenado = df[col_salario].notna().mean()

    if cv < 0.10 and llenado > 0.99:
        return {
            "sospechoso": True,
            "razon": (
                f"distribución uniforme por país (CV={cv:.3f}) y {llenado:.0%} de salarios "
                "presentes: patrón de dataset generado, no scrapeado"
            ),
        }
    return {"sospechoso": False, "razon": f"CV={cv:.2f}, salarios presentes={llenado:.0%}"}


def descargar(ref: str, destino: Path) -> Path:
    """Descarga vía la API de Kaggle usando ~/.kaggle/kaggle.json."""
    destino.mkdir(parents=True, exist_ok=True)
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    auth = base64.b64encode(f"{creds['username']}:{creds['key']}".encode()).decode()
    zip_path = destino / f"{ref.split('/')[1]}.zip"
    if not zip_path.exists():
        subprocess.run(
            [
                "curl",
                "-sL",
                "-H",
                f"Authorization: Basic {auth}",
                "-o",
                str(zip_path),
                f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
            ],
            check=True,
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destino)
    return destino


_PAISES = {"United States": "US", "USA": "US", "US": "US", "Mexico": "MX", "México": "MX"}
_PERIODOS = {"year": "año", "yearly": "año", "month": "mes", "hour": "hora"}


def map_mannacharya_frame(crudo: pd.DataFrame, *, snapshot_date: date) -> pd.DataFrame:
    out = pd.DataFrame(columns=CANONICAL_COLUMNS, index=crudo.index)
    out["posting_id"] = crudo["id"].map(
        lambda i: hashlib.sha1(f"kaggle_mannacharya:{i}".encode()).hexdigest()
    )
    out["snapshot_date"] = snapshot_date
    out["source"] = "kaggle_mannacharya"
    out["title_raw"] = crudo["title"]
    out["title_norm"] = crudo["title"].astype(str).str.strip().str.lower()
    out["company"] = crudo["company"]
    out["country"] = crudo["country"].map(_PAISES)
    out["city"] = crudo["city"]
    out["salary_min_raw"] = crudo["salary_min"]
    out["salary_max_raw"] = crudo["salary_max"]
    out["salary_currency"] = crudo["salary_currency"].fillna("USD")
    out["salary_period"] = crudo["salary_period"].map(_PERIODOS).fillna("año")
    # dtype=object + bool de Python explícito: una columna bool nativa de pandas
    # devuelve numpy.bool_ vía .loc, que falla comparaciones `is True`/`is False`.
    out["salary_is_predicted"] = pd.Series(False, index=out.index, dtype=object)
    out["salary_observed"] = crudo["salary_min"].notna().astype(object)
    out["description_text"] = crudo["description_text"]
    out["description_lang"] = "en"
    out["posted_date"] = pd.to_datetime(crudo["posted_date"], errors="coerce").dt.date
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def main() -> None:
    for nombre, ref in REFS.items():
        descargar(ref, DATA_DIR / "raw" / "kaggle" / nombre)
        print(f"✅ {nombre}: {ref}")


if __name__ == "__main__":
    main()
