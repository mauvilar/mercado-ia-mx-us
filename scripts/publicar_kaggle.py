"""Sube una versión nueva del dataset procesado a Kaggle.

Corrección sobre el spec original (Task 20): el spec proponía un solo POST multipart
(`curl -F file=@vacantes.csv ... /api/v1/datasets/create/version`) con auth Basic a mano.
Eso no corresponde a como funciona la API real de Kaggle:

- `datasets/create/version` versiona un dataset que YA EXISTE — no lo crea. Requiere que
  `dataset-metadata.json` esté presente junto a los archivos y que su `id` apunte a un
  dataset ya publicado (la primera creación es un paso manual, ver checklist del PR).
- Subir un archivo no es una llamada: el cliente oficial primero sube cada archivo como
  blob (con soporte de reintento/reanudación para archivos grandes) y recibe un token de
  vuelta, y sólo entonces crea la versión referenciando esos tokens. Un POST directo con
  `-F file=@...` a `create/version` no es la forma que ese endpoint espera.
- Ese protocolo de dos pasos es un detalle interno (los nombres de las clases en el SDK
  actual sugieren que está generado, no es un contrato REST público estable). Reimplementar
  eso a mano en curl es exactamente el tipo de cosa que se rompe en silencio con el próximo
  cambio de Kaggle.

Por eso este script no habla HTTP directo: llama al CLI oficial (`pip install kaggle`,
paquete `kaggle` en PyPI), que sí es un contrato estable y documentado
(https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md). El CLI resuelve el
flujo de dos pasos, la conversión de credenciales y el reintento por nosotros.

Autenticación: el CLI de Kaggle detecta KAGGLE_USERNAME / KAGGLE_KEY del entorno con
prioridad sobre ~/.kaggle/kaggle.json (ver docs de arriba). El workflow ya inyecta esas
dos variables desde los secrets del repo, así que no hace falta escribir el archivo.

Límite de tamaño de Kaggle: 200 GB por dataset (kaggle.com/docs/datasets, revisado al
escribir este script). `vacantes.csv` pesa ~20 MB — tres órdenes de magnitud por debajo,
nada que vigilar por ahora. Si el dataset creciera mucho más, el CLI ya trae soporte de
subida reanudable; un curl a mano no lo tiene.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "vacantes.csv"
SLUG = "vacantes-ia-mexico-estados-unidos"


def _validar_credenciales() -> None:
    faltantes = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.environ.get(v)]
    if faltantes:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltantes)}")


def escribir_metadata(carpeta: Path) -> None:
    usuario = os.environ["KAGGLE_USERNAME"]
    (carpeta / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Vacantes de IA: México y Estados Unidos",
                "id": f"{usuario}/{SLUG}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
    )


def main() -> None:
    _validar_credenciales()
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"No existe {DATASET_CSV}. Corre `make build` primero.")

    fecha = date.today().isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        # Carpeta aparte con sólo lo que se publica: data/processed/ también trae
        # vacantes.parquet y un .gitkeep que no son parte del dataset público.
        carpeta = Path(tmp)
        shutil.copy2(DATASET_CSV, carpeta / DATASET_CSV.name)
        escribir_metadata(carpeta)
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(carpeta), "-m", f"snapshot {fecha}"],
            check=True,
        )
    print(f"✅ Versión del {fecha} publicada en Kaggle")


if __name__ == "__main__":
    main()
