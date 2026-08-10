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
import subprocess
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "vacantes.csv"
SLUG = "vacantes-ia-mexico-estados-unidos"

# Columnas que NO se publican. `description_text` son los textos íntegros de las vacantes
# tal como los devuelve Adzuna (18 de los 20 MB del CSV) y `url` apunta a sus fichas:
# republicarlos sería redistribuir contenido de un tercero, no resultados propios, y los
# términos de la API de Adzuna permiten usar los datos, no reeditarlos como dataset
# descargable. Además ninguna de las dos hace falta para analizar: son insumo del pipeline,
# y lo que se deriva de ellas (skills, seniority, tier) sí se publica.
COLUMNAS_PRIVADAS = ["description_text", "url"]

DESCRIPCION = """\
Vacantes de Inteligencia Artificial en México y Estados Unidos, recolectadas de la API de
Adzuna cada semana y combinadas con un dataset real de Kaggle. Se publica porque no existía:
el mejor dataset público de la categoría tiene 12 filas de México y ninguna con salario, y el
más popular resultó ser data generada (doce países con ~4,300 filas cada uno y 100% de los
salarios presentes).

**Qué trae de distinto**

- `salary_is_predicted` está separado del salario observado. Adzuna *modela* sueldos cuando
  la vacante no los publica: en una corrida, Nueva York devolvió 151 vacantes y sólo 1 con
  salario real. Aquí ese filtro ya está aplicado; `salary_observed` sólo es verdadero cuando
  el empleador publicó la cifra.
- `tier` clasifica cada vacante en `nucleo` (el título declara que es de IA), `anillo` (el
  título no lo dice pero la descripción exige 2 o más skills de IA aplicada) o `fuera`. Sirve
  para separar la señal del ruido del buscador.
- Salarios normalizados: periodicidad → moneda → poder adquisitivo, con el factor PA.NUS.PPP
  del Banco Mundial aplicado sobre el monto en moneda local.
- `skills` y `seniority` extraídos del texto de cada vacante, no del título.

**Advertencias**

El lado mexicano tiene muy pocos salarios publicados — la opacidad del mercado es parte de lo
que este dataset documenta. Cualquier corte con n<30 no debería resumirse en una mediana.

El texto íntegro de las vacantes y sus URLs no se incluyen: son contenido de la fuente, no
resultados de este trabajo.

Código, metodología y análisis: https://github.com/mauvilar/mercado-ia-mx-us
"""


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
                # Kaggle exige entre 20 y 80 caracteres en el subtítulo.
                "subtitle": "Vacantes de IA con salario, clasificadas y normalizadas a PPP",
                "description": DESCRIPCION,
                "keywords": [
                    "jobs",
                    "salary",
                    "artificial intelligence",
                    "mexico",
                    "united states",
                ],
            },
            indent=2,
        )
    )


def preparar_csv_publicable(destino: Path) -> tuple[int, int]:
    """Escribe el CSV sin las columnas de contenido ajeno. Devuelve (filas, columnas)."""
    import pandas as pd

    df = pd.read_csv(DATASET_CSV, low_memory=False)
    df = df.drop(columns=[c for c in COLUMNAS_PRIVADAS if c in df.columns])
    df.to_csv(destino, index=False)
    return len(df), len(df.columns)


def main() -> None:
    _validar_credenciales()
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"No existe {DATASET_CSV}. Corre `make build` primero.")

    fecha = date.today().isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        # Carpeta aparte con sólo lo que se publica: data/processed/ también trae
        # vacantes.parquet y un .gitkeep que no son parte del dataset público.
        carpeta = Path(tmp)
        salida = carpeta / DATASET_CSV.name
        filas, columnas = preparar_csv_publicable(salida)
        escribir_metadata(carpeta)
        print(
            f"Publicando {filas:,} filas × {columnas} columnas "
            f"({salida.stat().st_size / 1e6:.1f} MB), sin {', '.join(COLUMNAS_PRIVADAS)}"
        )
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(carpeta), "-m", f"snapshot {fecha}"],
            check=True,
        )
    print(f"✅ Versión del {fecha} publicada en Kaggle")


if __name__ == "__main__":
    main()
