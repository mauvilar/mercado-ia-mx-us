#!/usr/bin/env bash
# Copia los cuatro notebooks ejecutados al clon de Proyectos-data-science.
set -euo pipefail
ORIGEN="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="$(cd "$ORIGEN/../../notebooks-source" && pwd)/mercado-ia-mx-us"  # el clon vive junto a proyectos/ en ~/Dev/data-science
mkdir -p "$DESTINO"
cp "$ORIGEN"/notebooks/01_recoleccion_y_calidad.ipynb \
   "$ORIGEN"/notebooks/02_brecha_salarial.ipynb \
   "$ORIGEN"/notebooks/03_skills_y_transparencia.ipynb \
   "$ORIGEN"/notebooks/04_modelo_imputacion.ipynb \
   "$DESTINO"/
echo "Copiado a $DESTINO:"; ls -la "$DESTINO"
