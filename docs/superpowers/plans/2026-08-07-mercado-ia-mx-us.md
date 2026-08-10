# Mercado de trabajo de IA MX vs EE.UU. — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un dataset propio y vivo de vacantes de IA en México y Estados Unidos, y analizarlo para responder cuánto paga el trabajo de IA en cada zona y por qué el mercado mexicano es opaco.

**Architecture:** Un pipeline de tres capas. `src/data/` ingiere de la API de Adzuna y de dos datasets reales de Kaggle hacia snapshots inmutables en `data/raw/`. `src/features/` los normaliza (periodicidad → moneda → PPP), clasifica (núcleo/anillo), extrae seniority y skills del texto, y deduplica hacia `data/processed/`. Encima, `src/analysis/` y `src/models/` alimentan cuatro notebooks narrativos. Un GitHub Action semanal repite la ingesta y versiona el dataset en Kaggle, de modo que el corpus crece solo.

**Tech Stack:** Python 3.12 con `uv` · pandas · numpy · scipy · scikit-learn · matplotlib · seaborn · requests · pyarrow · pyyaml · pytest · ruff · mypy

**Spec:** `docs/superpowers/specs/2026-08-07-mercado-ia-mx-us-design.md`

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/utils/config.py` | Cargar `.env` y los YAML de `config/`. Única puerta a la configuración |
| `src/utils/logging.py` | Logger con formato consistente para CLI y Action |
| `src/data/schema.py` | Esquema canónico, tipos, y `validate_frame()`. La única fuente de verdad de las columnas |
| `src/data/adzuna.py` | Cliente HTTP: paginación, rate limit, backoff. No sabe nada del esquema canónico |
| `src/data/mapping.py` | Traduce respuestas crudas (Adzuna, Kaggle) al esquema canónico. Aísla el formato de cada fuente |
| `src/data/kaggle_sources.py` | Descarga y valida los dos datasets reales de Kaggle |
| `src/data/collect.py` | CLI de una corrida: orquesta, escribe snapshot + `manifest.json` |
| `src/features/normalize.py` | Periodicidad → moneda → PPP. Funciones puras |
| `src/features/taxonomy.py` | Clasificación núcleo / anillo / fuera |
| `src/features/extract.py` | Seniority y skills desde la descripción |
| `src/features/dedupe.py` | Colapso de duplicados entre fuentes |
| `src/features/build.py` | CLI: `data/raw/*` → `data/processed/vacantes.parquet` |
| `src/analysis/stats.py` | Mediana con IC bootstrap, regla n ≥ 30 |
| `src/analysis/hypothesis.py` | Mann-Whitney y Welch con reporte legible |
| `src/models/impute.py` | Entrenamiento del imputador |
| `src/models/evaluate.py` | Hold-out mexicano, baseline, criterio de publicación |

Separo `adzuna.py` (habla HTTP) de `mapping.py` (habla esquema) a propósito: cuando Adzuna cambie un campo, sólo se toca `mapping.py`, y los tests de mapeo corren sobre JSON fijo sin red.

---

## Fase 0 — Andamio y compuerta de viabilidad

### Task 1: Andamio del proyecto

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.env.example`, `src/__init__.py`, `src/utils/__init__.py`, `src/utils/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Create: `config/cities.yml`, `config/skills.yml`, `config/taxonomy.yml`
- Create: `data/raw/.gitkeep`, `data/interim/.gitkeep`, `data/processed/.gitkeep`, `reports/figures/.gitkeep`

- [ ] **Step 1: Crear `pyproject.toml`**

```toml
[project]
name = "mercado-ia-mx-us"
version = "0.1.0"
description = "Mercado de trabajo de IA: México vs Estados Unidos"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12,<3.13"
authors = [{ name = "Mau Vilar", email = "unicemau@gmail.com" }]

dependencies = [
    "pandas>=2.2,<2.3",
    "numpy>=1.26,<2.2",
    "scipy>=1.13,<1.15",
    "scikit-learn>=1.5,<1.6",
    "requests>=2.32,<3.0",
    "pyarrow>=17.0,<19.0",
    "pyyaml>=6.0,<7.0",
    "python-dotenv>=1.0,<2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<9.0",
    "pytest-cov>=5.0,<6.0",
    "ruff>=0.7,<0.9",
    "mypy>=1.13,<2.0",
    "pandas-stubs>=2.2,<2.3",
    "types-requests>=2.32,<3.0",
    "types-PyYAML>=6.0,<7.0",
]
notebook = [
    "jupyterlab>=4.3,<5.0",
    "ipykernel>=6.29,<7.0",
    "matplotlib>=3.9,<4.0",
    "seaborn>=0.13,<0.14",
    "nbconvert>=7.16,<8.0",
]

[project.scripts]
mia-collect = "src.data.collect:main"
mia-build = "src.features.build:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

- [ ] **Step 2: Crear `Makefile`**

```makefile
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY_VERSION := 3.12
RUN := uv run

.PHONY: help
help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Crear el venv e instalar todo
	uv python install $(PY_VERSION)
	uv sync --all-extras
	@echo "✅ Listo. Copia .env.example a .env y pon tus llaves de Adzuna."

.PHONY: probe
probe: ## Compuerta día 0: medir cobertura salarial de Adzuna MX
	$(RUN) python -m src.data.probe

.PHONY: collect
collect: ## Una corrida de recolección -> data/raw/
	$(RUN) mia-collect

.PHONY: build
build: ## data/raw/ -> data/processed/vacantes.parquet
	$(RUN) mia-build

.PHONY: test
test: ## Correr las pruebas
	$(RUN) pytest -v

.PHONY: lint
lint: ## ruff + mypy
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests
	$(RUN) mypy src

.PHONY: notebooks
notebooks: ## Ejecutar los 4 notebooks de punta a punta
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

- [ ] **Step 3: Crear `.env.example`**

```bash
# developer.adzuna.com -> registro gratuito
ADZUNA_APP_ID=
ADZUNA_APP_KEY=

# ~/.kaggle/kaggle.json
KAGGLE_USERNAME=
KAGGLE_KEY=
```

- [ ] **Step 4: Crear `config/cities.yml`**

```yaml
# Cada entrada mapea los alias que devuelven las APIs a una ciudad canónica y su metro.
MX:
  - canonical: Ciudad de México
    metro: Valle de México
    aliases: [Ciudad de Mexico, CDMX, Mexico City, Distrito Federal, Cuauhtémoc, Cuauhtemoc,
              Miguel Hidalgo, Benito Juárez, Benito Juarez, Álvaro Obregón, Alvaro Obregon,
              Naucalpan, Tlalnepantla]
  - canonical: Querétaro
    metro: Querétaro
    aliases: [Queretaro, Santiago de Querétaro, Santiago de Queretaro, El Marqués, El Marques,
              Corregidora, San Juan del Río, San Juan del Rio]
  - canonical: Monterrey
    metro: Monterrey
    aliases: [San Pedro Garza García, San Pedro Garza Garcia, Guadalupe, Apodaca,
              Santa Catarina, San Nicolás, San Nicolas]
  - canonical: Guadalajara
    metro: Guadalajara
    aliases: [Zapopan, Tlaquepaque, Tonalá, Tonala, Tlajomulco]
US:
  - canonical: San Francisco
    metro: SF Bay Area
    aliases: [South San Francisco, Palo Alto, Mountain View, Menlo Park, Santa Clara,
              San Jose, Sunnyvale, Cupertino, Redwood City, Oakland, Berkeley]
  - canonical: New York
    metro: NYC
    aliases: [New York City, Manhattan, Brooklyn, Queens, Jersey City]
  - canonical: Seattle
    metro: Seattle
    aliases: [Bellevue, Redmond, Kirkland]
  - canonical: Austin
    metro: Austin
    aliases: [Round Rock, Cedar Park]
  - canonical: Boston
    metro: Boston
    aliases: [Cambridge, Somerville, Waltham]
  - canonical: Los Angeles
    metro: Los Angeles
    aliases: [Santa Monica, Pasadena, Burbank, El Segundo, Culver City]
```

- [ ] **Step 5: Crear `config/skills.yml`**

```yaml
# grupo "ia_aplicada": son los que disparan la clasificación de anillo (§5.1 del spec)
ia_aplicada:
  llm: [llm, large language model, gpt, claude, gemini, modelo de lenguaje]
  rag: [rag, retrieval augmented, retrieval-augmented]
  # 'lora' se quitó: colisiona con LoRa/LoRaWAN, muy común en vacantes de IoT/embebidos
  fine_tuning: [fine-tuning, fine tuning, finetuning, peft, ajuste fino]
  # 'torch' a secas se quitó: colisiona con soplete/torch de oxicorte en vacantes industriales
  pytorch: [pytorch]
  tensorflow: [tensorflow, keras]
  langchain: [langchain, llamaindex, llama-index]
  vector_db: [pinecone, weaviate, qdrant, chroma, milvus, pgvector, vector database,
              base de datos vectorial]
  embeddings: [embedding, embeddings, sentence-transformers]
  # 'transformer' singular se quitó: colisiona con transformador eléctrico en mantenimiento industrial
  transformers: [transformers, hugging face, huggingface]
  prompt_eng: [prompt engineering, ingeniería de prompts]
  mlops: [mlops, mlflow, kubeflow, model serving, feature store]
  # 'mcp' a secas se quitó: colisiona con Microsoft Certified Professional
  agents: [ai agent, agentes de ia, agentic, multi-agent, model context protocol]

# grupo "soporte": se miden para las primas salariales pero NO disparan el anillo
soporte:
  python: [python]
  sql: [sql, postgresql, mysql, bigquery, snowflake]
  spark: [spark, pyspark, databricks]
  aws: [aws, sagemaker, bedrock]
  gcp: [gcp, google cloud, vertex ai]
  azure: [azure, azure openai]
  docker: [docker, kubernetes, k8s]
  airflow: [airflow, dagster, prefect]
```

- [ ] **Step 6: Crear `config/taxonomy.yml`**

```yaml
# NÚCLEO: el título declara el trabajo de IA. Requiere término_ia Y término_rol.
nucleo:
  terminos_ia:
    - ai
    - a\.i\.
    - artificial intelligence
    - inteligencia artificial
    - ml
    - machine learning
    - aprendizaje autom[áa]tico
    - deep learning
    - llm
    - nlp
    - natural language
    - computer vision
    - visi[óo]n por computadora
    - mlops
    - gen ?ai
    - generative ai
    - ia generativa
  terminos_rol:
    - engineer
    - ingenier[oa]
    - developer
    - desarrollador[a]?
    - scientist
    - cient[íi]fic[oa]
    - architect
    - arquitect[oa]
    - specialist
    - especialista

# ANILLO: no cumple núcleo pero la descripción trae >= min_skills del grupo ia_aplicada.
anillo:
  min_skills: 2
```

- [ ] **Step 7: Escribir el test que falla**

`tests/test_config.py`:

```python
from src.utils.config import load_cities, load_skills, load_taxonomy


def test_cities_incluye_las_cuatro_ciudades_mexicanas():
    cities = load_cities()
    canonicas = {c["canonical"] for c in cities["MX"]}
    assert canonicas == {"Ciudad de México", "Querétaro", "Monterrey", "Guadalajara"}


def test_alias_de_cdmx_incluye_alcaldias():
    cities = load_cities()
    cdmx = next(c for c in cities["MX"] if c["canonical"] == "Ciudad de México")
    assert "Cuauhtémoc" in cdmx["aliases"]


def test_skills_separa_ia_aplicada_de_soporte():
    skills = load_skills()
    assert "rag" in skills["ia_aplicada"]
    assert "python" in skills["soporte"]
    assert "python" not in skills["ia_aplicada"]


def test_taxonomia_exige_dos_skills_para_anillo():
    tax = load_taxonomy()
    assert tax["anillo"]["min_skills"] == 2
```

- [ ] **Step 8: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.utils.config'`

- [ ] **Step 9: Implementar `src/utils/config.py`**

```python
"""Única puerta a la configuración del proyecto."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def load_cities() -> dict[str, Any]:
    return _load_yaml("cities.yml")


@cache
def load_skills() -> dict[str, Any]:
    return _load_yaml("skills.yml")


@cache
def load_taxonomy() -> dict[str, Any]:
    return _load_yaml("taxonomy.yml")


def adzuna_credentials() -> tuple[str, str]:
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError(
            "Faltan ADZUNA_APP_ID / ADZUNA_APP_KEY. "
            "Sácalas gratis en developer.adzuna.com y ponlas en .env"
        )
    return app_id, app_key
```

Crear también `src/__init__.py`, `src/utils/__init__.py` y `tests/__init__.py` vacíos.

- [ ] **Step 10: Correr el test y verificar que pasa**

Run: `make setup && uv run pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml Makefile .env.example config/ src/ tests/ data/ reports/
git commit -m "feat: andamio del proyecto, configuración y léxicos"
```

---

### Task 2: Compuerta de viabilidad (día 0)

Esta tarea existe para responder **una sola pregunta antes de invertir la semana**: ¿cuántas vacantes mexicanas de IA con salario publicado y no predicho entrega Adzuna? El resto del plan depende de la respuesta.

**Files:**
- Create: `src/data/__init__.py`, `src/data/probe.py`
- Test: manual, es un script exploratorio

- [ ] **Step 1: Implementar `src/data/probe.py`**

```python
"""Compuerta día 0: mide la cobertura salarial real de Adzuna antes de construir el pipeline.

No usa el esquema canónico a propósito — su trabajo es decirnos si vale la pena escribirlo.
"""

from __future__ import annotations

import time
from collections import Counter

import requests

from src.utils.config import adzuna_credentials

BASE = "https://api.adzuna.com/v1/api/jobs"
CONSULTAS = ["AI engineer", "machine learning engineer", "data scientist", "inteligencia artificial"]
ZONAS = {
    "mx": ["Ciudad de Mexico", "Queretaro", "Monterrey", "Guadalajara"],
    "us": ["San Francisco", "New York", "Austin"],
}


def sondear(pais: str, que: str, donde: str, app_id: str, app_key: str) -> list[dict]:
    resp = requests.get(
        f"{BASE}/{pais}/search/1",
        params={
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": 50,
            "what": que,
            "where": donde,
            "max_days_old": 60,
            "content-type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def main() -> None:
    app_id, app_key = adzuna_credentials()
    resumen: Counter[str] = Counter()
    periodos: Counter[str] = Counter()

    for pais, zonas in ZONAS.items():
        for donde in zonas:
            for que in CONSULTAS:
                filas = sondear(pais, que, donde, app_id, app_key)
                for f in filas:
                    tiene = f.get("salary_min") is not None
                    # OJO: Adzuna manda el flag como string "0"/"1", no como bool
                    predicho = str(f.get("salary_is_predicted", "0")) == "1"
                    resumen[f"{pais}|{donde}|total"] += 1
                    if tiene and not predicho:
                        resumen[f"{pais}|{donde}|observado"] += 1
                    if tiene and predicho:
                        resumen[f"{pais}|{donde}|predicho"] += 1
                    if tiene:
                        # para verificar si Adzuna anualiza: un sueldo MX de 6 cifras es anual
                        periodos[f"{pais}|{'6+cifras' if f['salary_min'] >= 100000 else '<6cifras'}"] += 1
                time.sleep(1.5)  # cortesía con el rate limit del tier gratuito

    print("\n=== COBERTURA POR ZONA ===")
    zonas_vistas = sorted({k.rsplit("|", 1)[0] for k in resumen})
    for z in zonas_vistas:
        total = resumen[f"{z}|total"]
        obs = resumen[f"{z}|observado"]
        pred = resumen[f"{z}|predicho"]
        pct = f"{obs / total:.0%}" if total else "—"
        print(f"  {z:<28} total={total:>4}  observado={obs:>4} ({pct})  predicho={pred:>4}")

    print("\n=== MAGNITUD (¿Adzuna anualiza?) ===")
    for k, v in sorted(periodos.items()):
        print(f"  {k:<20} {v}")

    mx_obs = sum(v for k, v in resumen.items() if k.startswith("mx|") and k.endswith("|observado"))
    print(f"\n>>> Salarios MX observados en una sola corrida: {mx_obs}")
    print(">>> COMPUERTA: si es 0, hay que reforzar con OCC/Computrabajo (§4 del spec).")
    print(">>> Si es >= 10, acumulando semanalmente se llega a n>=30 por ciudad en ~6 semanas.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Correr la sonda**

Run: `make probe`
Expected: una tabla de cobertura por zona y el conteo de salarios MX observados.

- [ ] **Step 3: Registrar el veredicto**

Crear `docs/superpowers/specs/2026-08-XX-viabilidad-dia0.md` con la salida pegada y la decisión tomada:
- `mx_obs >= 10` → seguir con el plan tal cual.
- `0 < mx_obs < 10` → seguir, pero ampliar `CONSULTAS` y ciudades en `collect.py`, y asumir que el modelo de imputación carga más peso.
- `mx_obs == 0` → **parar y replanificar**: revisar términos de OCC/Computrabajo y añadir una fuente antes de la Fase 1.

Anotar también si los salarios mexicanos vienen anualizados (mayoría de 6 cifras) o mensuales — eso define el default de `salary_period` en la Task 5.

- [ ] **Step 4: Commit**

```bash
git add src/data/ docs/superpowers/specs/
git commit -m "feat: sonda de viabilidad de cobertura salarial (compuerta día 0)"
```

---

## Fase 1 — Ingesta

### Task 3: Esquema canónico

**Files:**
- Create: `src/data/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_schema.py`:

```python
import pytest

from src.data.schema import CANONICAL_COLUMNS, SchemaError, empty_frame, validate_frame


def test_empty_frame_tiene_todas_las_columnas():
    df = empty_frame()
    assert list(df.columns) == CANONICAL_COLUMNS


def test_validate_frame_acepta_un_frame_valido():
    df = empty_frame()
    validate_frame(df)  # no debe lanzar


def test_validate_frame_rechaza_columnas_faltantes():
    df = empty_frame().drop(columns=["salary_is_predicted"])
    with pytest.raises(SchemaError, match="salary_is_predicted"):
        validate_frame(df)


def test_candado_de_integridad_predicho_no_puede_ser_observado():
    """El error que hundiría el análisis entero: un salario modelado por Adzuna
    colándose al conjunto 'observado'. Ver §4.1 del spec."""
    df = empty_frame()
    df.loc[0, CANONICAL_COLUMNS] = None
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "salary_is_predicted"] = True
    df.loc[0, "salary_observed"] = True
    with pytest.raises(SchemaError, match="predicho"):
        validate_frame(df)


def test_frame_con_predicho_y_no_observado_es_valido():
    df = empty_frame()
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "salary_is_predicted"] = True
    df.loc[0, "salary_observed"] = False
    validate_frame(df)


def test_validate_frame_rechaza_tier_invalido():
    df = empty_frame()
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "tier"] = "inventado"
    with pytest.raises(SchemaError, match="tier"):
        validate_frame(df)
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.schema'`

- [ ] **Step 3: Implementar `src/data/schema.py`**

```python
"""Esquema canónico. Única fuente de verdad de las columnas del dataset."""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "posting_id",
    "snapshot_date",
    "source",
    "title_raw",
    "title_norm",
    "company",
    "company_posting_count",
    "company_is_multinational",
    "category",
    "country",
    "state",
    "city",
    "metro",
    "is_remote",
    "remote_scope",
    "posted_date",
    "salary_min_raw",
    "salary_max_raw",
    "salary_currency",
    "salary_period",
    "salary_is_predicted",
    "salary_observed",
    "salary_annual_local",
    "salary_annual_usd",
    "salary_annual_usd_ppp",
    "seniority",
    "tier",
    "skills",
    "description_text",
    "description_lang",
    "url",
]

SOURCES = {"adzuna_mx", "adzuna_us", "kaggle_mannacharya", "aijobs_net", "indeed"}
COUNTRIES = {"MX", "US"}
CURRENCIES = {"MXN", "USD"}
PERIODS = {"hora", "mes", "año"}
SENIORITIES = {"intern", "jr", "mid", "sr", "lead", "staff", "principal", "unknown"}
TIERS = {"nucleo", "anillo", "fuera"}
REMOTE_SCOPES = {"local", "nacional", "us_desde_mx", "global"}

_ENUMS = {
    "source": SOURCES,
    "country": COUNTRIES,
    "salary_currency": CURRENCIES,
    "salary_period": PERIODS,
    "seniority": SENIORITIES,
    "tier": TIERS,
    "remote_scope": REMOTE_SCOPES,
}


class SchemaError(ValueError):
    """El frame no cumple el esquema canónico."""


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def validate_frame(df: pd.DataFrame) -> None:
    faltantes = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if faltantes:
        raise SchemaError(f"Faltan columnas: {', '.join(faltantes)}")

    for col, permitidos in _ENUMS.items():
        valores = set(df[col].dropna().unique()) - permitidos
        if valores:
            raise SchemaError(f"Valores inválidos en {col}: {sorted(valores)}")

    # Candado de integridad (§4.1 del spec): un salario predicho por Adzuna
    # jamás puede contar como observado.
    contaminadas = df[
        df["salary_is_predicted"].fillna(False).astype(bool)
        & df["salary_observed"].fillna(False).astype(bool)
    ]
    if len(contaminadas):
        raise SchemaError(
            f"{len(contaminadas)} filas con salario predicho marcadas como observadas. "
            "Ver §4.1 del spec: Adzuna modela sueldos y no pueden entrar al conjunto observado."
        )
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/schema.py tests/test_schema.py
git commit -m "feat: esquema canónico con candado contra salarios predichos"
```

---

### Task 4: Cliente de Adzuna

**Files:**
- Create: `src/data/adzuna.py`
- Test: `tests/test_adzuna.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_adzuna.py`:

```python
from unittest.mock import Mock, patch

import pytest

from src.data.adzuna import AdzunaClient, AdzunaError


def _resp(json_data, status=200):
    m = Mock()
    m.status_code = status
    m.json.return_value = json_data
    m.raise_for_status = Mock()
    return m


@patch("src.data.adzuna.requests.get")
def test_search_devuelve_los_resultados(mock_get):
    mock_get.return_value = _resp({"count": 2, "results": [{"id": "1"}, {"id": "2"}]})
    client = AdzunaClient("id", "key")
    out = client.search("mx", what="AI engineer", where="Ciudad de Mexico", page=1)
    assert [r["id"] for r in out] == ["1", "2"]


@patch("src.data.adzuna.requests.get")
def test_search_manda_las_credenciales_y_el_pais_en_la_url(mock_get):
    mock_get.return_value = _resp({"results": []})
    AdzunaClient("mi_id", "mi_key").search("mx", what="x", where="y", page=3)
    url = mock_get.call_args[0][0]
    params = mock_get.call_args[1]["params"]
    assert url.endswith("/jobs/mx/search/3")
    assert params["app_id"] == "mi_id"
    assert params["app_key"] == "mi_key"


@patch("src.data.adzuna.time.sleep")
@patch("src.data.adzuna.requests.get")
def test_reintenta_con_backoff_ante_429(mock_get, mock_sleep):
    mock_get.side_effect = [
        _resp({}, status=429),
        _resp({}, status=429),
        _resp({"results": [{"id": "ok"}]}),
    ]
    out = AdzunaClient("id", "key").search("mx", what="x", where="y", page=1)
    assert out == [{"id": "ok"}]
    assert mock_get.call_count == 3
    assert [c[0][0] for c in mock_sleep.call_args_list] == [2, 4]


@patch("src.data.adzuna.time.sleep")
@patch("src.data.adzuna.requests.get")
def test_se_rinde_tras_agotar_reintentos(mock_get, mock_sleep):
    mock_get.return_value = _resp({}, status=429)
    with pytest.raises(AdzunaError, match="429"):
        AdzunaClient("id", "key").search("mx", what="x", where="y", page=1)


@patch("src.data.adzuna.time.sleep")
@patch("src.data.adzuna.requests.get")
def test_search_all_pagina_hasta_que_se_vacia(mock_get, mock_sleep):
    mock_get.side_effect = [
        _resp({"results": [{"id": "a"}]}),
        _resp({"results": [{"id": "b"}]}),
        _resp({"results": []}),
    ]
    out = AdzunaClient("id", "key").search_all("mx", what="x", where="y", max_pages=5)
    assert [r["id"] for r in out] == ["a", "b"]


@patch("src.data.adzuna.time.sleep")
@patch("src.data.adzuna.requests.get")
def test_search_all_respeta_max_pages(mock_get, mock_sleep):
    mock_get.return_value = _resp({"results": [{"id": "a"}]})
    out = AdzunaClient("id", "key").search_all("mx", what="x", where="y", max_pages=3)
    assert len(out) == 3
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_adzuna.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.adzuna'`

- [ ] **Step 3: Implementar `src/data/adzuna.py`**

```python
"""Cliente HTTP de la API de Adzuna. No conoce el esquema canónico — eso vive en mapping.py."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

BASE = "https://api.adzuna.com/v1/api/jobs"
MAX_REINTENTOS = 3
PAUSA_ENTRE_PAGINAS = 1.5

log = logging.getLogger(__name__)


class AdzunaError(RuntimeError):
    """Adzuna no respondió algo usable."""


class AdzunaClient:
    def __init__(self, app_id: str, app_key: str, results_per_page: int = 50) -> None:
        self.app_id = app_id
        self.app_key = app_key
        self.results_per_page = results_per_page

    def search(
        self, country: str, *, what: str, where: str, page: int, max_days_old: int = 60
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": self.results_per_page,
            "what": what,
            "where": where,
            "max_days_old": max_days_old,
            "content-type": "application/json",
        }
        espera = 2
        for intento in range(MAX_REINTENTOS):
            resp = requests.get(f"{BASE}/{country}/search/{page}", params=params, timeout=30)
            if resp.status_code == 429:
                if intento == MAX_REINTENTOS - 1:
                    break
                log.warning("Adzuna 429, esperando %ss (intento %s)", espera, intento + 1)
                time.sleep(espera)
                espera *= 2
                continue
            resp.raise_for_status()
            return resp.json().get("results", [])
        raise AdzunaError(f"Adzuna devolvió 429 tras {MAX_REINTENTOS} intentos ({country}/{where})")

    def search_all(
        self, country: str, *, what: str, where: str, max_pages: int = 5
    ) -> list[dict[str, Any]]:
        acumulado: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            filas = self.search(country, what=what, where=where, page=page)
            if not filas:
                break
            acumulado.extend(filas)
            time.sleep(PAUSA_ENTRE_PAGINAS)
        return acumulado
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_adzuna.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/adzuna.py tests/test_adzuna.py
git commit -m "feat: cliente de Adzuna con backoff y paginación"
```

---

### Task 5: Mapeo al esquema canónico

**Files:**
- Create: `src/data/mapping.py`
- Test: `tests/test_mapping.py`, `tests/fixtures/adzuna_mx.json`

- [ ] **Step 1: Crear el fixture `tests/fixtures/adzuna_mx.json`**

```json
{
  "id": "4839201",
  "title": "AI Engineer (LLM / RAG)",
  "company": { "display_name": "Nubank México" },
  "location": {
    "display_name": "Cuauhtémoc, Ciudad de México",
    "area": ["Mexico", "Ciudad de México", "Cuauhtémoc"]
  },
  "category": { "label": "IT Jobs", "tag": "it-jobs" },
  "salary_min": 780000.0,
  "salary_max": 960000.0,
  "salary_is_predicted": "0",
  "created": "2026-07-28T09:14:22Z",
  "description": "Buscamos AI Engineer con experiencia en LLM, RAG y PyTorch para construir agentes.",
  "redirect_url": "https://www.adzuna.com.mx/details/4839201",
  "contract_time": "full_time"
}
```

- [ ] **Step 2: Escribir el test que falla**

`tests/test_mapping.py`:

```python
import json
from datetime import date
from pathlib import Path

from src.data.mapping import map_adzuna_row
from src.data.schema import CANONICAL_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "adzuna_mx.json"


def _fila():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_mapea_todas_las_columnas_canonicas():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert set(out) == set(CANONICAL_COLUMNS)


def test_posting_id_es_estable_y_depende_de_la_fuente():
    a = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    b = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 9, 1))
    assert a["posting_id"] == b["posting_id"]  # no depende de la fecha del snapshot
    assert len(a["posting_id"]) == 40  # sha1 hex


def test_salario_publicado_y_no_predicho_cuenta_como_observado():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_is_predicted"] is False
    assert out["salary_observed"] is True
    assert out["salary_min_raw"] == 780000.0
    assert out["salary_currency"] == "MXN"


def test_flag_predicho_llega_como_string_uno():
    """Adzuna manda "1"/"0" como string, no como booleano. Si esto se rompe,
    los salarios modelados se cuelan al análisis (§4.1 del spec)."""
    fila = _fila()
    fila["salary_is_predicted"] = "1"
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_is_predicted"] is True
    assert out["salary_observed"] is False


def test_sin_salario_no_es_observado():
    fila = _fila()
    del fila["salary_min"]
    del fila["salary_max"]
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["salary_observed"] is False
    assert out["salary_min_raw"] is None


def test_ciudad_se_canoniza_desde_la_alcaldia():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["city"] == "Ciudad de México"
    assert out["metro"] == "Valle de México"


def test_source_refleja_el_pais():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["source"] == "adzuna_mx"


def test_detecta_remoto_en_el_titulo():
    fila = _fila()
    fila["title"] = "Remote AI Engineer"
    out = map_adzuna_row(fila, country="MX", snapshot_date=date(2026, 8, 7))
    assert out["is_remote"] is True


def test_detecta_idioma_de_la_descripcion():
    out = map_adzuna_row(_fila(), country="MX", snapshot_date=date(2026, 8, 7))
    assert out["description_lang"] == "es"
```

- [ ] **Step 3: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_mapping.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.mapping'`

- [ ] **Step 4: Implementar `src/data/mapping.py`**

```python
"""Traduce respuestas crudas de cada fuente al esquema canónico.

Aísla el formato de la fuente: cuando Adzuna cambie un campo, sólo se toca este archivo.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any

from src.data.schema import CANONICAL_COLUMNS
from src.utils.config import load_cities

MONEDA_POR_PAIS = {"MX": "MXN", "US": "USD"}
# Confirmado en la sonda del día 0: Adzuna anualiza los importes en ambos países.
PERIODO_ADZUNA = "año"

_PALABRAS_ES = {" de ", " para ", " con ", " en ", " y ", "experiencia", "conocimientos"}
_REMOTO = re.compile(r"\b(remote|remoto|home office|teletrabajo|trabajo a distancia)\b", re.I)


def _canonizar_ciudad(country: str, area: list[str], display: str) -> tuple[str | None, str | None]:
    """Devuelve (ciudad_canónica, metro). None si la ubicación no es una de las zonas objetivo."""
    candidatos = [*area, display]
    texto = " | ".join(str(c) for c in candidatos)
    for entrada in load_cities().get(country, []):
        nombres = [entrada["canonical"], *entrada["aliases"]]
        for nombre in nombres:
            if re.search(rf"\b{re.escape(nombre)}\b", texto, re.I):
                return entrada["canonical"], entrada["metro"]
    return None, None


def _detectar_idioma(texto: str) -> str:
    bajo = f" {texto.lower()} "
    return "es" if sum(p in bajo for p in _PALABRAS_ES) >= 2 else "en"


def _a_fecha(valor: str | None) -> date | None:
    if not valor:
        return None
    try:
        return datetime.fromisoformat(valor.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def map_adzuna_row(raw: dict[str, Any], *, country: str, snapshot_date: date) -> dict[str, Any]:
    source = f"adzuna_{country.lower()}"
    posting_id = hashlib.sha1(f"{source}:{raw.get('id')}".encode()).hexdigest()

    area = raw.get("location", {}).get("area", []) or []
    display = raw.get("location", {}).get("display_name", "") or ""
    ciudad, metro = _canonizar_ciudad(country, area, display)

    titulo = raw.get("title", "") or ""
    descripcion = raw.get("description", "") or ""
    texto = f"{titulo} {display} {descripcion}"

    salary_min = raw.get("salary_min")
    salary_max = raw.get("salary_max")
    # Adzuna manda el flag como string "0"/"1". Tratarlo como bool de Python
    # convertiría "0" en True y contaminaría todo el análisis.
    predicho = str(raw.get("salary_is_predicted", "0")) == "1"
    tiene_salario = salary_min is not None

    fila: dict[str, Any] = dict.fromkeys(CANONICAL_COLUMNS)
    fila.update(
        posting_id=posting_id,
        snapshot_date=snapshot_date,
        source=source,
        title_raw=titulo,
        title_norm=titulo.strip().lower(),
        company=(raw.get("company") or {}).get("display_name"),
        category=(raw.get("category") or {}).get("label"),
        country=country,
        state=area[1] if len(area) > 1 else None,
        city=ciudad,
        metro=metro,
        is_remote=bool(_REMOTO.search(texto)),
        posted_date=_a_fecha(raw.get("created")),
        salary_min_raw=salary_min,
        salary_max_raw=salary_max,
        salary_currency=MONEDA_POR_PAIS[country],
        salary_period=PERIODO_ADZUNA if tiene_salario else None,
        salary_is_predicted=predicho,
        salary_observed=bool(tiene_salario and not predicho),
        description_text=descripcion,
        description_lang=_detectar_idioma(descripcion),
        url=raw.get("redirect_url"),
    )
    return fila
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_mapping.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/data/mapping.py tests/test_mapping.py tests/fixtures/
git commit -m "feat: mapeo de Adzuna al esquema canónico"
```

---

### Task 6: CLI de recolección con manifest

**Files:**
- Create: `src/data/collect.py`
- Test: `tests/test_collect.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_collect.py`:

```python
import json
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data.collect import ejecutar_corrida


@patch("src.data.collect.AdzunaClient")
def test_escribe_snapshot_y_manifest(MockClient, tmp_path):
    MockClient.return_value.search_all.return_value = [
        {
            "id": "1",
            "title": "AI Engineer",
            "company": {"display_name": "Acme"},
            "location": {"display_name": "Querétaro", "area": ["Mexico", "Querétaro"]},
            "category": {"label": "IT Jobs"},
            "salary_min": 600000.0,
            "salary_max": 800000.0,
            "salary_is_predicted": "0",
            "created": "2026-08-01T00:00:00Z",
            "description": "LLM y RAG",
            "redirect_url": "http://x",
        }
    ]
    manifest = ejecutar_corrida(
        destino=tmp_path, snapshot_date=date(2026, 8, 7), app_id="i", app_key="k"
    )

    parquet = tmp_path / "2026-08-07" / "adzuna.parquet"
    assert parquet.exists()
    df = pd.read_parquet(parquet)
    assert len(df) > 0
    assert df["salary_observed"].any()

    escrito = json.loads((tmp_path / "2026-08-07" / "manifest.json").read_text())
    assert escrito["snapshot_date"] == "2026-08-07"
    assert escrito["filas"] == len(df)
    assert escrito["consultas_fallidas"] == []
    assert manifest["filas"] == len(df)


@patch("src.data.collect.AdzunaClient")
def test_una_consulta_fallida_no_tumba_la_corrida(MockClient, tmp_path):
    from src.data.adzuna import AdzunaError

    MockClient.return_value.search_all.side_effect = AdzunaError("429 eterno")
    manifest = ejecutar_corrida(
        destino=tmp_path, snapshot_date=date(2026, 8, 7), app_id="i", app_key="k"
    )
    assert manifest["filas"] == 0
    assert len(manifest["consultas_fallidas"]) > 0
    assert (tmp_path / "2026-08-07" / "manifest.json").exists()


@patch("src.data.collect.AdzunaClient")
def test_no_sobrescribe_un_snapshot_existente(MockClient, tmp_path):
    MockClient.return_value.search_all.return_value = []
    destino = tmp_path / "2026-08-07"
    destino.mkdir(parents=True)
    (destino / "adzuna.parquet").write_text("intocable")

    ejecutar_corrida(destino=tmp_path, snapshot_date=date(2026, 8, 7), app_id="i", app_key="k")
    assert (destino / "adzuna.parquet").read_text() == "intocable"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_collect.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.collect'`

- [ ] **Step 3: Implementar `src/data/collect.py`**

```python
"""CLI de una corrida de recolección. Escribe un snapshot inmutable + su manifest."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adzuna import AdzunaClient
from src.data.mapping import map_adzuna_row
from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.utils.config import DATA_DIR, adzuna_credentials, load_cities
from src.utils.logging import setup_logging

CONSULTAS = [
    "AI engineer",
    "machine learning engineer",
    "LLM engineer",
    "MLOps engineer",
    "data scientist",
    "NLP engineer",
    "computer vision engineer",
    "inteligencia artificial",
    "aprendizaje automático",
]

log = logging.getLogger(__name__)


def _zonas() -> list[tuple[str, str]]:
    """(country, where) para cada ciudad objetivo, más el remoto nacional."""
    pares: list[tuple[str, str]] = []
    for country, entradas in load_cities().items():
        for e in entradas:
            pares.append((country, e["canonical"]))
    pares += [("MX", "remote"), ("US", "remote")]
    return pares


def ejecutar_corrida(
    *, destino: Path, snapshot_date: date, app_id: str, app_key: str
) -> dict[str, Any]:
    carpeta = destino / snapshot_date.isoformat()
    carpeta.mkdir(parents=True, exist_ok=True)
    parquet = carpeta / "adzuna.parquet"

    client = AdzunaClient(app_id, app_key)
    filas: list[dict[str, Any]] = []
    fallidas: list[dict[str, str]] = []

    for country, donde in _zonas():
        for que in CONSULTAS:
            try:
                crudas = client.search_all(country.lower(), what=que, where=donde, max_pages=5)
            except Exception as exc:  # noqa: BLE001 - un hueco no debe tumbar la corrida
                log.warning("Consulta fallida %s/%s/%s: %s", country, donde, que, exc)
                fallidas.append({"country": country, "where": donde, "what": que, "error": str(exc)})
                continue
            for cruda in crudas:
                filas.append(map_adzuna_row(cruda, country=country, snapshot_date=snapshot_date))

    df = pd.DataFrame(filas, columns=CANONICAL_COLUMNS)
    if len(df):
        validate_frame(df)

    total_consultas = len(_zonas()) * len(CONSULTAS)
    manifest = {
        "snapshot_date": snapshot_date.isoformat(),
        "filas": int(len(df)),
        "filas_con_salario_observado": int(df["salary_observed"].fillna(False).sum())
        if len(df)
        else 0,
        "consultas_totales": total_consultas,
        "consultas_fallidas": fallidas,
        "tasa_exito": round(1 - len(fallidas) / total_consultas, 4) if total_consultas else 0.0,
    }

    if parquet.exists():
        log.warning("El snapshot %s ya existe, no se sobrescribe.", parquet)
    else:
        df.to_parquet(parquet, index=False)

    (carpeta / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("Snapshot %s: %s filas, %s observadas", snapshot_date, manifest["filas"],
             manifest["filas_con_salario_observado"])
    return manifest


def main() -> None:
    setup_logging()
    app_id, app_key = adzuna_credentials()
    ejecutar_corrida(
        destino=DATA_DIR / "raw", snapshot_date=date.today(), app_id=app_id, app_key=app_key
    )


if __name__ == "__main__":
    main()
```

Y `src/utils/logging.py`:

```python
import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_collect.py -v`
Expected: 3 passed

- [ ] **Step 5: Primera corrida real**

Run: `make collect`
Expected: `data/raw/<hoy>/adzuna.parquet` y `manifest.json`. Revisar `filas_con_salario_observado` — es el número que la compuerta del día 0 predijo.

- [ ] **Step 6: Commit**

```bash
git add src/data/collect.py src/utils/logging.py tests/test_collect.py
git commit -m "feat: CLI de recolección con snapshots inmutables y manifest"
```

---

### Task 7: Fuentes de Kaggle

**Files:**
- Create: `src/data/kaggle_sources.py`
- Test: `tests/test_kaggle_sources.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_kaggle_sources.py`:

```python
from datetime import date

import pandas as pd

from src.data.kaggle_sources import map_mannacharya_frame, verificar_no_sintetico
from src.data.schema import CANONICAL_COLUMNS


def test_detecta_dataset_sintetico_por_uniformidad():
    """El chequeo que descartó m0sm71: 12 países con ~4,300 filas cada uno
    y 100% de salarios presentes. Ver §1.2 del spec."""
    sintetico = pd.DataFrame(
        {"country": ["A"] * 4300 + ["B"] * 4290 + ["C"] * 4310, "salary": range(12900)}
    )
    veredicto = verificar_no_sintetico(sintetico, col_pais="country", col_salario="salary")
    assert veredicto["sospechoso"] is True
    assert "uniforme" in veredicto["razon"]


def test_acepta_un_dataset_real():
    real = pd.DataFrame(
        {
            "country": ["US"] * 2000 + ["UK"] * 94 + ["MX"] * 12,
            "salary": [1.0] * 400 + [None] * 1706,
        }
    )
    veredicto = verificar_no_sintetico(real, col_pais="country", col_salario="salary")
    assert veredicto["sospechoso"] is False


def test_mapea_mannacharya_al_esquema_canonico():
    crudo = pd.DataFrame(
        [
            {
                "id": "x1",
                "title": "Machine Learning Engineer",
                "company": "OpenAI",
                "location": "San Francisco, California, United States",
                "city": "San Francisco",
                "country": "United States",
                "salary_min": 200000.0,
                "salary_max": 300000.0,
                "salary_currency": "USD",
                "salary_period": "year",
                "tags": "python, pytorch",
                "description_text": "We need PyTorch and LLM experience",
                "posted_date": "2026-05-10",
            }
        ]
    )
    out = map_mannacharya_frame(crudo, snapshot_date=date(2026, 8, 7))
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.loc[0, "source"] == "kaggle_mannacharya"
    assert out.loc[0, "country"] == "US"
    assert out.loc[0, "salary_observed"] is True
    # Este dataset no trae salarios modelados, así que nunca es predicho.
    assert out.loc[0, "salary_is_predicted"] is False
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_kaggle_sources.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/data/kaggle_sources.py`**

```python
"""Descarga y validación de los dos datasets reales de Kaggle.

Incluye el detector de datasets sintéticos que descartó m0sm71 (§1.2 del spec).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
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


def verificar_no_sintetico(
    df: pd.DataFrame, *, col_pais: str, col_salario: str
) -> dict[str, Any]:
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
            ["curl", "-sL", "-H", f"Authorization: Basic {auth}", "-o", str(zip_path),
             f"https://www.kaggle.com/api/v1/datasets/download/{ref}"],
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
    out["salary_is_predicted"] = False  # este dataset no modela sueldos
    out["salary_observed"] = crudo["salary_min"].notna()
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
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_kaggle_sources.py -v`
Expected: 3 passed

- [ ] **Step 5: Descargar los datasets reales y correr el detector**

Run:
```bash
uv run python -m src.data.kaggle_sources
uv run python -c "
import pandas as pd
from src.data.kaggle_sources import verificar_no_sintetico
df = pd.read_csv('data/raw/kaggle/kaggle_mannacharya/aijobs_dataset.csv', low_memory=False)
print(verificar_no_sintetico(df, col_pais='country', col_salario='salary_min'))
"
```
Expected: `{'sospechoso': False, ...}` — guardar la salida, va al notebook 01.

- [ ] **Step 6: Commit**

```bash
git add src/data/kaggle_sources.py tests/test_kaggle_sources.py
git commit -m "feat: ingesta de Kaggle con detector de datasets sintéticos"
```

---

## Fase 2 — Features

### Task 8: Normalización de salarios

**Files:**
- Create: `src/features/__init__.py`, `src/features/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_normalize.py`:

```python
import pytest

from src.features.normalize import a_anual, a_ppp, a_usd


def test_mensual_a_anual():
    assert a_anual(45000, "mes") == 540000


def test_anual_se_queda_igual():
    assert a_anual(540000, "año") == 540000


def test_por_hora_a_anual_con_2080_horas():
    assert a_anual(50, "hora") == 104000


def test_periodo_desconocido_revienta():
    with pytest.raises(ValueError, match="periodo"):
        a_anual(100, "quincena")


def test_mxn_a_usd():
    assert a_usd(540000, "MXN", fx_usd_mxn=18.0) == pytest.approx(30000.0)


def test_usd_se_queda_igual():
    assert a_usd(30000, "USD", fx_usd_mxn=18.0) == 30000


def test_ppp_divide_el_monto_local_por_el_factor_del_pais():
    """PA.NUS.PPP del Banco Mundial es 'moneda local por dólar internacional'.
    El ajuste se hace sobre el monto LOCAL, no sobre el ya convertido a USD."""
    factores = {"MX": 10.0, "US": 1.0}
    assert a_ppp(540000, "MX", factores) == pytest.approx(54000.0)
    assert a_ppp(150000, "US", factores) == pytest.approx(150000.0)


def test_ppp_hace_visible_que_el_tipo_de_cambio_subestima_mexico():
    """A 18 MXN/USD un sueldo de 540k MXN son 30k USD nominales,
    pero 54k dólares internacionales en poder adquisitivo."""
    factores = {"MX": 10.0, "US": 1.0}
    assert a_usd(540000, "MXN", fx_usd_mxn=18.0) < a_ppp(540000, "MX", factores)


def test_none_se_propaga_sin_reventar():
    assert a_anual(None, "mes") is None
    assert a_usd(None, "MXN", fx_usd_mxn=18.0) is None
    assert a_ppp(None, "MX", {"MX": 10.0}) is None
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/features/normalize.py`**

```python
"""Normalización salarial: periodicidad → moneda → poder adquisitivo. Funciones puras."""

from __future__ import annotations

import json
from typing import Any

import requests

HORAS_LABORALES_ANUALES = 2080
MULTIPLICADOR = {"año": 1, "mes": 12, "hora": HORAS_LABORALES_ANUALES}


def a_anual(monto: float | None, periodo: str) -> float | None:
    if monto is None:
        return None
    if periodo not in MULTIPLICADOR:
        raise ValueError(f"periodo desconocido: {periodo!r}. Válidos: {sorted(MULTIPLICADOR)}")
    return monto * MULTIPLICADOR[periodo]


def a_usd(monto: float | None, moneda: str, *, fx_usd_mxn: float) -> float | None:
    if monto is None:
        return None
    if moneda == "USD":
        return monto
    if moneda == "MXN":
        return monto / fx_usd_mxn
    raise ValueError(f"moneda desconocida: {moneda!r}")


def a_ppp(monto_local: float | None, pais: str, factores: dict[str, float]) -> float | None:
    """Convierte un monto en moneda local a dólares internacionales.

    PA.NUS.PPP del Banco Mundial se expresa como 'unidades de moneda local por
    dólar internacional', así que la conversión es una división del monto local.
    """
    if monto_local is None:
        return None
    if pais not in factores:
        raise ValueError(f"sin factor PPP para {pais!r}")
    return monto_local / factores[pais]


def obtener_fx_usd_mxn() -> float:
    """Tipo de cambio del día vía Frankfurter (BCE, sin llave)."""
    r = requests.get(
        "https://api.frankfurter.app/latest", params={"from": "USD", "to": "MXN"}, timeout=20
    )
    r.raise_for_status()
    return float(r.json()["rates"]["MXN"])


def obtener_factores_ppp() -> dict[str, float]:
    """PA.NUS.PPP del Banco Mundial, año más reciente disponible, para MEX y USA."""
    params: dict[str, Any] = {"format": "json", "date": "2022:2026", "per_page": 100}
    r = requests.get(
        "https://api.worldbank.org/v2/country/MEX;USA/indicator/PA.NUS.PPP",
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    registros: list[dict[str, Any]] = json.loads(r.text)[1]
    factores: dict[str, float] = {}
    for reg in sorted(registros, key=lambda x: x["date"], reverse=True):
        pais = {"MX": "MX", "US": "US"}.get(reg["country"]["id"])
        if pais and reg["value"] is not None and pais not in factores:
            factores[pais] = float(reg["value"])
    factores.setdefault("US", 1.0)
    return factores
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/features/ tests/test_normalize.py
git commit -m "feat: normalización de periodicidad, moneda y PPP"
```

---

### Task 9: Taxonomía núcleo / anillo

**Files:**
- Create: `src/features/taxonomy.py`
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_taxonomy.py`:

```python
from src.features.taxonomy import clasificar


def test_titulo_explicito_es_nucleo():
    assert clasificar("AI Engineer", skills=[]) == "nucleo"
    assert clasificar("Machine Learning Engineer", skills=[]) == "nucleo"
    assert clasificar("Ingeniero de Inteligencia Artificial", skills=[]) == "nucleo"
    assert clasificar("LLM Engineer", skills=[]) == "nucleo"


def test_titulo_generico_con_dos_skills_de_ia_es_anillo():
    """El hallazgo §1.4 del spec: en México el trabajo de IA viene disfrazado."""
    assert clasificar("Senior Software Engineer", skills=["rag", "pytorch"]) == "anillo"


def test_titulo_generico_con_una_sola_skill_queda_fuera():
    """Umbral de 2 para no cazar vacantes que mencionan 'AI' de relleno."""
    assert clasificar("Senior Software Engineer", skills=["pytorch"]) == "fuera"


def test_skills_de_soporte_no_disparan_el_anillo():
    assert clasificar("Backend Developer", skills=["python", "sql", "docker"]) == "fuera"


def test_nucleo_gana_sobre_anillo():
    assert clasificar("AI Engineer", skills=["rag", "llm"]) == "nucleo"


def test_termino_de_ia_sin_termino_de_rol_no_es_nucleo():
    """'AI Product Manager' no es un rol de ingeniería de IA."""
    assert clasificar("AI Product Manager", skills=[]) == "fuera"


def test_no_confunde_palabras_que_contienen_ai():
    """'Maintenance' contiene 'ai' — el \\b del regex debe evitar el falso positivo."""
    assert clasificar("Maintenance Engineer", skills=[]) == "fuera"
    assert clasificar("Retail Engineer", skills=[]) == "fuera"


def test_gen_ai_no_caza_nitrogen_ni_hydrogen():
    """Regresión: 'gen ?ai' sin anclas casaba dentro de 'nitroGEN AIr'. Querétaro y
    Monterrey están llenos de vacantes industriales con esas palabras."""
    assert clasificar("Nitrogen Air Systems Engineer", skills=[]) == "fuera"
    assert clasificar("Hydrogen Airflow Engineer", skills=[]) == "fuera"
    assert clasificar("GenAI Engineer", skills=[]) == "nucleo"


def test_reconoce_el_titulo_con_puntos():
    """Regresión: '\\ba\\.i\\.\\b' era un patrón muerto — el \\b tras un punto no casa nunca."""
    assert clasificar("Especialista en A.I.", skills=[]) == "nucleo"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/features/taxonomy.py`**

```python
"""Clasificación núcleo / anillo / fuera. Ver §5.1 del spec."""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils.config import load_skills, load_taxonomy


@lru_cache(maxsize=1)
def _patrones() -> tuple[re.Pattern[str], re.Pattern[str], int, frozenset[str]]:
    tax = load_taxonomy()
    ia = re.compile("|".join(tax["nucleo"]["terminos_ia"]), re.I)
    rol = re.compile("|".join(tax["nucleo"]["terminos_rol"]), re.I)
    return ia, rol, tax["anillo"]["min_skills"], frozenset(load_skills()["ia_aplicada"])


def clasificar(title: str, *, skills: list[str]) -> str:
    ia, rol, min_skills, skills_ia = _patrones()
    titulo = title or ""

    if ia.search(titulo) and rol.search(titulo):
        return "nucleo"

    if len([s for s in skills if s in skills_ia]) >= min_skills:
        return "anillo"

    return "fuera"
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/features/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: taxonomía núcleo/anillo sobre título y skills"
```

---

### Task 10: Extracción de seniority y skills

**Files:**
- Create: `src/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_extract.py`:

```python
from src.features.extract import extraer_seniority, extraer_skills


def test_extrae_skills_por_alias():
    texto = "Buscamos experiencia en RAG, PyTorch y bases de datos vectoriales"
    out = extraer_skills(texto)
    assert "rag" in out
    assert "pytorch" in out
    assert "vector_db" in out


def test_extrae_skills_en_ingles_y_espanol():
    assert "llm" in extraer_skills("experience with large language models")
    assert "llm" in extraer_skills("experiencia con modelos de lenguaje")


def test_no_duplica_skills():
    out = extraer_skills("PyTorch, pytorch, TORCH")
    assert out.count("pytorch") == 1


def test_devuelve_lista_ordenada_para_ser_determinista():
    a = extraer_skills("pytorch rag llm")
    b = extraer_skills("llm rag pytorch")
    assert a == b


def test_seniority_desde_el_titulo():
    assert extraer_seniority("Senior AI Engineer", "") == "sr"
    assert extraer_seniority("Junior ML Engineer", "") == "jr"
    assert extraer_seniority("Staff Machine Learning Engineer", "") == "staff"
    assert extraer_seniority("Lead AI Engineer", "") == "lead"
    assert extraer_seniority("Principal AI Engineer", "") == "principal"


def test_seniority_en_espanol():
    assert extraer_seniority("Ingeniero de IA Senior", "") == "sr"
    assert extraer_seniority("Ingeniero de IA Jr.", "") == "jr"


def test_seniority_cae_a_la_descripcion_si_el_titulo_no_dice():
    assert extraer_seniority("AI Engineer", "Se requieren 8+ años de experiencia") == "sr"
    assert extraer_seniority("AI Engineer", "Se requieren 1-2 años de experiencia") == "jr"
    assert extraer_seniority("AI Engineer", "Se requieren 4 años de experiencia") == "mid"


def test_seniority_desconocido_cuando_no_hay_senal():
    assert extraer_seniority("AI Engineer", "Únete a nuestro equipo") == "unknown"


def test_el_titulo_gana_sobre_la_descripcion():
    assert extraer_seniority("Junior AI Engineer", "10 años de experiencia") == "jr"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/features/extract.py`**

```python
"""Extracción de seniority y skills desde el texto de la vacante.

Se lee la descripción y no sólo el título porque el título miente (§1.4 del spec).
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils.config import load_skills

_SENIORITY_TITULO = [
    ("principal", r"\bprincipal\b"),
    ("staff", r"\bstaff\b"),
    ("lead", r"\b(lead|l[íi]der|head of)\b"),
    ("sr", r"\b(senior|sr\.?|s[ée]nior)\b"),
    ("jr", r"\b(junior|jr\.?|entry[- ]level|trainee|becari[oa])\b"),
    ("intern", r"\b(intern|internship|pasant[íi]a|practicante)\b"),
    ("mid", r"\b(mid[- ]level|semi[- ]?senior|intermedio)\b"),
]
_ANIOS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|años?)", re.I)


@lru_cache(maxsize=1)
def _lexico() -> list[tuple[str, re.Pattern[str]]]:
    lexico = load_skills()
    salida: list[tuple[str, re.Pattern[str]]] = []
    for grupo in ("ia_aplicada", "soporte"):
        for skill, alias in lexico[grupo].items():
            patron = "|".join(rf"\b{re.escape(a)}\b" for a in alias)
            salida.append((skill, re.compile(patron, re.I)))
    return salida


def extraer_skills(texto: str) -> list[str]:
    if not texto:
        return []
    encontradas = {skill for skill, patron in _lexico() if patron.search(texto)}
    return sorted(encontradas)


def extraer_seniority(title: str, description: str) -> str:
    titulo = title or ""
    for nivel, patron in _SENIORITY_TITULO:
        if re.search(patron, titulo, re.I):
            return nivel

    descripcion = description or ""
    for nivel, patron in _SENIORITY_TITULO:
        if re.search(patron, descripcion, re.I):
            return nivel

    match = _ANIOS.search(descripcion)
    if match:
        anios = int(match.group(1))
        if anios <= 2:
            return "jr"
        if anios <= 5:
            return "mid"
        return "sr"

    return "unknown"
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_extract.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/features/extract.py tests/test_extract.py
git commit -m "feat: extracción de seniority y skills desde la descripción"
```

---

### Task 11: Deduplicación

**Files:**
- Create: `src/features/dedupe.py`
- Test: `tests/test_dedupe.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_dedupe.py`:

```python
import pandas as pd

from src.features.dedupe import deduplicar


def _fila(**kw):
    base = {
        "posting_id": "a",
        "title_norm": "ai engineer",
        "company": "Acme",
        "city": "Ciudad de México",
        "source": "adzuna_mx",
        "salary_observed": False,
        "salary_annual_local": None,
        "snapshot_date": pd.Timestamp("2026-08-07").date(),
    }
    base.update(kw)
    return base


def test_colapsa_la_misma_vacante_de_dos_fuentes():
    df = pd.DataFrame([_fila(posting_id="a"), _fila(posting_id="b", source="kaggle_mannacharya")])
    assert len(deduplicar(df)) == 1


def test_conserva_la_fila_con_salario_observado():
    df = pd.DataFrame([
        _fila(posting_id="a", salary_observed=False),
        _fila(posting_id="b", salary_observed=True, salary_annual_local=600000.0),
    ])
    out = deduplicar(df)
    assert len(out) == 1
    assert out.iloc[0]["salary_observed"]


def test_no_colapsa_vacantes_de_empresas_distintas():
    df = pd.DataFrame([_fila(company="Acme"), _fila(posting_id="b", company="Globex")])
    assert len(deduplicar(df)) == 2


def test_no_colapsa_la_misma_empresa_en_ciudades_distintas():
    df = pd.DataFrame([_fila(city="Ciudad de México"), _fila(posting_id="b", city="Querétaro")])
    assert len(deduplicar(df)) == 2


def test_es_idempotente():
    df = pd.DataFrame([_fila(posting_id="a"), _fila(posting_id="b")])
    una = deduplicar(df)
    dos = deduplicar(una)
    assert len(una) == len(dos) == 1
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_dedupe.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/features/dedupe.py`**

```python
"""Colapso de vacantes duplicadas entre agregadores."""

from __future__ import annotations

import re

import pandas as pd


def _llave(fila: pd.Series) -> str:
    titulo = re.sub(r"[^a-z0-9]+", "", str(fila.get("title_norm", "")).lower())
    empresa = re.sub(r"[^a-z0-9]+", "", str(fila.get("company", "")).lower())
    ciudad = re.sub(r"[^a-z0-9]+", "", str(fila.get("city", "")).lower())
    return f"{titulo}|{empresa}|{ciudad}"


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (título, empresa, ciudad).

    Ante duplicados se prefiere la fila con salario observado: es la que aporta
    información al análisis.
    """
    if df.empty:
        return df

    trabajo = df.copy()
    trabajo["_llave"] = trabajo.apply(_llave, axis=1)
    trabajo["_prioridad"] = trabajo["salary_observed"].fillna(False).astype(int)

    trabajo = trabajo.sort_values("_prioridad", ascending=False, kind="stable")
    salida = trabajo.drop_duplicates(subset="_llave", keep="first")
    return salida.drop(columns=["_llave", "_prioridad"]).reset_index(drop=True)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_dedupe.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/features/dedupe.py tests/test_dedupe.py
git commit -m "feat: deduplicación entre fuentes, prefiriendo salario observado"
```

---

## Fase 3 — Construcción del dataset

### Task 12: Pipeline `raw` → `processed`

**Files:**
- Create: `src/features/build.py`
- Test: `tests/test_build.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_build.py`:

```python
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS
from src.features.build import construir


def _snapshot(tmp_path, filas):
    carpeta = tmp_path / "raw" / "2026-08-07"
    carpeta.mkdir(parents=True)
    pd.DataFrame(filas, columns=CANONICAL_COLUMNS).to_parquet(carpeta / "adzuna.parquet")
    return tmp_path


def _cruda(**kw):
    base = dict.fromkeys(CANONICAL_COLUMNS)
    base.update(
        posting_id="a",
        snapshot_date=date(2026, 8, 7),
        source="adzuna_mx",
        title_raw="Senior Software Engineer",
        title_norm="senior software engineer",
        company="Acme",
        country="MX",
        city="Ciudad de México",
        metro="Valle de México",
        salary_min_raw=600000.0,
        salary_max_raw=800000.0,
        salary_currency="MXN",
        salary_period="año",
        salary_is_predicted=False,
        salary_observed=True,
        description_text="Necesitamos RAG y PyTorch para agentes",
        description_lang="es",
    )
    base.update(kw)
    return base


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_fx_usd_mxn", return_value=18.0)
def test_construye_el_dataset_procesado(_fx, _ppp, tmp_path):
    raiz = _snapshot(tmp_path, [_cruda()])
    df = construir(raiz)

    assert len(df) == 1
    fila = df.iloc[0]
    assert fila["tier"] == "anillo"                  # título genérico + 2 skills de IA
    assert set(["rag", "pytorch"]) <= set(fila["skills"])
    assert fila["seniority"] == "sr"
    assert fila["salary_annual_local"] == 700000.0   # punto medio de min y max
    assert fila["salary_annual_usd"] == pytest.approx(38888.9, rel=1e-3)
    assert fila["salary_annual_usd_ppp"] == pytest.approx(70000.0)
    assert fila["remote_scope"] == "local"


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_fx_usd_mxn", return_value=18.0)
def test_remote_scope_detecta_el_arbitraje_hacia_estados_unidos(_fx, _ppp, tmp_path):
    """Pregunta 6 del spec: un remoto desde México que paga en dólares o menciona
    un equipo en EE.UU. es la señal de arbitraje."""
    raiz = _snapshot(
        tmp_path,
        [_cruda(posting_id="a", is_remote=True,
                description_text="Remoto desde México para un equipo en United States, pago en USD"),
         _cruda(posting_id="b", title_norm="ml engineer", is_remote=True,
                description_text="Trabajo remoto desde cualquier parte de la república"),
         _cruda(posting_id="c", title_norm="ai engineer", is_remote=False)],
    )
    df = construir(raiz).set_index("posting_id")
    assert df.loc["a", "remote_scope"] == "us_desde_mx"
    assert df.loc["b", "remote_scope"] == "nacional"
    assert df.loc["c", "remote_scope"] == "local"


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_fx_usd_mxn", return_value=18.0)
def test_calcula_el_conteo_de_vacantes_por_empresa(_fx, _ppp, tmp_path):
    raiz = _snapshot(
        tmp_path,
        [_cruda(posting_id="a", title_norm="ai engineer"),
         _cruda(posting_id="b", title_norm="ml engineer"),
         _cruda(posting_id="c", company="Globex", title_norm="ai engineer")],
    )
    df = construir(raiz)
    acme = df[df["company"] == "Acme"]
    assert (acme["company_posting_count"] == 2).all()


@patch("src.features.build.obtener_factores_ppp", return_value={"MX": 10.0, "US": 1.0})
@patch("src.features.build.obtener_fx_usd_mxn", return_value=18.0)
def test_el_salario_predicho_nunca_llega_a_las_columnas_normalizadas(_fx, _ppp, tmp_path):
    raiz = _snapshot(
        tmp_path,
        [_cruda(salary_is_predicted=True, salary_observed=False)],
    )
    df = construir(raiz)
    assert pd.isna(df.iloc[0]["salary_annual_local"])
```

Añadir `import pytest` al inicio del archivo.

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/features/build.py`**

```python
"""CLI: data/raw/* → data/processed/vacantes.parquet"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, validate_frame
from src.features.dedupe import deduplicar
from src.features.extract import extraer_seniority, extraer_skills
from src.features.normalize import a_anual, a_ppp, a_usd, obtener_factores_ppp, obtener_fx_usd_mxn
from src.features.taxonomy import clasificar
from src.utils.config import DATA_DIR
from src.utils.logging import setup_logging

MULTINACIONALES = {
    "mastercard", "amazon", "google", "microsoft", "ibm", "oracle", "sap", "accenture",
    "deloitte", "pwc", "kpmg", "ey", "nvidia", "intel", "meta", "apple", "salesforce",
    "hp", "dell", "cisco", "bosch", "siemens", "ge", "bbva", "santander", "citi",
}

# Señales de que un remoto publicado en México es en realidad para un equipo de EE.UU.
_ARBITRAJE = re.compile(
    r"\b(usd|us\$|d[óo]lares|u\.?s\.?[- ]based|united states|ee\.?\s?uu|latam|nearshore)\b", re.I
)
_GLOBAL = re.compile(r"\b(worldwide|anywhere in the world|globally|fully distributed)\b", re.I)

log = logging.getLogger(__name__)


def _es_multinacional(nombre: object) -> bool:
    texto = str(nombre or "").lower()
    return any(m in texto for m in MULTINACIONALES)


def _remote_scope(fila: pd.Series) -> str:
    """Alcance del remoto. Sostiene la pregunta 6 del spec (arbitraje MX → EE.UU.)."""
    if not bool(fila.get("is_remote")):
        return "local"
    texto = f"{fila.get('title_raw') or ''} {fila.get('description_text') or ''}"
    if _GLOBAL.search(texto):
        return "global"
    if fila.get("country") == "MX" and _ARBITRAJE.search(texto):
        return "us_desde_mx"
    return "nacional"


def _punto_medio(fila: pd.Series) -> float | None:
    lo, hi = fila["salary_min_raw"], fila["salary_max_raw"]
    if pd.isna(lo):
        return None
    if pd.isna(hi):
        return float(lo)
    return (float(lo) + float(hi)) / 2


def construir(raiz_datos: Path) -> pd.DataFrame:
    parquets = sorted((raiz_datos / "raw").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No hay snapshots en {raiz_datos / 'raw'}. Corre `make collect`.")

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    log.info("Cargadas %s filas de %s snapshots", len(df), len(parquets))

    # --- extracción sobre el texto ---
    texto = df["title_raw"].fillna("") + " " + df["description_text"].fillna("")
    df["skills"] = texto.map(extraer_skills)
    df["seniority"] = [
        extraer_seniority(t, d)
        for t, d in zip(df["title_raw"].fillna(""), df["description_text"].fillna(""), strict=True)
    ]
    df["tier"] = [
        clasificar(t, skills=s)
        for t, s in zip(df["title_raw"].fillna(""), df["skills"], strict=True)
    ]

    # --- normalización salarial: sólo sobre lo observado ---
    fx = obtener_fx_usd_mxn()
    ppp = obtener_factores_ppp()
    log.info("FX USD/MXN=%.2f  PPP=%s", fx, ppp)

    observado = df["salary_observed"].fillna(False).astype(bool)
    medio = df.apply(_punto_medio, axis=1).where(observado)
    df["salary_annual_local"] = [
        a_anual(m, p) if pd.notna(m) else None
        for m, p in zip(medio, df["salary_period"].fillna("año"), strict=True)
    ]
    df["salary_annual_usd"] = [
        a_usd(m, c, fx_usd_mxn=fx) if m is not None else None
        for m, c in zip(df["salary_annual_local"], df["salary_currency"].fillna("USD"), strict=True)
    ]
    df["salary_annual_usd_ppp"] = [
        a_ppp(m, p, ppp) if m is not None and pd.notna(p) else None
        for m, p in zip(df["salary_annual_local"], df["country"], strict=True)
    ]

    # --- atributos de empresa y de modalidad ---
    df["company_is_multinational"] = df["company"].map(_es_multinacional)
    df["remote_scope"] = df.apply(_remote_scope, axis=1)

    df = deduplicar(df)
    df["company_posting_count"] = df.groupby("company")["posting_id"].transform("count")

    df = df[CANONICAL_COLUMNS]
    validate_frame(df)
    return df


def main() -> None:
    setup_logging()
    df = construir(DATA_DIR)
    salida = DATA_DIR / "processed" / "vacantes.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida, index=False)
    df.to_csv(DATA_DIR / "processed" / "vacantes.csv", index=False)

    obs = df["salary_observed"].fillna(False).sum()
    log.info("✅ %s filas → %s (%s con salario observado)", len(df), salida, obs)
    print("\nSalarios observados por ciudad:")
    print(
        df[df["salary_observed"].fillna(False)]
        .groupby(["country", "city"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_build.py -v`
Expected: 3 passed

- [ ] **Step 5: Correr el pipeline completo y verificar la suite**

Run: `make build && make test && make lint`
Expected: `data/processed/vacantes.parquet` creado, toda la suite en verde, ruff y mypy limpios. Anotar el conteo de salarios observados por ciudad — es el insumo del criterio de éxito §11.1.

- [ ] **Step 6: Commit**

```bash
git add src/features/build.py tests/test_build.py
git commit -m "feat: pipeline raw -> processed con normalización y features"
```

---

## Fase 4 — Análisis estadístico

### Task 13: Estadística con IC y regla n ≥ 30

**Files:**
- Create: `src/analysis/__init__.py`, `src/analysis/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_stats.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.analysis.stats import mediana_ci, resumir_por


def test_mediana_ci_devuelve_mediana_y_cotas():
    datos = np.arange(1, 101, dtype=float)
    med, lo, hi = mediana_ci(datos, n_boot=2000, seed=42)
    assert med == pytest.approx(50.5, abs=1.0)
    assert lo < med < hi


def test_mediana_ci_es_reproducible_con_semilla():
    datos = np.random.default_rng(0).normal(100, 20, 200)
    assert mediana_ci(datos, n_boot=1000, seed=7) == mediana_ci(datos, n_boot=1000, seed=7)


def test_mediana_ci_con_muestra_vacia_devuelve_nan():
    med, lo, hi = mediana_ci(np.array([]), n_boot=100, seed=1)
    assert np.isnan(med) and np.isnan(lo) and np.isnan(hi)


def test_resumir_por_marca_los_grupos_con_n_insuficiente():
    """Regla dura del spec §6: n < 30 no se grafica, se marca."""
    df = pd.DataFrame({
        "city": ["CDMX"] * 40 + ["Querétaro"] * 12,
        "salary_annual_usd_ppp": list(np.linspace(50000, 90000, 40)) + list(np.linspace(40000, 60000, 12)),
    })
    out = resumir_por(df, ["city"], valor="salary_annual_usd_ppp", min_n=30)

    cdmx = out[out["city"] == "CDMX"].iloc[0]
    qro = out[out["city"] == "Querétaro"].iloc[0]
    assert cdmx["suficiente"] is True
    assert qro["suficiente"] is False
    assert qro["n"] == 12
    assert pd.isna(qro["mediana"])  # no se reporta un número que no se sostiene


def test_resumir_por_ignora_los_nulos_al_contar():
    df = pd.DataFrame({
        "city": ["CDMX"] * 50,
        "salary_annual_usd_ppp": [60000.0] * 20 + [None] * 30,
    })
    out = resumir_por(df, ["city"], valor="salary_annual_usd_ppp", min_n=30)
    assert out.iloc[0]["n"] == 20
    assert out.iloc[0]["suficiente"] is False
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/analysis/stats.py`**

```python
"""Estadística descriptiva del proyecto.

Mediana en vez de media porque los salarios tienen cola derecha larga, y regla
dura de n >= 30 para no publicar números que no se sostienen (§6 del spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_N = 30


def mediana_ci(
    valores: np.ndarray | pd.Series, *, n_boot: int = 10_000, alpha: float = 0.10, seed: int = 42
) -> tuple[float, float, float]:
    """Mediana con intervalo de confianza bootstrap. Devuelve (mediana, inferior, superior)."""
    datos = np.asarray(pd.Series(valores).dropna(), dtype=float)
    if datos.size == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    muestras = rng.choice(datos, size=(n_boot, datos.size), replace=True)
    medianas = np.median(muestras, axis=1)
    lo, hi = np.quantile(medianas, [alpha / 2, 1 - alpha / 2])
    return (float(np.median(datos)), float(lo), float(hi))


def resumir_por(
    df: pd.DataFrame,
    grupos: list[str],
    *,
    valor: str = "salary_annual_usd_ppp",
    min_n: int = MIN_N,
    n_boot: int = 10_000,
) -> pd.DataFrame:
    """Un renglón por grupo con n, mediana, IC, IQR y la bandera de suficiencia."""
    filas: list[dict[str, object]] = []
    for llaves, sub in df.groupby(grupos, dropna=False):
        llaves = llaves if isinstance(llaves, tuple) else (llaves,)
        datos = sub[valor].dropna()
        n = int(datos.size)
        suficiente = n >= min_n

        if suficiente:
            med, lo, hi = mediana_ci(datos, n_boot=n_boot)
            q1, q3 = float(datos.quantile(0.25)), float(datos.quantile(0.75))
        else:
            med = lo = hi = q1 = q3 = np.nan

        filas.append(
            {**dict(zip(grupos, llaves, strict=True)),
             "n": n, "suficiente": suficiente,
             "mediana": med, "ic_inf": lo, "ic_sup": hi, "q1": q1, "q3": q3}
        )
    return pd.DataFrame(filas).sort_values("n", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_stats.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/analysis/ tests/test_stats.py
git commit -m "feat: mediana con IC bootstrap y regla n>=30"
```

---

### Task 14: Pruebas de hipótesis

**Files:**
- Create: `src/analysis/hypothesis.py`
- Test: `tests/test_hypothesis.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_hypothesis.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_hypothesis.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/analysis/hypothesis.py`**

```python
"""Pruebas de hipótesis con reporte legible, al estilo de los notebooks de TripleTen."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MIN_N = 30


def _limpio(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(pd.Series(x).dropna(), dtype=float)


def mann_whitney(
    a: np.ndarray | pd.Series, b: np.ndarray | pd.Series, *,
    etiqueta_a: str, etiqueta_b: str, alpha: float = ALPHA,
) -> dict[str, Any]:
    """H0: las dos muestras vienen de la misma distribución. No asume normalidad."""
    xa, xb = _limpio(a), _limpio(b)
    if xa.size < MIN_N or xb.size < MIN_N:
        return {
            "n_a": int(xa.size), "n_b": int(xb.size), "p_valor": None, "rechaza_h0": None,
            "conclusion": (
                f"Muestra insuficiente para comparar {etiqueta_a} (n={xa.size}) "
                f"con {etiqueta_b} (n={xb.size}). Se requiere n>={MIN_N} en ambos."
            ),
        }

    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    rechaza = bool(p < alpha)
    mayor, menor = (etiqueta_a, etiqueta_b) if np.median(xa) > np.median(xb) else (etiqueta_b, etiqueta_a)
    conclusion = (
        f"Se rechaza H0 (p={p:.2e}): la mediana de {mayor} es significativamente mayor que la de {menor}."
        if rechaza
        else f"No se rechaza H0 (p={p:.3f}): no hay evidencia de diferencia entre {etiqueta_a} y {etiqueta_b}."
    )
    return {
        "n_a": int(xa.size), "n_b": int(xb.size), "u": float(u), "p_valor": float(p),
        "rechaza_h0": rechaza, "mediana_a": float(np.median(xa)), "mediana_b": float(np.median(xb)),
        "conclusion": conclusion,
    }


def welch_log(
    a: np.ndarray | pd.Series, b: np.ndarray | pd.Series, *,
    etiqueta_a: str, etiqueta_b: str, alpha: float = ALPHA,
) -> dict[str, Any]:
    """Welch sobre log-salarios: compara razones y no diferencias absolutas."""
    xa, xb = _limpio(a), _limpio(b)
    xa, xb = xa[xa > 0], xb[xb > 0]
    if xa.size < MIN_N or xb.size < MIN_N:
        return {
            "n_a": int(xa.size), "n_b": int(xb.size), "p_valor": None, "rechaza_h0": None,
            "razon_medianas": float("nan"),
            "conclusion": f"Muestra insuficiente ({etiqueta_a} n={xa.size}, {etiqueta_b} n={xb.size}).",
        }

    t, p = stats.ttest_ind(np.log(xa), np.log(xb), equal_var=False)
    razon = float(np.median(xa) / np.median(xb))
    rechaza = bool(p < alpha)
    return {
        "n_a": int(xa.size), "n_b": int(xb.size), "t": float(t), "p_valor": float(p),
        "rechaza_h0": rechaza, "razon_medianas": razon,
        "conclusion": (
            f"{'Se rechaza' if rechaza else 'No se rechaza'} H0 (p={p:.2e}). "
            f"La mediana de {etiqueta_a} es {razon:.2f}x la de {etiqueta_b}."
        ),
    }
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_hypothesis.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/analysis/hypothesis.py tests/test_hypothesis.py
git commit -m "feat: Mann-Whitney y Welch sobre log-salarios con reporte legible"
```

---

## Fase 5 — Notebooks

Los notebooks no se hacen con TDD: su corrección se verifica ejecutándolos de punta a punta. La regla para los cuatro es la misma — **cargan desde `data/processed/vacantes.parquet` y no llaman a ninguna API**, para que cualquiera pueda reproducirlos. Cada uno arranca con el encabezado de tus proyectos de TripleTen y cierra cada sección con un bloque de conclusiones en markdown.

Celda de encabezado, idéntica en los cuatro (cambiando el título):

```markdown
# Mercado de trabajo de IA: México vs Estados Unidos
## 01 — Recolección y calidad de los datos
### DA: *Mau Vilar*
#### https://github.com/mauvilar
```

Celda de arranque, idéntica en los cuatro:

```python
import sys; sys.path.insert(0, "..")
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from src.analysis.stats import resumir_por, mediana_ci
from src.analysis.hypothesis import mann_whitney, welch_log

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)
df = pd.read_parquet("../data/processed/vacantes.parquet")
print(f"{len(df):,} vacantes | {df['salary_observed'].sum():,} con salario observado")
df.head()
```

### Task 15: Notebook 01 — Recolección y calidad

**Files:**
- Create: `notebooks/01_recoleccion_y_calidad.ipynb`

- [ ] **Step 1: Crear el esqueleto de secciones**

Secciones en markdown, en este orden:
`## Auditoría de fuentes` → `### El dataset que no usamos, y por qué` → `### Las fuentes que sí` → `## Descripción de los datos` → `#### Datos nulos y duplicados` → `## Cobertura geográfica` → `## Cobertura salarial` → `## Resumen de la calidad de los datos`

- [ ] **Step 2: Implementar «El dataset que no usamos, y por qué»**

Es la apertura del proyecto: la autopsia del dataset sintético.

```python
from src.data.kaggle_sources import verificar_no_sintetico

sospechoso = pd.read_csv("../data/raw/kaggle/_descartado/ai_jobs_dataset_2026.csv", low_memory=False)
conteos = sospechoso["Country"].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
conteos.plot.barh(ax=ax[0], color="#c44").set_title("m0sm71 — filas por país (descartado)")
real = pd.read_parquet("../data/processed/vacantes.parquet")["country"].value_counts()
real.plot.barh(ax=ax[1], color="#4a4").set_title("Nuestro dataset — filas por país")
plt.tight_layout()

print(verificar_no_sintetico(sospechoso, col_pais="Country", col_salario="Salary Range"))
print(f"Salarios presentes: {sospechoso['Salary Range'].notna().mean():.0%}")
```

Conclusión en markdown, con este contenido: doce países con ~4,300 filas cada uno y 100 % de salarios presentes es la firma de un dataset generado; ningún agregador real produce esa uniformidad, y la referencia real tiene 19 % de salarios y está sesgada a San Francisco. Por eso se descartó antes de usarlo.

- [ ] **Step 3: Implementar «Cobertura geográfica» y «Cobertura salarial»**

```python
cobertura = (
    df.assign(observado=df["salary_observed"].fillna(False))
      .groupby(["country", "city"])
      .agg(vacantes=("posting_id", "count"), con_salario=("observado", "sum"))
      .assign(tasa=lambda d: d["con_salario"] / d["vacantes"])
      .sort_values("vacantes", ascending=False)
)
display(cobertura.style.format({"tasa": "{:.1%}"}))

fig, ax = plt.subplots()
cobertura.reset_index().pivot_table(index="city", columns="country", values="tasa").plot.barh(ax=ax)
ax.set_xlabel("Proporción de vacantes que publican salario")
ax.set_title("Transparencia salarial por ciudad")
```

- [ ] **Step 4: Validación cruzada contra Indeed**

El spec (§4) lista el conector de Indeed como fuente de validación: sirve para comprobar que la
cobertura de Adzuna es real y no un artefacto de su índice. No entra al dataset, sólo se contrasta.

Pedirle a Claude Code, en la sesión, que corra `search_jobs` para «AI Engineer» en cada una de las
cuatro ciudades mexicanas y pegar los resultados en `data/raw/validacion_indeed.csv` con las
columnas `city,title,company,posted_on,compensation`. Después:

```python
indeed = pd.read_csv("../data/raw/validacion_indeed.csv")
comparacion = pd.DataFrame({
    "adzuna": df[df["country"] == "MX"].groupby("city").size(),
    "indeed": indeed.groupby("city").size(),
    "indeed_con_salario": indeed[indeed["compensation"].notna()].groupby("city").size(),
}).fillna(0).astype(int)
display(comparacion)
```

Conclusión en markdown: si Indeed también devuelve ~0 salarios en México, la opacidad es del
mercado y no un sesgo de Adzuna — que es exactamente la afirmación que sostiene la tesis del
proyecto. Si Indeed sí publicara salarios donde Adzuna no, habría que revisar la fuente.

- [ ] **Step 5: Escribir «Resumen de la calidad de los datos»**

Bloque en markdown que responda: cuántas vacantes hay, de qué fuentes, cuántas por ciudad, qué proporción publica salario en cada país, cuántos duplicados se colapsaron, y **cuáles cortes ya alcanzan n ≥ 30 y cuáles no**. Este último punto define qué puede afirmar el notebook 02.

- [ ] **Step 6: Ejecutar el notebook completo**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_recoleccion_y_calidad.ipynb`
Expected: sin excepciones.

- [ ] **Step 7: Commit**

```bash
git add notebooks/01_recoleccion_y_calidad.ipynb
git commit -m "feat(nb): 01 auditoría de fuentes y calidad de datos"
```

---

### Task 16: Notebook 02 — Brecha salarial

**Files:**
- Create: `notebooks/02_brecha_salarial.ipynb`

- [ ] **Step 1: Crear el esqueleto de secciones**

`## Salarios en México` → `### Por ciudad` → `### Por seniority` → `## Salarios en Estados Unidos` → `### Por metro` → `## La brecha` → `### Brecha nominal` → `### Brecha ajustada por poder adquisitivo` → `## Pruebas de hipótesis` → `## El arbitraje del remoto` → `## Resumen del análisis salarial`

- [ ] **Step 2: Implementar «Por ciudad» con la regla de suficiencia**

```python
ia = df[df["tier"].isin(["nucleo", "anillo"]) & df["salary_observed"].fillna(False)]
mx = ia[ia["country"] == "MX"]

resumen_mx = resumir_por(mx, ["city"], valor="salary_annual_local", min_n=30)
display(resumen_mx)

graficables = resumen_mx[resumen_mx["suficiente"]]
insuficientes = resumen_mx[~resumen_mx["suficiente"]]

fig, ax = plt.subplots()
ax.errorbar(
    graficables["mediana"], graficables["city"],
    xerr=[graficables["mediana"] - graficables["ic_inf"], graficables["ic_sup"] - graficables["mediana"]],
    fmt="o", capsize=5, color="#2a6",
)
ax.set_xlabel("Salario anual mediano (MXN)")
ax.set_title("México — mediana con IC 90 %, sólo ciudades con n ≥ 30")
for _, r in graficables.iterrows():
    ax.annotate(f"n={r['n']}", (r["mediana"], r["city"]), xytext=(6, 6), textcoords="offset points")

if len(insuficientes):
    print("Muestra insuficiente, NO se grafican:")
    print(insuficientes[["city", "n"]].to_string(index=False))
```

- [ ] **Step 3: Implementar «Brecha ajustada por poder adquisitivo»**

```python
comparativa = resumir_por(ia, ["country", "metro"], valor="salary_annual_usd_ppp", min_n=30)
comparativa = comparativa[comparativa["suficiente"]].sort_values("mediana")

fig, ax = plt.subplots(figsize=(11, 6))
colores = comparativa["country"].map({"MX": "#2a6", "US": "#36c"})
ax.barh(comparativa["metro"], comparativa["mediana"], color=colores)
ax.set_xlabel("Salario anual mediano (dólares internacionales, ajustado por PPP)")
ax.set_title("Brecha ajustada por poder adquisitivo")

nominal = resumir_por(ia, ["country"], valor="salary_annual_usd", min_n=30)
ppp = resumir_por(ia, ["country"], valor="salary_annual_usd_ppp", min_n=30)
print("Razón US/MX nominal:",
      nominal.set_index("country")["mediana"]["US"] / nominal.set_index("country")["mediana"]["MX"])
print("Razón US/MX en PPP:",
      ppp.set_index("country")["mediana"]["US"] / ppp.set_index("country")["mediana"]["MX"])
```

- [ ] **Step 4: Implementar «Pruebas de hipótesis»**

```python
cdmx = mx[mx["city"] == "Ciudad de México"]["salary_annual_local"]
qro = mx[mx["city"] == "Querétaro"]["salary_annual_local"]
print(mann_whitney(cdmx, qro, etiqueta_a="CDMX", etiqueta_b="Querétaro")["conclusion"])

print(welch_log(
    ia[ia["country"] == "US"]["salary_annual_usd_ppp"],
    ia[ia["country"] == "MX"]["salary_annual_usd_ppp"],
    etiqueta_a="Estados Unidos", etiqueta_b="México",
)["conclusion"])
```

- [ ] **Step 5: Implementar «El arbitraje del remoto»**

```python
print(resumir_por(mx, ["remote_scope"], valor="salary_annual_local", min_n=30).to_string(index=False))

hacia_us = mx[mx["remote_scope"] == "us_desde_mx"]["salary_annual_local"]
local = mx[mx["remote_scope"] == "local"]["salary_annual_local"]
print(mann_whitney(hacia_us, local, etiqueta_a="Remoto para EE.UU.", etiqueta_b="Presencial en México")["conclusion"])
```

- [ ] **Step 6: Escribir «Resumen del análisis salarial»**

Markdown que responda las preguntas 1, 2 y 6 del spec con los números obtenidos, y que diga explícitamente qué cortes quedaron sin muestra suficiente en vez de omitirlos.

- [ ] **Step 7: Ejecutar el notebook completo**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_brecha_salarial.ipynb`
Expected: sin excepciones.

- [ ] **Step 8: Commit**

```bash
git add notebooks/02_brecha_salarial.ipynb
git commit -m "feat(nb): 02 brecha salarial nominal y ajustada por PPP"
```

---

### Task 17: Notebook 03 — Skills y transparencia

**Files:**
- Create: `notebooks/03_skills_y_transparencia.ipynb`

- [ ] **Step 1: Crear el esqueleto de secciones**

`## Qué skills pide el mercado` → `### México` → `### Estados Unidos` → `## Primas salariales por skill` → `## Núcleo vs anillo` → `### Cuántas vacantes de IA no se llaman así` → `## Transparencia salarial` → `### Quién publica sueldo` → `## Resumen de skills y transparencia`

- [ ] **Step 2: Implementar «Primas salariales por skill»**

```python
explotado = ia.explode("skills").dropna(subset=["skills"])

primas = []
for pais in ["MX", "US"]:
    sub = explotado[explotado["country"] == pais]
    base = sub["salary_annual_usd_ppp"].median()
    for skill, grupo in sub.groupby("skills"):
        datos = grupo["salary_annual_usd_ppp"].dropna()
        if len(datos) >= 30:
            primas.append({"country": pais, "skill": skill, "n": len(datos),
                           "mediana": datos.median(), "prima_pct": datos.median() / base - 1})
primas = pd.DataFrame(primas).sort_values("prima_pct", ascending=False)
display(primas)

pivote = primas.pivot(index="skill", columns="country", values="prima_pct").dropna()
pivote.sort_values("US").plot.barh(figsize=(10, 7))
plt.xlabel("Prima sobre la mediana del país")
plt.title("¿Las mismas skills pagan en los dos mercados?")
```

- [ ] **Step 3: Implementar «Cuántas vacantes de IA no se llaman así»**

```python
mezcla = (
    df[df["tier"].isin(["nucleo", "anillo"])]
    .groupby(["country", "tier"]).size().unstack(fill_value=0)
)
mezcla["% anillo"] = mezcla["anillo"] / mezcla.sum(axis=1)
display(mezcla.style.format({"% anillo": "{:.1%}"}))
mezcla[["nucleo", "anillo"]].plot.bar(stacked=True)
plt.title("Vacantes de IA que el título declara vs las que no")
```

- [ ] **Step 4: Implementar «Quién publica sueldo»**

```python
transparencia = (
    df.assign(observado=df["salary_observed"].fillna(False))
      .groupby(["country", "company_is_multinational"])["observado"].agg(["mean", "count"])
      .rename(columns={"mean": "tasa_publicacion", "count": "vacantes"})
)
display(transparencia.style.format({"tasa_publicacion": "{:.1%}"}))

por_sector = (
    df.assign(observado=df["salary_observed"].fillna(False))
      .groupby(["country", "category"])["observado"].agg(["mean", "count"])
      .query("count >= 30").sort_values("mean", ascending=False)
)
display(por_sector.style.format({"mean": "{:.1%}"}))
```

- [ ] **Step 5: Escribir «Resumen de skills y transparencia»**

Markdown que responda las preguntas 3, 4 y 5 del spec.

- [ ] **Step 6: Ejecutar el notebook completo**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_skills_y_transparencia.ipynb`
Expected: sin excepciones.

- [ ] **Step 7: Commit**

```bash
git add notebooks/03_skills_y_transparencia.ipynb
git commit -m "feat(nb): 03 primas por skill y transparencia salarial"
```

---

## Fase 6 — Modelo de imputación

### Task 18: Entrenamiento y evaluación

**Files:**
- Create: `src/models/__init__.py`, `src/models/impute.py`, `src/models/evaluate.py`
- Test: `tests/test_impute.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/test_impute.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import baseline_pais_seniority, evaluar_holdout_mx, mdape
from src.models.impute import FEATURES, entrenar, particionar


def _corpus(n_us=400, n_mx=60, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for pais, n, base in [("US", n_us, 150_000), ("MX", n_mx, 55_000)]:
        for _ in range(n):
            sen = rng.choice(["jr", "mid", "sr"])
            mult = {"jr": 0.7, "mid": 1.0, "sr": 1.4}[sen]
            filas.append({
                "country": pais, "seniority": sen, "tier": "nucleo",
                "metro": "SF Bay Area" if pais == "US" else "Valle de México",
                "category": "IT Jobs", "title_norm": "ai engineer",
                "is_remote": False, "remote_scope": "local",
                "company_posting_count": 3, "company_is_multinational": True,
                "skills": ["llm", "pytorch"],
                "salary_annual_usd_ppp": base * mult * rng.normal(1, 0.12),
                "salary_observed": True,
            })
    return pd.DataFrame(filas)


def test_particionar_saca_todo_mexico_del_entrenamiento():
    """El hold-out del spec §7: el modelo no puede ver ni una fila mexicana."""
    train, holdout = particionar(_corpus())
    assert (train["country"] != "MX").all()
    assert (holdout["country"] == "MX").all()
    assert len(holdout) == 60


def test_particionar_solo_usa_salarios_observados():
    corpus = _corpus()
    corpus.loc[:9, "salary_observed"] = False
    train, holdout = particionar(corpus)
    assert len(train) + len(holdout) == len(corpus) - 10


def test_entrenar_devuelve_un_modelo_que_predice():
    train, _ = particionar(_corpus())
    modelo = entrenar(train)
    pred = modelo.predict(train[FEATURES].head(5))
    assert len(pred) == 5
    assert np.all(pred > 0)


def test_mdape_calcula_la_mediana_del_error_porcentual():
    real = np.array([100.0, 100.0, 100.0])
    pred = np.array([110.0, 90.0, 150.0])
    assert mdape(real, pred) == pytest.approx(0.10)


def test_baseline_usa_la_mediana_por_pais_y_seniority():
    corpus = _corpus()
    train, holdout = particionar(corpus)
    pred = baseline_pais_seniority(train, holdout)
    assert len(pred) == len(holdout)
    assert np.all(pred > 0)


def test_evaluar_holdout_emite_un_veredicto_de_publicacion():
    train, holdout = particionar(_corpus())
    modelo = entrenar(train)
    res = evaluar_holdout_mx(modelo, train, holdout, umbral_mdape=0.35)
    assert set(res) >= {"mdape_modelo", "mdape_baseline", "supera_baseline", "publicable"}
    assert isinstance(res["publicable"], bool)


def test_no_es_publicable_si_no_supera_el_umbral():
    train, holdout = particionar(_corpus())
    modelo = entrenar(train)
    res = evaluar_holdout_mx(modelo, train, holdout, umbral_mdape=0.0001)
    assert res["publicable"] is False
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `uv run pytest tests/test_impute.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/models/impute.py`**

```python
"""Imputación de salarios para vacantes que no los publican (§7 del spec)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAS = [
    "country", "seniority", "tier", "metro", "category", "title_norm", "remote_scope",
]
NUMERICAS = ["company_posting_count"]
BOOLEANAS = ["is_remote", "company_is_multinational"]
SKILLS_MULTIHOT = ["llm", "rag", "pytorch", "mlops", "vector_db", "agents", "python", "sql"]
FEATURES = CATEGORICAS + NUMERICAS + BOOLEANAS + [f"skill_{s}" for s in SKILLS_MULTIHOT]

TARGET = "salary_annual_usd_ppp"


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas multi-hot de skills que el modelo espera."""
    out = df.copy()
    listas = out["skills"].apply(lambda s: s if isinstance(s, list) else [])
    for skill in SKILLS_MULTIHOT:
        out[f"skill_{skill}"] = listas.apply(lambda s, k=skill: k in s)
    for col in BOOLEANAS:
        out[col] = out[col].fillna(False).astype(bool)
    for col in NUMERICAS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in CATEGORICAS:
        out[col] = out[col].fillna("desconocido").astype(str)
    return out


def particionar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento = observados fuera de México. Hold-out = TODOS los observados de México."""
    observados = preparar(df[df["salary_observed"].fillna(False).astype(bool)].copy())
    observados = observados[observados[TARGET].notna()]
    train = observados[observados["country"] != "MX"].reset_index(drop=True)
    holdout = observados[observados["country"] == "MX"].reset_index(drop=True)
    return train, holdout


class ModeloSalarial:
    """Envuelve el pipeline para entrenar en log y predecir en escala original."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.pipeline.predict(X[FEATURES]))


def entrenar(train: pd.DataFrame, *, seed: int = 42) -> ModeloSalarial:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS)],
        remainder="passthrough",
    )
    pipeline = Pipeline([
        ("pre", pre),
        ("gbm", HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=400,
            l2_regularization=1.0, random_state=seed)),
    ])
    pipeline.fit(train[FEATURES], np.log(train[TARGET]))
    return ModeloSalarial(pipeline)
```

- [ ] **Step 4: Implementar `src/models/evaluate.py`**

```python
"""Hold-out mexicano, baseline y criterio de publicación (§7 del spec)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.impute import TARGET, ModeloSalarial

UMBRAL_MDAPE = 0.35


def mdape(real: np.ndarray, pred: np.ndarray) -> float:
    """Mediana del error porcentual absoluto. Robusta a los outliers salariales."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.median(np.abs((real - pred) / real)))


def baseline_pais_seniority(train: pd.DataFrame, objetivo: pd.DataFrame) -> np.ndarray:
    """El rival a vencer: la mediana por (país, seniority). Si el modelo no lo supera, no aporta."""
    medianas = train.groupby("seniority")[TARGET].median()
    global_ = float(train[TARGET].median())
    return objetivo["seniority"].map(medianas).fillna(global_).to_numpy(dtype=float)


def evaluar_holdout_mx(
    modelo: ModeloSalarial, train: pd.DataFrame, holdout: pd.DataFrame,
    *, umbral_mdape: float = UMBRAL_MDAPE,
) -> dict[str, Any]:
    if holdout.empty:
        return {
            "n_holdout": 0, "mdape_modelo": float("nan"), "mdape_baseline": float("nan"),
            "supera_baseline": False, "publicable": False,
            "veredicto": "Sin salarios mexicanos observados: no hay con qué validar. No se publica.",
        }

    real = holdout[TARGET].to_numpy(dtype=float)
    e_modelo = mdape(real, modelo.predict(holdout))
    e_baseline = mdape(real, baseline_pais_seniority(train, holdout))

    supera = bool(e_modelo < e_baseline)
    publicable = bool(e_modelo <= umbral_mdape and supera)

    if publicable:
        veredicto = (
            f"PUBLICABLE. MdAPE={e_modelo:.1%} sobre {len(holdout)} salarios mexicanos que el "
            f"modelo nunca vio, contra {e_baseline:.1%} del baseline."
        )
    else:
        motivo = []
        if e_modelo > umbral_mdape:
            motivo.append(f"MdAPE={e_modelo:.1%} supera el umbral de {umbral_mdape:.0%}")
        if not supera:
            motivo.append(f"no le gana al baseline ({e_baseline:.1%})")
        veredicto = (
            f"NO PUBLICABLE: {' y '.join(motivo)}. El modelo entrenado en EE.UU. no transfiere "
            "al mercado mexicano; eso se reporta como hallazgo y las estimaciones no se publican."
        )

    return {
        "n_holdout": int(len(holdout)), "mdape_modelo": e_modelo, "mdape_baseline": e_baseline,
        "supera_baseline": supera, "publicable": publicable, "veredicto": veredicto,
    }
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `uv run pytest tests/test_impute.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/models/ tests/test_impute.py
git commit -m "feat: imputación con hold-out mexicano y criterio de publicación"
```

---

### Task 19: Notebook 04 — El modelo

**Files:**
- Create: `notebooks/04_modelo_imputacion.ipynb`

- [ ] **Step 1: Crear el esqueleto de secciones**

`## Por qué un modelo` → `## Preparación` → `#### Partición: el hold-out mexicano` → `## Entrenamiento` → `## Evaluación contra el hold-out` → `### El baseline a vencer` → `### Veredicto` → `## Qué mueve el salario` → `## Resumen del modelo`

- [ ] **Step 2: Implementar partición, entrenamiento y veredicto**

```python
from src.models.impute import FEATURES, entrenar, particionar
from src.models.evaluate import evaluar_holdout_mx

train, holdout = particionar(df)
print(f"Entrenamiento (sin México): {len(train):,}")
print(f"Hold-out mexicano observado: {len(holdout):,}")

modelo = entrenar(train)
resultado = evaluar_holdout_mx(modelo, train, holdout)
for k, v in resultado.items():
    print(f"{k}: {v}")
```

- [ ] **Step 3: Graficar predicho vs real sobre el hold-out**

```python
if len(holdout):
    pred = modelo.predict(holdout)
    real = holdout["salary_annual_usd_ppp"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real, pred, alpha=0.6)
    lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
    ax.plot(lim, lim, "--", color="#888")
    ax.set_xlabel("Salario real (USD PPP)"); ax.set_ylabel("Salario predicho")
    ax.set_title(f"Hold-out México — MdAPE {resultado['mdape_modelo']:.1%} (n={len(holdout)})")
```

- [ ] **Step 4: Implementar «Qué mueve el salario» con importancia por permutación**

```python
from sklearn.inspection import permutation_importance

imp = permutation_importance(
    modelo.pipeline, train[FEATURES], np.log(train["salary_annual_usd_ppp"]),
    n_repeats=10, random_state=42, scoring="neg_mean_absolute_error",
)
(pd.Series(imp.importances_mean, index=FEATURES)
   .sort_values().tail(15).plot.barh(figsize=(9, 6)))
plt.title("Importancia por permutación (MAE sobre log-salario)")
```

- [ ] **Step 5: Escribir «Resumen del modelo»**

Markdown que reporte el veredicto tal cual salió. **Si `publicable` es `False`, ese es el resultado del notebook** y así se escribe: el modelo entrenado en EE.UU. no transfiere al mercado mexicano, las estimaciones no se publican, y la razón más probable es que la estructura salarial mexicana no es una reescala de la estadounidense. Un resultado negativo bien documentado es un resultado.

- [ ] **Step 6: Ejecutar el notebook completo**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelo_imputacion.ipynb`
Expected: sin excepciones.

- [ ] **Step 7: Commit**

```bash
git add notebooks/04_modelo_imputacion.ipynb
git commit -m "feat(nb): 04 modelo de imputación y veredicto del hold-out"
```

---

## Fase 7 — Automatización y publicación

### Task 20: GitHub Action semanal

**Files:**
- Create: `.github/workflows/collect.yml`, `scripts/publicar_kaggle.py`

- [ ] **Step 1: Crear `scripts/publicar_kaggle.py`**

```python
"""Sube una versión nueva del dataset procesado a Kaggle."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SLUG = "vacantes-ia-mexico-estados-unidos"


def _auth() -> str:
    usuario = os.environ["KAGGLE_USERNAME"]
    llave = os.environ["KAGGLE_KEY"]
    return base64.b64encode(f"{usuario}:{llave}".encode()).decode()


def escribir_metadata() -> None:
    usuario = os.environ["KAGGLE_USERNAME"]
    (PROCESSED / "dataset-metadata.json").write_text(
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
    escribir_metadata()
    subprocess.run(
        ["curl", "-sS", "-X", "POST",
         "-H", f"Authorization: Basic {_auth()}",
         "-F", f"file=@{PROCESSED / 'vacantes.csv'}",
         "https://www.kaggle.com/api/v1/datasets/create/version"],
        check=True, cwd=PROCESSED,
    )
    print(f"✅ Versión del {date.today().isoformat()} publicada en Kaggle")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Crear `.github/workflows/collect.yml`**

```yaml
name: Recolección semanal

on:
  schedule:
    - cron: "0 13 * * 1"   # lunes 13:00 UTC = 7:00 CDMX
  workflow_dispatch:

jobs:
  recolectar:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Instalar dependencias
        run: uv sync --all-extras

      - name: Recolectar
        env:
          ADZUNA_APP_ID: ${{ secrets.ADZUNA_APP_ID }}
          ADZUNA_APP_KEY: ${{ secrets.ADZUNA_APP_KEY }}
        run: uv run mia-collect

      - name: Reconstruir el dataset
        run: uv run mia-build

      - name: Pruebas
        run: uv run pytest -q

      - name: Commitear el snapshot
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -f data/raw data/processed
          git diff --staged --quiet || git commit -m "data: snapshot $(date +%F)"
          git push

      - name: Publicar en Kaggle
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: uv run python scripts/publicar_kaggle.py
```

- [ ] **Step 3: Ajustar `.gitignore` para que el Action pueda versionar los datos**

El `.gitignore` actual excluye `data/`. El Action usa `git add -f`, pero conviene dejarlo explícito:

```
data/interim/*
!data/**/.gitkeep
.env
__pycache__/
.venv/
.ipynb_checkpoints/
.DS_Store
```

`data/raw/` y `data/processed/` dejan de estar ignorados: son el activo del proyecto y su historial es la serie de tiempo.

- [ ] **Step 4: Cargar los secretos y probar el Action a mano**

```bash
gh secret set ADZUNA_APP_ID
gh secret set ADZUNA_APP_KEY
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY
gh workflow run "Recolección semanal"
gh run watch
```
Expected: el run termina en verde y aparece un commit `data: snapshot <fecha>`.

- [ ] **Step 5: Commit**

```bash
git add .github/ scripts/ .gitignore
git commit -m "feat: recolección semanal automática y publicación en Kaggle"
```

---

### Task 21: README

**Files:**
- Modify: `README.md` (la Task 1 dejó un placeholder mínimo: `pyproject.toml` declara `readme = "README.md"` y hatchling no construye sin él)

- [ ] **Step 1: Escribir el README**

Secciones obligatorias, en este orden:

1. **Título y una línea** de qué responde el proyecto.
2. **Los hallazgos**, con los números reales que salieron de los notebooks. Tres o cuatro viñetas, sin adornos.
3. **Por qué existe este dataset** — el hallazgo §1.1 del spec: Kaggle no tenía datos salariales de México, así que se construyeron.
4. **Fuentes descartadas y por qué** — la tabla del §4 del spec, incluyendo el conteo por país de `m0sm71` como evidencia. Esta sección es deliberada: documenta que las fuentes se auditaron antes de usarse.
5. **Advertencias metodológicas** — la regla n ≥ 30, el filtro de `salary_is_predicted`, y el veredicto del modelo de imputación tal como haya quedado.
6. **Cómo reproducirlo** — `make setup`, copiar `.env.example`, `make collect`, `make build`, `make notebooks`.
7. **Estructura del repo** y **stack**.
8. **Licencia** y enlace al dataset publicado en Kaggle.

- [ ] **Step 2: Verificar que las instrucciones funcionan desde cero**

```bash
git clone <repo> /tmp/verificacion && cd /tmp/verificacion
make setup && cp .env.example .env  # llenar llaves
make build && make test
```
Expected: la suite pasa siguiendo sólo el README.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README con hallazgos, fuentes descartadas y reproducción"
```

---

### Task 22: Entrada en el portafolio

**Files:**
- Create: `~/Desktop/DA/Portafolio/notebooks-source/mercado-ia-mx-us/` (copia de los 4 notebooks)
- Modify: `~/Desktop/DA/Portafolio/portfolio-ds/scripts/parse-notebooks.mjs`
- Modify: `~/Desktop/DA/Portafolio/notebooks-source/README.md`

`src/data/projects.json` **se genera**, no se edita a mano: `npm run build` corre primero
`scripts/parse-notebooks.mjs`, que recorre `notebooks-source/`, produce **una entrada por
notebook** y toma título, resumen y portada de su mapa `PROJECT_OVERRIDES`.

- [ ] **Step 1: Copiar los notebooks**

```bash
mkdir -p ~/Desktop/DA/Portafolio/notebooks-source/mercado-ia-mx-us
cp notebooks/*.ipynb ~/Desktop/DA/Portafolio/notebooks-source/mercado-ia-mx-us/
```

- [ ] **Step 2: Extender la llave de override para carpetas multi-notebook**

En `scripts/parse-notebooks.mjs` la llave de override es la carpeta, salvo para `Telecom*`, que
usa `carpeta/archivo`. Con cuatro notebooks en una carpeta los cuatro colisionarían en la misma
llave y en el mismo slug. Cambiar la condición (alrededor de la línea 331):

```js
      const overrideKey = folder.startsWith("Telecom") || folder.startsWith("mercado-ia")
        ? `${folder}/${baseName}`
        : folder;
```

- [ ] **Step 3: Añadir los cuatro overrides**

Dentro de `PROJECT_OVERRIDES`, con el mismo shape que las entradas existentes y usando el verde
de marca `#00DF81` como en el resto del portafolio:

```js
  "mercado-ia-mx-us/01_recoleccion_y_calidad": {
    title: "Mercado de IA — Auditoría de fuentes y calidad",
    summary:
      "Por qué ningún dataset público servía: auditoría de las fuentes candidatas, detección de un dataset sintético por su distribución uniforme, y construcción de un corpus propio de vacantes de IA en México y Estados Unidos.",
    cover: { gradientFrom: "#00DF81", gradientTo: "#6366F1", icon: "search" },
  },
  "mercado-ia-mx-us/02_brecha_salarial": {
    title: "Mercado de IA — Brecha salarial México vs Estados Unidos",
    summary:
      "Cuánto paga el trabajo de IA en CDMX, Querétaro, Monterrey y Guadalajara frente a los principales metros estadounidenses. Medianas con intervalos de confianza bootstrap, ajuste por poder adquisitivo y pruebas de hipótesis.",
    cover: { gradientFrom: "#00DF81", gradientTo: "#0EA5E9", icon: "chart-bar" },
  },
  "mercado-ia-mx-us/03_skills_y_transparencia": {
    title: "Mercado de IA — Primas por skill y transparencia salarial",
    summary:
      "Qué habilidades cargan prima salarial en cada mercado, cuántas vacantes son de IA sin llamarse así, y qué tipo de empresa publica su sueldo y cuál no.",
    cover: { gradientFrom: "#6366F1", gradientTo: "#00DF81", icon: "sparkles" },
  },
  "mercado-ia-mx-us/04_modelo_imputacion": {
    title: "Mercado de IA — Modelo de imputación salarial",
    summary:
      "Gradient boosting entrenado con salarios de Estados Unidos y validado contra un hold-out de salarios mexicanos que el modelo nunca vio, con un criterio explícito para decidir si sus estimaciones se publican.",
    cover: { gradientFrom: "#0EA5E9", gradientTo: "#00DF81", icon: "cpu" },
  },
```

- [ ] **Step 4: Añadir el renglón al índice de notebooks**

En `~/Desktop/DA/Portafolio/notebooks-source/README.md`, dentro de la tabla «Índice de proyectos»,
con el mismo formato de tres columnas que los `Sp_*`:

```markdown
| [mercado-ia-mx-us](mercado-ia-mx-us/) | Mercado de trabajo de IA: México vs Estados Unidos | Proyecto propio de punta a punta: recolección vía API, construcción de un dataset publicado en Kaggle, análisis salarial por zona con intervalos de confianza y pruebas de hipótesis, y un modelo de imputación validado contra hold-out. |
```

Actualizar también la sección «Stack utilizado» del mismo README para incluir `requests`,
`pyarrow` y `uv`, que este proyecto añade a la lista existente.

- [ ] **Step 5: Regenerar y verificar que el sitio compila**

Run:
```bash
cd ~/Desktop/DA/Portafolio/portfolio-ds && npm run build
python3 -c "
import json; d=json.load(open('src/data/projects.json'))
nuevos=[p for p in d if p['folder']=='mercado-ia-mx-us']
print(f'entradas nuevas: {len(nuevos)}')
[print(' ', p['slug']) for p in nuevos]
assert len(nuevos)==4, 'deben ser 4 slugs distintos, uno por notebook'
assert len({p['slug'] for p in nuevos})==4, 'los slugs colisionaron: revisa el overrideKey'
"
```
Expected: build sin errores, 4 entradas con slugs distintos, y 18 proyectos en total.

- [ ] **Step 6: Commit**

```bash
git -C ~/Desktop/DA/Portafolio/portfolio-ds add -A
git -C ~/Desktop/DA/Portafolio/portfolio-ds commit -m "feat: proyecto mercado de IA MX vs EE.UU."
```

> Nota: `~/Desktop/DA/Portafolio/` puede no ser un repo git. Verificar con
> `git -C ~/Desktop/DA/Portafolio/portfolio-ds rev-parse --git-dir` antes de commitear; si no lo
> es, basta con dejar los archivos en su lugar.

---

## Verificación final

- [ ] `make test` — toda la suite en verde
- [ ] `make lint` — ruff y mypy limpios
- [ ] `make notebooks` — los cuatro corren de punta a punta
- [ ] El README documenta las fuentes descartadas con su evidencia
- [ ] El dataset está publicado en Kaggle y el Action corrió al menos una vez en verde
- [ ] Criterio §11.1 del spec: revisar cuántas ciudades mexicanas alcanzaron n ≥ 30. Si no llegan, el README lo dice explícitamente en vez de omitirlo
