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
collect: ## Una corrida completa (Adzuna + fuentes anexas) -> data/raw/
	$(RUN) python -m src.data.collect todo

.PHONY: collect-analisis
collect-analisis: ## Sólo México y EE.UU. en Adzuna: la corrida semanal barata
	$(RUN) python -m src.data.collect analisis

.PHONY: collect-calibracion
collect-calibracion: ## Sólo los países de calibración. Se levanta una vez, no cada semana
	$(RUN) python -m src.data.collect calibracion

.PHONY: collect-fuentes
collect-fuentes: ## Sólo USAJOBS y Jooble, sin gastar cuota de Adzuna
	$(RUN) python -m src.data.collect fuentes

.PHONY: build
build: ## data/raw/ -> data/processed/vacantes.parquet
	$(RUN) python -m src.features.build

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
