"""Imputación de salarios para vacantes que no los publican (§7 del spec)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAS = [
    "country",
    "seniority",
    "tier",
    "metro",
    "category",
    "title_norm",
    "remote_scope",
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
    """Entrenamiento = observados fuera de México. Hold-out = TODOS los observados de México.

    Se restringe a vacantes de IA (`tier` distinto de "fuera") porque el modelo imputa
    salarios de IA. Sin ese filtro el hold-out mexicano se llena de ruido del buscador de
    Adzuna — maestras de matemáticas y ejecutivos de canal PYME — y el veredicto de
    publicación se emitiría contra la población equivocada.
    """
    de_ia = df[df["tier"].isin(["nucleo", "anillo"])]
    observados = preparar(de_ia[de_ia["salary_observed"].fillna(False).astype(bool)].copy())
    observados = observados[observados[TARGET].notna() & (observados[TARGET] > 0)]
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
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAS,
            )
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [
            ("pre", pre),
            (
                "gbm",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=400,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(train[FEATURES], np.log(train[TARGET]))
    return ModeloSalarial(pipeline)
