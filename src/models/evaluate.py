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
    modelo: ModeloSalarial,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    umbral_mdape: float = UMBRAL_MDAPE,
) -> dict[str, Any]:
    if holdout.empty:
        return {
            "n_holdout": 0,
            "mdape_modelo": float("nan"),
            "mdape_baseline": float("nan"),
            "supera_baseline": False,
            "publicable": False,
            "veredicto": (
                "Sin salarios mexicanos observados: no hay con qué validar. No se publica."
            ),
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
        "n_holdout": int(len(holdout)),
        "mdape_modelo": e_modelo,
        "mdape_baseline": e_baseline,
        "supera_baseline": supera,
        "publicable": publicable,
        "veredicto": veredicto,
    }
