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
