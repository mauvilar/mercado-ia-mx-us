import json
import logging
from datetime import date
from unittest.mock import patch

import pandas as pd
import requests

from src.data.collect import ejecutar_corrida, recolectar_jooble, recolectar_usajobs


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


# ── Credenciales fuera de lo que se guarda ──────────────────────────────────
# El 502 de Nueva York del snapshot 2026-08-10 dejó la app_key de Adzuna en el
# manifest, que se versiona en un repo público: `str(exc)` de requests trae la URL
# completa. Estas pruebas leen lo que quedó en disco y en el log, no el valor de
# retorno, porque lo que se filtra es lo que se escribe.


def _http_error(url: str) -> requests.HTTPError:
    """El HTTPError que levanta `raise_for_status`, con el mensaje que arma requests."""
    resp = requests.Response()
    resp.status_code, resp.reason, resp.url = 502, "Bad Gateway", url
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        return exc
    raise AssertionError("raise_for_status no levantó el error")


@patch("src.data.collect.AdzunaClient")
def test_el_manifest_no_guarda_las_credenciales_de_adzuna(MockClient, tmp_path, caplog):
    url = (
        "https://api.adzuna.com/v1/api/jobs/us/search/2"
        "?app_id=id-secreto&app_key=llave-secreta&what=AI+engineer&where=New+York"
    )
    MockClient.return_value.search_all.side_effect = _http_error(url)
    with caplog.at_level(logging.WARNING):
        ejecutar_corrida(
            destino=tmp_path,
            snapshot_date=date(2026, 8, 7),
            app_id="id-secreto",
            app_key="llave-secreta",
        )

    texto = (tmp_path / "2026-08-07" / "manifest.json").read_text()
    for secreto in ("id-secreto", "llave-secreta"):
        assert secreto not in texto
        assert secreto not in caplog.text
    error = json.loads(texto)["consultas_fallidas"][0]["error"]
    assert error == (
        "502 Server Error: Bad Gateway for url: "
        "https://api.adzuna.com/v1/api/jobs/us/search/2?<redactado>"
    )


@patch("src.data.collect.JoobleClient")
@patch("src.data.collect.jooble_key", return_value="llave-jooble")
def test_el_manifest_no_guarda_la_llave_de_jooble(_llave, MockClient, tmp_path, caplog):
    """Jooble lleva la llave en la ruta, donde no la alcanza el filtro del query string."""
    MockClient.return_value.search_all.side_effect = _http_error(
        "https://jooble.org/api/llave-jooble"
    )
    with caplog.at_level(logging.WARNING):
        recolectar_jooble(destino=tmp_path, snapshot_date=date(2026, 8, 7))

    texto = (tmp_path / "2026-08-07" / "manifest_jooble.json").read_text()
    assert "llave-jooble" not in texto
    assert "llave-jooble" not in caplog.text
    assert json.loads(texto)["consultas_fallidas"][0]["error"].endswith(
        "jooble.org/api/<redactado>"
    )


@patch("src.data.collect.UsajobsClient")
@patch("src.data.collect.usajobs_credentials", return_value=("yo@ejemplo.com", "llave-usa"))
def test_el_manifest_no_guarda_las_credenciales_de_usajobs(_cred, MockClient, tmp_path, caplog):
    """USAJOBS manda la llave en cabeceras: si un error llegara a citarlas, tampoco pasan."""
    MockClient.return_value.search_all.side_effect = RuntimeError(
        "cabeceras {'User-Agent': 'yo@ejemplo.com', 'Authorization-Key': 'llave-usa'}"
    )
    with caplog.at_level(logging.WARNING):
        recolectar_usajobs(destino=tmp_path, snapshot_date=date(2026, 8, 7))

    texto = (tmp_path / "2026-08-07" / "manifest_usajobs.json").read_text()
    for secreto in ("yo@ejemplo.com", "llave-usa"):
        assert secreto not in texto
        assert secreto not in caplog.text
