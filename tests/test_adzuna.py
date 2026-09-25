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


@patch("src.data.adzuna.requests.get")
def test_where_vacio_se_omite_para_buscar_a_nivel_nacional(mock_get):
    """Así se recolectan los países de calibración: sin detalle de ciudad.

    Mandar where="" explícitamente devuelve cero resultados, así que el parámetro
    tiene que desaparecer de la petición, no ir vacío.
    """
    mock_get.return_value = _resp({"results": []})
    AdzunaClient("id", "key").search("br", what="AI engineer", where="", page=1)
    params = mock_get.call_args[1]["params"]
    assert "where" not in params
    assert params["what"] == "AI engineer"


@patch("src.data.adzuna.requests.get")
def test_where_con_valor_si_viaja_en_la_peticion(mock_get):
    mock_get.return_value = _resp({"results": []})
    AdzunaClient("id", "key").search("mx", what="x", where="Monterrey", page=1)
    assert mock_get.call_args[1]["params"]["where"] == "Monterrey"


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
