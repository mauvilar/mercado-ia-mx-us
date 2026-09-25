"""Ningún mensaje de error que se guarde o se registre puede llevar una credencial.

Los errores se arman con los mismos objetos de `requests` y `urllib3` que produce una
corrida real (HTTPError desde `raise_for_status`, MaxRetryError envuelto en
ConnectionError), para que la prueba vea el formato exacto y no una imitación.
"""

import urllib3
from requests import ConnectionError, HTTPError, Request, Response
from urllib3.exceptions import MaxRetryError

from src.data.adzuna import BASE as BASE_ADZUNA
from src.data.jooble import BASE as BASE_JOOBLE
from src.utils.redaccion import REDACTADO, describir_error, redactar

APP_ID = "idfalso1"
APP_KEY = "0123456789abcdef0123456789abcdef"
LLAVE_JOOBLE = "aaaa1111-bb22-cc33-dd44-eeeeee555555"


def _url_adzuna() -> str:
    """La URL tal como la arma `requests` para AdzunaClient.search."""
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 50,
        "what": "AI engineer",
        "where": "New York",
    }
    return str(Request("GET", f"{BASE_ADZUNA}/us/search/2", params=params).prepare().url)


def _http_error(url: str, status: int = 502, reason: str = "Bad Gateway") -> HTTPError:
    """El HTTPError que levanta `raise_for_status`, con su mensaje verdadero."""
    resp = Response()
    resp.status_code = status
    resp.reason = reason
    resp.url = url
    try:
        resp.raise_for_status()
    except HTTPError as exc:
        return exc
    raise AssertionError("raise_for_status no levantó el error")


def _error_de_conexion(host: str, ruta: str) -> ConnectionError:
    """Lo que levanta `requests` cuando el servidor no contesta: urllib3 cita la ruta."""
    pool = urllib3.HTTPSConnectionPool(host, 443)
    return ConnectionError(MaxRetryError(pool, ruta, reason=OSError("sin red")))


def _sin_credenciales(texto: str) -> bool:
    return all(s not in texto for s in (APP_ID, APP_KEY, LLAVE_JOOBLE))


def test_el_error_de_adzuna_trae_la_llave_si_no_se_redacta():
    """Premisa de todo el archivo: sin redactar, el mensaje filtra las dos credenciales."""
    mensaje = str(_http_error(_url_adzuna()))
    assert APP_ID in mensaje
    assert APP_KEY in mensaje


def test_http_error_de_adzuna_sin_credenciales():
    mensaje = describir_error(_http_error(_url_adzuna()))
    assert _sin_credenciales(mensaje)
    # Lo útil para depurar sobrevive: el código, el host y la ruta.
    assert mensaje.startswith("502 Server Error: Bad Gateway for url: ")
    assert mensaje.endswith(f"{BASE_ADZUNA}/us/search/2?{REDACTADO}")


def test_error_de_conexion_de_adzuna_sin_credenciales():
    """urllib3 cita la URL relativa, sin esquema ni host: el query string igual se va."""
    ruta = _url_adzuna().removeprefix("https://api.adzuna.com")
    mensaje = describir_error(_error_de_conexion("api.adzuna.com", ruta))
    assert _sin_credenciales(mensaje)
    assert f"/v1/api/jobs/us/search/2?{REDACTADO} (Caused by" in mensaje


def test_http_error_de_jooble_sin_la_llave_en_la_ruta():
    """Jooble no usa query string: la llave es el último tramo de la ruta."""
    mensaje = describir_error(_http_error(f"{BASE_JOOBLE}/{LLAVE_JOOBLE}", 403, "Forbidden"))
    assert _sin_credenciales(mensaje)
    assert mensaje.endswith(f"jooble.org/api/{REDACTADO}")


def test_error_de_conexion_de_jooble_sin_la_llave():
    mensaje = describir_error(_error_de_conexion("jooble.org", f"/api/{LLAVE_JOOBLE}"))
    assert _sin_credenciales(mensaje)
    assert f"url: /api/{REDACTADO} (Caused by" in mensaje


def test_los_secretos_explicitos_se_borran_donde_aparezcan():
    """Segunda capa: un formato que ningún patrón prevé tampoco los deja pasar."""
    texto = f"cabecera Authorization-Key: {APP_KEY}; usuario {APP_ID}"
    limpio = redactar(texto, [APP_KEY, APP_ID])
    assert _sin_credenciales(limpio)
    assert limpio == f"cabecera Authorization-Key: {REDACTADO}; usuario {REDACTADO}"


def test_un_secreto_vacio_o_nulo_no_borra_todo():
    """`str.replace("", x)` intercalaría x entre cada letra: los vacíos se ignoran."""
    assert redactar("sin nada que ocultar", ["", None]) == "sin nada que ocultar"


def test_un_mensaje_sin_credenciales_queda_igual():
    mensaje = "Adzuna devolvió 429 tras 3 intentos (us/New York)"
    assert redactar(mensaje) == mensaje


def test_redactar_es_idempotente():
    una = describir_error(_http_error(_url_adzuna()), APP_ID, APP_KEY)
    assert redactar(una, [APP_ID, APP_KEY]) == una
