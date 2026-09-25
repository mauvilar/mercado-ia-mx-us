"""Quitar credenciales de un texto antes de guardarlo en disco o mandarlo al log.

El caso que motivó este módulo: `requests` arma el mensaje de un HTTPError con la URL
completa de la petición, y la de Adzuna lleva `app_id` y `app_key` en el query string.
`collect.py` guardaba ese mensaje tal cual en el manifest del snapshot, que se versiona
en un repo público, y así se publicó la llave. Jooble es peor: su llave va en la ruta.

Dos capas, porque ninguna basta sola:

- Patrones: cualquier query string y la llave de Jooble en su ruta. Cubren lo que se
  sabe que filtra aunque quien llama olvide pasar los secretos.
- Secretos explícitos: los valores que quien llama conoce se borran donde aparezcan,
  también en formatos que los patrones no prevén.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REDACTADO = "<redactado>"

# Un query string con al menos un par nombre=valor, hasta el primer espacio o comilla.
# No exige esquema ni host porque urllib3 cita la URL relativa:
# "Max retries exceeded with url: /v1/api/jobs/us/search/2?app_id=...".
_QUERY_STRING = re.compile(r"\?[\w.~%+-]+=[^\s#'\")]*")

# La llave de Jooble es el último tramo de la ruta. Dos formas de citarla:
# la URL absoluta de un HTTPError y la ruta suelta de un error de conexión de urllib3,
# que separa el host ("host='jooble.org', port=443): ... with url: /api/<llave>").
_RUTA_JOOBLE = re.compile(
    r"(jooble\.org(?::\d+)?/api/|host='jooble\.org'[^/]*url: /api/)[^\s/?#'\")]+"
)


def redactar(texto: str, secretos: Iterable[str | None] = ()) -> str:
    """`texto` sin query strings, sin la llave de Jooble y sin ninguno de `secretos`."""
    # De mayor a menor largo: si un secreto contiene a otro, el corto no debe partir al
    # largo y dejar visible un pedazo. Los vacíos se descartan, porque reemplazar ""
    # intercalaría la marca entre cada letra del texto.
    for secreto in sorted({s for s in secretos if s}, key=len, reverse=True):
        texto = texto.replace(secreto, REDACTADO)
    texto = _QUERY_STRING.sub(f"?{REDACTADO}", texto)
    return _RUTA_JOOBLE.sub(rf"\g<1>{REDACTADO}", texto)


def describir_error(exc: BaseException, *secretos: str | None) -> str:
    """`str(exc)` listo para guardarse en un manifest o mandarse al log."""
    return redactar(str(exc), secretos)
