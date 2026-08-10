"""Compuerta día 0: mide la cobertura salarial real de Adzuna antes de construir el pipeline.

No usa el esquema canónico a propósito — su trabajo es decirnos si vale la pena escribirlo.
"""

from __future__ import annotations

import time
from collections import Counter

import requests

from src.utils.config import adzuna_credentials

BASE = "https://api.adzuna.com/v1/api/jobs"
CONSULTAS = [
    "AI engineer",
    "machine learning engineer",
    "data scientist",
    "inteligencia artificial",
]
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
                        periodos[
                            f"{pais}|{'6+cifras' if f['salary_min'] >= 100000 else '<6cifras'}"
                        ] += 1
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
