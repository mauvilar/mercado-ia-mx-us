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
