from src.features.extract import extraer_seniority, extraer_skills


def test_extrae_skills_por_alias():
    texto = "Buscamos experiencia en RAG, PyTorch y bases de datos vectoriales"
    out = extraer_skills(texto)
    assert "rag" in out
    assert "pytorch" in out
    assert "vector_db" in out


def test_extrae_skills_en_ingles_y_espanol():
    assert "llm" in extraer_skills("experience with large language models")
    assert "llm" in extraer_skills("experiencia con modelos de lenguaje")


def test_no_duplica_skills():
    out = extraer_skills("PyTorch, pytorch, TORCH")
    assert out.count("pytorch") == 1


def test_devuelve_lista_ordenada_para_ser_determinista():
    a = extraer_skills("pytorch rag llm")
    b = extraer_skills("llm rag pytorch")
    assert a == b


def test_seniority_desde_el_titulo():
    assert extraer_seniority("Senior AI Engineer", "") == "sr"
    assert extraer_seniority("Junior ML Engineer", "") == "jr"
    assert extraer_seniority("Staff Machine Learning Engineer", "") == "staff"
    assert extraer_seniority("Lead AI Engineer", "") == "lead"
    assert extraer_seniority("Principal AI Engineer", "") == "principal"


def test_seniority_en_espanol():
    assert extraer_seniority("Ingeniero de IA Senior", "") == "sr"
    assert extraer_seniority("Ingeniero de IA Jr.", "") == "jr"


def test_seniority_cae_a_la_descripcion_si_el_titulo_no_dice():
    assert extraer_seniority("AI Engineer", "Se requieren 8+ años de experiencia") == "sr"
    assert extraer_seniority("AI Engineer", "Se requieren 1-2 años de experiencia") == "jr"
    assert extraer_seniority("AI Engineer", "Se requieren 4 años de experiencia") == "mid"


def test_seniority_desconocido_cuando_no_hay_senal():
    assert extraer_seniority("AI Engineer", "Únete a nuestro equipo") == "unknown"


def test_el_titulo_gana_sobre_la_descripcion():
    assert extraer_seniority("Junior AI Engineer", "10 años de experiencia") == "jr"
