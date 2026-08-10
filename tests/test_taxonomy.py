from src.features.taxonomy import clasificar


def test_titulo_explicito_es_nucleo():
    assert clasificar("AI Engineer", skills=[]) == "nucleo"
    assert clasificar("Machine Learning Engineer", skills=[]) == "nucleo"
    assert clasificar("Ingeniero de Inteligencia Artificial", skills=[]) == "nucleo"
    assert clasificar("LLM Engineer", skills=[]) == "nucleo"


def test_titulo_generico_con_dos_skills_de_ia_es_anillo():
    """El hallazgo §1.4 del spec: en México el trabajo de IA viene disfrazado."""
    assert clasificar("Senior Software Engineer", skills=["rag", "pytorch"]) == "anillo"


def test_titulo_generico_con_una_sola_skill_queda_fuera():
    """Umbral de 2 para no cazar vacantes que mencionan 'AI' de relleno."""
    assert clasificar("Senior Software Engineer", skills=["pytorch"]) == "fuera"


def test_skills_de_soporte_no_disparan_el_anillo():
    assert clasificar("Backend Developer", skills=["python", "sql", "docker"]) == "fuera"


def test_nucleo_gana_sobre_anillo():
    assert clasificar("AI Engineer", skills=["rag", "llm"]) == "nucleo"


def test_termino_de_ia_sin_termino_de_rol_no_es_nucleo():
    """'AI Product Manager' no es un rol de ingeniería de IA."""
    assert clasificar("AI Product Manager", skills=[]) == "fuera"


def test_no_confunde_palabras_que_contienen_ai():
    """'Maintenance' contiene 'ai' — el \\b del regex debe evitar el falso positivo."""
    assert clasificar("Maintenance Engineer", skills=[]) == "fuera"
    assert clasificar("Retail Engineer", skills=[]) == "fuera"


def test_gen_ai_no_caza_nitrogen_ni_hydrogen():
    """Regresión: 'gen ?ai' sin anclas casaba dentro de 'nitroGEN AIr'. Querétaro y
    Monterrey están llenos de vacantes industriales con esas palabras."""
    assert clasificar("Nitrogen Air Systems Engineer", skills=[]) == "fuera"
    assert clasificar("Hydrogen Airflow Engineer", skills=[]) == "fuera"
    assert clasificar("GenAI Engineer", skills=[]) == "nucleo"


def test_reconoce_el_titulo_con_puntos():
    """Regresión: '\\ba\\.i\\.\\b' era un patrón muerto — el \\b tras un punto no casa nunca."""
    assert clasificar("Especialista en A.I.", skills=[]) == "nucleo"
