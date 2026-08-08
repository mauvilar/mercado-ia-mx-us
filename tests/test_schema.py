import pytest

from src.data.schema import CANONICAL_COLUMNS, SchemaError, empty_frame, validate_frame


def test_empty_frame_tiene_todas_las_columnas():
    df = empty_frame()
    assert list(df.columns) == CANONICAL_COLUMNS


def test_validate_frame_acepta_un_frame_valido():
    df = empty_frame()
    validate_frame(df)  # no debe lanzar


def test_validate_frame_rechaza_columnas_faltantes():
    df = empty_frame().drop(columns=["salary_is_predicted"])
    with pytest.raises(SchemaError, match="salary_is_predicted"):
        validate_frame(df)


def test_candado_de_integridad_predicho_no_puede_ser_observado():
    """El error que hundiría el análisis entero: un salario modelado por Adzuna
    colándose al conjunto 'observado'. Ver §4.1 del spec."""
    df = empty_frame()
    df.loc[0, CANONICAL_COLUMNS] = None
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "salary_is_predicted"] = True
    df.loc[0, "salary_observed"] = True
    with pytest.raises(SchemaError, match="predicho"):
        validate_frame(df)


def test_frame_con_predicho_y_no_observado_es_valido():
    df = empty_frame()
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "salary_is_predicted"] = True
    df.loc[0, "salary_observed"] = False
    validate_frame(df)


def test_validate_frame_rechaza_tier_invalido():
    df = empty_frame()
    df.loc[0, "posting_id"] = "abc"
    df.loc[0, "tier"] = "inventado"
    with pytest.raises(SchemaError, match="tier"):
        validate_frame(df)
