import json

import numpy as np
import pandas as pd

from entrenamiento import datos
from entrenamiento.metricas import Costos, elegir_umbral, metricas_en_umbral


def test_metricas_y_costo_en_un_caso_a_mano():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    m = metricas_en_umbral(y, scores, 0.5, Costos(fraude_no_detectado=100, bloqueo_legitimo=10))
    # Bloquea 0.9 (fraude) y 0.8 (legítima): 1 VP, 1 FP, 1 FN
    assert (m["fraudes_detectados"], m["bloqueos_legitimos"], m["fraudes_no_detectados"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["costo"] == 1 * 100 + 1 * 10


def test_umbral_barato_cuando_el_fraude_es_caro():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.3, 0.2, 0.1, 0.1])
    # Dejar pasar el fraude cuesta 1000; bloquear una legítima, 1: conviene bajar el umbral hasta 0.3
    assert elegir_umbral(y, scores, Costos(1000, 1)) <= 0.3


def test_particion_temporal_no_mezcla_fechas(csv_sintetico):
    df = datos.cargar_transacciones(csv_sintetico)
    corte_val, corte_test = datos.cortes_temporales(df["trans_date_trans_time"])
    particion = datos.asignar_particion(df, corte_val, corte_test)
    fechas = df["trans_date_trans_time"]
    assert fechas[particion == "train"].max() < fechas[particion == "val"].min()
    assert fechas[particion == "val"].max() < fechas[particion == "test"].min()
    assert abs((particion == "train").mean() - 0.6) < 0.01


def test_dias_desde_anterior_solo_mira_hacia_atras(csv_sintetico):
    df = datos.cargar_transacciones(csv_sintetico)
    primeras = df[df["posicion"] == 0]
    assert (primeras["dias_desde_anterior"] == 0).all()
    assert (df["dias_desde_anterior"] >= 0).all()


def test_pipeline_genera_artefactos_y_evalua_hipotesis(pipeline_entrenado):
    resumen = pipeline_entrenado["resumen"]
    assert set(resumen["resultados"]) == {"C_v2", "A", "C_v1", "sin_modelo"}
    assert set(resumen["hipotesis"]) == {"H1", "H2"}

    params = json.loads((pipeline_entrenado["artefactos"] / "parametros.json").read_text(encoding="utf-8"))
    assert min(params["cat2cod"].values()) == 1  # el 0 queda para relleno
    assert params["arquitectura"]["padding_idx"] == 0
    assert 0 < params["umbral_operativo"] < 1

    for figura in ("historial_entrenamiento.png", "costo_vs_umbral.png", "costo_por_modelo.png"):
        assert (pipeline_entrenado["reportes"] / figura).stat().st_size > 0


def test_umbral_se_elige_en_validacion(pipeline_entrenado):
    c_v2 = pipeline_entrenado["resumen"]["resultados"]["C_v2"]
    assert c_v2["umbral"] == c_v2["validacion"]["umbral"] == c_v2["prueba"]["umbral"]
