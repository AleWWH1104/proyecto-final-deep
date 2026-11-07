from datetime import datetime

import numpy as np

from app.ml import preprocesamiento as pp
from entrenamiento import datos

CAT2COD = {"food_dining": 1, "shopping_net": 2}
ESCALADO = pp.Escalado(media=(3.0, 12.0, 0.5), desviacion=(1.0, 6.0, 0.4))


def test_relleno_no_se_confunde_con_una_categoria():
    fechas = [datetime(2020, 1, 1, 10), datetime(2020, 1, 2, 22)]
    ventana = pp.ventana_desde_historial([10.0, 900.0], ["food_dining", "shopping_net"], fechas,
                                         CAT2COD, ESCALADO, longitud=5)

    assert ventana.shape == (1, 5, 5)
    relleno, reales = ventana[0, :3], ventana[0, 3:]
    assert np.all(relleno == 0)                       # categoría 0 y es_real 0
    assert list(reales[:, pp.INDICE_CATEGORIA]) == [1, 2]
    assert np.all(reales[:, 4] == 1)


def test_categoria_desconocida_usa_el_codigo_de_relleno():
    assert pp.codificar_categorias(["no_existe"], CAT2COD)[0] == pp.CODIGO_RELLENO


def test_orden_de_entrada_no_importa():
    montos, categorias = [50.0, 20.0, 700.0], ["food_dining", "food_dining", "shopping_net"]
    fechas = [datetime(2020, 1, 2), datetime(2020, 1, 1), datetime(2020, 1, 3, 3)]
    desordenada = pp.ventana_desde_historial(montos, categorias, fechas, CAT2COD, ESCALADO, 4)
    orden = [1, 0, 2]
    ordenada = pp.ventana_desde_historial([montos[i] for i in orden], [categorias[i] for i in orden],
                                          [fechas[i] for i in orden], CAT2COD, ESCALADO, 4)
    np.testing.assert_array_equal(desordenada, ordenada)


def test_servicio_y_entrenamiento_arman_la_misma_ventana(csv_sintetico):
    """La ventana que arma la API para una tarjeta es idéntica a la del entrenamiento."""
    df = datos.cargar_transacciones(csv_sintetico)
    cat2cod = pp.construir_cat2cod(df["category"])
    escalado = pp.ajustar_escalado(df["amt"], df["hora"], df["dias_desde_anterior"])
    features = pp.calcular_features(df["amt"], pp.codificar_categorias(df["category"], cat2cod),
                                    df["hora"], df["dias_desde_anterior"], escalado)

    tarjeta = df[df["cc_num"] == df["cc_num"].iloc[0]]
    fila = tarjeta.index[25]  # transacción 26 de la tarjeta: la ventana de 10 está completa
    desde_entrenamiento = pp.armar_ventanas(features, df["posicion"].to_numpy(), np.array([fila]), 10)

    historial = tarjeta.loc[:fila]
    desde_servicio = pp.ventana_desde_historial(
        historial["amt"].tolist(), historial["category"].tolist(),
        historial["trans_date_trans_time"].dt.to_pydatetime().tolist(), cat2cod, escalado, 10)

    np.testing.assert_allclose(desde_servicio, desde_entrenamiento, rtol=1e-6)


def test_escalado_estandariza_entrenamiento():
    rng = np.random.default_rng(1)
    montos, horas, dias = rng.lognormal(3, 1, 500), rng.integers(0, 24, 500), rng.exponential(1, 500)
    escalado = pp.ajustar_escalado(montos, horas, dias)
    f = pp.calcular_features(montos, np.ones(500), horas, dias, escalado)
    continuas = f[:, [0, 2, 3]]
    np.testing.assert_allclose(continuas.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(continuas.std(axis=0), 1, atol=1e-4)
