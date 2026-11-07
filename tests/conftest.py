"""Datos sintéticos con el formato de Sparkov y un modelo entrenado con ellos.

Los tests no dependen del CSV real (1.3 M filas) ni de los artefactos entrenados: generan
un CSV pequeño, corren el pipeline completo y prueban el servicio con ese modelo.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CATEGORIAS = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
              "shopping_net", "shopping_pos", "travel"]


def generar_csv_sintetico(ruta: Path, n_tarjetas: int = 30, tx_por_tarjeta: int = 80, semilla: int = 0) -> Path:
    rng = np.random.default_rng(semilla)
    inicio = pd.Timestamp("2019-01-01")
    filas = []
    for tarjeta in range(n_tarjetas):
        cc_num = 4_000_000_000_000 + tarjeta
        # Fechas repartidas en 18 meses, ordenadas
        offsets = np.sort(rng.uniform(0, 540 * 24 * 3600, tx_por_tarjeta))
        for j, segundos in enumerate(offsets):
            filas.append({
                "trans_date_trans_time": inicio + pd.Timedelta(seconds=float(segundos)),
                "cc_num": cc_num,
                "amt": round(float(rng.lognormal(3.5, 1.0)), 2),
                "category": CATEGORIAS[rng.integers(len(CATEGORIAS))],
                "is_fraud": 0,
                "trans_num": f"t{tarjeta}_{j}",
            })
    df = pd.DataFrame(filas)
    # Fraude: montos altos en compras en línea de madrugada
    fraude = rng.random(len(df)) < 0.06
    df.loc[fraude, "is_fraud"] = 1
    df.loc[fraude, "amt"] = rng.uniform(500, 1200, fraude.sum()).round(2)
    df.loc[fraude, "category"] = rng.choice(["shopping_net", "misc_net", "grocery_pos"], fraude.sum())
    df.loc[fraude, "trans_date_trans_time"] = (
        df.loc[fraude, "trans_date_trans_time"].dt.normalize() + pd.Timedelta(hours=2))
    df = df.sort_values("trans_date_trans_time")
    df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def csv_sintetico(tmp_path_factory) -> Path:
    return generar_csv_sintetico(tmp_path_factory.mktemp("datos") / "sintetico.csv")


@pytest.fixture(scope="session")
def pipeline_entrenado(csv_sintetico, tmp_path_factory) -> dict:
    """Corre el pipeline completo (C v2 + línea base A + C v1) con 2 épocas."""
    from entrenamiento.entrenador import ConfigEntrenamiento
    from entrenamiento.entrenar import ejecutar

    base = tmp_path_factory.mktemp("modelo")
    raiz = Path(__file__).resolve().parents[1]
    resumen = ejecutar(csv_sintetico, base / "artefactos", base / "reportes",
                       ConfigEntrenamiento(epocas=2, batch_size=64),
                       dir_v1=raiz / "artefactos" / "v1_proyecto1")
    return {"resumen": resumen, "artefactos": base / "artefactos", "reportes": base / "reportes"}


@pytest.fixture
def cliente(pipeline_entrenado, tmp_path):
    """Cliente HTTP del servicio, con el modelo sintético y una base de datos temporal."""
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    os.environ["RUTA_ARTEFACTOS"] = str(pipeline_entrenado["artefactos"])
    os.environ["RUTA_BASE_DATOS"] = str(tmp_path / "predicciones.db")
    get_settings.cache_clear()
    with TestClient(app) as c:  # el "with" ejecuta el lifespan, igual que en producción
        yield c
    get_settings.cache_clear()
