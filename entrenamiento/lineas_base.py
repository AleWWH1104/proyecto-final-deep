"""Líneas base contra las que se compara el modelo C v2.

- Modelo A: gradient boosting sobre 4 variables agregadas, sin orden (Proyecto 1).
- Modelo C v1: el artefacto entregado en el Proyecto 1, evaluado con su preprocesamiento original.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier

from app.ml import preprocesamiento as pp
from app.ml.modelo import ModeloLSTMEmbedding, puntuar

COLUMNAS_AGREGADAS = ["monto_prom_24h", "n_tx_ultima_hora", "monto_max_dia", "diversidad_comercio"]


# ---------- Modelo A ----------

def construir_agregados(df: pd.DataFrame) -> np.ndarray:
    """Agregados por tarjeta con ventanas móviles hacia atrás (incluyen la transacción actual).

    `df` debe estar ordenado por tarjeta y fecha (ver datos.cargar_transacciones).
    """
    por_fecha = df.set_index("trans_date_trans_time")
    por_tarjeta = por_fecha.groupby("cc_num", sort=True)["amt"]
    monto_prom_24h = por_tarjeta.rolling("24h").mean().to_numpy()
    n_tx_ultima_hora = por_tarjeta.rolling("1h").count().to_numpy()
    monto_max_dia = por_tarjeta.rolling("1D").max().to_numpy()

    indicadores = pd.get_dummies(por_fecha["category"], dtype=float)
    columnas = list(indicadores.columns)
    indicadores["cc_num"] = por_fecha["cc_num"].to_numpy()
    conteos = indicadores.groupby("cc_num", sort=True)[columnas].rolling("24h").sum()
    diversidad = (conteos.to_numpy() > 0).sum(axis=1)

    return np.column_stack([monto_prom_24h, n_tx_ultima_hora, monto_max_dia, diversidad]).astype(np.float64)


def entrenar_modelo_a(X_train: np.ndarray, y_train: np.ndarray, semilla: int) -> GradientBoostingClassifier:
    # Igual que en el Proyecto 1. No se escalan las variables: los árboles no dependen de la escala.
    modelo = GradientBoostingClassifier(random_state=semilla)
    modelo.fit(X_train, y_train)
    return modelo


# ---------- Modelo C v1 (Proyecto 1) ----------

def ventanas_v1(df: pd.DataFrame, particion: np.ndarray, ruta_parametros: Path) -> dict[str, np.ndarray]:
    """Reproduce el preprocesamiento del Proyecto 1: variables crudas y historial reiniciado en cada partición."""
    params = json.loads(ruta_parametros.read_text(encoding="utf-8"))
    cat2cod_v1 = params["cat2cod"]
    longitud = params["longitud_secuencia"]

    grupo = [df["cc_num"], pd.Series(particion, index=df.index)]
    posicion = df.groupby(grupo).cumcount().to_numpy()
    dias = (df.groupby(grupo)["trans_date_trans_time"].diff().dt.total_seconds().div(86400).fillna(0.0))

    features = np.column_stack([
        df["amt"].to_numpy(dtype=np.float64),
        df["category"].map(cat2cod_v1).fillna(-1).to_numpy(dtype=np.float64),
        df["hora"].to_numpy(dtype=np.float64),
        dias.to_numpy(dtype=np.float64),
    ]).astype(np.float32)

    return {
        nombre: pp.armar_ventanas(features, posicion, np.flatnonzero(particion == nombre), longitud)
        for nombre in ("val", "test")
    }


def cargar_modelo_v1(ruta_modelo: Path, ruta_parametros: Path) -> ModeloLSTMEmbedding:
    params = json.loads(ruta_parametros.read_text(encoding="utf-8"))
    modelo = ModeloLSTMEmbedding(n_features=len(params["columnas_evento"]),
                                 n_categorias=params["n_categorias"])
    modelo.load_state_dict(torch.load(ruta_modelo, map_location="cpu", weights_only=True))
    return modelo.eval()


def puntuar_v1(df: pd.DataFrame, particion: np.ndarray, directorio_v1: Path) -> dict[str, np.ndarray]:
    ruta_parametros = directorio_v1 / "preproc_params.json"
    modelo = cargar_modelo_v1(directorio_v1 / "modelo_C.pt", ruta_parametros)
    return {nombre: puntuar(modelo, X) for nombre, X in ventanas_v1(df, particion, ruta_parametros).items()}
