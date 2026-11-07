"""Carga del CSV de Sparkov, partición temporal y ventanas de entrenamiento del modelo C v2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml import preprocesamiento as pp

COLUMNAS_CSV = ["trans_date_trans_time", "cc_num", "amt", "category", "is_fraud", "trans_num"]
PARTICIONES = ("train", "val", "test")


@dataclass
class Conjunto:
    X: np.ndarray        # (n, longitud, n_features)
    y: np.ndarray        # (n,) 0/1
    indices: np.ndarray  # filas del DataFrame que se evalúan


def cargar_transacciones(ruta_csv: Path) -> pd.DataFrame:
    """Lee el CSV, lo ordena por tarjeta y fecha y calcula hora y días desde la transacción anterior.

    Los días se calculan sobre el historial completo de la tarjeta: solo miran hacia atrás, como en producción.
    """
    df = pd.read_csv(ruta_csv, usecols=COLUMNAS_CSV)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort").reset_index(drop=True)

    por_tarjeta = df.groupby("cc_num", sort=False)
    df["posicion"] = por_tarjeta.cumcount()
    df["hora"] = df["trans_date_trans_time"].dt.hour
    df["dias_desde_anterior"] = (
        por_tarjeta["trans_date_trans_time"].diff().dt.total_seconds().div(86400).fillna(0.0)
    )
    return df


def cortes_temporales(fechas: pd.Series, frac_train: float = 0.6,
                      frac_val: float = 0.2) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fechas de corte entre entrenamiento/validación y validación/prueba (igual que en el Proyecto 1)."""
    ordenadas = np.sort(fechas.to_numpy())
    n = len(ordenadas)
    return pd.Timestamp(ordenadas[int(n * frac_train)]), pd.Timestamp(ordenadas[int(n * (frac_train + frac_val))])


def asignar_particion(df: pd.DataFrame, corte_val: pd.Timestamp, corte_test: pd.Timestamp) -> np.ndarray:
    fechas = df["trans_date_trans_time"]
    return np.select([fechas < corte_val, fechas < corte_test], ["train", "val"], default="test")


def construir_conjuntos(features: np.ndarray, posicion: np.ndarray, etiquetas: np.ndarray,
                        particion: np.ndarray, longitud: int) -> dict[str, Conjunto]:
    """Una ventana por transacción; cada transacción pertenece a la partición de su fecha.

    La ventana puede incluir transacciones previas de otra partición (son pasado conocido),
    pero la etiqueta que se predice es siempre la de la transacción de esa partición.
    """
    conjuntos = {}
    for nombre in PARTICIONES:
        indices = np.flatnonzero(particion == nombre)
        conjuntos[nombre] = Conjunto(
            X=pp.armar_ventanas(features, posicion, indices, longitud),
            y=etiquetas[indices].astype(np.float32),
            indices=indices,
        )
    return conjuntos
