"""Preprocesamiento compartido por el entrenamiento y el servicio.

El entrenamiento y la API usan exactamente estas funciones para convertir transacciones en la
ventana que recibe el modelo. Si se usaran dos implementaciones, podrían diferir sin que nadie lo note.

Cada transacción se describe con 5 variables, en este orden:
    0. log_monto   log(1 + monto), estandarizado
    1. categoria   código entero de la categoría de comercio (1..N; 0 = relleno / desconocida)
    2. hora        hora del día (0-23), estandarizada
    3. log_dias    log(1 + días desde la transacción anterior de la tarjeta), estandarizado
    4. es_real     1 si es una transacción real, 0 si es relleno
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

COLUMNAS_FEATURES = ("log_monto", "categoria", "hora", "log_dias", "es_real")
INDICE_CATEGORIA = 1
CODIGO_RELLENO = 0


@dataclass(frozen=True)
class Escalado:
    """Media y desviación de [log_monto, hora, log_dias], calculadas solo con entrenamiento."""

    media: tuple[float, float, float]
    desviacion: tuple[float, float, float]


def _continuas(montos, horas, dias) -> np.ndarray:
    return np.column_stack([
        np.log1p(np.asarray(montos, dtype=np.float64)),
        np.asarray(horas, dtype=np.float64),
        np.log1p(np.asarray(dias, dtype=np.float64)),
    ])


def ajustar_escalado(montos, horas, dias) -> Escalado:
    continuas = _continuas(montos, horas, dias)
    desviacion = continuas.std(axis=0)
    desviacion[desviacion == 0] = 1.0
    return Escalado(media=tuple(continuas.mean(axis=0).tolist()),
                    desviacion=tuple(desviacion.tolist()))


def construir_cat2cod(categorias) -> dict[str, int]:
    """Asigna 1..N en orden alfabético; el 0 queda reservado para relleno."""
    return {c: i + 1 for i, c in enumerate(sorted(set(categorias)))}


def codificar_categorias(categorias, cat2cod: dict[str, int]) -> np.ndarray:
    return np.array([cat2cod.get(c, CODIGO_RELLENO) for c in categorias], dtype=np.float64)


def calcular_features(montos, codigos, horas, dias, escalado: Escalado) -> np.ndarray:
    """Devuelve una matriz (n, 5) float32 con las variables de cada transacción."""
    continuas = (_continuas(montos, horas, dias) - np.array(escalado.media)) / np.array(escalado.desviacion)
    n = continuas.shape[0]
    return np.column_stack([
        continuas[:, 0],
        np.asarray(codigos, dtype=np.float64),
        continuas[:, 1],
        continuas[:, 2],
        np.ones(n),
    ]).astype(np.float32)


def armar_ventanas(features: np.ndarray, posicion_en_tarjeta: np.ndarray,
                   indices: np.ndarray, longitud: int) -> np.ndarray:
    """Arma una ventana con las últimas `longitud` transacciones de la tarjeta para cada fila de `indices`.

    `features` debe estar ordenado por tarjeta y por fecha. `posicion_en_tarjeta[i]` indica cuántas
    transacciones previas de la misma tarjeta hay antes de la fila i. La última posición de cada
    ventana es la transacción evaluada; si faltan transacciones previas, se rellena con ceros al inicio.
    """
    indices = np.asarray(indices)
    ventanas = np.zeros((len(indices), longitud, features.shape[1]), dtype=features.dtype)
    posiciones = np.asarray(posicion_en_tarjeta)[indices]
    for atras in range(longitud):
        validos = posiciones >= atras
        ventanas[validos, longitud - 1 - atras, :] = features[indices[validos] - atras]
    return ventanas


def horas_y_dias(fechas: Sequence[datetime]) -> tuple[np.ndarray, np.ndarray]:
    """Hora del día y días desde la transacción anterior (0 para la primera). `fechas` ya ordenadas."""
    horas = np.array([f.hour for f in fechas], dtype=np.float64)
    dias = np.zeros(len(fechas), dtype=np.float64)
    for i in range(1, len(fechas)):
        dias[i] = (fechas[i] - fechas[i - 1]).total_seconds() / 86400
    return horas, dias


def ventana_desde_historial(montos: Sequence[float], categorias: Sequence[str],
                            fechas: Sequence[datetime], cat2cod: dict[str, int],
                            escalado: Escalado, longitud: int) -> np.ndarray:
    """Ventana (1, longitud, 5) para evaluar la transacción más reciente de un historial.

    Se usa en el servicio. Ordena por fecha; la transacción más reciente es la que se evalúa.
    """
    orden = sorted(range(len(fechas)), key=lambda i: fechas[i])
    fechas_ord = [fechas[i] for i in orden]
    horas, dias = horas_y_dias(fechas_ord)
    features = calcular_features(
        [montos[i] for i in orden],
        codificar_categorias([categorias[i] for i in orden], cat2cod),
        horas, dias, escalado,
    )
    n = len(fechas_ord)
    return armar_ventanas(features, np.arange(n), np.array([n - 1]), longitud)
