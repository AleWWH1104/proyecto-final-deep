"""Lógica de predicción: transacciones de una tarjeta → puntaje de riesgo y decisión.

No sabe nada de HTTP (eso es de app/api) ni reimplementa el preprocesamiento (usa app/ml,
el mismo código del entrenamiento).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.ml.cargador import ArtefactosModelo
from app.ml.modelo import puntuar
from app.ml.preprocesamiento import ventana_desde_historial


class TransaccionEntrada(Protocol):
    monto: float
    categoria: str
    fecha_hora: datetime


class CategoriaDesconocidaError(ValueError):
    """La categoría no estaba en los datos de entrenamiento; el modelo no tiene cómo representarla."""


@dataclass(frozen=True)
class ResultadoPrediccion:
    puntaje: float
    decision: str
    umbral: float
    version_modelo: str
    latencia_ms: float


def categorias_validas(artefactos: ArtefactosModelo) -> list[str]:
    return sorted(artefactos.parametros.cat2cod)


def predecir(transacciones: Sequence[TransaccionEntrada], artefactos: ArtefactosModelo) -> ResultadoPrediccion:
    inicio = time.perf_counter()
    parametros = artefactos.parametros

    desconocidas = sorted({t.categoria for t in transacciones} - set(parametros.cat2cod))
    if desconocidas:
        raise CategoriaDesconocidaError(
            f"Categorías no válidas: {desconocidas}. Válidas: {categorias_validas(artefactos)}")

    ventana = ventana_desde_historial(
        montos=[t.monto for t in transacciones],
        categorias=[t.categoria for t in transacciones],
        # Hora local tal como llega; si trae zona horaria se ignora para no mezclar fechas con y sin zona.
        fechas=[t.fecha_hora.replace(tzinfo=None) for t in transacciones],
        cat2cod=parametros.cat2cod,
        escalado=parametros.escalado,
        longitud=parametros.longitud_secuencia,
    )
    puntaje = float(puntuar(artefactos.modelo, ventana)[0])
    umbral = parametros.umbral_operativo

    return ResultadoPrediccion(
        puntaje=puntaje,
        decision="bloquear" if puntaje >= umbral else "aprobar",
        umbral=umbral,
        version_modelo=parametros.version_modelo,
        latencia_ms=round((time.perf_counter() - inicio) * 1000, 2),
    )
