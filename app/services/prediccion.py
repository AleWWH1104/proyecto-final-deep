"""Lógica de predicción: transacciones de una tarjeta → puntaje de riesgo y decisión.

No sabe nada de HTTP (eso es de app/api) ni reimplementa el preprocesamiento (usa app/ml,
el mismo código del entrenamiento).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from app.ml.cargador import ArtefactosModelo
from app.ml.modelo import puntuar
from app.ml.preprocesamiento import ventana_desde_historial
from app.repositories.registro_predicciones import RegistroPrediccion, RepositorioPredicciones


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
    id_prediccion: int | None = None


def categorias_validas(artefactos: ArtefactosModelo) -> list[str]:
    return sorted(artefactos.parametros.cat2cod)


def predecir(transacciones: Sequence[TransaccionEntrada], artefactos: ArtefactosModelo,
             repositorio: RepositorioPredicciones | None = None) -> ResultadoPrediccion:
    """Evalúa la transacción más reciente y, si hay repositorio, guarda el registro de la predicción."""
    inicio = time.perf_counter()
    parametros = artefactos.parametros

    desconocidas = sorted({t.categoria for t in transacciones} - set(parametros.cat2cod))
    if desconocidas:
        raise CategoriaDesconocidaError(
            f"Categorías no válidas: {desconocidas}. Válidas: {categorias_validas(artefactos)}")

    # Hora local tal como llega; si trae zona horaria se ignora para no mezclar fechas con y sin zona.
    fechas = [t.fecha_hora.replace(tzinfo=None) for t in transacciones]
    ventana = ventana_desde_historial(
        montos=[t.monto for t in transacciones],
        categorias=[t.categoria for t in transacciones],
        fechas=fechas,
        cat2cod=parametros.cat2cod,
        escalado=parametros.escalado,
        longitud=parametros.longitud_secuencia,
    )
    puntaje = float(puntuar(artefactos.modelo, ventana)[0])
    umbral = parametros.umbral_operativo

    resultado = ResultadoPrediccion(
        puntaje=puntaje,
        decision="bloquear" if puntaje >= umbral else "aprobar",
        umbral=umbral,
        version_modelo=parametros.version_modelo,
        latencia_ms=round((time.perf_counter() - inicio) * 1000, 2),
    )
    if repositorio is None:
        return resultado

    # La evaluada es la última al ordenar por fecha, igual que en ventana_desde_historial.
    evaluada = transacciones[sorted(range(len(fechas)), key=lambda i: fechas[i])[-1]]
    id_prediccion = repositorio.guardar(RegistroPrediccion(
        puntaje=resultado.puntaje,
        decision=resultado.decision,
        latencia_ms=resultado.latencia_ms,
        version_modelo=resultado.version_modelo,
        n_transacciones=len(transacciones),
        monto=evaluada.monto,
        categoria=evaluada.categoria,
    ))
    return replace(resultado, id_prediccion=id_prediccion)
