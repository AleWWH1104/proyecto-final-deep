"""Monitoreo: resumen del uso real del servicio, al lado de lo que el modelo vio al entrenar.

Si la tasa de bloqueo, los montos o las categorías se alejan mucho de la referencia
(parametros.json → referencia), el tráfico ya no se parece al de entrenamiento y conviene revisar el modelo.
Las latencias son la evidencia de H3 (p95 < 300 ms).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.ml.cargador import ArtefactosModelo
from app.repositories.registro_predicciones import RepositorioPredicciones

N_ULTIMAS = 20


@dataclass(frozen=True)
class PrediccionReciente:
    id: int
    fecha: str
    puntaje: float
    decision: str


@dataclass(frozen=True)
class ResumenMonitoreo:
    total_predicciones: int
    tasa_bloqueo: float | None
    tasa_bloqueo_referencia: float | None
    monto_mediana: float | None
    monto_mediana_referencia: float | None
    latencia_p50_ms: float | None
    latencia_p95_ms: float | None
    frecuencia_categorias: dict[str, float] = field(default_factory=dict)
    frecuencia_categorias_referencia: dict[str, float] = field(default_factory=dict)
    ultimas_predicciones: list[PrediccionReciente] = field(default_factory=list)


def _redondear(valor: float, decimales: int = 4) -> float:
    return round(float(valor), decimales)


def resumir(repositorio: RepositorioPredicciones, artefactos: ArtefactosModelo) -> ResumenMonitoreo:
    referencia = artefactos.parametros.referencia
    registros = repositorio.todas()
    total = len(registros)

    # Sin predicciones todavía: solo la referencia (las métricas de uso quedan en None).
    tasa_bloqueo = monto_mediana = p50 = p95 = None
    frecuencia: dict[str, float] = {}
    if total:
        latencias = np.array([r.latencia_ms for r in registros])
        tasa_bloqueo = _redondear(sum(r.decision == "bloquear" for r in registros) / total)
        monto_mediana = _redondear(np.median([r.monto for r in registros]), 2)
        p50 = _redondear(np.percentile(latencias, 50), 2)
        p95 = _redondear(np.percentile(latencias, 95), 2)
        conteo = Counter(r.categoria for r in registros)
        frecuencia = {cat: _redondear(conteo.get(cat, 0) / total) for cat in sorted(artefactos.parametros.cat2cod)}

    return ResumenMonitoreo(
        total_predicciones=total,
        tasa_bloqueo=tasa_bloqueo,
        tasa_bloqueo_referencia=_redondear(referencia["tasa_bloqueo_val"]) if "tasa_bloqueo_val" in referencia else None,
        monto_mediana=monto_mediana,
        monto_mediana_referencia=referencia.get("monto_mediana"),
        latencia_p50_ms=p50,
        latencia_p95_ms=p95,
        frecuencia_categorias=frecuencia,
        frecuencia_categorias_referencia={
            cat: _redondear(f) for cat, f in referencia.get("frecuencia_categorias", {}).items()},
        ultimas_predicciones=[
            PrediccionReciente(id=r.id, fecha=r.fecha, puntaje=r.puntaje, decision=r.decision)
            for r in repositorio.ultimas(N_ULTIMAS)
        ],
    )
