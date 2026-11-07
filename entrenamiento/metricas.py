"""Métricas de clasificación y de costo del negocio."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

# Rejilla de umbrales: 0.010, 0.015, ..., 0.990 (la misma resolución que en el Proyecto 1)
UMBRALES = np.round(np.linspace(0.01, 0.99, 197), 3)


@dataclass(frozen=True)
class Costos:
    fraude_no_detectado: float = 4200.0
    bloqueo_legitimo: float = 180.0


def metricas_en_umbral(y: np.ndarray, scores: np.ndarray, umbral: float, costos: Costos) -> dict:
    y = np.asarray(y).astype(int)
    bloquea = np.asarray(scores) >= umbral
    vp = int(np.sum(bloquea & (y == 1)))
    fp = int(np.sum(bloquea & (y == 0)))
    fn = int(np.sum(~bloquea & (y == 1)))
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "umbral": float(umbral),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fraudes_detectados": vp,
        "fraudes_no_detectados": fn,
        "bloqueos_legitimos": fp,
        "tasa_bloqueo": float(bloquea.mean()),
        "costo": fn * costos.fraude_no_detectado + fp * costos.bloqueo_legitimo,
    }


def curva_costo(y: np.ndarray, scores: np.ndarray, costos: Costos) -> list[dict]:
    return [metricas_en_umbral(y, scores, u, costos) for u in UMBRALES]


def elegir_umbral(y_val: np.ndarray, scores_val: np.ndarray, costos: Costos) -> float:
    """Umbral de mínimo costo en validación. Nunca se elige con el conjunto de prueba."""
    curva = curva_costo(y_val, scores_val, costos)
    return min(curva, key=lambda r: r["costo"])["umbral"]


def costo_sin_modelo(y: np.ndarray, costos: Costos) -> float:
    """Dejar pasar todas las transacciones: cada fraude se paga completo."""
    return float(np.sum(y) * costos.fraude_no_detectado)


def resumen_modelo(y_val, scores_val, y_test, scores_test, costos: Costos) -> dict:
    """Elige el umbral en validación y reporta validación y prueba con ese umbral fijo."""
    umbral = elegir_umbral(y_val, scores_val, costos)
    return {
        "umbral": umbral,
        "validacion": {"auc_pr": float(average_precision_score(y_val, scores_val)),
                       **metricas_en_umbral(y_val, scores_val, umbral, costos)},
        "prueba": {"auc_pr": float(average_precision_score(y_test, scores_test)),
                   **metricas_en_umbral(y_test, scores_test, umbral, costos)},
        "prueba_umbral_0_5": metricas_en_umbral(y_test, scores_test, 0.5, costos),
    }
