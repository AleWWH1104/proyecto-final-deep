"""Carga de artefactos: pesos del modelo y parámetros de preprocesamiento."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from app.ml.modelo import ModeloLSTMEmbedding
from app.ml.preprocesamiento import Escalado


@dataclass(frozen=True)
class ParametrosModelo:
    version_modelo: str
    cat2cod: dict[str, int]
    longitud_secuencia: int
    escalado: Escalado
    umbral_operativo: float
    referencia: dict  # distribución de entrenamiento/validación, para monitoreo


@dataclass(frozen=True)
class ArtefactosModelo:
    modelo: ModeloLSTMEmbedding
    parametros: ParametrosModelo


def cargar_artefactos(ruta_modelo: Path, ruta_parametros: Path) -> ArtefactosModelo:
    datos = json.loads(ruta_parametros.read_text(encoding="utf-8"))
    parametros = ParametrosModelo(
        version_modelo=datos["version_modelo"],
        cat2cod=datos["cat2cod"],
        longitud_secuencia=datos["longitud_secuencia"],
        escalado=Escalado(media=tuple(datos["escalado"]["media"]),
                          desviacion=tuple(datos["escalado"]["desviacion"])),
        umbral_operativo=float(datos["umbral_operativo"]),
        referencia=datos.get("referencia", {}),
    )

    modelo = ModeloLSTMEmbedding(**datos["arquitectura"])
    modelo.load_state_dict(torch.load(ruta_modelo, map_location="cpu", weights_only=True))
    modelo.eval()
    return ArtefactosModelo(modelo=modelo, parametros=parametros)
