"""Entrenamiento del modelo C v2 con selección de la mejor época en validación."""

import copy
import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from app.ml.modelo import ModeloLSTMEmbedding, puntuar
from app.ml.preprocesamiento import CODIGO_RELLENO, INDICE_CATEGORIA

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigEntrenamiento:
    # Arquitectura: la misma del Modelo C del Proyecto 1
    dim_embedding: int = 4
    hidden_size: int = 32
    # Optimización: la misma del Proyecto 1
    epocas: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    semilla: int = 42


def fijar_semillas(semilla: int) -> None:
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def entrenar(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
             n_categorias: int, config: ConfigEntrenamiento) -> tuple[ModeloLSTMEmbedding, dict, list[dict]]:
    """Devuelve el modelo de la época con mejor AUC-PR en validación, su arquitectura y el historial."""
    fijar_semillas(config.semilla)
    arquitectura = {
        "n_features": int(X_train.shape[2]),
        "n_categorias": n_categorias,
        "dim_embedding": config.dim_embedding,
        "hidden_size": config.hidden_size,
        "indice_categoria": INDICE_CATEGORIA,
        "padding_idx": CODIGO_RELLENO,
    }
    modelo = ModeloLSTMEmbedding(**arquitectura)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.semilla),
    )
    criterio = nn.BCELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config.lr)

    historial, mejor_auc, mejor_estado = [], -1.0, None
    for epoca in range(1, config.epocas + 1):
        modelo.train()
        perdida_acumulada = 0.0
        for X_lote, y_lote in loader:
            optimizador.zero_grad()
            perdida = criterio(modelo(X_lote), y_lote)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item() * len(y_lote)

        scores_val = puntuar(modelo, X_val)
        p = np.clip(scores_val, 1e-7, 1 - 1e-7)
        perdida_val = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
        auc_val = float(average_precision_score(y_val, scores_val))
        historial.append({"epoca": epoca, "perdida_train": perdida_acumulada / len(y_train),
                          "perdida_val": perdida_val, "auc_pr_val": auc_val})
        log.info("Época %d/%d - pérdida train %.4f - pérdida val %.4f - AUC-PR val %.4f",
                 epoca, config.epocas, historial[-1]["perdida_train"], perdida_val, auc_val)

        if auc_val > mejor_auc:
            mejor_auc, mejor_estado = auc_val, copy.deepcopy(modelo.state_dict())

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    return modelo, arquitectura, historial


def config_a_dict(config: ConfigEntrenamiento) -> dict:
    return asdict(config)
