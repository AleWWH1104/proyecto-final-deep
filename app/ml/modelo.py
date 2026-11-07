"""Arquitectura del Modelo C: LSTM con embedding de categoría de comercio.

La misma clase sirve para el modelo v1 del Proyecto 1 (`padding_idx=None`, sin código de relleno)
y para el v2 de producción (`padding_idx=0`: el relleno y las categorías desconocidas son un vector cero).
"""

import numpy as np
import torch
import torch.nn as nn


class ModeloLSTMEmbedding(nn.Module):
    """Entrada (batch, longitud, n_features); salida: probabilidad de fraude de la última transacción."""

    def __init__(self, n_features: int, n_categorias: int, dim_embedding: int = 4,
                 hidden_size: int = 32, indice_categoria: int = 1, padding_idx: int | None = None):
        super().__init__()
        self.indice_categoria = indice_categoria
        self.embedding = nn.Embedding(n_categorias, dim_embedding, padding_idx=padding_idx)
        self.lstm = nn.LSTM(n_features - 1 + dim_embedding, hidden_size, batch_first=True)
        self.salida = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.indice_categoria
        codigo = x[:, :, idx].long().clamp(min=0)
        emb = self.embedding(codigo)
        continuas = torch.cat([x[:, :, :idx], x[:, :, idx + 1:]], dim=2)
        _, (h_n, _) = self.lstm(torch.cat([continuas, emb], dim=2))
        return torch.sigmoid(self.salida(h_n[-1])).squeeze(1)


@torch.no_grad()
def puntuar(modelo: nn.Module, ventanas: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Puntaje de riesgo (0-1) para cada ventana."""
    modelo.eval()
    X = torch.from_numpy(np.ascontiguousarray(ventanas, dtype=np.float32))
    partes = [modelo(X[i:i + batch_size]).numpy() for i in range(0, len(X), batch_size)]
    return np.concatenate(partes) if partes else np.empty(0, dtype=np.float32)
