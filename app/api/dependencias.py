"""Dependencias que las rutas reciben con Depends()."""

from fastapi import Request

from app.ml.cargador import ArtefactosModelo
from app.repositories.registro_predicciones import RepositorioPredicciones


def get_artefactos(request: Request) -> ArtefactosModelo:
    """Devuelve el modelo cargado una sola vez al arrancar (ver lifespan en main.py)."""
    return request.app.state.artefactos


def get_repositorio(request: Request) -> RepositorioPredicciones:
    """Devuelve el registro de predicciones creado al arrancar (ver lifespan en main.py)."""
    return request.app.state.repositorio
