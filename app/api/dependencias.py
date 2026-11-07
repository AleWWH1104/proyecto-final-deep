"""Dependencias que las rutas reciben con Depends()."""

from fastapi import Request

from app.ml.cargador import ArtefactosModelo


def get_artefactos(request: Request) -> ArtefactosModelo:
    """Devuelve el modelo cargado una sola vez al arrancar (ver lifespan en main.py)."""
    return request.app.state.artefactos
