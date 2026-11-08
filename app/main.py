"""Punto de entrada: crea la app, carga el modelo al arrancar y registra las rutas."""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from app.api import rutas_web
from app.api.rutas import router
from app.core.config import get_settings
from app.ml.cargador import cargar_artefactos
from app.ml.modelo import puntuar
from app.repositories.registro_predicciones import RepositorioPredicciones


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.artefactos = cargar_artefactos(settings.ruta_modelo, settings.ruta_parametros)
    # Calentamiento: la primera llamada a PyTorch es lenta; se hace aquí y no en la primera petición real.
    puntuar(app.state.artefactos.modelo, np.zeros((1, app.state.artefactos.parametros.longitud_secuencia, 5)))
    app.state.repositorio = RepositorioPredicciones(settings.ruta_base_datos)
    yield


app = FastAPI(title=get_settings().nombre_app, lifespan=lifespan)
app.include_router(router)
app.include_router(rutas_web.router)
