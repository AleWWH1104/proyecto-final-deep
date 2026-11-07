"""Punto de entrada: crea la app, carga el modelo al arrancar y registra las rutas."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.rutas import router
from app.core.config import get_settings
from app.ml.cargador import cargar_artefactos
from app.repositories.registro_predicciones import RepositorioPredicciones


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.artefactos = cargar_artefactos(settings.ruta_modelo, settings.ruta_parametros)
    app.state.repositorio = RepositorioPredicciones(settings.ruta_base_datos)
    yield


app = FastAPI(title=get_settings().nombre_app, lifespan=lifespan)
app.include_router(router)
