"""Rutas HTTP. Solo reciben, validan y delegan: la lógica vive en app/services."""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencias import get_artefactos
from app.api.esquemas import RespuestaPrediccion, RespuestaSalud, SolicitudPrediccion
from app.ml.cargador import ArtefactosModelo
from app.services import prediccion

router = APIRouter()

Artefactos = Annotated[ArtefactosModelo, Depends(get_artefactos)]


@router.post("/predict", response_model=RespuestaPrediccion, tags=["predicción"],
             summary="Puntaje de riesgo y decisión para la transacción más reciente de una tarjeta")
def predecir(solicitud: SolicitudPrediccion, artefactos: Artefactos) -> RespuestaPrediccion:
    try:
        resultado = prediccion.predecir(solicitud.transacciones, artefactos)
    except prediccion.CategoriaDesconocidaError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RespuestaPrediccion(**asdict(resultado))


@router.get("/categorias", response_model=list[str], tags=["predicción"],
            summary="Categorías de comercio que acepta el modelo")
def categorias(artefactos: Artefactos) -> list[str]:
    return prediccion.categorias_validas(artefactos)


@router.get("/health", response_model=RespuestaSalud, tags=["sistema"])
def salud(artefactos: Artefactos) -> RespuestaSalud:
    return RespuestaSalud(
        estado="ok",
        modelo_cargado=artefactos is not None,
        version_modelo=artefactos.parametros.version_modelo,
    )
