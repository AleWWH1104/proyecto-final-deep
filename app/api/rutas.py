"""Rutas HTTP. Solo reciben, validan y delegan: la lógica vive en app/services."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencias import get_artefactos
from app.api.esquemas import RespuestaSalud
from app.ml.cargador import ArtefactosModelo

router = APIRouter()


@router.get("/health", response_model=RespuestaSalud, tags=["sistema"])
def salud(artefactos: Annotated[ArtefactosModelo, Depends(get_artefactos)]) -> RespuestaSalud:
    return RespuestaSalud(
        estado="ok",
        modelo_cargado=artefactos is not None,
        version_modelo=artefactos.parametros.version_modelo,
    )
