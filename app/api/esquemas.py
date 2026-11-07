"""Esquemas de entrada y salida de la API (contrato con el cliente)."""

from pydantic import BaseModel


class RespuestaSalud(BaseModel):
    estado: str
    modelo_cargado: bool
    version_modelo: str
