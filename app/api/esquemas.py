"""Esquemas de entrada y salida de la API (contrato con el cliente, ver docs/plan_trabajo.md)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_TRANSACCIONES = 20


class Transaccion(BaseModel):
    monto: float = Field(gt=0, description="Monto de la transacción en quetzales.")
    categoria: str = Field(description="Categoría de comercio; ver GET /categorias.")
    fecha_hora: datetime = Field(description="Fecha y hora local de la transacción (ISO 8601, sin zona horaria).")


class SolicitudPrediccion(BaseModel):
    transacciones: list[Transaccion] = Field(
        min_length=1, max_length=MAX_TRANSACCIONES,
        description="Transacciones recientes de UNA tarjeta. Se evalúa la más reciente; "
                    "el modelo usa hasta 10 (envíe 11 para que el intervalo entre la primera y la anterior sea exacto).",
    )

    model_config = ConfigDict(json_schema_extra={"examples": [{"transacciones": [
        {"monto": 54.71, "categoria": "food_dining", "fecha_hora": "2020-03-11T13:19:42"},
        {"monto": 152.81, "categoria": "shopping_pos", "fecha_hora": "2020-03-12T06:54:16"},
        {"monto": 356.49, "categoria": "grocery_pos", "fecha_hora": "2020-03-14T02:34:35"},
    ]}]})


class RespuestaPrediccion(BaseModel):
    id_prediccion: int | None = Field(default=None, description="Id del registro de la predicción.")
    puntaje: float = Field(ge=0, le=1, description="Riesgo de fraude de la transacción más reciente.")
    decision: Literal["aprobar", "bloquear"]
    umbral: float = Field(description="Se bloquea si puntaje >= umbral.")
    version_modelo: str
    latencia_ms: float


class PrediccionReciente(BaseModel):
    id: int
    fecha: str = Field(description="Momento en que se hizo la predicción (UTC, ISO 8601).")
    puntaje: float
    decision: Literal["aprobar", "bloquear"]


class RespuestaMonitoreo(BaseModel):
    total_predicciones: int
    tasa_bloqueo: float | None = Field(description="Fracción de predicciones bloqueadas.")
    tasa_bloqueo_referencia: float | None = Field(description="Tasa de bloqueo en validación.")
    monto_mediana: float | None
    monto_mediana_referencia: float | None = Field(description="Mediana de montos en entrenamiento.")
    latencia_p50_ms: float | None
    latencia_p95_ms: float | None = Field(description="Evidencia de H3: debe ser < 300 ms.")
    frecuencia_categorias: dict[str, float]
    frecuencia_categorias_referencia: dict[str, float]
    ultimas_predicciones: list[PrediccionReciente]


class RespuestaSalud(BaseModel):
    estado: str
    modelo_cargado: bool
    version_modelo: str
