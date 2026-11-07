"""Configuración del servicio.

Los valores se pueden sobrescribir con variables de entorno o con un archivo .env
(por ejemplo, RUTA_ARTEFACTOS=/otra/carpeta).
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    nombre_app: str = "Servicio de riesgo de fraude"
    ruta_artefactos: Path = RAIZ_PROYECTO / "artefactos"
    ruta_base_datos: Path = RAIZ_PROYECTO / "datos" / "predicciones.db"

    model_config = SettingsConfigDict(env_file=".env")

    @property
    def ruta_modelo(self) -> Path:
        return self.ruta_artefactos / "modelo.pt"

    @property
    def ruta_parametros(self) -> Path:
        return self.ruta_artefactos / "parametros.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
