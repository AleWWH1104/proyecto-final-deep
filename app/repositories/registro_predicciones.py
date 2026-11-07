"""Registro de predicciones en SQLite: guardar cada predicción y leerla para el monitoreo.

Solo persistencia, sin lógica de negocio. No guarda datos personales: el contrato de la API no
incluye número de tarjeta ni nada que identifique al cliente.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS predicciones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha           TEXT    NOT NULL,
    puntaje         REAL    NOT NULL,
    decision        TEXT    NOT NULL,
    latencia_ms     REAL    NOT NULL,
    version_modelo  TEXT    NOT NULL,
    n_transacciones INTEGER NOT NULL,
    monto           REAL    NOT NULL,
    categoria       TEXT    NOT NULL
)
"""


@dataclass(frozen=True)
class RegistroPrediccion:
    """Una predicción tal como queda guardada (monto y categoría son de la transacción evaluada)."""
    puntaje: float
    decision: str
    latencia_ms: float
    version_modelo: str
    n_transacciones: int
    monto: float
    categoria: str
    id: int | None = None
    fecha: str | None = None


class RepositorioPredicciones:
    def __init__(self, ruta_base_datos: Path):
        self.ruta = Path(ruta_base_datos)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self._conectar() as conexion:
            conexion.execute(ESQUEMA)

    def _conectar(self) -> sqlite3.Connection:
        # Una conexión por operación: FastAPI atiende las rutas síncronas en varios hilos.
        conexion = sqlite3.connect(self.ruta)
        conexion.row_factory = sqlite3.Row
        return conexion

    def guardar(self, registro: RegistroPrediccion) -> int:
        fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conectar() as conexion:
            cursor = conexion.execute(
                "INSERT INTO predicciones (fecha, puntaje, decision, latencia_ms, version_modelo, "
                "n_transacciones, monto, categoria) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (fecha, registro.puntaje, registro.decision, registro.latencia_ms, registro.version_modelo,
                 registro.n_transacciones, registro.monto, registro.categoria),
            )
            return cursor.lastrowid

    def todas(self) -> list[RegistroPrediccion]:
        with self._conectar() as conexion:
            filas = conexion.execute("SELECT * FROM predicciones ORDER BY id").fetchall()
        return [RegistroPrediccion(**dict(fila)) for fila in filas]

    def ultimas(self, n: int) -> list[RegistroPrediccion]:
        with self._conectar() as conexion:
            filas = conexion.execute("SELECT * FROM predicciones ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [RegistroPrediccion(**dict(fila)) for fila in filas]
