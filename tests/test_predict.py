import json
import os
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = RAIZ / "docs" / "ejemplos"


def _ejemplo(nombre: str) -> dict:
    return json.loads((EJEMPLOS / f"{nombre}.json").read_text(encoding="utf-8"))


def _una_transaccion(**cambios) -> dict:
    tx = {"monto": 25.0, "categoria": "food_dining", "fecha_hora": "2020-05-01T12:00:00"}
    return {"transacciones": [{**tx, **cambios}]}


# ---------- Contrato de la API (modelo sintético de conftest.py) ----------

@pytest.mark.parametrize("nombre", ["fraude", "legitima"])
def test_predict_responde_segun_el_contrato(cliente, nombre):
    respuesta = cliente.post("/predict", json=_ejemplo(nombre)["solicitud"])

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert 0 <= cuerpo["puntaje"] <= 1
    assert cuerpo["decision"] == ("bloquear" if cuerpo["puntaje"] >= cuerpo["umbral"] else "aprobar")
    assert cuerpo["version_modelo"] == "C-v2"
    assert cuerpo["latencia_ms"] >= 0


@pytest.mark.parametrize("solicitud", [
    _una_transaccion(categoria="no_existe"),
    _una_transaccion(monto=0),
    _una_transaccion(monto=-10),
    _una_transaccion(fecha_hora="ayer"),
    {"transacciones": []},
    {"transacciones": _una_transaccion()["transacciones"] * 21},
    {},
], ids=["categoria", "monto_cero", "monto_negativo", "fecha", "vacia", "mas_de_20", "sin_campo"])
def test_predict_rechaza_entradas_invalidas(cliente, solicitud):
    assert cliente.post("/predict", json=solicitud).status_code == 422


def test_una_sola_transaccion_es_valida(cliente):
    """Tarjeta sin historial: se evalúa con relleno."""
    assert cliente.post("/predict", json=_una_transaccion()).status_code == 200


def test_categorias(cliente):
    respuesta = cliente.get("/categorias")
    assert respuesta.status_code == 200
    categorias = respuesta.json()
    assert categorias == sorted(categorias) and "food_dining" in categorias


# ---------- Modelo de producción (artefactos/modelo.pt) ----------

@pytest.fixture
def cliente_produccion(tmp_path):
    if not (RAIZ / "artefactos" / "modelo.pt").exists():
        pytest.skip("No hay modelo entrenado en artefactos/")
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    os.environ["RUTA_ARTEFACTOS"] = str(RAIZ / "artefactos")
    os.environ["RUTA_BASE_DATOS"] = str(tmp_path / "predicciones.db")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.mark.parametrize("nombre", ["fraude", "legitima"])
def test_modelo_de_produccion_reproduce_los_ejemplos(cliente_produccion, nombre):
    """La API da el mismo puntaje que el pipeline de entrenamiento para casos reales de prueba."""
    ejemplo = _ejemplo(nombre)
    cuerpo = cliente_produccion.post("/predict", json=ejemplo["solicitud"]).json()

    assert cuerpo["decision"] == ejemplo["decision_esperada"]
    assert cuerpo["puntaje"] == pytest.approx(ejemplo["puntaje_esperado"], abs=1e-4)
