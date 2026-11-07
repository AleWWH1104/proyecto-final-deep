import json
from pathlib import Path

EJEMPLOS = Path(__file__).resolve().parents[1] / "docs" / "ejemplos"


def _solicitud(nombre: str) -> dict:
    return json.loads((EJEMPLOS / f"{nombre}.json").read_text(encoding="utf-8"))["solicitud"]


def test_monitoreo_sin_predicciones(cliente):
    cuerpo = cliente.get("/monitoreo").json()

    assert cuerpo["total_predicciones"] == 0
    assert cuerpo["tasa_bloqueo"] is None and cuerpo["latencia_p95_ms"] is None
    assert cuerpo["ultimas_predicciones"] == []


def test_monitoreo_cuenta_las_predicciones(cliente):
    for nombre in ["fraude", "legitima"]:
        assert cliente.post("/predict", json=_solicitud(nombre)).status_code == 200

    respuesta = cliente.get("/monitoreo")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total_predicciones"] == 2
    assert 0 <= cuerpo["tasa_bloqueo"] <= 1
    assert cuerpo["latencia_p50_ms"] <= cuerpo["latencia_p95_ms"]
    assert sum(cuerpo["frecuencia_categorias"].values()) == 1
    assert cuerpo["monto_mediana_referencia"] is not None


def test_id_prediccion_aparece_en_las_ultimas(cliente):
    id_prediccion = cliente.post("/predict", json=_solicitud("fraude")).json()["id_prediccion"]

    ultimas = cliente.get("/monitoreo").json()["ultimas_predicciones"]

    assert isinstance(id_prediccion, int)
    assert ultimas[0]["id"] == id_prediccion


def test_predicciones_invalidas_no_se_registran(cliente):
    cliente.post("/predict", json={"transacciones": [
        {"monto": 10.0, "categoria": "no_existe", "fecha_hora": "2020-05-01T12:00:00"}]})

    assert cliente.get("/monitoreo").json()["total_predicciones"] == 0


def test_registro_guarda_la_transaccion_mas_reciente(cliente):
    """La evaluada es la más reciente por fecha, aunque no venga de última en la lista."""
    cliente.post("/predict", json={"transacciones": [
        {"monto": 999.0, "categoria": "travel", "fecha_hora": "2020-05-02T12:00:00"},
        {"monto": 10.0, "categoria": "food_dining", "fecha_hora": "2020-05-01T12:00:00"},
    ]})

    cuerpo = cliente.get("/monitoreo").json()

    assert cuerpo["monto_mediana"] == 999.0
    assert cuerpo["frecuencia_categorias"]["travel"] == 1
