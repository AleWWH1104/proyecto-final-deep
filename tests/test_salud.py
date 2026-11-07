def test_health_reporta_modelo_cargado(cliente):
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["version_modelo"] == "C-v2"
