def test_inicio_devuelve_la_pagina(cliente):
    respuesta = cliente.get("/")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    assert 'fetch("/predict"' in respuesta.text
    assert "Cargar ejemplo sospechoso" in respuesta.text
