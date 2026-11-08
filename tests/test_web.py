def test_inicio_devuelve_la_pagina(cliente):
    respuesta = cliente.get("/")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    assert 'fetch("/predict"' in respuesta.text
    assert "Compra sospechosa" in respuesta.text


def test_panel_de_monitoreo_devuelve_la_pagina(cliente):
    respuesta = cliente.get("/monitoreo.html")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("text/html")
    assert 'fetch("/monitoreo")' in respuesta.text
