# Plan de trabajo

Entregables (según las instrucciones del curso):

1. **Repositorio en GitHub.** El `README.md` **es el informe técnico**; no se pide un informe aparte. Lleva las 7 secciones
   (problemática, planificación, supuestos, modelo, validación de hipótesis, arquitectura, evidencias), las instrucciones
   de ejecución y la declaración de uso de IA.
2. **Presentación en PDF** dentro del repositorio (`docs/presentacion.pdf`): 10 minutos, más 5 de preguntas.
3. **Servicio desplegado** con enlace público. Es la parte que más vale: 20 puntos de evidencias.

Fechas clave: **listos el lunes 9 de noviembre** (el orden de presentación se sortea). Entrega límite: domingo 15 de noviembre, 23:59.

## División en 3 partes

Cada parte tiene un responsable, pero **todos deben poder explicar todo**: en las preguntas (15 pts) cualquiera puede recibir cualquier pregunta.

### Parte 1: Modelo y validación · responsable: Iris Ayala

Cubre las secciones 4 y 5 del README (20 pts de la rúbrica).

- [ ] Descargar `fraudTrain.csv` (Kaggle, Sparkov) en `datos/`.
- [ ] Correr `entrenamiento/entrenar.py`: reentrena C v2, línea base A y evalúa C v1.
- [ ] Revisar que la reproducción de C v1 dé el AUC-PR del Proyecto 1 (≈ 0.983 en prueba).
- [ ] Leer `reportes/metricas.json` y las figuras; redactar la **sección 4** (proyecto de origen, arquitectura, las 4 correcciones y por qué) y la **sección 5** (H1, H2, H3: hipótesis, métrica, criterio, resultado frente a la línea base).
- [ ] Preparar respuestas sobre el modelo: por qué LSTM, por qué el embedding, por qué el umbral, qué pasa con categorías raras, por qué los datos simulados son un límite.

### Parte 2: Servicio, despliegue y monitoreo · responsable: Anggie Quezada

Cubre las secciones 6 y 7 del README (30 pts de la rúbrica).

- [ ] API: `POST /predict`, `GET /health`, `GET /monitoreo` (FastAPI, en capas).
- [ ] Registro de cada predicción en SQLite (puntaje, decisión, latencia, versión del modelo).
- [ ] Tests (`uv run pytest`) y GitHub Actions que los corra en cada push.
- [ ] Dockerfile y despliegue público (Hugging Face Spaces o Render).
- [ ] **Sección 6:** diagrama datos → entrenamiento → servicio → monitoreo, más las herramientas y por qué.
- [ ] **Sección 7:** enlace, capturas, registros reales de uso, resultado de los tests y latencia p95 (H3).

### Parte 3: Producto, interfaz y comunicación · responsable: (integrante 3)

Cubre las secciones 1–3 del README y la presentación (10 + 15 pts de la rúbrica).

- [ ] **Sección 1, problemática:** el área de riesgos decide en línea si **aprueba o bloquea** una transacción; qué cuesta equivocarse.
- [ ] **Sección 2, planificación:** esta tabla de etapas, fechas y responsables.
- [ ] **Sección 3, supuestos:** datos simulados (Sparkov), el cliente envía el historial reciente de la tarjeta, costos fijos de Q4,200/Q180, hora local de la transacción.
- [ ] Interfaz web (`/`): formulario claro para que alguien ajeno al equipo pruebe el modelo con ejemplos precargados (normal / sospechoso).
- [ ] Conseguir **usuarios externos** que prueben el servicio, para tener registros de uso reales como evidencia.
- [ ] Presentación PDF de unas 10 diapositivas y guion de 10 minutos, con la demo en vivo y un plan B (video o capturas).

## Etapas

| Etapa | Fecha | Responsable |
|---|---|---|
| Estructura del repo, hipótesis pre-registradas | 6 nov | Todo el equipo |
| Reentrenamiento C v2 y evaluación contra líneas base | 6–7 nov | Parte 1 |
| API, registro de predicciones y tests | 7 nov | Parte 2 |
| Interfaz web | 7–8 nov | Parte 3 |
| Docker, despliegue público y monitoreo | 8 nov | Parte 2 |
| Pruebas con usuarios externos y capturas | 8 nov | Parte 3 |
| README (7 secciones) y presentación PDF | 8 nov | Todo el equipo |
| Ensayo de la presentación con preguntas cruzadas | 8 nov (noche) | Todo el equipo |
| **Presentación** | **9 o 13 nov** | Todo el equipo |
| Ajustes según retroalimentación, versión final | hasta 15 nov 23:59 | Todo el equipo |
