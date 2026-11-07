# Plan de trabajo

> **Léanlo completo antes de empezar.** Explica qué hay en el repo, qué le toca a cada quien, qué archivos tocar y
> cuándo está "terminado". Cualquiera puede recibir cualquier pregunta en la presentación, así que también lean las
> partes de los demás.

## 0. Contexto en 2 minutos

**Qué pide el curso:** llevar a producción un modelo de los proyectos 1–3. "Producción" significa que **alguien ajeno al
equipo pueda usarlo** (API o página web). Hay 3 entregables:

1. **Repositorio en GitHub.** El `README.md` **es el informe**, con 7 secciones (no hay informe aparte).
2. **Presentación en PDF** en el repo: 10 minutos más 5 de preguntas, ante un "comité" que decide si el modelo se queda en producción.
3. **Servicio desplegado** con enlace público.

Fechas: **listos el lunes 9 de noviembre**; entrega límite el domingo 15 de noviembre, 23:59.

**Qué elegimos:** el **Modelo C del Proyecto 1** (fraude con tarjeta). Es una LSTM que ve las últimas 10 transacciones
de una tarjeta y da un **puntaje de riesgo** de 0 a 1. Si el puntaje es ≥ **0.025**, se **bloquea** la transacción; si no, se **aprueba**.

**Qué ya está hecho (Parte 1, Iris):**
- El modelo se corrigió y reentrenó (C v2). Está en `artefactos/modelo.pt` + `artefactos/parametros.json`. **No hace falta el CSV para nada de lo que sigue.**
- Resultados en prueba: AUC-PR **0.989**; deja pasar 25 de 1,538 fraudes; cuesta Q185,640, contra Q1,345,860 de la línea base. Detalle en el `README.md`, secciones 4 y 5.
- La estructura del servicio (FastAPI en capas) ya existe, con `GET /health` funcionando.
- Hay dos ejemplos reales listos para usar en `docs/ejemplos/fraude.json` y `docs/ejemplos/legitima.json`.

## 1. Preparar el entorno (todos)

```bash
git clone <url-del-repo>
cd proyecto-final-deep
uv sync                 # instala todo; si no tienen uv: https://docs.astral.sh/uv/getting-started/installation/
uv run pytest           # deben pasar todos los tests
uv run fastapi dev      # abrir http://127.0.0.1:8000/docs y probar GET /health
```

**Cómo trabajamos con git: por turnos, todos en `main`.** Una persona trabaja a la vez y la siguiente empieza
cuando la anterior terminó y subió sus cambios. Así nadie se pisa.

**Orden:** Iris (✅ terminado) → **Erick** → **Anggie**.

```bash
git pull                     # SIEMPRE antes de empezar, para tener lo último
# ... trabajar ...
uv run pytest                # debe pasar antes de subir
git add .
git commit -m "mensaje corto y claro en inglés"
git push
```

Hagan commits pequeños mientras avanzan (no uno solo al final) y avisen en el grupo cuando terminen su turno.

**Reglas de la estructura** (para que no se mezclen responsabilidades):

| Carpeta | Qué va ahí | Qué NO va ahí |
|---|---|---|
| `app/api/` | rutas HTTP y esquemas Pydantic (entrada/salida) | lógica de negocio, SQL |
| `app/services/` | lógica: armar la predicción, decidir, calcular el monitoreo | detalles HTTP, SQL directo |
| `app/repositories/` | leer y escribir en SQLite | lógica de negocio |
| `app/ml/` | modelo y preprocesamiento (**ya está; no modificar**) | — |
| `app/web/` | HTML/CSS/JS de la interfaz | lógica del modelo |

## 2. Contrato de la API (Erick lo implementa, Anggie lo consume)

Erick debe respetar exactamente este formato, porque la página web de Anggie se construye encima de él.

### `POST /predict`

Entrada: las transacciones recientes de **una** tarjeta, de 1 a 20. **La más reciente (por fecha) es la que se evalúa.**

```json
{
  "transacciones": [
    {"monto": 54.71, "categoria": "food_dining", "fecha_hora": "2020-03-11T13:19:42"},
    {"monto": 356.49, "categoria": "grocery_pos", "fecha_hora": "2020-03-14T02:34:35"}
  ]
}
```

- `monto`: número > 0, en quetzales.
- `categoria`: una de las 14 categorías de `artefactos/parametros.json` → `cat2cod` (`entertainment`, `food_dining`,
  `gas_transport`, `grocery_net`, `grocery_pos`, `health_fitness`, `home`, `kids_pets`, `misc_net`, `misc_pos`,
  `personal_care`, `shopping_net`, `shopping_pos`, `travel`). Una categoría distinta responde **422**.
- `fecha_hora`: ISO 8601, hora local de la transacción, sin zona horaria.

Salida:

```json
{
  "id_prediccion": 17,
  "puntaje": 0.944,
  "decision": "bloquear",
  "umbral": 0.025,
  "version_modelo": "C-v2",
  "latencia_ms": 4.2
}
```

### `GET /categorias`

`["entertainment", "food_dining", ...]`: la lista para el selector de la página.

### `GET /monitoreo`

Resumen del uso real, comparado con lo que el modelo vio al entrenar (`parametros.json` → `referencia`):

```json
{
  "total_predicciones": 120,
  "tasa_bloqueo": 0.05,          "tasa_bloqueo_referencia": 0.0068,
  "monto_mediana": 61.2,         "monto_mediana_referencia": 47.41,
  "latencia_p50_ms": 3.1,        "latencia_p95_ms": 8.7,
  "frecuencia_categorias": {"food_dining": 0.2, "...": 0.0},
  "ultimas_predicciones": [{"id": 120, "fecha": "...", "puntaje": 0.01, "decision": "aprobar"}]
}
```

### `GET /health` (ya existe)

---

## 3. Parte 1: Modelo y validación · **Iris Ayala**

Cubre las secciones 4 y 5 del README (20 pts).

- [x] Pipeline de entrenamiento reproducible (`entrenamiento/`), con las 4 correcciones al Proyecto 1.
- [x] Hipótesis pre-registradas (`docs/hipotesis.md`) y evaluadas: H1 ✅ y H2 ✅.
- [x] Artefactos (`artefactos/`), reportes y figuras (`reportes/`), ejemplos reales (`docs/ejemplos/`).
- [x] Secciones 4 y 5 del README (borrador con resultados reales).
- [ ] Revisar y ajustar la redacción de las secciones 4 y 5.
- [ ] Cuando Erick tenga el servicio desplegado: completar **H3** (latencia p95 de `/monitoreo`) en la sección 5.
- [ ] Al terminar cada turno, hacer `git pull` y probar que todo funcione (`uv run pytest` y la página).
- [ ] Preparar respuestas: por qué LSTM, por qué embedding, por qué umbral 0.025, por qué los 4 cambios, límites con datos simulados.

## 4. Parte 2: Servicio, despliegue y monitoreo · **Erick**

Cubre las secciones 6 y 7 del README (30 pts). **Es la parte más técnica y va primero, porque la página de Anggie depende de `/predict`.**

### 4.1 `POST /predict` y `GET /categorias`

| Archivo | Qué hacer |
|---|---|
| `app/api/esquemas.py` | `Transaccion` (monto > 0, categoria, fecha_hora), `SolicitudPrediccion` (lista de 1 a 20) y `RespuestaPrediccion`, según el contrato de la sección 2. |
| `app/services/prediccion.py` *(nuevo)* | Función que recibe la solicitud y los artefactos y: (1) valida que las categorías existan en `parametros.cat2cod`; (2) arma la ventana con **`ventana_desde_historial`** de `app/ml/preprocesamiento.py` (**no reimplementar el preprocesamiento**); (3) calcula el puntaje con `puntuar` de `app/ml/modelo.py`; (4) decide `bloquear` si puntaje ≥ `parametros.umbral_operativo`; (5) mide la latencia; (6) guarda el registro con el repositorio. |
| `app/api/rutas.py` | Rutas `POST /predict` y `GET /categorias`, que solo llaman al servicio. |

Prueba manual: en `/docs`, pegar `solicitud` de `docs/ejemplos/fraude.json`; debe dar `puntaje ≈ 0.944` y `bloquear`.
Con `legitima.json` debe dar `aprobar`.

### 4.2 Registro de predicciones (SQLite)

| Archivo | Qué hacer |
|---|---|
| `app/repositories/registro_predicciones.py` *(nuevo)* | Crear la tabla si no existe y guardar cada predicción: id, fecha, puntaje, decisión, latencia_ms, version_modelo, n_transacciones, monto y categoría de la transacción evaluada. Funciones para leer el resumen. Usar `sqlite3` de Python (no hace falta un ORM). Ruta: `Settings.ruta_base_datos`. |
| `app/main.py` | En el `lifespan`, crear el repositorio y guardarlo en `app.state` (igual que el modelo). |

**No** guardar datos personales: no hay número de tarjeta en el contrato, y así debe quedar.

### 4.3 `GET /monitoreo`

| Archivo | Qué hacer |
|---|---|
| `app/services/monitoreo.py` *(nuevo)* | Calcular con el repositorio: total, tasa de bloqueo, mediana de montos, frecuencia de categorías, latencia p50/p95 y últimas predicciones. Ponerlos al lado de `parametros.referencia`. |
| `app/api/rutas.py` | `GET /monitoreo`. |

Este endpoint es la evidencia de **H3** (latencia p95 < 300 ms) y de "registros de uso".

### 4.4 Tests

`tests/test_predict.py` *(nuevo)*, usando el fixture `cliente` de `tests/conftest.py` (ya existe; usa un modelo sintético y una base temporal):
- fraude y legítima responden 200, con `puntaje` entre 0 y 1 y `decision` en {aprobar, bloquear};
- categoría inválida → 422; lista vacía → 422; más de 20 → 422; monto ≤ 0 → 422;
- después de 2 predicciones, `/monitoreo` reporta `total_predicciones == 2`.

### 4.5 Docker y despliegue

- `Dockerfile` en la raíz. Base `python:3.12-slim`. Copiar uv (`COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`),
  copiar `pyproject.toml`, `uv.lock`, `app/` y `artefactos/`, y correr `uv sync --frozen --no-default-groups`. Ese flag
  instala solo lo del servicio, sin pandas ni sklearn. Comando: `fastapi run app/main.py --host 0.0.0.0 --port <puerto>`.
- Probar en local: `docker build -t fraude .` y `docker run -p 8000:8000 fraude`.
- Desplegar gratis. Recomendado: **Hugging Face Spaces** (tipo *Docker*, puerto **7860**, 16 GB de RAM); alternativa: Render.
- ⚠️ En los planes gratuitos **el disco se borra al reiniciar**, y con él la base SQLite. Antes de sacar las evidencias,
  guarden capturas y exporten `/monitoreo` a un JSON en `docs/evidencias/`.
- `.github/workflows/tests.yml`: GitHub Actions que corre `uv sync` y `uv run pytest` en cada push.

### 4.6 README

- **Sección 6:** diagrama datos → entrenamiento → servicio → monitoreo (Mermaid en el README), más una tabla de herramientas
  con su porqué: FastAPI, PyTorch CPU, SQLite, uv, Docker, Hugging Face Spaces, GitHub Actions.
- **Sección 7:** enlace, capturas de `/docs` y `/monitoreo`, captura de GitHub Actions en verde, latencia p95 → pasar a Iris para H3.
- Enlaces del servicio al inicio del README.

**Terminado cuando:** el enlace público responde `/predict` con los dos ejemplos, `/monitoreo` muestra datos, los tests pasan en GitHub Actions y las secciones 6 y 7 están escritas.

## 5. Parte 3: Producto, interfaz y comunicación · **Anggie Quezada**

Cubre las secciones 1–3 del README, la presentación y el apoyo a las evidencias (10 + 15 pts, más parte de los 20 de evidencias).

### 5.1 Interfaz web (`/`)

Una página **simple y clara** para que alguien que no conoce el proyecto pruebe el modelo.

| Archivo | Qué hacer |
|---|---|
| `app/web/index.html` *(nuevo)* | Formulario con una tabla de transacciones (monto, categoría, fecha y hora), botones **"Cargar ejemplo normal"** y **"Cargar ejemplo sospechoso"** (copiar `solicitud` de `docs/ejemplos/*.json`), botón **Evaluar** que hace `fetch("/predict")`, y el resultado grande: puntaje y decisión en color, con una frase que explique qué significa. El selector de categoría se llena con `GET /categorias`. HTML, CSS y JavaScript simples, sin frameworks. |
| `app/web/monitoreo.html` *(opcional)* | Página que muestre `GET /monitoreo` de forma legible. |
| `app/api/rutas_web.py` *(nuevo)* | `GET /` que devuelve `index.html` (`FileResponse`). |
| `app/main.py` | Una línea: `app.include_router(rutas_web.router)`. |

Mientras Erick hace su turno, Anggie puede ir armando la presentación y el texto de las secciones 1–3 **fuera del repo**
(por ejemplo, en Google Docs o Canva) y pasarlo al README cuando empiece su turno.

### 5.2 Usuarios externos (evidencia)

Cuando el servicio esté desplegado, conseguir al menos **5 personas ajenas al equipo** que lo usen. Anotar quién lo probó,
cuándo y qué comentó, en `docs/evidencias/pruebas_usuarios.md`, con capturas. Sus predicciones quedan en `/monitoreo`.

### 5.3 README, secciones 1–3

Hay instrucciones en cada sección del `README.md` (bloques _TODO (Anggie)_). Datos útiles: costos de Q4,200 (fraude no
detectado) y Q180 (bloqueo de un cliente legítimo); usuario: área de riesgos / sistema de autorización del banco.

### 5.4 Presentación (`docs/presentacion.pdf`)

Unas 10 diapositivas para 10 minutos. La audiencia es un **comité técnico y de negocio** que decide si el modelo se queda en producción. Propuesta:

1. Problema y lo que cuesta (Q4,200 vs Q180).
2. Usuario y decisión: aprobar o bloquear en línea.
3. Supuestos.
4. Modelo: de dónde viene, la arquitectura en un diagrama, los 4 cambios.
5. Hipótesis y resultados: tabla de costos, gráfica `reportes/costo_por_modelo.png`.
6. Por qué el umbral 0.025: gráfica `reportes/costo_vs_umbral.png`.
7. Arquitectura del sistema (el diagrama de Erick).
8. **Demo en vivo**: ejemplo normal → aprobar; ejemplo sospechoso → bloquear; `/monitoreo`.
9. Evidencias: usuarios externos, tests, latencia.
10. Recomendación al comité, límites y próximos pasos (reentrenar con datos reales, monitorear deriva).

Tener un **plan B** para la demo (video corto o capturas) por si falla el internet.

**Terminado cuando:** la página funciona en el enlace público, hay registros de al menos 5 usuarios externos, las secciones 1–3 están escritas y el PDF está en `docs/`.

---

## 6. Etapas

| Etapa | Fecha | Responsable | Estado |
|---|---|---|---|
| Estructura del repo e hipótesis pre-registradas | 6 nov | Iris | ✅ |
| Reentrenamiento C v2 y evaluación contra líneas base | 6 nov | Iris | ✅ |
| README: borrador de las secciones 4–5 | 6 nov | Iris | ✅ |
| **Turno Erick:** `/predict`, `/categorias`, SQLite, `/monitoreo`, tests, Docker, despliegue, GitHub Actions | 7 nov | Erick | |
| **Turno Anggie:** interfaz web, pruebas con usuarios externos, capturas | 8 nov | Anggie | |
| README (secciones 1–3, 6–7), H3 y presentación PDF | 8 nov | Todos | |
| Ensayo con preguntas cruzadas | 8 nov (noche) | Todos | |
| **Presentación** | **9 o 13 nov** | Todos | |
| Ajustes según retroalimentación | hasta 15 nov 23:59 | Todos | |

## 7. Preguntas probables del comité (todos deben saber responderlas)

- ¿Por qué una LSTM y no un modelo más simple? → Porque el orden aporta: en el Proyecto 1, barajar la secuencia bajó el AUC-PR de 0.95 a 0.48. Además, la línea base A cuesta 7 veces más.
- ¿Por qué el umbral es tan bajo (0.025)? → Porque un fraude cuesta 23 veces más que un bloqueo, y se eligió en validación.
- ¿Qué pasa si llega una categoría nueva? → La API responde 422; habría que reentrenar.
- ¿Cómo saben si el modelo se degrada? → `/monitoreo` compara la tasa de bloqueo, los montos y las categorías contra la referencia de entrenamiento.
- ¿Funciona con datos reales? → No está probado: Sparkov es simulado. Es el principal límite y el siguiente paso.
- ¿Qué cambiaron del Proyecto 1 y por qué? → Las 4 correcciones (README, sección 4).
