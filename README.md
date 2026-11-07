# Proyecto Final: Detección de fraude con tarjeta en producción

Deep Learning y Sistemas Inteligentes 2026 · Universidad del Valle de Guatemala
**Integrantes:** Iris Ayala · Anggie Quezada · Erick Guerra

Servicio que recibe las transacciones recientes de una tarjeta y devuelve, en línea, un **puntaje de riesgo de fraude**
y la decisión **aprobar / bloquear**. El modelo es una LSTM con embedding de categoría de comercio (Modelo C del
Proyecto 1), corregida y reentrenada para producción.

- **Servicio en línea:** _TODO (Erick): enlace público_
- **Documentación interactiva de la API:** _TODO (Erick): enlace/docs_
- **Presentación:** [`docs/presentacion.pdf`](docs/presentacion.pdf) _TODO (Anggie)_

---

## 1. Problemática

> _TODO (Anggie)._ Responder en 1–2 párrafos: **cuál es el problema** (fraude con tarjeta), **quién es el usuario**
> (el área de riesgos del banco, o el sistema de autorización que la llama) y **qué decisión depende del modelo**
> (aprobar o bloquear una transacción en el momento). Incluir lo que cuesta equivocarse: un fraude no detectado
> cuesta en promedio **Q4,200** y bloquear a un cliente legítimo, **Q180**.

## 2. Planificación

> _TODO (Anggie)._ Tabla de etapas, fechas y responsables. Partir de [`docs/plan_trabajo.md`](docs/plan_trabajo.md)
> y actualizarla con las fechas reales en que se terminó cada cosa.

## 3. Supuestos

> _TODO (Anggie)._ Una viñeta por supuesto, con su consecuencia si no se cumple. Mínimo:
>
> - **Datos:** Sparkov es un conjunto **simulado**; el fraude real puede verse distinto.
> - **Usuario/entrada:** quien llama a la API envía las últimas transacciones de la tarjeta (hasta 20; la última es la
>   que se evalúa), con fecha y hora **local** de la transacción y una de las 14 categorías conocidas.
> - **Costos:** Q4,200 y Q180 son fijos y promedio; si cambian, cambia el umbral óptimo.
> - **Entorno:** el servicio corre en CPU; sin GPU; una tarjeta nueva (sin historial) se evalúa con lo que haya.

## 4. Modelo planteado

**Proyecto de origen:** Proyecto 1, _"El orden revela: monitoreo transaccional"_. Se
eligió el **Modelo C**, que fue el mejor del proyecto: AUC-PR de 0.983 en prueba contra 0.726 de la línea base sin secuencia.

**Arquitectura** ([`app/ml/modelo.py`](app/ml/modelo.py)), 5,469 parámetros:

```
últimas 10 transacciones de la tarjeta  →  (10 × 5)
   ├─ categoría de comercio → Embedding(15 × 4)     (código 0 = relleno, vector cero)
   └─ log(1+monto), hora, log(1+días desde la anterior), es_real  (estandarizados)
                     ↓ concatenar (10 × 8)
              LSTM (1 capa, 32 unidades)
                     ↓ último estado oculto
              Lineal(32 → 1) + sigmoide  →  puntaje de riesgo ∈ [0, 1]
                     ↓
       puntaje ≥ 0.025  →  BLOQUEAR      ·      puntaje < 0.025  →  APROBAR
```

**Cambios respecto al Proyecto 1, y por qué.** Al revisar el notebook original para llevarlo a producción encontramos
cuatro problemas de método. Los corregimos sin cambiar la arquitectura ni los hiperparámetros:

| #   | Problema en el Proyecto 1                                                                                  | Por qué importa                                                                                           | Corrección                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | El umbral 0.06 se eligió minimizando el costo **en el conjunto de prueba**.                                | El costo reportado era optimista: se "ajustó" con los datos que debían ser nuevos.                        | El umbral se elige en **validación**; prueba solo se usa para reportar.                                                     |
| 2   | El relleno de secuencias (ceros) y las categorías desconocidas compartían el código 0 con `entertainment`. | Una tarjeta con poco historial "parecía" tener compras de entretenimiento de Q0.                          | Código 0 reservado para relleno (`padding_idx=0`), categorías 1–14 y variable `es_real`.                                    |
| 3   | Variables sin escalar (montos de miles junto a horas 0–23).                                                | Entrenamiento menos estable; el monto domina la entrada.                                                  | `log(1+monto)`, hora y `log(1+días)`, estandarizados con media y desviación **de entrenamiento**.                           |
| 4   | El historial de cada tarjeta se reiniciaba al inicio de validación y de prueba.                            | En producción la tarjeta ya tiene historial: el modelo se evaluaba en condiciones distintas a las reales. | Variables calculadas sobre el historial completo, siempre hacia atrás en el tiempo; las etiquetas nunca cruzan particiones. |

Además, el modelo guardado es el de **la época con mejor AUC-PR en validación** (época 8 de 10), no el de la última.

**Datos y partición.** Sparkov `fraudTrain.csv`: 1,296,675 transacciones de 983 tarjetas, con 0.58 % de fraude.
La partición es temporal, la misma del Proyecto 1:

| Conjunto      | Transacciones | Periodo                            |
| ------------- | ------------- | ---------------------------------- |
| Entrenamiento | 778,005       | 2019-01-01 → 2019-11-29            |
| Validación    | 259,335       | 2019-11-29 → 2020-03-06            |
| Prueba        | 259,335       | 2020-03-06 → 2020-06-21 (107 días) |

El diccionario de categorías, el escalado, la época y el umbral se ajustan **solo** con entrenamiento y validación.
Entrenamiento y servicio usan **la misma función** de preprocesamiento ([`app/ml/preprocesamiento.py`](app/ml/preprocesamiento.py)),
y un test verifica que ambos arman la misma ventana.

![Entrenamiento](reportes/historial_entrenamiento.png)

## 5. Validación de hipótesis

Las hipótesis, métricas y criterios de éxito se registraron **antes de reentrenar**, en [`docs/hipotesis.md`](docs/hipotesis.md)
(ver la fecha del commit). El script de entrenamiento los evalúa automáticamente ([`entrenamiento/hipotesis.py`](entrenamiento/hipotesis.py)).

**Resultados en prueba.** Cada modelo usa el umbral de mínimo costo **elegido en validación**:

| Modelo                                            | AUC-PR    | Umbral | Recall    | Precisión | Fraudes no detectados | Bloqueos legítimos | Costo en prueba |
| ------------------------------------------------- | --------- | ------ | --------- | --------- | --------------------- | ------------------ | --------------- |
| Sin modelo (dejar pasar todo)                     | —         | —      | 0         | —         | 1,538                 | 0                  | Q6,459,600      |
| **A**: agregados + gradient boosting (línea base) | 0.703     | 0.040  | 0.850     | 0.385     | 231                   | 2,087              | Q1,345,860      |
| **C v1**: Proyecto 1, tal como se entregó         | 0.983     | 0.045  | 0.978     | 0.731     | 34                    | 555                | Q242,700        |
| **C v2**: producción                              | **0.989** | 0.025  | **0.984** | **0.772** | **25**                | **448**            | **Q185,640**    |

![Costo por modelo](reportes/costo_por_modelo.png)

| Hipótesis                                             | Criterio (definido antes)                         | Resultado                                       | ¿Cumple?    |
| ----------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- | ----------- |
| **H1:** C v2 reduce el costo frente a la línea base A | costo(C v2) ≤ 50 % de costo(A) **y** AUC-PR +0.10 | costo = **13.8 %** del de A; AUC-PR **+0.286**  | ✅ Sí       |
| **H2:** las correcciones no empeoran el modelo        | AUC-PR(C v2) ≥ AUC-PR(C v1) − 0.01                | **+0.006**                                      | ✅ Sí       |
| **H3:** el servicio responde a tiempo                 | latencia p95 < 300 ms                             | _TODO (Erick): medir en el servicio desplegado_ | _pendiente_ |

**Lectura de los resultados**

- **Frente a la línea base A**, C v2 deja pasar 25 fraudes en lugar de 231 y bloquea 448 clientes legítimos en lugar
  de 2,087. El costo baja de Q1.35 M a Q0.19 M en los 107 días de prueba: unos **Q325,000 menos por mes**.
- **Frente a no tener modelo**, el ahorro es de Q6.27 M en el periodo, unos **Q1.76 M por mes**.
- **Frente al Proyecto 1**, C v2 mejora un poco en todo (+0.006 AUC-PR, 9 fraudes menos y 107 bloqueos menos), con un
  costo Q57,060 menor. La razón principal para usar C v2 no es esa mejora, sino que **se evaluó con método correcto**:
  el Proyecto 1 reportaba Q235,620 con un umbral elegido mirando la prueba; con un umbral elegido en validación, C v1 cuesta Q242,700.
- **El umbral importa tanto como el modelo:** el mismo C v2 con el umbral por defecto de 0.5 cuesta Q449,160, 2.4 veces
  más, porque deja pasar 106 fraudes. Como un fraude cuesta 23 veces más que un bloqueo, conviene un umbral bajo.
- **Reproducibilidad:** el pipeline reproduce exactamente el AUC-PR de C v1 que reportó el Proyecto 1 (0.9827). El
  modelo A da 0.703 en lugar de 0.726 porque ahora sus agregados también usan el historial completo de la tarjeta.

![Costo vs umbral](reportes/costo_vs_umbral.png)

**Límites conocidos:** los datos son simulados; las categorías con pocos fraudes (travel, grocery_net) tienen menor
recall; no se probó contra mecanismos de fraude nuevos. El monitoreo (sección 6) existe para detectar cuándo el
tráfico real se aleja de lo que vio el modelo.

## 6. Herramientas y arquitectura

> _TODO (Erick)._ Diagrama del sistema: **datos → entrenamiento → servicio → monitoreo** (puede ser Mermaid aquí
> mismo o una imagen en `docs/`), más una tabla de herramientas con el porqué de cada una.

**Estructura del código** (capas con una sola responsabilidad cada una):

```
app/                         SERVICIO (lo que se despliega)
  main.py                    crea la app y carga el modelo una vez al arrancar (lifespan)
  core/config.py             configuración por variables de entorno
  api/                       rutas HTTP y esquemas de entrada/salida (solo validan y delegan)
  services/                  lógica: predicción, decisión, monitoreo
  repositories/              persistencia del registro de predicciones (SQLite)
  ml/                        arquitectura, preprocesamiento compartido y carga de artefactos
  web/                       interfaz web
entrenamiento/               PIPELINE: CSV → modelo + métricas + figuras
  entrenar.py                comando principal
  datos.py · entrenador.py · lineas_base.py · metricas.py · hipotesis.py · reportes.py
  exportar_ejemplos.py       genera docs/ejemplos/*.json desde el conjunto de prueba
artefactos/                  modelo.pt + parametros.json (C v2) y v1_proyecto1/ (línea base)
reportes/                    metricas.json y figuras del último entrenamiento
docs/                        hipótesis, plan de trabajo, ejemplos, presentación
tests/                       pruebas automáticas (pytest)
```

## 7. Evidencias del pase a producción

> _TODO (Erick + Anggie)._ Enlace al servicio, capturas de la interfaz y de `/docs`, registros reales de uso
> (`/monitoreo`, con usuarios externos), resultado de `pytest` / GitHub Actions y la latencia p95 (H3).

---

## Cómo ejecutar

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 automáticamente).

```bash
uv sync                          # instala dependencias (servicio + entrenamiento + tests)
uv run fastapi dev               # servicio en http://127.0.0.1:8000  (documentación en /docs)
uv run pytest                    # pruebas (no necesitan el CSV)
```

**Reentrenar** (opcional; el modelo entrenado ya está en `artefactos/`): descargar `fraudTrain.csv` del dataset
[Sparkov en Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) en `datos/` y correr:

```bash
uv run python -m entrenamiento.entrenar          # ~4 min en CPU; regenera artefactos/ y reportes/
uv run python -m entrenamiento.exportar_ejemplos # regenera docs/ejemplos/
```

Semilla fija (42); la partición y todos los ajustes se recalculan desde el CSV.

## Uso de inteligencia artificial

> _TODO (los tres)._ Cada integrante declara para qué usó asistentes de IA. Ejemplo:
>
> - **Iris:** Claude Code para revisar el método del Proyecto 1 (detectó los 4 problemas de la sección 4), estructurar el
>   pipeline de entrenamiento y redactar borradores de las secciones 4 y 5. Revisamos y ejecutamos todo el código.
> - **Erick:** …
> - **Anggie:** …
