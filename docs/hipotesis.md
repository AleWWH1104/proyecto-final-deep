# Hipótesis y criterios de éxito (pre-registro)

**Fecha de registro:** 6 de noviembre de 2026, **antes** de reentrenar el modelo con el pipeline de producción.
El commit que agrega este archivo es la evidencia de que los criterios se fijaron antes de ver resultados.
Los mismos valores están en `entrenamiento/hipotesis.py`, y el script de entrenamiento los evalúa automáticamente.

## Contexto

El modelo de origen es el **Modelo C del Proyecto 1** (LSTM con embedding de categoría de comercio). Para llevarlo a
producción se corrigen cuatro problemas de método detectados al revisar el notebook original:

| # | Problema en el Proyecto 1 | Corrección (modelo C v2) |
|---|---|---|
| 1 | El umbral operativo (0.06) se eligió minimizando el costo **en el conjunto de prueba**, así que el ahorro reportado era optimista. | El umbral se elige en **validación** y el costo se reporta en prueba con ese umbral fijo. |
| 2 | El relleno de secuencias usaba ceros y el código 0 era la categoría `entertainment`, así que una tarjeta nueva "parecía" tener compras de entretenimiento de Q0. Las categorías desconocidas (-1) también caían en `entertainment` por el `clamp`. | El código 0 queda reservado para relleno (`padding_idx=0`, vector cero), las categorías van de 1 a 14 y se agrega la variable `es_real` (1 = transacción real, 0 = relleno). |
| 3 | Las variables entraban sin escalar (montos de hasta miles de quetzales junto a horas 0–23). | Se usan `log(1+monto)`, la hora y `log(1+días)`, estandarizados con media y desviación **de entrenamiento**. |
| 4 | Las secuencias y los "días desde la transacción anterior" se reiniciaban al inicio de validación y de prueba, algo que no pasa en producción, donde la tarjeta ya tiene historial. | Las variables se calculan sobre el historial completo de cada tarjeta (siempre hacia atrás en el tiempo) y la partición se aplica a la transacción evaluada. Las etiquetas nunca cruzan particiones. |

Además, el modelo final ya no es el de la última época: es **el de la época con mejor AUC-PR en validación**.

## Datos y partición

Sparkov `fraudTrain.csv`, con la misma partición temporal del Proyecto 1: 60 % entrenamiento, 20 % validación y
20 % prueba, en orden de fecha. Prueba se usa **solo** para reportar resultados; ninguna decisión (época, umbral) se toma con ella.

## Costos del negocio

Los mismos del Proyecto 1: un fraude no detectado cuesta **Q4,200** y bloquear una transacción legítima cuesta **Q180**.

## Hipótesis

### H1: el modelo secuencial vale la pena frente a la línea base

> El modelo C v2 reduce el costo esperado del fraude frente a la línea base sin secuencia (Modelo A: gradient
> boosting sobre 4 variables agregadas), porque el orden y el ritmo de las transacciones contienen señal que los agregados no resumen.

- **Métrica principal:** costo total en prueba (Q), con el umbral de cada modelo elegido en validación con la misma regla (mínimo costo).
- **Métrica secundaria:** AUC-PR en prueba.
- **Criterio de éxito:** costo(C v2) ≤ **50 %** del costo(A) **y** AUC-PR(C v2) − AUC-PR(A) ≥ **0.10**.
- **Referencia adicional:** "sin modelo" (dejar pasar todo, costo = fraudes × Q4,200).

### H2: las correcciones no empeoran el modelo (no inferioridad)

> Las correcciones 2–4 hacen el modelo coherente con producción sin perder capacidad de detección.

- **Métrica:** AUC-PR en prueba.
- **Línea base:** el Modelo C v1, tal como se entregó en el Proyecto 1 (`artefactos/v1_proyecto1/`), evaluado con su propio preprocesamiento.
- **Criterio de éxito:** AUC-PR(C v2) ≥ AUC-PR(C v1) − **0.01**.

### H3: el servicio responde a tiempo para decidir en línea

> El servicio desplegado responde lo bastante rápido para usarse en la autorización de una transacción.

- **Métrica:** latencia p95 de `POST /predict`, medida en los registros del servicio desplegado.
- **Criterio de éxito:** p95 < **300 ms**.
