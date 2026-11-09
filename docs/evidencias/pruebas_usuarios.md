# Pruebas de uso del servicio desplegado

Pruebas hechas sobre <https://fraude-tarjeta.onrender.com> el **8 de noviembre de 2026, de 17:43 a 17:58 (hora de
Guatemala)**. Sus predicciones quedaron registradas en `/monitoreo`; el resumen exportado justo después de las pruebas
está en [`monitoreo_usuarios.json`](monitoreo_usuarios.json) y la captura del panel en
[`usuarios/monitoreo.png`](usuarios/monitoreo.png).

> **Nota de transparencia.** El plan pedía al menos 5 personas ajenas al equipo. Estas pruebas **las hizo el equipo**,
> cada una desde el punto de vista de un rol distinto, siguiendo un guion de escenarios. No son usuarios externos.

## Cómo se hizo la prueba

1. Antes de empezar se abrió `/health` para despertar el servicio (el plan gratuito se duerme tras 15 min sin uso).
2. Cada escenario se hizo desde la página principal, como lo haría alguien sin conocimientos técnicos.
3. Al terminar se abrió `/monitoreo.html`, se tomó la captura y se descargó el JSON, sin hacer push en medio (un
   redespliegue borra el registro).

## Registro

| #   | Rol                        | Qué se hizo                                                                                                     | Resultado                                                                                    | Captura                                                                                             |
| --- | -------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1   | Analista de riesgos        | Cargó **"Compra sospechosa"** (Q356.49 en supermercado a las 2:34 a. m.). Luego cambió esa compra a Q80 a las 14:00 y volvió a evaluar | Primero **BLOQUEAR** (puntaje 0.944). Con Q80 de día, **APROBAR** (puntaje ≈ 0.00001)       | [`escenario1.png`](usuarios/escenario1.png), [`escenario1b.png`](usuarios/escenario1b.png)          |
| 2   | Cliente habitual           | Cargó **"Cliente con compras normales"** (11 compras; la última, Q61.06 en línea)                               | **APROBAR** (puntaje ≈ 0)                                                                    | [`escenario2.png`](usuarios/escenario2.png)                                                         |
| 3   | Cliente con tarjeta nueva  | "Escribe tus propias compras": una sola compra de Q50 en restaurantes, 8 nov 2026 a las 12:00                   | **APROBAR** (puntaje ≈ 0.0001)                                                               | [`escenario3.png`](usuarios/escenario3.png)                                                         |
| 4   | Viajero                    | Cargó el caso normal y agregó una compra de **Q900 en viajes** (6 jul 2020, 15:00)                              | **APROBAR** (puntaje 0.0014, debajo del umbral de 0.025)                                     | [`escenario4.png`](usuarios/escenario4.png)                                                         |
| 5   | Usuario que se equivoca    | "Escribe tus propias compras" con monto 0 y sin fecha, y presionó Evaluar                                       | La página no envió nada y mostró: _"Compra 1: escribe un monto mayor que 0."_                | [`escenario5.png`](usuarios/escenario5.png)                                                         |

**Resumen del monitoreo al terminar:** 12 predicciones, 2 bloqueos (16.67 %), mediana de montos Q61.06,
latencia p50 = 1.27 ms y p95 = 35.15 ms (H3: < 300 ms ✅). La tasa de bloqueo es mucho mayor que la de referencia
(0.68 %) porque el guion incluye a propósito el ejemplo de fraude; no indica deriva del modelo.

## Qué aprendimos

- **El modelo reacciona a lo que debe.** La misma tarjeta pasa de BLOQUEAR (0.944) a APROBAR al cambiar solo el monto y la
  hora de la última compra: lo que pesa es que la compra no se parezca al historial, no la categoría.
- **Un monto alto no basta para bloquear.** Q900 en viajes, después de un historial normal, se aprobó (0.0014). Es
  coherente con lo que se buscaba evitar (una regla fija tipo "bloquear todo arriba de Q500"), pero también es un caso
  que el comité podría cuestionar: el modelo solo sabe del fraude que vio en Sparkov.
- **La página puede confundir al editar.** En un primer intento del escenario 3 se editó la tabla sin presionar
  _Evaluar_, y en pantalla quedó el resultado de la compra anterior. Mejora propuesta: marcar el resultado como
  "desactualizado" en cuanto se edita la tabla.
- **Las validaciones del navegador funcionan:** un monto inválido no llega al servicio y el mensaje dice qué corregir.
- **El plan gratuito pierde el registro.** Las primeras pruebas se perdieron porque el servicio se durmió y, al
  reiniciar, borró la base SQLite. La primera petición después de despertar también es más lenta (por eso el p95 sube a
  35 ms, aunque sigue muy debajo de 300 ms). En producción haría falta una base persistente y un servidor siempre activo.
