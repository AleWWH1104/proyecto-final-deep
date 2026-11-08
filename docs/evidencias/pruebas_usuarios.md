# Pruebas con usuarios externos

Personas **ajenas al equipo** que usaron el servicio desplegado en <https://fraude-tarjeta.onrender.com>.
Sus predicciones quedaron registradas en `/monitoreo`; el resumen exportado justo después de las pruebas está en
[`monitoreo_usuarios.json`](monitoreo_usuarios.json).

## Cómo se hizo la prueba

1. Antes de empezar se abrió `/health` para despertar el servicio (el plan gratuito se duerme tras 15 min sin uso).
2. A cada persona se le compartió el enlace sin más explicación que: _"esta página decide si una compra con tarjeta
   se aprueba o se bloquea; pruébela"_.
3. Se le pidió: (a) cargar el ejemplo normal y evaluarlo, (b) cargar el ejemplo sospechoso y evaluarlo, y (c) cambiar
   algo (un monto, una categoría o una hora) y ver si cambia la decisión.
4. Se anotó qué entendió, qué le costó y qué sugirió.

## Registro

| #   | Persona (nombre o iniciales) | Perfil / relación | Fecha y hora | Qué probó | Comentario | Captura |
| --- | ---------------------------- | ----------------- | ------------ | --------- | ---------- | ------- |
| 1   |                              |                   |              |           |            |         |
| 2   |                              |                   |              |           |            |         |
| 3   |                              |                   |              |           |            |         |
| 4   |                              |                   |              |           |            |         |
| 5   |                              |                   |              |           |            |         |

Capturas en [`usuarios/`](usuarios/).

## Qué aprendimos

- _Pendiente: resumir los comentarios repetidos y qué se cambió (o se cambiaría) por ellos._
