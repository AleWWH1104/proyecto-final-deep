"""Exporta ejemplos reales del conjunto de prueba con el formato de entrada de POST /predict.

Sirven para la demo, para la interfaz web y para los tests del servicio, sin necesitar el CSV.
Cada ejemplo trae el puntaje calculado con `ventana_desde_historial`, la misma función que usará la API.

Uso:
    uv run python -m entrenamiento.exportar_ejemplos
"""

import argparse
import json
from pathlib import Path

import numpy as np

from app.ml.cargador import cargar_artefactos
from app.ml.modelo import puntuar
from app.ml.preprocesamiento import ventana_desde_historial
from entrenamiento import datos

N_HISTORIAL = 11  # 10 de la ventana + 1 anterior para que los "días desde la anterior" sean exactos


def _historial(df, fila: int) -> list[dict]:
    tarjeta = df.loc[:fila]
    tarjeta = tarjeta[tarjeta["cc_num"] == df.at[fila, "cc_num"]].tail(N_HISTORIAL)
    return [{"monto": float(r.amt), "categoria": r.category,
             "fecha_hora": r.trans_date_trans_time.isoformat()} for r in tarjeta.itertuples()]


def _puntaje(artefactos, transacciones: list[dict]) -> float:
    from datetime import datetime
    p = artefactos.parametros
    ventana = ventana_desde_historial(
        [t["monto"] for t in transacciones], [t["categoria"] for t in transacciones],
        [datetime.fromisoformat(t["fecha_hora"]) for t in transacciones],
        p.cat2cod, p.escalado, p.longitud_secuencia)
    return float(puntuar(artefactos.modelo, ventana)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("datos/fraudTrain.csv"))
    parser.add_argument("--salida", type=Path, default=Path("docs/ejemplos"))
    parser.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args()

    artefactos = cargar_artefactos(Path("artefactos/modelo.pt"), Path("artefactos/parametros.json"))
    umbral = artefactos.parametros.umbral_operativo
    df = datos.cargar_transacciones(args.csv)
    corte_val, corte_test = datos.cortes_temporales(df["trans_date_trans_time"])
    en_prueba = datos.asignar_particion(df, corte_val, corte_test) == "test"
    rng = np.random.default_rng(args.semilla)

    # Primer fraude de una racha (la transacción anterior de la tarjeta fue legítima): el caso que importa detectar
    anterior_legitima = df.groupby("cc_num")["is_fraud"].shift(1).eq(0)
    candidatos = {
        "fraude": np.flatnonzero(en_prueba & (df["is_fraud"] == 1) & anterior_legitima & (df["posicion"] >= 10)),
        "legitima": np.flatnonzero(en_prueba & (df["is_fraud"] == 0) & (df["posicion"] >= 10)),
    }

    args.salida.mkdir(parents=True, exist_ok=True)
    for nombre, filas in candidatos.items():
        for fila in rng.permutation(filas)[:200]:
            transacciones = _historial(df, int(fila))
            puntaje = _puntaje(artefactos, transacciones)
            # Ejemplos claros para la demo: un fraude que el modelo bloquea y una compra normal que aprueba
            if (nombre == "fraude") == (puntaje >= umbral):
                ejemplo = {
                    "descripcion": f"Transacción {'fraudulenta' if nombre == 'fraude' else 'legítima'} real "
                                   f"del conjunto de prueba (Sparkov). La última es la que se evalúa.",
                    "etiqueta_real": nombre,
                    "puntaje_esperado": round(puntaje, 6),
                    "decision_esperada": "bloquear" if puntaje >= umbral else "aprobar",
                    "solicitud": {"transacciones": transacciones},
                }
                ruta = args.salida / f"{nombre}.json"
                ruta.write_text(json.dumps(ejemplo, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"{ruta}: puntaje {puntaje:.4f} (umbral {umbral})")
                break


if __name__ == "__main__":
    main()
