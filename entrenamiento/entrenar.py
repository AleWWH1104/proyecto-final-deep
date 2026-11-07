"""Pipeline completo: datos → entrenamiento de C v2 → umbral → comparación con líneas base → artefactos.

Uso:
    uv run python -m entrenamiento.entrenar --csv datos/fraudTrain.csv

Genera:
    artefactos/modelo.pt, artefactos/parametros.json  (lo que carga el servicio)
    reportes/metricas.json, reportes/*.png            (evidencia para el README y la presentación)
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from app.ml import preprocesamiento as pp
from app.ml.modelo import puntuar
from entrenamiento import datos, hipotesis, lineas_base, reportes
from entrenamiento.entrenador import ConfigEntrenamiento, config_a_dict, entrenar
from entrenamiento.metricas import (Costos, costo_sin_modelo, curva_costo, metricas_en_umbral,
                                    resumen_modelo)

log = logging.getLogger(__name__)

VERSION_MODELO = "C-v2"
LONGITUD_SECUENCIA = 10  # la misma del Proyecto 1


def ejecutar(ruta_csv: Path, dir_artefactos: Path, dir_reportes: Path,
             config: ConfigEntrenamiento = ConfigEntrenamiento(), costos: Costos = Costos(),
             dir_v1: Path | None = None, con_modelo_a: bool = True) -> dict:
    inicio = time.perf_counter()

    # 1. Datos y partición temporal
    df = datos.cargar_transacciones(ruta_csv)
    corte_val, corte_test = datos.cortes_temporales(df["trans_date_trans_time"])
    particion = datos.asignar_particion(df, corte_val, corte_test)
    es_train = particion == "train"
    log.info("Transacciones: %d | train %d | val %d | test %d", len(df), es_train.sum(),
             (particion == "val").sum(), (particion == "test").sum())

    # 2. Preprocesamiento ajustado SOLO con entrenamiento
    cat2cod = pp.construir_cat2cod(df.loc[es_train, "category"])
    escalado = pp.ajustar_escalado(df.loc[es_train, "amt"], df.loc[es_train, "hora"],
                                   df.loc[es_train, "dias_desde_anterior"])
    features = pp.calcular_features(df["amt"], pp.codificar_categorias(df["category"], cat2cod),
                                    df["hora"], df["dias_desde_anterior"], escalado)
    etiquetas = df["is_fraud"].to_numpy()
    conjuntos = datos.construir_conjuntos(features, df["posicion"].to_numpy(), etiquetas,
                                          particion, LONGITUD_SECUENCIA)
    tr, va, te = conjuntos["train"], conjuntos["val"], conjuntos["test"]

    # 3. Entrenamiento de C v2 (mejor época por AUC-PR en validación)
    modelo, arquitectura, historial = entrenar(tr.X, tr.y, va.X, va.y,
                                               n_categorias=len(cat2cod) + 1, config=config)
    scores = {"val": puntuar(modelo, va.X), "test": puntuar(modelo, te.X)}

    # 4. Resultados: umbral elegido en validación, reportado en prueba
    resultados = {"C_v2": resumen_modelo(va.y, scores["val"], te.y, scores["test"], costos)}
    resultados["sin_modelo"] = {"prueba": {"costo": costo_sin_modelo(te.y, costos),
                                           "fraudes_no_detectados": int(te.y.sum())}}

    if con_modelo_a:
        log.info("Entrenando línea base A (agregados + gradient boosting)...")
        agregados = lineas_base.construir_agregados(df)
        modelo_a = lineas_base.entrenar_modelo_a(agregados[es_train], etiquetas[es_train], config.semilla)
        scores_a = {n: modelo_a.predict_proba(agregados[particion == n])[:, 1] for n in ("val", "test")}
        resultados["A"] = resumen_modelo(va.y, scores_a["val"], te.y, scores_a["test"], costos)

    if dir_v1 is not None:
        log.info("Evaluando C v1 (artefacto del Proyecto 1)...")
        scores_v1 = lineas_base.puntuar_v1(df, particion, dir_v1)
        resultados["C_v1"] = resumen_modelo(va.y, scores_v1["val"], te.y, scores_v1["test"], costos)
        # Referencia: el umbral 0.06 del Proyecto 1 (elegido con prueba, por eso no se usa para decidir)
        umbral_v1_original = json.loads((dir_v1 / "preproc_params.json").read_text())["umbral_operativo"]
        resultados["C_v1"]["prueba_umbral_original"] = metricas_en_umbral(
            te.y, scores_v1["test"], round(umbral_v1_original, 3), costos)

    # 5. Hipótesis pre-registradas
    veredictos = {}
    if "A" in resultados:
        veredictos["H1"] = hipotesis.evaluar_h1(resultados["C_v2"], resultados["A"])
    if "C_v1" in resultados:
        veredictos["H2"] = hipotesis.evaluar_h2(resultados["C_v2"], resultados["C_v1"])

    # 6. Artefactos para el servicio
    umbral = resultados["C_v2"]["umbral"]
    dir_artefactos.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), dir_artefactos / "modelo.pt")
    parametros = {
        "version_modelo": VERSION_MODELO,
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columnas_features": list(pp.COLUMNAS_FEATURES),
        "longitud_secuencia": LONGITUD_SECUENCIA,
        "cat2cod": cat2cod,
        "escalado": {"media": list(escalado.media), "desviacion": list(escalado.desviacion)},
        "arquitectura": arquitectura,
        "entrenamiento": config_a_dict(config),
        "umbral_operativo": umbral,
        "costos": {"fraude_no_detectado": costos.fraude_no_detectado,
                   "bloqueo_legitimo": costos.bloqueo_legitimo},
        "particion": {"corte_validacion": str(corte_val), "corte_prueba": str(corte_test)},
        "referencia": _referencia_monitoreo(df, es_train, scores["val"], umbral),
    }
    (dir_artefactos / "parametros.json").write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. Reportes
    resumen = {
        "version_modelo": VERSION_MODELO,
        "datos": {"transacciones": len(df), "tarjetas": int(df["cc_num"].nunique()),
                  "tasa_fraude": float(etiquetas.mean()),
                  **{f"n_{n}": int((particion == n).sum()) for n in datos.PARTICIONES},
                  **parametros["particion"]},
        "historial_entrenamiento": historial,
        "resultados": resultados,
        "hipotesis": veredictos,
        "duracion_segundos": round(time.perf_counter() - inicio, 1),
    }
    dir_reportes.mkdir(parents=True, exist_ok=True)
    (dir_reportes / "metricas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    reportes.generar_figuras(dir_reportes, historial, curva_costo(va.y, scores["val"], costos),
                             umbral, resultados)
    log.info("Listo en %.0f s. Hipótesis: %s", resumen["duracion_segundos"],
             {h: v["cumple"] for h, v in veredictos.items()})
    return resumen


def _referencia_monitoreo(df, es_train, scores_val, umbral) -> dict:
    """Lo que el monitoreo compara contra el tráfico real para detectar cambios (deriva)."""
    montos = df.loc[es_train, "amt"]
    frecuencias = df.loc[es_train, "category"].value_counts(normalize=True)
    return {
        "monto_mediana": float(montos.median()),
        "monto_p95": float(montos.quantile(0.95)),
        "frecuencia_categorias": {c: float(f) for c, f in frecuencias.sort_index().items()},
        "puntaje_mediana_val": float(np.median(scores_val)),
        "tasa_bloqueo_val": float(np.mean(scores_val >= umbral)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("datos/fraudTrain.csv"))
    parser.add_argument("--artefactos", type=Path, default=Path("artefactos"))
    parser.add_argument("--reportes", type=Path, default=Path("reportes"))
    parser.add_argument("--v1", type=Path, default=Path("artefactos/v1_proyecto1"),
                        help="carpeta con el modelo C del Proyecto 1 (línea base de H2)")
    parser.add_argument("--epocas", type=int, default=ConfigEntrenamiento.epocas)
    parser.add_argument("--sin-modelo-a", action="store_true", help="omite la línea base A (más rápido)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ejecutar(args.csv, args.artefactos, args.reportes, ConfigEntrenamiento(epocas=args.epocas),
             dir_v1=args.v1 if args.v1.exists() else None, con_modelo_a=not args.sin_modelo_a)


if __name__ == "__main__":
    main()
