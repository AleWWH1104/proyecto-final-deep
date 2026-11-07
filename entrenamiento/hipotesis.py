"""Criterios de éxito pre-registrados (ver docs/hipotesis.md). No cambiar después de entrenar."""

# H1: C v2 frente a la línea base A (agregados + gradient boosting)
H1_RAZON_COSTO_MAXIMA = 0.50   # costo(C v2) / costo(A) en prueba
H1_MEJORA_AUC_PR_MINIMA = 0.10  # AUC-PR(C v2) - AUC-PR(A) en prueba

# H2: no inferioridad de C v2 frente a C v1 (Proyecto 1)
H2_MARGEN_AUC_PR = 0.01         # AUC-PR(C v2) >= AUC-PR(C v1) - margen

# H3: latencia del servicio (se evalúa con los registros del servicio, no aquí)
H3_LATENCIA_P95_MS = 300


def evaluar_h1(resultado_c: dict, resultado_a: dict) -> dict:
    razon_costo = resultado_c["prueba"]["costo"] / resultado_a["prueba"]["costo"]
    mejora_auc = resultado_c["prueba"]["auc_pr"] - resultado_a["prueba"]["auc_pr"]
    return {
        "razon_costo": razon_costo,
        "mejora_auc_pr": mejora_auc,
        "criterio": f"razon_costo <= {H1_RAZON_COSTO_MAXIMA} y mejora_auc_pr >= {H1_MEJORA_AUC_PR_MINIMA}",
        "cumple": bool(razon_costo <= H1_RAZON_COSTO_MAXIMA and mejora_auc >= H1_MEJORA_AUC_PR_MINIMA),
    }


def evaluar_h2(resultado_c_v2: dict, resultado_c_v1: dict) -> dict:
    diferencia = resultado_c_v2["prueba"]["auc_pr"] - resultado_c_v1["prueba"]["auc_pr"]
    return {
        "diferencia_auc_pr": diferencia,
        "criterio": f"diferencia_auc_pr >= -{H2_MARGEN_AUC_PR}",
        "cumple": bool(diferencia >= -H2_MARGEN_AUC_PR),
    }
