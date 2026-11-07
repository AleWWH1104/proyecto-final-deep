"""Figuras para el README y la presentación (PNG, fondo claro)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
REJILLA = "#e4e3df"
SERIE_1 = "#2a78d6"   # azul: el modelo de producción
SERIE_2 = "#eb6834"   # naranja: segunda serie
NEUTRO = "#a8a7a1"    # gris: referencias

NOMBRES = {"sin_modelo": "Sin modelo", "A": "A · agregados (línea base)",
           "C_v1": "C v1 · Proyecto 1", "C_v2": "C v2 · producción"}


def _estilo(ax, titulo: str) -> None:
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12, fontweight="bold")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TEXTO_SECUNDARIO, labelsize=9)
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def _guardar(fig, ruta: Path) -> None:
    fig.patch.set_facecolor(SUPERFICIE)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def figura_historial(historial: list[dict], ruta: Path) -> None:
    epocas = [h["epoca"] for h in historial]
    mejor = max(historial, key=lambda h: h["auc_pr_val"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _estilo(ax1, "AUC-PR en validación por época")
    ax1.plot(epocas, [h["auc_pr_val"] for h in historial], color=SERIE_1, linewidth=2, marker="o", markersize=5)
    ax1.annotate(f"mejor época: {mejor['epoca']} ({mejor['auc_pr_val']:.3f})",
                 (mejor["epoca"], mejor["auc_pr_val"]), textcoords="offset points", xytext=(-8, -22),
                 ha="right", color=TEXTO, fontsize=9)
    ax1.set_xlabel("época", color=TEXTO_SECUNDARIO)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    _estilo(ax2, "Pérdida (entropía cruzada)")
    ax2.plot(epocas, [h["perdida_train"] for h in historial], color=SERIE_1, linewidth=2, label="entrenamiento")
    ax2.plot(epocas, [h["perdida_val"] for h in historial], color=SERIE_2, linewidth=2, label="validación")
    ax2.legend(frameon=False, labelcolor=TEXTO, fontsize=9)
    ax2.set_xlabel("época", color=TEXTO_SECUNDARIO)
    _guardar(fig, ruta)


def figura_costo_umbral(curva: list[dict], umbral: float, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _estilo(ax, "Costo esperado vs. umbral (C v2, validación)")
    ax.plot([c["umbral"] for c in curva], [c["costo"] / 1000 for c in curva], color=SERIE_1, linewidth=2)
    elegido = next(c for c in curva if c["umbral"] == umbral)
    ax.axvline(umbral, color=NEUTRO, linestyle="--", linewidth=1)
    ax.plot([umbral], [elegido["costo"] / 1000], "o", color=SERIE_1, markersize=8,
            markeredgecolor=SUPERFICIE, markeredgewidth=2)
    ax.annotate(f"umbral elegido {umbral:.3f}\nQ{elegido['costo']:,.0f}", (umbral, elegido["costo"] / 1000),
                textcoords="offset points", xytext=(12, 12), color=TEXTO, fontsize=9)
    ax.set_xlabel("umbral de bloqueo", color=TEXTO_SECUNDARIO)
    ax.set_ylabel("costo (miles de Q)", color=TEXTO_SECUNDARIO)
    _guardar(fig, ruta)


def figura_costo_modelos(resultados: dict, ruta: Path) -> None:
    # "Sin modelo" es un orden de magnitud mayor: va como nota para no aplastar las barras que se comparan.
    orden = [m for m in ("A", "C_v1", "C_v2") if m in resultados]
    costos = [resultados[m]["prueba"]["costo"] for m in orden]

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(orden) + 1.6))
    _estilo(ax, "Costo del fraude en prueba (umbral elegido en validación)")
    ax.text(0, 1.02, f"Referencia sin modelo (dejar pasar todo): Q{resultados['sin_modelo']['prueba']['costo']:,.0f}",
            transform=ax.transAxes, color=TEXTO_SECUNDARIO, fontsize=9)
    ax.grid(axis="y", visible=False)
    colores = [SERIE_1 if m == "C_v2" else NEUTRO for m in orden]
    barras = ax.barh([NOMBRES[m] for m in orden], costos, color=colores, height=0.55)
    ax.invert_yaxis()
    for barra, costo in zip(barras, costos):
        ax.text(barra.get_width(), barra.get_y() + barra.get_height() / 2, f"  Q{costo:,.0f}",
                va="center", color=TEXTO, fontsize=9)
    ax.set_xlim(0, max(costos) * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"Q{v / 1e3:,.0f} mil"))
    ax.set_title(ax.get_title(loc="left"), loc="left", color=TEXTO, fontsize=12, fontweight="bold", pad=22)
    _guardar(fig, ruta)


def generar_figuras(directorio: Path, historial: list[dict], curva_val: list[dict],
                    umbral: float, resultados: dict) -> None:
    figura_historial(historial, directorio / "historial_entrenamiento.png")
    figura_costo_umbral(curva_val, umbral, directorio / "costo_vs_umbral.png")
    figura_costo_modelos(resultados, directorio / "costo_por_modelo.png")
