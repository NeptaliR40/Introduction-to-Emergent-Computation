"""
Utilidades de visualizacion construidas directamente sobre `matplotlib`.

Este modulo genera las cuatro familias de figuras del repositorio:

1. **Diagramas de arquitectura** (`dibujar_red`, `dibujar_red_mcp`): dibujan la
   red tal como aparece en las laminas de clase --- neuronas lineales como
   cuadrados y no lineales como circulos, sinapsis etiquetadas con su peso.
2. **Regiones y fronteras de decision** (`frontera_decision`,
   `evolucion_fronteras`, `haz_de_soluciones`): el plano x1-x2 dividido por la
   recta w0 + w1 x1 + w2 x2 = 0.
3. **Curvas de aprendizaje** (`curva_aprendizaje`, `curvas_comparadas`).
4. **Mapas de retina y superficies de error** (`mapa_retina`, `rejilla_retinas`,
   `contorno_error`, `superficie_error_3d`).
5. **Perceptron multicapa, Kohonen y Hopfield** (`region_continua`,
   `grupos_en_plano`, `malla_kohonen`, `mapa_calor`).

Criterios de diseno
-------------------
* Paleta categorica validada para vision con deficiencia de color (CVD): se
  usan como maximo tres tonos simultaneos --- azul, naranja y aqua --- que
  superan los umbrales de separacion perceptual en todos los pares.
* La identidad de clase **nunca** depende solo del color: cada clase lleva
  ademas una forma de marcador distinta (circulo / cuadrado).
* Ejes y rejilla recesivos, marcas finas, etiquetas directas cuando son pocas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: los experimentos guardan archivos
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# --- paleta -----------------------------------------------------------------
AZUL = "#2a78d6"      # categoria 1  (clase +1)
NARANJA = "#eb6834"   # categoria 2  (clase -1 / clase 0)
AQUA = "#1baf7a"      # categoria 3  (acento: solucion optima, referencia)
VIOLETA = "#4a3aa7"   # acento secundario (trayectorias)
TINTA = "#0b0b0b"     # texto primario y frontera de decision
TINTA_2 = "#52514e"   # texto secundario
GRIS = "#c9c8c3"      # rejilla
SUPERFICIE = "#fcfcfb"
AMARILLO = "#eda100"  # categoria 4 (solo en figuras de grupos)
MAGENTA = "#e87ba4"   # categoria 5 (solo en figuras de grupos)

#: Orden categorico fijo para figuras con mas de tres grupos (Kohonen).  Cada
#: grupo lleva ademas su propio marcador: la identidad nunca depende solo del color.
COLORES_GRUPO = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA]
MARCADORES_GRUPO = ["o", "s", "^", "D", "v"]

COLORES_CLASE = {1.0: AZUL, -1.0: NARANJA, 0.0: NARANJA}
MARCADORES_CLASE = {1.0: "o", -1.0: "s", 0.0: "s"}


def estilo() -> None:
    """Aplica el estilo comun a todas las figuras del repositorio."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": TINTA_2,
            "axes.labelcolor": TINTA,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIS,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def guardar(fig, ruta) -> Path:
    """Guarda la figura creando el directorio si hace falta y la cierra."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 1. Diagramas de arquitectura
# ---------------------------------------------------------------------------

def _dibujar_nodo(ax, xy, etiqueta, lineal=False, radio=0.28, color=TINTA, relleno="white"):
    """Dibuja una unidad de proceso.

    `claseRN02.md`: "aqui a las neuronas lineales se las representa por un
    cuadrado, y a las no lineales por un circulo".
    """
    if lineal:
        forma = Rectangle(
            (xy[0] - radio, xy[1] - radio), 2 * radio, 2 * radio,
            facecolor=relleno, edgecolor=color, linewidth=1.6, zorder=3,
        )
    else:
        forma = Circle(xy, radio, facecolor=relleno, edgecolor=color, linewidth=1.6, zorder=3)
    ax.add_patch(forma)
    ax.text(xy[0], xy[1], etiqueta, ha="center", va="center", fontsize=9, color=TINTA, zorder=4)
    return forma


def _dibujar_sinapsis(ax, origen, destino, peso=None, radio=0.28, color=GRIS, ancho=1.2):
    """Traza la conexion entre dos unidades, opcionalmente etiquetada con su peso."""
    dx, dy = destino[0] - origen[0], destino[1] - origen[1]
    dist = np.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    p0 = (origen[0] + ux * radio, origen[1] + uy * radio)
    p1 = (destino[0] - ux * radio, destino[1] - uy * radio)
    flecha = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9,
        linewidth=ancho, color=color, zorder=2, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(flecha)
    if peso is not None:
        t = 0.62
        ax.text(
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]) + 0.13,
            f"{peso:g}",
            fontsize=8, color=TINTA_2, ha="center", va="bottom", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=SUPERFICIE, edgecolor="none", alpha=0.9),
        )


def dibujar_red(
    capas: list[list[str]],
    titulo: str = "",
    pesos: list[np.ndarray] | None = None,
    capas_lineales: set[int] | None = None,
    etiquetas_capas: list[str] | None = None,
    nota: str = "",
    figsize=(8.5, 5.5),
):
    """Dibuja una red totalmente conectada capa a capa.

    Parameters
    ----------
    capas:
        Lista de capas; cada capa es la lista de etiquetas de sus unidades.
        La primera capa es la de entrada (se dibuja como unidades lineales,
        porque "lo que esperamos de un sensor es que indique precisamente lo que
        esta percibiendo").
    pesos:
        Lista con una matriz por cada par de capas consecutivas, de forma
        (n_destino, n_origen).  Si se da, cada sinapsis se rotula con su peso.
    capas_lineales:
        Indices de las capas cuyas unidades son lineales (cuadrados).  Por
        omision, solo la capa 0.
    etiquetas_capas:
        Titulo bajo cada capa ("Capa 0", "Capa 1", ...).
    """
    estilo()
    capas_lineales = {0} if capas_lineales is None else set(capas_lineales)
    fig, ax = plt.subplots(figsize=figsize)

    alto = max(len(c) for c in capas)
    posiciones: list[list[tuple[float, float]]] = []
    for k, capa in enumerate(capas):
        n = len(capa)
        ys = np.linspace((alto - n) / 2.0, (alto + n) / 2.0, n) if n > 1 else np.array([alto / 2.0])
        # se invierte para que la primera unidad declarada quede arriba
        posiciones.append([(k * 2.6, float(y)) for y in ys[::-1]])

    # sinapsis primero para que queden por debajo de los nodos
    for k in range(len(capas) - 1):
        W = None if pesos is None else np.atleast_2d(pesos[k])
        for j, destino in enumerate(posiciones[k + 1]):
            for i, origen in enumerate(posiciones[k]):
                peso = None if W is None else float(W[j, i])
                color = GRIS if peso is None else (AZUL if peso > 0 else (NARANJA if peso < 0 else GRIS))
                ancho = 1.2 if peso is None else 1.0 + min(abs(peso), 3.0) * 0.6
                _dibujar_sinapsis(ax, origen, destino, peso, color=color, ancho=ancho)

    for k, capa in enumerate(capas):
        for etiqueta, xy in zip(capa, posiciones[k]):
            _dibujar_nodo(ax, xy, etiqueta, lineal=(k in capas_lineales))

    if etiquetas_capas:
        for k, texto in enumerate(etiquetas_capas):
            ax.text(k * 2.6, -0.85, texto, ha="center", va="top", fontsize=9, color=TINTA_2)

    if nota:
        ax.text(
            0.5, -0.06, nota, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=TINTA_2,
        )

    leyenda = [
        Line2D([0], [0], color=AZUL, lw=2, label="peso excitador (w > 0)"),
        Line2D([0], [0], color=NARANJA, lw=2, label="peso inhibidor (w < 0)"),
    ]
    if pesos is not None:
        ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2)

    ax.set_title(titulo)
    ax.set_xlim(-1.0, (len(capas) - 1) * 2.6 + 1.0)
    ax.set_ylim(-1.4, alto + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def dibujar_red_mcp(red, titulo: str = "", figsize=(8.5, 5.0)):
    """Dibuja una `RedMCP` respetando su grafo de conexiones reales.

    A diferencia de `dibujar_red`, aqui las conexiones no son completas: cada
    neurona se enlaza unicamente con las fuentes que declaro.
    """
    estilo()
    fig, ax = plt.subplots(figsize=figsize)

    # nivel topologico de cada nodo
    nivel = {nombre: 0 for nombre in red.entradas}
    for nombre, _, fuentes in red.nodos:
        nivel[nombre] = 1 + max(nivel[f] for f in fuentes)

    por_nivel: dict[int, list[str]] = {}
    for nombre, k in nivel.items():
        por_nivel.setdefault(k, []).append(nombre)

    alto = max(len(v) for v in por_nivel.values())
    pos: dict[str, tuple[float, float]] = {}
    for k in sorted(por_nivel):
        nombres = por_nivel[k]
        n = len(nombres)
        ys = np.linspace((alto - n) / 2.0, (alto + n) / 2.0, n) if n > 1 else np.array([alto / 2.0])
        for nombre, y in zip(nombres, ys[::-1]):
            pos[nombre] = (k * 2.6, float(y))

    for nombre, neurona, fuentes in red.nodos:
        for peso, fuente in zip(neurona.pesos, fuentes):
            color = AZUL if peso > 0 else (NARANJA if peso < 0 else GRIS)
            _dibujar_sinapsis(ax, pos[fuente], pos[nombre], float(peso), color=color,
                              ancho=1.0 + min(abs(float(peso)), 3.0) * 0.5)

    for nombre in red.entradas:
        _dibujar_nodo(ax, pos[nombre], nombre, lineal=True)
    for nombre, neurona, _ in red.nodos:
        _dibujar_nodo(ax, pos[nombre], nombre, lineal=False, radio=0.32)
        ax.text(
            pos[nombre][0], pos[nombre][1] - 0.48, f"θ = {neurona.umbral:g}",
            ha="center", va="top", fontsize=8, color=TINTA_2,
        )

    ax.legend(
        handles=[
            Line2D([0], [0], color=AZUL, lw=2, label="conexion excitadora"),
            Line2D([0], [0], color=NARANJA, lw=2, label="conexion inhibidora"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.03), ncols=2,
    )
    ax.set_title(titulo or red.nombre)
    ax.set_xlim(-1.0, max(p[0] for p in pos.values()) + 1.0)
    ax.set_ylim(-1.4, alto + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Regiones y fronteras de decision
# ---------------------------------------------------------------------------

def _dibujar_patrones(ax, X, d, etiquetas=("clase +1", "clase -1")):
    """Dispersa los patrones usando color Y forma (identidad nunca solo color)."""
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    for valor, etiqueta in zip(sorted(set(d.tolist()), reverse=True), etiquetas):
        mascara = d == valor
        ax.scatter(
            X[mascara, 0], X[mascara, 1],
            s=130,
            c=COLORES_CLASE.get(valor, VIOLETA),
            marker=MARCADORES_CLASE.get(valor, "^"),
            edgecolors="white", linewidths=1.6, zorder=4,
            label=f"{etiqueta}  (d = {valor:g})",
        )


def frontera_decision(
    w,
    X,
    d,
    titulo: str = "",
    etiquetas_ejes=("x1", "x2"),
    etiquetas_clases=("clase +1", "clase -1"),
    margen: float = 0.8,
    predictor=None,
    anotar_patrones: bool = True,
    ax=None,
    figsize=(6.0, 5.2),
):
    """Dibuja los patrones, las regiones de decision y la recta frontera.

    `ICE-claseRN03.md`: "esto divide al plano formado por x1 y x2 en dos
    regiones ... la frontera entre ambas esta dada por la ecuacion lineal de la
    recta w0 + w1 x1 + w2 x2 = 0".

    Parameters
    ----------
    w:
        Vector de pesos extendido (w0, w1, w2) con w0 el sesgo/inclinacion.
    predictor:
        Funcion opcional f(X) -> salida, usada para colorear las regiones.  Si
        no se da, la region se colorea por el signo de w . x.
    """
    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).ravel()

    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    rejilla = np.column_stack([gx.ravel(), gy.ravel()])

    if predictor is None:
        z = w[0] + rejilla @ w[1:]
        z = np.where(z >= 0, 1.0, -1.0)
    else:
        z = np.asarray(predictor(rejilla), dtype=float).ravel()
    z = z.reshape(gx.shape)

    ax.contourf(
        gx, gy, z, levels=[-np.inf, -1e-9, 1e-9, np.inf],
        colors=[NARANJA, "#f5f4f1", AZUL], alpha=0.14, zorder=0,
    )

    # recta frontera w0 + w1 x1 + w2 x2 = 0
    if abs(w[2]) > 1e-12:
        xs = np.array([x_min, x_max])
        ys = -(w[0] + w[1] * xs) / w[2]
        ax.plot(xs, ys, color=TINTA, lw=2.2, zorder=3, label="frontera  w·x = 0")
    elif abs(w[1]) > 1e-12:
        xv = -w[0] / w[1]
        ax.axvline(xv, color=TINTA, lw=2.2, zorder=3, label="frontera  w·x = 0")

    _dibujar_patrones(ax, X, d, etiquetas_clases)

    if anotar_patrones and X.shape[0] <= 8:
        for fila in range(X.shape[0]):
            ax.annotate(
                f"({X[fila, 0]:g}, {X[fila, 1]:g})",
                (X[fila, 0], X[fila, 1]),
                textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=8, color=TINTA_2, zorder=5,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(etiquetas_ejes[0])
    ax.set_ylabel(etiquetas_ejes[1])
    ax.set_title(titulo)
    ax.legend(loc="best")
    return fig, ax


def evolucion_fronteras(
    trayectoria,
    X,
    d,
    titulo: str = "",
    max_rectas: int = 12,
    etiquetas_clases=("clase +1", "clase -1"),
    figsize=(6.2, 5.4),
):
    """Superpone las fronteras que la red fue adoptando durante el aprendizaje.

    Las rectas antiguas se dibujan mas transparentes; la final, solida.  Permite
    *ver* como la regla perceptronica va rotando y desplazando el hiperplano
    hasta dejar todos los patrones del lado correcto.
    """
    estilo()
    fig, ax = plt.subplots(figsize=figsize)
    X = np.asarray(X, dtype=float)

    pasos = [np.asarray(w, dtype=float).ravel() for w in trayectoria]
    if len(pasos) > max_rectas:
        indices = np.linspace(0, len(pasos) - 1, max_rectas).astype(int)
        pasos = [pasos[i] for i in indices]

    x_min, x_max = X[:, 0].min() - 0.9, X[:, 0].max() + 0.9
    y_min, y_max = X[:, 1].min() - 0.9, X[:, 1].max() + 0.9
    xs = np.array([x_min, x_max])

    for k, w in enumerate(pasos):
        ultimo = k == len(pasos) - 1
        alfa = 0.18 + 0.72 * (k / max(len(pasos) - 1, 1))
        color = TINTA if ultimo else VIOLETA
        ancho = 2.4 if ultimo else 1.2
        if abs(w[2]) > 1e-12:
            ax.plot(xs, -(w[0] + w[1] * xs) / w[2], color=color, alpha=alfa, lw=ancho, zorder=2)
        elif abs(w[1]) > 1e-12:
            ax.axvline(-w[0] / w[1], color=color, alpha=alfa, lw=ancho, zorder=2)

    _dibujar_patrones(ax, X, d, etiquetas_clases)
    ax.plot([], [], color=VIOLETA, alpha=0.6, lw=1.2, label="fronteras intermedias")
    ax.plot([], [], color=TINTA, lw=2.4, label="frontera final")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(titulo)
    ax.legend(loc="best")
    return fig, ax


def haz_de_soluciones(soluciones, X, d, titulo: str = "", figsize=(6.2, 5.4)):
    """Dibuja muchas fronteras validas sobre el mismo conjunto de patrones.

    Ilustra la lamina "Algoritmo Perceptronico: Infinitas Soluciones" de
    `ICE-claseRN03.md`: "o no existe ninguna solucion, o existen infinitas".
    """
    estilo()
    fig, ax = plt.subplots(figsize=figsize)
    X = np.asarray(X, dtype=float)
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xs = np.linspace(x_min, x_max, 2)

    for w in soluciones:
        w = np.asarray(w, dtype=float).ravel()
        if abs(w[2]) > 1e-12:
            ax.plot(xs, -(w[0] + w[1] * xs) / w[2], color=VIOLETA, alpha=0.25, lw=1.1, zorder=2)
        elif abs(w[1]) > 1e-12:
            ax.axvline(-w[0] / w[1], color=VIOLETA, alpha=0.25, lw=1.1, zorder=2)

    _dibujar_patrones(ax, X, d)
    ax.plot([], [], color=VIOLETA, alpha=0.6, lw=1.4, label=f"{len(soluciones)} soluciones validas")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(titulo)
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Curvas de aprendizaje
# ---------------------------------------------------------------------------

def curva_aprendizaje(
    y,
    titulo: str = "",
    etiqueta_y: str = "error",
    etiqueta_x: str = "epoca",
    color: str = AZUL,
    escala_log: bool = False,
    referencia: float | None = None,
    etiqueta_referencia: str = "",
    ax=None,
    figsize=(6.4, 4.2),
):
    """Curva de una magnitud a lo largo de las epocas (error, ECM, exactitud)."""
    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y = np.asarray(y, dtype=float)
    x = np.arange(1, y.size + 1)
    ax.plot(x, y, color=color, marker="o" if y.size <= 30 else None, markersize=4, zorder=3)

    if referencia is not None:
        ax.axhline(referencia, color=AQUA, lw=1.6, ls="--", zorder=2,
                   label=etiqueta_referencia or f"referencia = {referencia:g}")
        ax.legend(loc="best")

    if escala_log:
        ax.set_yscale("log")
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)

    # etiqueta directa del valor final (etiquetado selectivo, no en cada punto)
    if y.size:
        ax.annotate(
            f"final: {y[-1]:.4g}",
            (x[-1], y[-1]), textcoords="offset points", xytext=(-6, 10),
            ha="right", fontsize=9, color=TINTA_2,
        )
    return fig, ax


def curvas_comparadas(
    series: dict[str, np.ndarray],
    titulo: str = "",
    etiqueta_y: str = "ECM",
    etiqueta_x: str = "epoca",
    escala_log: bool = True,
    x=None,
    figsize=(7.0, 4.4),
):
    """Varias curvas de aprendizaje en un mismo eje (maximo tres series).

    Se limita deliberadamente a tres tonos, que son los que superan los umbrales
    de separacion perceptual para todos los pares.  Cada serie lleva ademas su
    etiqueta directa al final de la linea.
    """
    estilo()
    fig, ax = plt.subplots(figsize=figsize)
    colores = [AZUL, NARANJA, AQUA, VIOLETA]
    estilos = ["-", "--", "-.", ":"]

    for k, (nombre, y) in enumerate(series.items()):
        y = np.asarray(y, dtype=float)
        eje_x = np.arange(1, y.size + 1) if x is None else np.asarray(x, dtype=float)
        ax.plot(eje_x, y, color=colores[k % len(colores)], ls=estilos[k % len(estilos)],
                marker="o" if y.size <= 30 else None, markersize=4,
                label=nombre, zorder=3)

    if escala_log:
        ax.set_yscale("log")
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.legend(loc="best")
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Retinas y superficies de error
# ---------------------------------------------------------------------------

def mapa_retina(vector, titulo: str = "", forma=(5, 5), ax=None, figsize=(2.6, 2.8), divergente=False):
    """Dibuja un patron de retina (o un vector de pesos) como imagen.

    Con `divergente=True` usa una escala azul-blanco-naranja centrada en cero,
    apropiada para pesos con signo: azul = peso positivo, naranja = negativo.
    """
    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    M = np.asarray(vector, dtype=float).reshape(forma)
    if divergente:
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

        mapa = LinearSegmentedColormap.from_list("div", [NARANJA, "#f0efec", AZUL])
        limite = max(abs(M).max(), 1e-9)
        norma = TwoSlopeNorm(vmin=-limite, vcenter=0.0, vmax=limite)
        im = ax.imshow(M, cmap=mapa, norm=norma)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        from matplotlib.colors import ListedColormap

        ax.imshow(M, cmap=ListedColormap(["#f0efec", AZUL]), vmin=-1, vmax=1)

    ax.set_xticks(range(forma[1]))
    ax.set_yticks(range(forma[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="white", linewidth=1.2)
    ax.set_title(titulo, fontsize=10)
    for borde in ax.spines.values():
        borde.set_visible(False)
    return fig, ax


def rejilla_retinas(patrones, titulos, forma=(5, 5), n_columnas=4, divergente=False, sup_titulo=""):
    """Rejilla de retinas (por ejemplo: prototipos limpios y versiones ruidosas)."""
    estilo()
    n = len(patrones)
    n_col = min(n_columnas, n)
    n_fil = int(np.ceil(n / n_col))
    fig, ejes = plt.subplots(n_fil, n_col, figsize=(2.3 * n_col, 2.5 * n_fil))
    ejes = np.atleast_1d(ejes).ravel()
    for k, (patron, titulo) in enumerate(zip(patrones, titulos)):
        mapa_retina(patron, titulo, forma=forma, ax=ejes[k], divergente=divergente)
    for eje in ejes[n:]:
        eje.axis("off")
    if sup_titulo:
        fig.suptitle(sup_titulo, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94) if sup_titulo else None)
    return fig, ejes


def contorno_error(Wi, Wj, E, trayectoria=None, indices=(1, 2), optimo=None,
                   titulo: str = "", figsize=(6.4, 5.2)):
    """Curvas de nivel del error cuadratico medio con la trayectoria del aprendizaje.

    La superficie de error de un ADALINE es un paraboloide: las curvas de nivel
    son elipses concentricas alrededor del minimo global.  La trayectoria
    muestra el "descenso del gradiente" literalmente bajando por ese cuenco.
    """
    estilo()
    fig, ax = plt.subplots(figsize=figsize)

    niveles = np.logspace(np.log10(max(E.min(), 1e-6) + 1e-9), np.log10(E.max() + 1e-9), 18)
    cs = ax.contourf(Wi, Wj, E, levels=niveles, cmap="Blues", alpha=0.85)
    ax.contour(Wi, Wj, E, levels=niveles, colors="white", linewidths=0.5)
    fig.colorbar(cs, ax=ax, label="ECM  E(w)")

    if trayectoria is not None:
        T = np.array([np.asarray(w, dtype=float).ravel() for w in trayectoria])
        ax.plot(T[:, indices[0]], T[:, indices[1]], color=NARANJA, lw=1.8, zorder=3,
                label="trayectoria del aprendizaje")
        ax.scatter(T[0, indices[0]], T[0, indices[1]], s=90, color=NARANJA, marker="o",
                   edgecolors="white", zorder=4, label="pesos iniciales")

    if optimo is not None:
        optimo = np.asarray(optimo, dtype=float).ravel()
        ax.scatter(optimo[indices[0]], optimo[indices[1]], s=160, color=AQUA, marker="*",
                   edgecolors="white", linewidths=1.2, zorder=5, label="minimo global (minimos cuadrados)")

    ax.set_xlabel(f"w{indices[0]}")
    ax.set_ylabel(f"w{indices[1]}")
    ax.set_title(titulo)
    ax.legend(loc="best")
    return fig, ax


def superficie_error_3d(Wi, Wj, E, indices=(1, 2), titulo: str = "", figsize=(7.0, 5.4)):
    """Vista tridimensional del paraboloide de error del ADALINE."""
    estilo()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Wi, Wj, E, cmap="Blues", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel(f"w{indices[0]}")
    ax.set_ylabel(f"w{indices[1]}")
    ax.set_zlabel("ECM  E(w)")
    ax.set_title(titulo)
    ax.view_init(elev=26, azim=-130)
    return fig, ax


def dibujar_activaciones(figsize=(8.6, 5.0)):
    """Panel con las funciones de activacion del catalogo de `claseRN01.md`."""
    from .activaciones import (Escalon, EscalonBipolar, EscalonBipolarConZona,
                               Gaussiana, Identidad, Sigmoide, SigmoideBipolar)

    estilo()
    v = np.linspace(-6, 6, 1000)
    paneles = [
        (Escalon(0.0), "Escalon (Heaviside)", r"$F(x)=\theta(x)$"),
        (EscalonBipolar(), "Escalon bipolar", r"$F(x)=2\theta(x)-1$"),
        (EscalonBipolarConZona(1.0), "Bipolar con indeterminacion", r"$\theta=1$"),
        (Sigmoide(), "Sigmoide", r"$F(x)=\frac{1}{1+e^{-x}}$"),
        (SigmoideBipolar(), "Sigmoide bipolar", r"$F(x)=\frac{1-e^{-x}}{1+e^{-x}}$"),
        (Gaussiana(0.0, 1.5), "Gaussiana", r"$F(x)=e^{-(x-\mu)^2/2\sigma^2}$"),
    ]
    fig, ejes = plt.subplots(2, 3, figsize=figsize)
    for eje, (phi, titulo, formula) in zip(ejes.ravel(), paneles):
        eje.plot(v, phi(v), color=AZUL, lw=2.2)
        eje.axhline(0, color=GRIS, lw=0.8)
        eje.axvline(0, color=GRIS, lw=0.8)
        eje.set_title(titulo, fontsize=10)
        eje.text(0.03, 0.06, formula, transform=eje.transAxes, fontsize=9, color=TINTA_2)
        eje.set_ylim(-1.35, 1.35)
    fig.suptitle("Funciones de activacion del modelo de neurona", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ejes


def tabla_a_figura(df, titulo: str = "", figsize=None, resaltar=None):
    """Renderiza un `DataFrame` pequeno como figura (util para tablas de verdad)."""
    estilo()
    n_fil, n_col = df.shape
    if figsize is None:
        figsize = (1.35 * n_col + 1.0, 0.45 * n_fil + 1.2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tabla = ax.table(
        cellText=[[f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v) for v in fila]
                  for fila in df.values],
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.45)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor(GRIS)
        if fila == 0:
            celda.set_facecolor("#eef4fd")
            celda.set_text_props(color=TINTA, fontweight="bold")
        elif resaltar and col == resaltar:
            celda.set_facecolor("#f7f9fc")
    ax.set_title(titulo, pad=16)
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Perceptron multicapa, Kohonen y Hopfield
# ---------------------------------------------------------------------------

def region_continua(predictor, X, d, titulo: str = "", etiquetas_ejes=("x1", "x2"),
                    margen: float = 0.35, ax=None, figsize=(6.0, 5.2)):
    """Salida continua de una red en el plano, con la curva de nivel 0.5.

    Para un MLP la frontera de decision ya no es una recta: es la curva
    y(x1, x2) = 0.5, que puede ser cualquier forma.  El fondo muestra la salida
    real de la red (escala secuencial de un solo tono).
    """
    from matplotlib.colors import LinearSegmentedColormap

    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    X = np.asarray(X, dtype=float)
    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    z = np.asarray(predictor(np.column_stack([gx.ravel(), gy.ravel()])), dtype=float).reshape(gx.shape)
    mapa = LinearSegmentedColormap.from_list("sec", ["#f5f4f1", "#86b6ef", "#1c5cab"])
    im = ax.contourf(gx, gy, z, levels=np.linspace(0, 1, 11), cmap=mapa, zorder=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="salida de la red  y")
    ax.contour(gx, gy, z, levels=[0.5], colors=[TINTA], linewidths=2.0, zorder=2)
    d = np.asarray(d, dtype=float).ravel()
    for valor, marcador, color, etiqueta in ((1.0, "o", AZUL, "d = 1"), (0.0, "s", NARANJA, "d = 0")):
        m = d == valor
        ax.scatter(X[m, 0], X[m, 1], s=140, c=color, marker=marcador, edgecolors="white",
                   linewidths=1.8, zorder=4, label=etiqueta)
    ax.plot([], [], color=TINTA, lw=2.0, label="frontera  y = 0.5")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(etiquetas_ejes[0])
    ax.set_ylabel(etiquetas_ejes[1])
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
    ax.grid(False)
    return fig, ax


def grupos_en_plano(X, grupos, titulo: str = "", nombres=None, ax=None, figsize=(6.4, 5.4)):
    """Dispersion de puntos coloreados Y marcados por grupo (maximo cinco)."""
    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    X = np.asarray(X, dtype=float)
    grupos = np.asarray(grupos).astype(int)
    for k, g in enumerate(np.unique(grupos)):
        m = grupos == g
        nombre = nombres[k] if nombres else f"grupo {g}"
        ax.scatter(X[m, 0], X[m, 1], s=26, c=COLORES_GRUPO[k % 5], marker=MARCADORES_GRUPO[k % 5],
                   edgecolors="white", linewidths=0.6, zorder=3, label=f"{nombre} ({m.sum()})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(titulo)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def malla_kohonen(W, filas, columnas, X=None, titulo: str = "", ax=None, figsize=(6.0, 5.4)):
    """Prototipos del mapa dibujados en el espacio de los datos, unidos como en la rejilla.

    Las lineas unen neuronas vecinas en la rejilla: si el mapa esta bien
    ordenado, la malla se despliega sobre los datos sin cruzarse.
    """
    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if X is not None:
        X = np.asarray(X, dtype=float)
        ax.scatter(X[:, 0], X[:, 1], s=10, c=GRIS, zorder=1, label="patrones")
    M = np.asarray(W, dtype=float).reshape(filas, columnas, -1)
    for f in range(filas):
        ax.plot(M[f, :, 0], M[f, :, 1], color=AZUL, lw=1.0, zorder=2)
    for c in range(columnas):
        ax.plot(M[:, c, 0], M[:, c, 1], color=AZUL, lw=1.0, zorder=2)
    ax.scatter(M[..., 0].ravel(), M[..., 1].ravel(), s=14, c=AZUL, edgecolors="white",
               linewidths=0.6, zorder=3, label="prototipos w_j")
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def mapa_calor(M, titulo: str = "", etiquetas_filas=None, etiquetas_columnas=None,
               anotar: bool = True, formato: str = "{:.2f}", divergente: bool = False,
               etiqueta_barra: str = "", ax=None, figsize=(5.4, 4.6)):
    """Matriz como mapa de calor: secuencial (un tono) o divergente (dos tonos + gris)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    estilo()
    creada = ax is None
    if creada:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    M = np.asarray(M, dtype=float)
    if divergente:
        mapa = LinearSegmentedColormap.from_list("div", [NARANJA, "#f0efec", AZUL])
        limite = max(np.abs(M).max(), 1e-9)
        im = ax.imshow(M, cmap=mapa, norm=TwoSlopeNorm(vmin=-limite, vcenter=0.0, vmax=limite))
    else:
        mapa = LinearSegmentedColormap.from_list("sec", ["#f5f4f1", "#86b6ef", "#1c5cab", "#0d366b"])
        im = ax.imshow(M, cmap=mapa)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=etiqueta_barra)
    if anotar and M.size <= 400:
        umbral = np.nanmax(np.abs(M)) * 0.6
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, formato.format(v), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > umbral and not divergente else TINTA)
    if etiquetas_filas is not None:
        ax.set_yticks(range(len(etiquetas_filas)))
        ax.set_yticklabels(etiquetas_filas)
    if etiquetas_columnas is not None:
        ax.set_xticks(range(len(etiquetas_columnas)))
        ax.set_xticklabels(etiquetas_columnas)
    ax.grid(False)
    ax.set_title(titulo)
    return fig, ax
