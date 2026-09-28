"""
Experimento 11 --- Red de Hopfield: reconocimiento de las letras A, B, C y D.

Actividad 5 de la guia de evaluacion.  Una red de Hopfield de 42 neuronas
almacena cuatro imagenes binarias de 7x6 pixeles y se evalua su capacidad de
recuperarlas a partir de versiones contaminadas con ruido.

Contenido
---------
1. Representacion de las imagenes como patrones bipolares de 42 elementos.
2. Diseno de la red y almacenamiento (regla de Hebb).
3. Recuperacion de patrones con ruido: ejemplos visuales y evolucion de la energia.
4. Evaluacion cuantitativa: tasa de reconocimiento en funcion del ruido.
5. Reconocimientos incorrectos: estados espurios, inversos y confusiones.
6. El diseno de los patrones importa: letras de trazo fino y solapamiento.
7. Regla de la pseudoinversa y capacidad de la red.

Ejecucion
---------
    python experimentos/exp11_hopfield.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import visual as vz
from ce_rna.hopfield import RedHopfield
from ce_rna.reportes import Reporte, encabezado_experimento

FORMA = ds.FORMA_7X6
NOMBRES = list("ABCD")
NIVELES_RUIDO = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21)
ENSAYOS = 200             # patrones ruidosos por letra y nivel de ruido
SEMILLA = 0


def evaluar(red: RedHopfield, patrones, niveles=NIVELES_RUIDO, ensayos=ENSAYOS) -> pd.DataFrame:
    """Recupera `ensayos` versiones ruidosas de cada patron por nivel de ruido."""
    filas = []
    for k in niveles:
        for mu, patron in enumerate(patrones):
            conteo = {"patron": 0, "otro": 0, "inverso": 0, "espurio": 0}
            iteraciones = []
            for e in range(ensayos):
                ruidoso = ds.contaminar(patron, k, semilla=100_000 * k + 1000 * mu + e)
                res = red.recuperar(ruidoso, semilla=e)
                iteraciones.append(res.iteraciones)
                indice, tipo = red.identificar(res.estado)
                if tipo == "patron":
                    conteo["patron" if indice == mu else "otro"] += 1
                else:
                    conteo[tipo] += 1
            filas.append({
                "pixeles_invertidos": k, "ruido_%": 100 * k / patron.size, "letra": NOMBRES[mu],
                "correcto_%": 100 * conteo["patron"] / ensayos,
                "otra_letra_%": 100 * conteo["otro"] / ensayos,
                "inverso_%": 100 * conteo["inverso"] / ensayos,
                "espurio_%": 100 * conteo["espurio"] / ensayos,
                "barridos_medios": float(np.mean(iteraciones)),
            })
    return pd.DataFrame(filas)


def curvas_por_letra(df: pd.DataFrame, titulo: str):
    series = {}
    for letra in NOMBRES:
        sub = df[df["letra"] == letra]
        series[letra] = sub["correcto_%"].to_numpy()
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.2, 4.4))
    colores = [vz.AZUL, vz.NARANJA, vz.AQUA, vz.AMARILLO]
    marcadores = ["o", "s", "^", "D"]
    x = df[df["letra"] == NOMBRES[0]]["pixeles_invertidos"].to_numpy()
    for k, (letra, y) in enumerate(series.items()):
        ax.plot(x, y, color=colores[k], marker=marcadores[k], markersize=6, lw=2.0, label=f"letra {letra}")
    ax.set_xlabel("pixeles invertidos (de 42)")
    ax.set_ylabel("recuperacion correcta (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(titulo)
    ax.legend(loc="lower left")
    return fig, ax


def main() -> None:
    encabezado_experimento("Experimento 11 - Red de Hopfield: letras A, B, C, D")

    reporte = Reporte(
        titulo="Experimento 11 --- Red de Hopfield: reconocimiento de las letras A, B, C y D",
        nombre_archivo="11_hopfield",
        resumen=(
            "Una red de Hopfield es una **memoria asociativa**: almacena un conjunto de patrones "
            "y, a partir de una version incompleta o contaminada de uno de ellos, evoluciona "
            "hasta recuperarlo. No hay salida deseada ni error que minimizar durante el uso: la "
            "red recorre su funcion de energia cuesta abajo hasta un minimo, y los patrones "
            "almacenados son esos minimos. El experimento disena una red de 42 neuronas para "
            "cuatro letras de 7x6 pixeles, mide su tasa de reconocimiento frente al ruido y "
            "analiza cuando y por que falla."
        ),
    )

    letras = ds.letras_abcd("gruesa")
    P = letras.X

    # ------------------------------------------------------------------
    # 1. Representacion
    # ------------------------------------------------------------------
    reporte.seccion("1. Representacion de las imagenes")
    fig, _ = vz.rejilla_retinas(list(P), [f"letra {n}" for n in NOMBRES], forma=FORMA,
                                sup_titulo="Los cuatro patrones almacenados (7x6 = 42 pixeles)")
    reporte.figura(vz.guardar(fig, FIGURAS / "11_letras.png"), "Imagenes originales.")
    reporte.texto(
        "Cada imagen es una matriz de 7 filas por 6 columnas. Se lee fila a fila y se convierte "
        "en un vector de 42 elementos con codificacion **bipolar**: pixel negro = +1, blanco = "
        "−1. La codificacion bipolar (y no binaria 0/1) es la natural en Hopfield: con 0/1 la "
        "regla de Hebb no refuerza las coincidencias de pixeles blancos, y el umbral de cada "
        "neurona tendria que compensar la actividad media."
    )
    tabla_vec = pd.DataFrame({
        "letra": NOMBRES,
        "vector (fila a fila, # = +1, . = -1)": [
            " ".join("".join("#" if v > 0 else "." for v in fila) for fila in p.reshape(FORMA)) for p in P],
        "pixeles negros": [(p > 0).sum() for p in P],
    })
    reporte.tabla(tabla_vec, "Patrones como vectores de 42 elementos", nombre_csv="11_patrones")
    pd.DataFrame(P.astype(int), index=NOMBRES).to_csv(_ruta.TABLAS / "11_patrones_bipolares.csv")

    # ------------------------------------------------------------------
    # 2. Diseno y almacenamiento
    # ------------------------------------------------------------------
    reporte.seccion("2. Diseno de la red y almacenamiento")
    red = RedHopfield(42, regla="hebb").almacenar(P)
    print(red.resumen())
    reporte.lista([
        "**42 neuronas**, una por pixel; el estado de la red *es* la imagen.",
        "Conexion **total y recurrente**: cada neurona recibe la salida de las otras 41. "
        "Pesos simetricos w_ij = w_ji y sin autoconexiones w_ii = 0 → 42·41/2 = 861 pesos distintos.",
        "Neuronas bipolares con funcion signo y umbral 0.",
        "Almacenamiento con la **regla de Hebb** en una sola pasada (no hay iteraciones de entrenamiento):",
    ])
    reporte.formula(r"w_{ij} = \frac{1}{N}\sum_{\mu=1}^{P} \xi_i^{\mu}\,\xi_j^{\mu}\quad (i \neq j), "
                    r"\qquad w_{ii} = 0, \qquad N = 42,\; P = 4")
    reporte.texto("Recuperacion con **dinamica asincrona**: en cada barrido se visitan las 42 "
                  "neuronas en orden aleatorio y cada una se actualiza con el estado mas reciente,")
    reporte.formula(r"s_i \leftarrow \operatorname{sgn}\Bigl(\sum_{j} w_{ij}\, s_j\Bigr)")
    reporte.texto("hasta que un barrido completo no cambia ninguna neurona (punto fijo). Con pesos "
                  "simetricos y actualizacion asincrona la energia")
    reporte.formula(r"E(\mathbf{s}) = -\tfrac{1}{2}\sum_{i,j} w_{ij}\, s_i\, s_j")
    reporte.texto(
        "no aumenta nunca (cada cambio de una neurona la reduce estrictamente), y como el numero "
        "de estados es finito la red **siempre converge**. Los patrones almacenados deberian ser "
        "minimos locales de E: los *atractores* de la dinamica."
    )
    fig, _ = vz.mapa_calor(red.W, titulo="Matriz de pesos W (42x42)", anotar=False, divergente=True,
                           etiqueta_barra="w_ij", figsize=(6.2, 5.4))
    reporte.figura(vz.guardar(fig, FIGURAS / "11_matriz_pesos.png"),
                   "Pesos simetricos con diagonal nula. Azul: pixeles que suelen coincidir; naranja: "
                   "pixeles que suelen ser opuestos.")
    fijos = pd.DataFrame({
        "letra": NOMBRES,
        "es_punto_fijo": [red.es_punto_fijo(p) for p in P],
        "energia": [red.energia(p) for p in P],
    })
    reporte.tabla(fijos, "Los cuatro patrones son puntos fijos de la red", nombre_csv="11_puntos_fijos")

    solap = (P @ P.T) / 42
    fig, _ = vz.mapa_calor(solap, titulo="Solapamiento entre letras  m = ξ·ξ'/N",
                           etiquetas_filas=NOMBRES, etiquetas_columnas=NOMBRES, divergente=True,
                           etiqueta_barra="solapamiento", figsize=(4.8, 4.0))
    reporte.figura(vz.guardar(fig, FIGURAS / "11_solapamiento.png"),
                   "Solapamiento entre pares de letras: 1 en la diagonal; cuanto mas cerca de 0 fuera de ella, mejor.")
    reporte.texto(
        f"El solapamiento maximo entre dos letras distintas es {np.max(np.abs(solap - np.eye(4))):.2f} "
        f"(entre {NOMBRES[np.unravel_index(np.argmax(np.abs(solap - np.eye(4))), solap.shape)[0]]} y "
        f"{NOMBRES[np.unravel_index(np.argmax(np.abs(solap - np.eye(4))), solap.shape)[1]]}). "
        "Esta magnitud es la que decide si la regla de Hebb funciona: el campo local de la neurona "
        "i cuando la red esta en el patron ν es"
    )
    reporte.formula(r"h_i = \sum_j w_{ij}\,\xi_j^{\nu} \approx \xi_i^{\nu} + "
                    r"\sum_{\mu \neq \nu} \xi_i^{\mu}\, m_{\mu\nu}")
    reporte.texto(
        "El primer termino (senal) empuja hacia el patron correcto; el segundo (**diafonia** o "
        "*crosstalk*) es la interferencia de las demas memorias. Si la diafonia supera a la senal "
        "en algun pixel, el patron deja de ser un punto fijo. Las letras se disenaron con trazo "
        "grueso precisamente para mantener bajos los solapamientos (seccion 6)."
    )

    # ------------------------------------------------------------------
    # 3. Recuperacion: ejemplos
    # ------------------------------------------------------------------
    reporte.seccion("3. Recuperacion de patrones contaminados")
    niveles_ejemplo = (4, 8, 12)
    patrones_fig, titulos_fig, filas_ej = [], [], []
    for mu, nombre in enumerate(NOMBRES):
        for k in niveles_ejemplo:
            ruidoso = ds.contaminar(P[mu], k, semilla=7 + 31 * mu + k)
            res = red.recuperar(ruidoso, semilla=mu)
            indice, tipo = red.identificar(res.estado)
            resultado = NOMBRES[indice] if tipo == "patron" else (f"-{NOMBRES[indice]}" if tipo == "inverso" else "espurio")
            patrones_fig += [ruidoso, res.estado]
            titulos_fig += [f"{nombre} con {k} px", f"→ {resultado} ({res.iteraciones} it.)"]
            filas_ej.append({"letra": nombre, "pixeles_invertidos": k,
                             "distancia_inicial": int(np.sum(ruidoso != P[mu])),
                             "resultado": resultado, "barridos": res.iteraciones,
                             "neuronas_cambiadas": res.cambios,
                             "E_inicial": res.energia[0], "E_final": res.energia[-1]})
    fig, _ = vz.rejilla_retinas(patrones_fig, titulos_fig, forma=FORMA, n_columnas=6,
                                sup_titulo="Pares (imagen contaminada → imagen recuperada)")
    reporte.figura(vz.guardar(fig, FIGURAS / "11_recuperacion_ejemplos.png"),
                   "Cada par muestra la imagen con ruido y el estado final de la red.")
    reporte.tabla(pd.DataFrame(filas_ej), "Resultados de los ejemplos", nombre_csv="11_ejemplos")

    # evolucion de la energia en un ejemplo, con estados intermedios
    ruidoso = ds.contaminar(P[1], 12, semilla=3)
    rng = np.random.default_rng(3)
    s = ruidoso.copy()
    energias, instantaneas = [red.energia(s)], [s.copy()]
    for barrido in range(3):
        for i in rng.permutation(42):
            h = red.W[i] @ s
            if h != 0:
                s[i] = np.sign(h)
            energias.append(red.energia(s))
        instantaneas.append(s.copy())
    fig, ejes = vz.plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ejes[0].plot(np.arange(len(energias)), energias, color=vz.AZUL, lw=2.0)
    for b in range(1, 4):
        ejes[0].axvline(42 * b, color=vz.GRIS, lw=1.0, ls="--")
    ejes[0].axhline(red.energia(P[1]), color=vz.AQUA, lw=1.4, ls="-.", label="energia del patron B")
    ejes[0].set_xlabel("actualizaciones individuales de neuronas (lineas: fin de cada barrido)")
    ejes[0].set_ylabel("energia E(s)")
    ejes[0].set_title("La energia nunca aumenta")
    ejes[0].legend(loc="upper right")
    ejes[1].axis("off")
    for k, estado in enumerate(instantaneas):
        ax_k = ejes[1].inset_axes([k * 0.25, 0.1, 0.23, 0.8])
        vz.mapa_retina(estado, "inicio" if k == 0 else f"barrido {k}", forma=FORMA, ax=ax_k)
    fig.tight_layout()
    indice_final, tipo_final = red.identificar(s)
    destino = f"la letra {NOMBRES[indice_final]}" if tipo_final == "patron" else f"un estado {tipo_final}"
    reporte.figura(vz.guardar(fig, FIGURAS / "11_energia.png"),
                   f"Letra B con 12 pixeles invertidos: la energia desciende en escalones hasta "
                   f"{destino} (E = {energias[-1]:.2f}).")
    reporte.texto(
        "La energia es una funcion de Lyapunov de la dinamica: cada vez que una neurona cambia, E "
        "disminuye; cuando ninguna puede cambiar, la red esta en un minimo local. Recordar es "
        "**descender por la superficie de energia** desde el punto de partida (la imagen "
        "contaminada) hasta el fondo del valle en que cae. La mayor parte de la correccion ocurre "
        "en el primer barrido."
    )

    # ------------------------------------------------------------------
    # 4. Evaluacion cuantitativa
    # ------------------------------------------------------------------
    reporte.seccion("4. Tasa de reconocimiento frente al ruido")
    df = evaluar(red, P)
    print(df.groupby("pixeles_invertidos")[["correcto_%", "otra_letra_%", "inverso_%", "espurio_%"]].mean())
    reporte.texto(
        f"Para cada letra y cada nivel de ruido (k pixeles invertidos elegidos al azar) se generan "
        f"{ENSAYOS} imagenes contaminadas distintas y se deja evolucionar la red. El resultado se "
        "clasifica en: recuperacion **correcta**, **otra letra** almacenada, el **inverso** de "
        "una letra, o un **estado espurio** (un minimo que no es ninguna letra)."
    )
    resumen = (df.groupby(["pixeles_invertidos", "ruido_%"])
               [["correcto_%", "otra_letra_%", "inverso_%", "espurio_%", "barridos_medios"]]
               .mean().reset_index())
    reporte.tabla(resumen, "Promedio sobre las cuatro letras", nombre_csv="11_reconocimiento")
    df.to_csv(_ruta.TABLAS / "11_reconocimiento_por_letra.csv", index=False)
    fig, _ = curvas_por_letra(df, "Recuperacion correcta segun el ruido (regla de Hebb)")
    reporte.figura(vz.guardar(fig, FIGURAS / "11_reconocimiento_ruido.png"),
                   f"Porcentaje de recuperaciones correctas por letra ({ENSAYOS} ensayos por punto).")
    tabla_letras = df.pivot(index="pixeles_invertidos", columns="letra", values="correcto_%").reset_index()
    reporte.tabla(tabla_letras, "Recuperacion correcta (%) por letra", nombre_csv="11_reconocimiento_tabla")

    umbral90 = {}
    for letra in NOMBRES:
        sub = df[(df["letra"] == letra) & (df["correcto_%"] >= 90)]
        umbral90[letra] = int(sub["pixeles_invertidos"].max()) if len(sub) else 0
    reporte.texto(
        "Maximo ruido con al menos 90 % de recuperacion correcta: "
        + ", ".join(f"**{l}**: {k} px ({100 * k / 42:.0f} %)" for l, k in umbral90.items()) + ". "
        "Con 21 pixeles invertidos (50 %) la imagen es ruido puro --- no tiene mas parecido con "
        "la letra original que con su inversa --- y ninguna memoria puede recuperarla; es la "
        "referencia de 'azar'. Las letras con mayor solapamiento con alguna otra son las que "
        "antes empiezan a fallar, porque sus valles de energia son mas estrechos y estan mas "
        "cerca de los valles vecinos."
    )

    # ------------------------------------------------------------------
    # 5. Errores
    # ------------------------------------------------------------------
    reporte.seccion("5. Reconocimientos incorrectos")
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.4, 4.4))
    x = resumen["pixeles_invertidos"].to_numpy()
    base = np.zeros_like(x, dtype=float)
    for columna, color, etiqueta in (("correcto_%", vz.AZUL, "letra correcta"),
                                     ("otra_letra_%", vz.NARANJA, "otra letra"),
                                     ("inverso_%", vz.AQUA, "inverso de una letra"),
                                     ("espurio_%", vz.AMARILLO, "estado espurio")):
        y = resumen[columna].to_numpy()
        ax.bar(x, y, bottom=base, width=1.6, color=color, edgecolor=vz.SUPERFICIE, linewidth=1.0, label=etiqueta)
        base = base + y
    ax.set_xlabel("pixeles invertidos")
    ax.set_ylabel("% de ensayos")
    ax.set_title("A donde converge la red")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=4)
    reporte.figura(vz.guardar(fig, FIGURAS / "11_tipos_de_error.png"),
                   "Desglose del estado final segun el nivel de ruido (promedio de las cuatro letras).")

    # catalogo de estados finales espurios
    espurios, vistos = [], set()
    for mu in range(4):
        for e in range(400):
            res = red.recuperar(ds.contaminar(P[mu], 14, semilla=999 + 400 * mu + e), semilla=e)
            if red.identificar(res.estado)[1] == "espurio":
                clave = tuple(res.estado.astype(int))
                if clave not in vistos:
                    vistos.add(clave)
                    espurios.append(res.estado)
    mezcla = np.sign(P[0] + P[1] + P[2])
    energias_espurios = [red.energia(e) for e in espurios]
    reporte.texto(
        "Tipos de error observados:"
    )
    reporte.lista([
        "**Otra letra**: la imagen contaminada quedo mas cerca (en distancia de Hamming) de otra "
        "letra que de la original, y la red la recupera --- correctamente desde su punto de vista. "
        "Es mas frecuente entre las letras mas solapadas.",
        "**Inverso**: por la simetria de la regla de Hebb, E(−ξ) = E(ξ), de modo que el negativo "
        "de cada letra tambien es un atractor. Solo se alcanza con ruido muy alto, cuando la "
        "imagen esta mas cerca del negativo que de la letra.",
        f"**Estados espurios**: minimos de energia que no corresponden a ninguna letra. Con 14 "
        f"pixeles de ruido se encontraron {len(espurios)} estados espurios distintos. Muchos son "
        "**mezclas** de varias letras, como sgn(ξ_A + ξ_B + ξ_C), cuya existencia predice la teoria.",
    ])
    if espurios:
        muestras = espurios[:6]
        titulos = []
        for s_e in muestras:
            m = red.solapamientos(s_e)
            titulos.append(" ".join(f"{n}:{v:+.1f}" for n, v in zip(NOMBRES, m)))
        fig, _ = vz.rejilla_retinas(muestras + [mezcla], titulos + ["sgn(A+B+C)"], forma=FORMA, n_columnas=7,
                                    sup_titulo="Estados espurios encontrados (con su solapamiento con cada letra)")
        reporte.figura(vz.guardar(fig, FIGURAS / "11_espurios.png"),
                       "Estados finales que no son ninguna letra; el ultimo es la mezcla teorica de tres letras.")
    reporte.texto(
        f"¿Es sgn(ξ_A + ξ_B + ξ_C) un punto fijo de esta red? **{red.es_punto_fijo(mezcla)}**. "
        f"Energia de la mezcla: {red.energia(mezcla):.2f}, frente a "
        f"{np.mean([red.energia(p) for p in P]):.2f} de media para las letras. "
        + (f"Los estados espurios encontrados tienen energias entre {min(energias_espurios):.2f} "
           f"y {max(energias_espurios):.2f}: algunos son valles **tan profundos o mas** que las "
           "propias letras (la energia de las letras va de "
           f"{min(red.energia(p) for p in P):.2f} a {max(red.energia(p) for p in P):.2f}). "
           "Lo que los hace poco frecuentes con ruido bajo no es su profundidad sino el tamano "
           "de su cuenca de atraccion: una imagen poco contaminada esta mucho mas cerca de su "
           "letra que de cualquier estado espurio." if energias_espurios else "")
    )

    # ------------------------------------------------------------------
    # 6. Diseno de los patrones
    # ------------------------------------------------------------------
    reporte.seccion("6. El diseno de los patrones importa: letras de trazo fino")
    finas = ds.letras_abcd("fina").X
    red_finas = RedHopfield(42, regla="hebb").almacenar(finas)
    solap_f = finas @ finas.T / 42
    fig, ejes = vz.plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ejes[0].axis("off")
    for k in range(4):
        ax_k = ejes[0].inset_axes([k * 0.25, 0.05, 0.23, 0.9])
        vz.mapa_retina(finas[k], f"{NOMBRES[k]} (trazo fino)", forma=FORMA, ax=ax_k)
    vz.mapa_calor(solap_f, titulo="Solapamiento (trazo fino)", etiquetas_filas=NOMBRES,
                  etiquetas_columnas=NOMBRES, divergente=True, ax=ejes[1])
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "11_letras_finas.png"),
                   "El dibujo 'natural' de las letras con trazo de un pixel y sus solapamientos.")
    fijos_f = pd.DataFrame({
        "letra": NOMBRES,
        "punto_fijo_trazo_fino": [red_finas.es_punto_fijo(p) for p in finas],
        "pixeles_inestables": [int(np.sum(np.sign(red_finas.W @ p) != p)) for p in finas],
        "punto_fijo_trazo_grueso": [red.es_punto_fijo(p) for p in P],
    })
    print(fijos_f.to_string(index=False))
    reporte.tabla(fijos_f, "¿Siguen siendo puntos fijos las letras de trazo fino?", nombre_csv="11_trazo_fino")
    df_finas = evaluar(red_finas, finas, niveles=(0, 4, 8, 12), ensayos=100)
    reporte.tabla(df_finas.pivot(index="pixeles_invertidos", columns="letra", values="correcto_%").reset_index(),
                  "Recuperacion correcta (%) con las letras de trazo fino y regla de Hebb",
                  nombre_csv="11_trazo_fino_reconocimiento")
    reporte.texto(
        f"El primer diseno probado fue el dibujo natural con trazo de un pixel. B, C y D comparten "
        f"la columna izquierda, las filas superior e inferior y --- en bipolar, donde el blanco "
        f"tambien cuenta --- casi todo el fondo: el solapamiento B-C llega a "
        f"{solap_f[1, 2]:.2f}. Con esos solapamientos la diafonia supera a la senal en varios "
        "pixeles, y **B, C y D dejan de ser puntos fijos**: la red no puede recuperarlas ni "
        "siquiera sin ruido, porque al presentarle la letra exacta la dinamica la aleja de ella. "
        "Solo la A, la menos parecida a las demas, sobrevive."
    )
    reporte.texto(
        "Cuatro patrones en 42 neuronas estan por debajo de la capacidad teorica (0.138·42 ≈ 5.8), "
        "pero esa cota vale para patrones **aleatorios** (solapamientos del orden de 1/√N ≈ 0.15). "
        "Las letras reales estan muy correlacionadas. Rediseñarlas con trazo grueso y formas "
        "diferenciadas redujo el solapamiento maximo a "
        f"{np.max(np.abs(solap - np.eye(4))):.2f} y basto para que la regla de Hebb funcione. "
        "La leccion: en una memoria de Hopfield con regla de Hebb, la **codificacion de los "
        "patrones** forma parte del diseno de la red."
    )

    # ------------------------------------------------------------------
    # 7. Pseudoinversa y capacidad
    # ------------------------------------------------------------------
    reporte.seccion("7. Alternativa: regla de la pseudoinversa, y capacidad")
    red_pinv_finas = RedHopfield(42, regla="pseudoinversa").almacenar(finas)
    red_pinv = RedHopfield(42, regla="pseudoinversa").almacenar(P)
    reporte.formula(r"W = \Xi\,(\Xi^{T}\Xi)^{-1}\,\Xi^{T}, \qquad \Xi = [\xi^1 \cdots \xi^P] \in \mathbb{R}^{N\times P}")
    reporte.texto(
        "La regla de la pseudoinversa (o de proyeccion) sustituye la suma de productos externos "
        "por el proyector ortogonal sobre el subespacio que generan los patrones. Tiene en "
        "cuenta sus correlaciones --- el factor (Ξ^T Ξ)^{-1} las 'descorrelaciona' --- y "
        "garantiza W ξ^μ = ξ^μ: todo patron linealmente independiente es punto fijo. El precio es "
        "que ya no es local ni de una sola pasada acumulativa: requiere invertir una matriz P x P "
        "con todos los patrones a la vez."
    )
    comparacion = []
    for nombre, r_, pats in (("Hebb, trazo grueso", red, P), ("pseudoinversa, trazo grueso", red_pinv, P),
                             ("Hebb, trazo fino", red_finas, finas),
                             ("pseudoinversa, trazo fino", red_pinv_finas, finas)):
        d_ = evaluar(r_, pats, niveles=(0, 4, 8, 12, 16), ensayos=100)
        fila = {"red": nombre}
        for k, v in d_.groupby("pixeles_invertidos")["correcto_%"].mean().items():
            fila[f"{k} px"] = v
        comparacion.append(fila)
        print(fila)
    reporte.tabla(pd.DataFrame(comparacion), "Recuperacion correcta media (%) segun regla y diseno",
                  nombre_csv="11_hebb_vs_pseudoinversa")

    rng = np.random.default_rng(SEMILLA)
    filas_cap = []
    for n_pat in (2, 4, 6, 8, 10, 12, 16):
        for regla in ("hebb", "pseudoinversa"):
            estables = []
            for rep in range(20):
                aleatorios = rng.choice([-1.0, 1.0], size=(n_pat, 42))
                r_ = RedHopfield(42, regla=regla).almacenar(aleatorios)
                estables.append(np.mean([r_.es_punto_fijo(p) for p in aleatorios]))
            filas_cap.append({"patrones_P": n_pat, "P/N": n_pat / 42, "regla": regla,
                              "patrones_estables_%": 100 * float(np.mean(estables))})
    df_cap = pd.DataFrame(filas_cap).pivot(index=["patrones_P", "P/N"], columns="regla",
                                           values="patrones_estables_%").reset_index()
    df_cap.columns = [str(c) for c in df_cap.columns]
    print(df_cap.to_string(index=False))
    reporte.tabla(df_cap, "Capacidad con patrones aleatorios de 42 elementos (20 repeticiones): "
                          "% de patrones que son puntos fijos", nombre_csv="11_capacidad")
    reporte.texto(
        "Con patrones aleatorios la regla de Hebb empieza a perder patrones en torno a P ≈ 0.1-0.14·N, "
        "en linea con la capacidad teorica 0.138·N ≈ 5.8. La pseudoinversa mantiene todos los "
        "patrones como puntos fijos hasta P < N, aunque con P grande sus cuencas de atraccion se "
        "estrechan y tolera menos ruido."
    )

    # ------------------------------------------------------------------
    # 8. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("8. Conclusiones")
    media8 = resumen.loc[resumen["pixeles_invertidos"] == 8, "correcto_%"].iloc[0]
    reporte.lista([
        "Una red de Hopfield de 42 neuronas almacena las cuatro letras de 7x6 con la regla de "
        "Hebb en una sola pasada, y las cuatro son atractores de la dinamica.",
        f"La red recupera las letras contaminadas: con 8 pixeles invertidos (19 %) la tasa media "
        f"de recuperacion correcta es {media8:.0f} %. El reconocimiento se degrada gradualmente "
        "con el ruido y llega al azar al 50 %.",
        "La recuperacion es un descenso de energia; los errores corresponden a caer en otro "
        "valle: otra letra, el inverso de una letra o un estado espurio (mezclas).",
        "La capacidad real depende de la correlacion entre patrones, no solo de su numero: el "
        "dibujo natural de las letras (trazo fino) hace fallar a la regla de Hebb con solo 4 "
        "patrones. Reducir el solapamiento en el diseno, o usar la regla de la pseudoinversa, lo "
        "resuelve.",
        "Contraste con los modelos supervisados: la red no aprende una correspondencia "
        "entrada → salida, sino que convierte cada patron en un estado estable; la 'respuesta' es "
        "el estado final completo, una memoria direccionable por contenido.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme escrito en {ruta}")


if __name__ == "__main__":
    main()
