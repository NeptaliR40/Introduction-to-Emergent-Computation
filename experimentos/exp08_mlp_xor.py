"""
Experimento 08 --- Perceptron multicapa: el XOR con retropropagacion.

Actividad 3 de la guia de evaluacion (primera parte).  El experimento 03 probo
que ninguna recta separa el XOR; el 01 lo resolvio componiendo tres neuronas de
McCulloch-Pitts **con los pesos puestos a mano**.  Aqui la red 2-2-1 de la clase
**aprende** esos pesos por si sola mediante backpropagation.

Contenido
---------
1. Arquitectura 2-2-1, funcion sigmoidal y codificacion de los patrones.
2. Entrenamiento: tabla de resultados del XOR y evolucion del error.
3. Que aprendio la capa oculta: el espacio oculto y la frontera no lineal.
4. Minimos locales: no todas las inicializaciones llegan a la solucion.
5. Estudio de parametros: razon de aprendizaje, neuronas ocultas, momento,
   modo de actualizacion y criterio de parada.

Ejecucion
---------
    python experimentos/exp08_mlp_xor.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import visual as vz
from ce_rna.perceptron_multicapa import PerceptronMulticapa
from ce_rna.reportes import Reporte, encabezado_experimento

GAMMA = 0.5
ERROR_OBJETIVO = 0.005     # E = (1/N) sum (1/2) e^2  ->  |e| ~ 0.1 por patron
MAX_EPOCAS = 10000
SEMILLA = 0
N_SEMILLAS = 20            # inicializaciones por configuracion en el estudio


def estudiar(X, d, n_semillas=N_SEMILLAS, **parametros) -> dict:
    """Entrena con `n_semillas` inicializaciones y resume exito y velocidad."""
    epocas, exitos = [], 0
    for s in range(n_semillas):
        red = PerceptronMulticapa(semilla=s, **parametros).entrenar(X, d)
        if red.motivo_parada == "error_objetivo" and np.array_equal(red.predecir(X), d):
            exitos += 1
            epocas.append(red.epocas_usadas)
    return {
        "exito_%": 100.0 * exitos / n_semillas,
        "epocas_mediana": float(np.median(epocas)) if epocas else float("nan"),
        "epocas_min": float(np.min(epocas)) if epocas else float("nan"),
        "epocas_max": float(np.max(epocas)) if epocas else float("nan"),
    }


def main() -> None:
    encabezado_experimento("Experimento 08 - Perceptron multicapa: XOR")

    reporte = Reporte(
        titulo="Experimento 08 --- Perceptron multicapa: el XOR con retropropagacion",
        nombre_archivo="08_mlp_xor",
        resumen=(
            "El XOR es el ejemplo clasico de problema **no linealmente separable**: el experimento "
            "03 comprobo que ninguna de 68 921 rectas clasifica sus cuatro patrones y que el "
            "perceptron simple oscila indefinidamente. La salida es anadir una **capa oculta** de "
            "neuronas no lineales. Este experimento entrena la arquitectura 2-2-1 presentada en "
            "clase con retropropagacion del error, verifica la tabla del XOR, muestra que "
            "representa la capa oculta y estudia como influyen la razon de aprendizaje, el numero "
            "de neuronas ocultas, el momento y el criterio de parada."
        ),
    )

    conjunto = ds.compuerta("XOR", "binaria")
    X, d = conjunto.X, conjunto.d

    # ------------------------------------------------------------------
    # 1. Arquitectura
    # ------------------------------------------------------------------
    reporte.seccion("1. Arquitectura y codificacion")
    reporte.tabla(conjunto.tabla(), "Patrones del XOR (codificacion binaria)", nombre_csv="08_patrones_xor")
    reporte.texto(
        "Se usa la codificacion **binaria** {0, 1} porque la sigmoide logistica tiene recorrido "
        "(0, 1): los objetivos 0 y 1 son sus asintotas. La salida se interpreta como clase 1 "
        "si y ≥ 0.5."
    )
    reporte.lista([
        "**Capa de entrada**: 2 unidades (x1, x2), que solo distribuyen la senal.",
        "**Capa oculta**: 2 neuronas sigmoidales con sesgo. Es el minimo que resuelve el XOR: "
        "cada neurona oculta traza una recta, y dos rectas bastan para aislar la franja que "
        "contiene (0,1) y (1,0).",
        "**Capa de salida**: 1 neurona sigmoidal con sesgo.",
        "Total: 2·(2+1) + 1·(2+1) = **9 pesos** ajustables.",
    ])
    reporte.texto("Propagacion hacia adelante (con y_0 = 1 como entrada de sesgo en cada capa):")
    reporte.formula(r"v_j^{(l)} = \sum_{i} w_{ji}^{(l)}\, y_i^{(l-1)}, \qquad "
                    r"y_j^{(l)} = \varphi\!\left(v_j^{(l)}\right) = \frac{1}{1+e^{-v_j^{(l)}}}")
    reporte.texto("Funcion de error y retropropagacion (regla Delta generalizada):")
    reporte.formula(r"E = \frac{1}{N}\sum_{p}\tfrac{1}{2}\sum_k \left(d_k^{p}-y_k^{p}\right)^2")
    reporte.formula(r"\delta_k^{(L)} = (d_k - y_k)\,y_k(1-y_k), \qquad "
                    r"\delta_j^{(l)} = y_j^{(l)}\left(1-y_j^{(l)}\right)\sum_k \delta_k^{(l+1)} w_{kj}^{(l+1)}")
    reporte.formula(r"\Delta w_{ji}^{(l)}(t) = \gamma\,\delta_j^{(l)}\,y_i^{(l-1)} + \alpha\,\Delta w_{ji}^{(l)}(t-1)")
    reporte.texto(
        "La derivada de la sigmoide se expresa con su propia salida, φ'(v) = y(1 − y); por eso la "
        "sigmoide es la activacion de referencia para backpropagation: es derivable en todo "
        "punto (el escalon no lo es) y su derivada es barata de calcular."
    )

    red = PerceptronMulticapa([2, 2, 1], razon_aprendizaje=GAMMA, max_epocas=MAX_EPOCAS,
                              error_objetivo=ERROR_OBJETIVO, semilla=SEMILLA)
    fig, _ = vz.dibujar_red(
        [["x0=1", "x1", "x2"], ["h1", "h2"], ["y"]],
        titulo="Perceptron multicapa 2-2-1 para el XOR",
        etiquetas_capas=["Capa de entrada", "Capa oculta (sigmoide)", "Salida (sigmoide)"],
        nota="Cada neurona oculta y la de salida tienen ademas su propio peso de sesgo.",
    )
    reporte.figura(vz.guardar(fig, FIGURAS / "08_arquitectura_221.png"), "Arquitectura 2-2-1.")

    # ------------------------------------------------------------------
    # 2. Entrenamiento
    # ------------------------------------------------------------------
    reporte.seccion("2. Entrenamiento y resultados")
    red.entrenar(X, d)
    print(red.resumen())
    salidas = red.propagar(X)
    tabla = pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "d": d,
        "h1": salidas[1][:, 0], "h2": salidas[1][:, 1],
        "y (salida real)": salidas[2][:, 0],
        "clase (y>=0.5)": red.predecir(X),
        "error d-y": d - salidas[2][:, 0],
    })
    print(tabla.to_string(index=False))
    reporte.texto(
        f"Parametros: γ = {GAMMA}, sin momento, modo estocastico (actualizacion tras cada patron, "
        f"orden barajado en cada epoca), pesos iniciales uniformes en [−1, 1] (semilla {SEMILLA}). "
        f"Criterio de parada: E ≤ {ERROR_OBJETIVO} (equivale a un error tipico de ~0.1 por "
        f"patron) o {MAX_EPOCAS} epocas."
    )
    reporte.tabla(tabla, "Tabla de resultados del XOR tras el entrenamiento", nombre_csv="08_tabla_xor")
    reporte.bloque(red.resumen())

    pesos = pd.DataFrame(
        [[f"h{j + 1}", *red.W[0][j]] for j in range(2)] + [["y", *red.W[1][0]]],
        columns=["neurona", "w0 (sesgo)", "w1", "w2"],
    )
    reporte.tabla(pesos, "Pesos aprendidos (para 'y', w1 y w2 conectan con h1 y h2)",
                  nombre_csv="08_pesos_xor")

    fig, _ = vz.curva_aprendizaje(red.curva_error(), titulo="Evolucion del error en el XOR (2-2-1)",
                                  etiqueta_y="E (error cuadratico medio)", escala_log=True,
                                  referencia=ERROR_OBJETIVO, etiqueta_referencia="criterio de parada")
    reporte.figura(vz.guardar(fig, FIGURAS / "08_error_xor.png"),
                   "El error permanece casi constante en una meseta y despues cae bruscamente: "
                   "es el momento en que las neuronas ocultas se especializan.")
    hist = red.historial_df()
    meseta = int(hist.loc[hist["error_entrenamiento"] < 0.9 * hist["error_entrenamiento"].iloc[0], "epoca"].min())
    reporte.texto(
        f"La curva tiene la forma tipica de backpropagation sobre el XOR: una **meseta** inicial "
        f"(E ≈ {hist['error_entrenamiento'].iloc[0]:.3f}, la red responde ≈ 0.5 a todo porque es lo "
        f"que minimiza el error mientras las ocultas son redundantes) que dura hasta la epoca "
        f"~{meseta}, seguida de una caida rapida. En la meseta el gradiente es pequeno pero no "
        f"nulo, y el descenso termina por romper la simetria entre las dos neuronas ocultas."
    )

    # ------------------------------------------------------------------
    # 3. Espacio oculto
    # ------------------------------------------------------------------
    reporte.seccion("3. Que aprendio la capa oculta")
    fig, ejes = vz.plt.subplots(1, 2, figsize=(12.0, 5.2))
    vz.region_continua(red.salida, X, d, titulo="Espacio de entrada: frontera y = 0.5", ax=ejes[0])
    H = salidas[1]
    w = red.W[1][0]
    ejes[1].scatter(H[d == 1, 0], H[d == 1, 1], s=140, c=vz.AZUL, marker="o", edgecolors="white",
                    linewidths=1.8, zorder=4, label="d = 1")
    ejes[1].scatter(H[d == 0, 0], H[d == 0, 1], s=140, c=vz.NARANJA, marker="s", edgecolors="white",
                    linewidths=1.8, zorder=4, label="d = 0")
    # los patrones que caen en el mismo punto del espacio oculto comparten etiqueta
    rotulos: list[tuple[np.ndarray, list[str]]] = []
    for k in range(4):
        texto = f"({X[k, 0]:g},{X[k, 1]:g})"
        for punto, textos in rotulos:
            if np.linalg.norm(punto - H[k]) < 0.03:
                textos.append(texto)
                break
        else:
            rotulos.append((H[k], [texto]))
    for punto, textos in rotulos:
        ejes[1].annotate(" y ".join(textos), punto, textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=8, color=vz.TINTA_2)
    xs = np.linspace(-0.05, 1.05, 50)
    if abs(w[2]) > 1e-9:
        ejes[1].plot(xs, -(w[0] + w[1] * xs) / w[2], color=vz.TINTA, lw=2.0, label="recta de la neurona de salida")
    ejes[1].set_xlim(-0.05, 1.05)
    ejes[1].set_ylim(-0.05, 1.05)
    ejes[1].set_xlabel("h1")
    ejes[1].set_ylabel("h2")
    ejes[1].set_title("Espacio oculto (h1, h2): ahora SI es separable")
    ejes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "08_espacio_oculto.png"),
                   "Izquierda: la frontera aprendida en el plano (x1, x2) ya no es una recta. "
                   "Derecha: los mismos cuatro patrones vistos por la neurona de salida.")

    papeles = []
    for j in range(2):
        h = (salidas[1][:, j] >= 0.5).astype(int).tolist()
        nombre = {(0, 1, 1, 1): "OR", (0, 0, 0, 1): "AND", (1, 1, 1, 0): "NAND",
                  (1, 0, 0, 0): "NOR"}.get(tuple(h), "otra")
        papeles.append((f"h{j + 1}", h, nombre))
    reporte.texto(
        "Umbralizando las neuronas ocultas en 0.5 sobre los patrones (0,0), (0,1), (1,0), (1,1): "
        + "; ".join(f"**{n}** → {h} (≈ {nom})" for n, h, nom in papeles) + ". "
        "La red ha redescubierto por si sola la descomposicion que el experimento 01 impuso a "
        "mano, XOR = OR ∧ ¬AND (o una equivalente): la capa oculta calcula dos funciones "
        "linealmente separables y la de salida las combina. En el espacio oculto los patrones "
        "(0,1) y (1,0) colapsan en la misma esquina y una sola recta los separa de los otros dos."
    )

    # ------------------------------------------------------------------
    # 4. Minimos locales
    # ------------------------------------------------------------------
    reporte.seccion("4. Minimos locales: no toda inicializacion converge")
    filas = []
    ejemplo_fallo = None
    for s in range(50):
        r = PerceptronMulticapa([2, 2, 1], razon_aprendizaje=GAMMA, max_epocas=MAX_EPOCAS,
                                error_objetivo=ERROR_OBJETIVO, semilla=s).entrenar(X, d)
        exito = r.motivo_parada == "error_objetivo"
        filas.append({"semilla": s, "converge": exito, "epocas": r.epocas_usadas,
                      "E_final": r.curva_error()[-1]})
        if not exito and ejemplo_fallo is None:
            ejemplo_fallo = r
    df_semillas = pd.DataFrame(filas)
    tasa = df_semillas["converge"].mean() * 100
    print(f"\nConvergencia 2-2-1 sobre 50 semillas: {tasa:.0f} %")
    reporte.texto(
        f"Se repitio el entrenamiento con 50 inicializaciones distintas (mismos parametros). "
        f"Convergieron **{tasa:.0f} %**; el resto agoto las {MAX_EPOCAS} epocas atrapado en un "
        f"**minimo local** de la superficie de error."
    )
    reporte.tabla(df_semillas.describe().loc[["mean", "min", "50%", "max"], ["epocas", "E_final"]]
                  .reset_index().rename(columns={"index": "estadistico"}),
                  "Resumen de las 50 corridas", nombre_csv="08_semillas_xor")
    if ejemplo_fallo is not None:
        y_fallo = ejemplo_fallo.salida(X)
        reporte.texto(
            "Ejemplo de corrida fallida: salidas "
            + ", ".join(f"{v:.3f}" for v in y_fallo)
            + f" (E = {ejemplo_fallo.curva_error()[-1]:.4f}). La red clasifica bien dos patrones y "
            "responde ≈ 0.5 a los otros dos: una neurona oculta se satura y deja de aportar "
            "gradiente, y la red queda con una sola frontera util --- justo lo que un perceptron "
            "simple podria hacer. El descenso por el gradiente es un metodo **local**: solo ve la "
            "pendiente donde esta, y no garantiza el minimo global."
        )
        fig, _ = vz.region_continua(ejemplo_fallo.salida, X, d,
                                    titulo="Corrida atrapada en un minimo local")
        reporte.figura(vz.guardar(fig, FIGURAS / "08_minimo_local.png"),
                       "Con una sola neurona oculta util, la frontera vuelve a ser practicamente una recta.")

    # ------------------------------------------------------------------
    # 5. Parametros
    # ------------------------------------------------------------------
    reporte.seccion("5. Estudio de parametros")
    reporte.texto(
        f"Cada configuracion se entrena con {N_SEMILLAS} inicializaciones distintas. Se considera "
        f"exito alcanzar E ≤ {ERROR_OBJETIVO} con los cuatro patrones bien clasificados; las "
        "epocas se resumen solo sobre las corridas exitosas."
    )
    base = dict(capas=[2, 2, 1], razon_aprendizaje=GAMMA, max_epocas=MAX_EPOCAS,
                error_objetivo=ERROR_OBJETIVO)

    reporte.seccion("5.1 Razon de aprendizaje γ", nivel=3)
    filas = []
    for g in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        filas.append({"gamma": g, **estudiar(X, d, **{**base, "razon_aprendizaje": g})})
        print(filas[-1])
    df_gamma = pd.DataFrame(filas)
    reporte.tabla(df_gamma, "Efecto de la razon de aprendizaje (2-2-1, sin momento)", nombre_csv="08_gamma")
    curvas = {}
    for g in (0.1, 0.5, 2.0):
        r = PerceptronMulticapa(**{**base, "razon_aprendizaje": g}, semilla=SEMILLA).entrenar(X, d)
        curvas[f"γ = {g}"] = r.curva_error()
    fig, _ = vz.curvas_comparadas(curvas, titulo="Error segun la razon de aprendizaje",
                                  etiqueta_y="E", escala_log=True)
    reporte.figura(vz.guardar(fig, FIGURAS / "08_gamma_curvas.png"),
                   "Con γ pequeno la meseta se alarga; con γ grande la caida llega antes.")
    reporte.texto(
        "γ fija el tamano de cada paso del descenso por el gradiente. Las epocas necesarias son "
        "aproximadamente inversamente proporcionales a γ: la meseta se recorre a velocidad "
        f"proporcional al paso. Con γ = 0.1 la tasa de exito baja sobre todo porque muchas "
        f"corridas **agotan las {MAX_EPOCAS} epocas** antes de salir de la meseta, no porque "
        "caigan en un minimo peor. Con γ muy grande (4) los pesos crecen deprisa, las sigmoides "
        "se saturan (φ' ≈ 0), aparecen corridas muy largas y la tasa de exito vuelve a bajar. "
        "El rango 0.5-2 ofrece el mejor compromiso."
    )

    reporte.seccion("5.2 Numero de neuronas ocultas", nivel=3)
    filas = []
    for h in (2, 3, 4, 8):
        filas.append({"neuronas_ocultas": h, "pesos": 3 * h + h + 1,
                      **estudiar(X, d, **{**base, "capas": [2, h, 1]})})
        print(filas[-1])
    df_ocultas = pd.DataFrame(filas)
    reporte.tabla(df_ocultas, "Efecto del tamano de la capa oculta (γ = 0.5)", nombre_csv="08_ocultas")
    reporte.texto(
        "Dos neuronas ocultas son suficientes pero es la configuracion mas fragil: si una de las "
        "dos se desperdicia, no hay repuesto. Con mas neuronas ocultas hay mas formas de repartir "
        "el trabajo, la superficie de error tiene menos minimos locales problematicos y la tasa de "
        "exito sube; a cambio hay mas pesos que ajustar. Para un problema de cuatro patrones no "
        "hay riesgo de sobreajuste --- todos los patrones posibles estan en el entrenamiento ---, "
        "asi que aqui la capacidad extra solo aporta robustez."
    )

    reporte.seccion("5.3 Momento y modo de actualizacion", nivel=3)
    filas = []
    for alpha in (0.0, 0.5, 0.9):
        filas.append({"configuracion": f"estocastico, α = {alpha}",
                      **estudiar(X, d, **{**base, "momento": alpha})})
    filas.append({"configuracion": "lote, α = 0, γ = 2",
                  **estudiar(X, d, **{**base, "modo": "lote", "razon_aprendizaje": 2.0})})
    filas.append({"configuracion": "lote, α = 0.9, γ = 2",
                  **estudiar(X, d, **{**base, "modo": "lote", "razon_aprendizaje": 2.0, "momento": 0.9})})
    df_momento = pd.DataFrame(filas)
    print(df_momento.to_string(index=False))
    reporte.tabla(df_momento, "Momento y modo de actualizacion", nombre_csv="08_momento")
    reporte.texto(
        "El momento α acumula una fraccion del paso anterior: en las mesetas, donde el gradiente "
        "apunta siempre en la misma direccion, el paso efectivo crece hasta γ/(1 − α), y la red "
        "sale antes. En modo por lotes se da un unico paso por epoca con el gradiente exacto de E; "
        "como cada epoca aporta 4 veces menos actualizaciones que el modo estocastico, necesita un "
        "γ mayor para avanzar al mismo ritmo."
    )

    reporte.seccion("5.4 Criterio de parada", nivel=3)
    filas = []
    for objetivo in (0.05, 0.02, 0.005, 0.001):
        r = PerceptronMulticapa([2, 2, 1], razon_aprendizaje=GAMMA, max_epocas=MAX_EPOCAS * 3,
                                error_objetivo=objetivo, semilla=SEMILLA).entrenar(X, d)
        y = r.salida(X)
        filas.append({"E_objetivo": objetivo, "epocas": r.epocas_usadas,
                      "clasifica_4": bool(np.array_equal(r.predecir(X), d)),
                      "max_|d-y|": float(np.max(np.abs(d - y))),
                      "y(0,0)": y[0], "y(0,1)": y[1], "y(1,0)": y[2], "y(1,1)": y[3]})
    df_parada = pd.DataFrame(filas)
    print(df_parada.to_string(index=False))
    reporte.tabla(df_parada, "Umbral de error como criterio de parada", nombre_csv="08_criterio_parada")
    reporte.texto(
        "La clasificacion correcta (y del lado bueno de 0.5) llega mucho antes que un error "
        "pequeno: un umbral holgado basta para *clasificar* pero deja salidas poco confiables, "
        "cerca de 0.5. Endurecer el criterio aumenta las epocas sin cambiar las clases, pero "
        "aleja las salidas de la frontera, es decir, aumenta el **margen**. Como la sigmoide solo "
        "alcanza 0 y 1 asintoticamente, pedir E → 0 exige pesos → ∞; por eso el criterio de "
        "parada debe ser un umbral razonable y no la anulacion del error, y siempre va "
        "acompanado de un maximo de epocas para cubrir las corridas atrapadas en minimos locales."
    )

    # ------------------------------------------------------------------
    # 6. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("6. Conclusiones")
    reporte.lista([
        "Una sola capa oculta de **dos** neuronas sigmoidales basta para resolver el XOR: la "
        "capa oculta transforma el plano en un espacio donde el problema es linealmente separable.",
        "La red **aprende** la descomposicion que en McCulloch-Pitts habia que disenar a mano; "
        "backpropagation extiende la regla Delta del ADALINE a capas ocultas repartiendo el "
        "error hacia atras con la regla de la cadena.",
        f"El descenso por el gradiente es local: con la 2-2-1 aproximadamente el {100 - tasa:.0f} % "
        "de las inicializaciones queda atrapado en un minimo local. Mas neuronas ocultas o el "
        "momento reducen el problema.",
        "La curva de error del XOR presenta una meseta seguida de una caida brusca; la razon de "
        "aprendizaje y el momento controlan sobre todo la duracion de esa meseta.",
        "El criterio de parada combina un umbral de error (calidad) y un maximo de epocas "
        "(seguridad); clasificar bien no es lo mismo que tener un error pequeno.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme escrito en {ruta}")


if __name__ == "__main__":
    main()
