"""
Experimento 09 --- Perceptron multicapa: aproximacion de una funcion.

Actividad 3 de la guia de evaluacion (segunda parte).  El ADALINE del
experimento 05 aproximaba el descodificador porque su salida era una funcion
**lineal** de las entradas.  Aqui la funcion objetivo es no lineal y no
monotona,

    f(x) = sin(x) + 0.5 sin(3x),     x en [-pi, pi],

y se aproxima con un MLP 1-H-1: capa oculta sigmoidal y neurona de salida
**lineal**, para que la salida pueda tomar cualquier valor real.

Contenido
---------
1. Datos: entrenamiento (30 puntos equiespaciados) y prueba (200 puntos nuevos).
2. Red principal 1-8-1: evolucion del error de entrenamiento y de prueba y
   grafica de la aproximacion.
3. Como construye la red la funcion: aportes de cada neurona oculta.
4. Comparacion de configuraciones: neuronas ocultas, razon de aprendizaje y
   momento.
5. Criterio de parada con datos ruidosos: parada temprana por validacion.
6. Limitacion: la red no extrapola fuera del intervalo de entrenamiento.

Ejecucion
---------
    python experimentos/exp09_mlp_aproximacion.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.activaciones import Identidad
from ce_rna.perceptron_multicapa import PerceptronMulticapa
from ce_rna.reportes import Reporte, encabezado_experimento

OCULTAS = 8
GAMMA = 0.1
ALPHA = 0.5
ESCALA = 2.0
MAX_EPOCAS = 8000
EPOCAS_ESTUDIO = 5000
ERROR_OBJETIVO = 1e-5
SEMILLA = 0
PARADA_EPOCAS = 20000
PACIENCIA = 3000


def red_1h1(h, **kwargs) -> PerceptronMulticapa:
    parametros = dict(razon_aprendizaje=GAMMA, momento=ALPHA, escala_inicial=ESCALA,
                      max_epocas=EPOCAS_ESTUDIO, error_objetivo=ERROR_OBJETIVO, semilla=SEMILLA)
    parametros.update(kwargs)
    return PerceptronMulticapa([1, h, 1], activacion_salida=Identidad(), **parametros)


def main() -> None:
    encabezado_experimento("Experimento 09 - Perceptron multicapa: aproximacion de funciones")

    reporte = Reporte(
        titulo="Experimento 09 --- Perceptron multicapa: aproximacion de una funcion",
        nombre_archivo="09_mlp_aproximacion",
        resumen=(
            "Un perceptron multicapa con una capa oculta de neuronas sigmoidales y una salida "
            "lineal es un **aproximador universal**: con suficientes neuronas ocultas puede "
            "aproximar cualquier funcion continua en un intervalo cerrado con el error que se "
            "desee (Cybenko, 1989; Hornik, Stinchcombe y White, 1989). El experimento entrena "
            "una red 1-H-1 con backpropagation para aproximar una funcion no lineal y no "
            "monotona, mide el error sobre puntos que la red **no vio** y estudia la influencia "
            "del numero de neuronas ocultas, la razon de aprendizaje, el momento y el criterio "
            "de parada."
        ),
    )

    entrenamiento, prueba = ds.aproximacion_funcion(n_entrenamiento=30, n_prueba=200, semilla=SEMILLA)
    Xe, de = entrenamiento.X, entrenamiento.d
    Xp, dp = prueba.X, prueba.d
    x_fino = np.linspace(-np.pi, np.pi, 400)[:, None]

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    reporte.seccion("1. Funcion objetivo y conjuntos de datos")
    reporte.formula(r"f(x) = \sin x + \tfrac{1}{2}\sin 3x, \qquad x \in [-\pi, \pi]")
    reporte.lista([
        "La funcion tiene cuatro extremos en el intervalo y mezcla dos frecuencias: una sola "
        "sigmoide (monotona) no puede reproducirla, y el numero de neuronas ocultas necesario "
        "es una pregunta abierta que el experimento responde empiricamente.",
        f"**Entrenamiento**: {entrenamiento.n_patrones} puntos equiespaciados en [−π, π].",
        f"**Prueba**: {prueba.n_patrones} puntos aleatorios del mismo intervalo, distintos de los "
        "de entrenamiento. Miden la **generalizacion**: lo que hace la red entre los puntos vistos.",
        "La entrada x se usa sin normalizar: en [−π, π] ya es de orden unidad, y la frecuencia "
        "3x requiere pendientes de las sigmoides del orden de 3, alcanzables con pesos "
        "iniciales en [−2, 2].",
    ])
    reporte.tabla(entrenamiento.tabla().round(4), "Conjunto de entrenamiento",
                  nombre_csv="09_entrenamiento")

    # ------------------------------------------------------------------
    # 2. Red principal
    # ------------------------------------------------------------------
    reporte.seccion(f"2. Red 1-{OCULTAS}-1: entrenamiento y aproximacion")
    reporte.lista([
        f"Arquitectura 1-{OCULTAS}-1: {OCULTAS} neuronas ocultas sigmoidales, salida lineal "
        f"(φ(v) = v), {3 * OCULTAS + 1} pesos en total.",
        "Salida lineal: en la retropropagacion su derivada es 1, de modo que δ de salida es "
        "directamente el error d − y, como en el ADALINE.",
        f"γ = {GAMMA}, momento α = {ALPHA}, modo estocastico con orden barajado.",
        f"Parada: E ≤ {ERROR_OBJETIVO:g} o {MAX_EPOCAS} epocas.",
    ])
    fig, _ = vz.dibujar_red(
        [["x0=1", "x"], [f"h{j + 1}" for j in range(OCULTAS)], ["y"]],
        titulo=f"MLP 1-{OCULTAS}-1 para aproximar f(x)",
        capas_lineales={0, 2},
        etiquetas_capas=["Entrada", "Capa oculta (sigmoide)", "Salida (lineal)"],
        nota="La neurona de salida es lineal (cuadrado): y = w0 + sum_j w_j h_j.",
        figsize=(8.5, 6.5),
    )
    reporte.figura(vz.guardar(fig, FIGURAS / "09_arquitectura.png"), "Arquitectura de la red.")

    red = red_1h1(OCULTAS, max_epocas=MAX_EPOCAS).entrenar(Xe, de, Xp, dp)
    print(red.resumen())
    y_ent, y_pru = red.salida(Xe), red.salida(Xp)
    metricas = pd.DataFrame([
        {"conjunto": "entrenamiento", "E (1/2 ECM)": mt.ecm(y_ent, de), "RMSE": mt.rmse(y_ent, de),
         "error_max": float(np.max(np.abs(y_ent - de))), "R2": mt.r2(y_ent, de)},
        {"conjunto": "prueba", "E (1/2 ECM)": mt.ecm(y_pru, dp), "RMSE": mt.rmse(y_pru, dp),
         "error_max": float(np.max(np.abs(y_pru - dp))), "R2": mt.r2(y_pru, dp)},
    ])
    print(metricas.to_string(index=False))
    reporte.bloque(red.resumen())
    reporte.tabla(metricas, "Error de entrenamiento y de prueba", nombre_csv="09_metricas")

    fig, _ = vz.curvas_comparadas(
        {"entrenamiento": red.curva_error(), "prueba": red.curva_error(prueba=True)},
        titulo=f"Evolucion del error (1-{OCULTAS}-1)", etiqueta_y="E", escala_log=True,
    )
    reporte.figura(vz.guardar(fig, FIGURAS / "09_error.png"),
                   "Error de entrenamiento y de prueba por epoca. Las dos curvas van juntas: "
                   "la red no memoriza los puntos, aprende la funcion.")
    reporte.texto(
        "La curva desciende a saltos: tramos casi planos (mesetas) separados por caidas. Cada "
        "caida corresponde a que una neurona oculta mas \"encuentra\" una parte de la funcion "
        "que aun no estaba explicada (una subida o bajada de sin 3x). Los errores de "
        "entrenamiento y de prueba son practicamente iguales durante todo el entrenamiento: con "
        "datos sin ruido y 30 puntos bien repartidos, ajustar los puntos equivale a ajustar la "
        "funcion."
    )

    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.4, 4.6))
    ax.plot(x_fino[:, 0], ds.funcion_objetivo(x_fino[:, 0]), color=vz.TINTA_2, lw=2.0, ls="--",
            label="f(x) real")
    ax.plot(x_fino[:, 0], red.salida(x_fino), color=vz.AZUL, lw=2.0, label=f"red 1-{OCULTAS}-1")
    ax.scatter(Xe[:, 0], de, s=36, c=vz.NARANJA, marker="s", edgecolors="white", zorder=4,
               label="puntos de entrenamiento")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Aproximacion obtenida")
    ax.legend(loc="upper left")
    reporte.figura(vz.guardar(fig, FIGURAS / "09_aproximacion.png"),
                   "La salida de la red (continua) y la funcion real son indistinguibles a simple vista.")

    # ------------------------------------------------------------------
    # 3. Aportes de las neuronas ocultas
    # ------------------------------------------------------------------
    reporte.seccion("3. Como construye la red la funcion")
    H = red.propagar(x_fino)[1]
    w_sal = red.W[1][0]
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.4, 4.6))
    for j in range(OCULTAS):
        ax.plot(x_fino[:, 0], w_sal[j + 1] * H[:, j], lw=1.2, color=vz.GRIS if j else vz.AQUA,
                label="aporte w_j·h_j(x) de cada neurona oculta" if j == 0 else None)
    ax.plot(x_fino[:, 0], red.salida(x_fino), color=vz.AZUL, lw=2.4, label="suma + sesgo = salida y")
    ax.set_xlabel("x")
    ax.set_ylabel("aporte")
    ax.set_title("Descomposicion de la salida en sigmoides")
    ax.legend(loc="lower left", fontsize=8)
    reporte.figura(vz.guardar(fig, FIGURAS / "09_aportes_ocultas.png"),
                   "Cada neurona oculta aporta un 'escalon suave' (una sigmoide desplazada, escalada "
                   "y con su propia pendiente); la salida es su suma ponderada.")
    capa = red.W[0]
    df_ocultas = pd.DataFrame({
        "neurona": [f"h{j + 1}" for j in range(OCULTAS)],
        "sesgo w_j0": capa[:, 0], "peso w_j1": capa[:, 1],
        "centro x = -w_j0/w_j1": -capa[:, 0] / capa[:, 1],
        "pendiente |w_j1|": np.abs(capa[:, 1]),
        "peso de salida": w_sal[1:],
    }).sort_values("centro x = -w_j0/w_j1")
    reporte.tabla(df_ocultas, "Parametros de cada neurona oculta (ordenadas por su centro)",
                  nombre_csv="09_neuronas_ocultas")
    reporte.texto(
        "Cada neurona oculta es una sigmoide φ(w_j1·x + w_j0) centrada en x = −w_j0/w_j1 con "
        "pendiente proporcional a |w_j1|. La neurona de salida suma esos escalones suaves con "
        "pesos de ambos signos: un escalon hacia arriba seguido de uno hacia abajo forma una "
        "'joroba', y con suficientes jorobas se construye cualquier curva continua. Es la "
        "intuicion del teorema de aproximacion universal, y explica por que los centros "
        "aprendidos se reparten por el intervalo donde la funcion cambia de pendiente."
    )

    # ------------------------------------------------------------------
    # 4. Comparacion de configuraciones
    # ------------------------------------------------------------------
    reporte.seccion("4. Comparacion de configuraciones")
    reporte.texto(f"Todas las corridas de esta seccion usan {EPOCAS_ESTUDIO} epocas como maximo.")

    reporte.seccion("4.1 Numero de neuronas ocultas", nivel=3)
    filas, ajustes = [], {}
    for h in (1, 2, 3, 4, 8, 16):
        for s in range(3):
            r = red_1h1(h, semilla=s).entrenar(Xe, de, Xp, dp)
            filas.append({"ocultas": h, "pesos": 3 * h + 1, "semilla": s,
                          "E_entrenamiento": r.curva_error()[-1],
                          "E_prueba": r.curva_error(prueba=True)[-1],
                          "R2_prueba": mt.r2(r.salida(Xp), dp)})
            if s == 0:
                ajustes[h] = r.salida(x_fino)
        print(filas[-1])
    df_h = pd.DataFrame(filas)
    resumen_h = (df_h.groupby(["ocultas", "pesos"])
                 .agg(E_entrenamiento_mediana=("E_entrenamiento", "median"),
                      E_prueba_mediana=("E_prueba", "median"),
                      E_prueba_mejor=("E_prueba", "min"),
                      R2_prueba_mediana=("R2_prueba", "median"))
                 .reset_index())
    df_h.to_csv(_ruta.TABLAS / "09_ocultas_corridas.csv", index=False)
    reporte.tabla(resumen_h, "Error segun el numero de neuronas ocultas (mediana de 3 semillas)",
                  nombre_csv="09_ocultas")
    vz.estilo()
    fig, ejes = vz.plt.subplots(2, 3, figsize=(12.0, 6.4), sharex=True, sharey=True)
    for ax, (h, y) in zip(ejes.ravel(), ajustes.items()):
        ax.plot(x_fino[:, 0], ds.funcion_objetivo(x_fino[:, 0]), color=vz.TINTA_2, lw=1.6, ls="--")
        ax.plot(x_fino[:, 0], y, color=vz.AZUL, lw=2.0)
        ax.scatter(Xe[:, 0], de, s=12, c=vz.NARANJA, marker="s", zorder=4)
        ax.set_title(f"1-{h}-1", fontsize=10)
    fig.suptitle("Aproximacion segun el numero de neuronas ocultas (linea discontinua: f real)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "09_comparacion_ocultas.png"),
                   "Con 1-4 neuronas la red solo captura la tendencia y parte de los extremos; "
                   "con 8 y 16 reproduce el armonico sin 3x.")
    reporte.texto(
        "Con una neurona oculta la red solo puede producir **una** sigmoide: una curva monotona "
        "que sigue la tendencia general (el salto de −1 a +1 en torno a x = 0). Con 2-4 neuronas "
        "la red anade algunos extremos pero, en las epocas disponibles, no reproduce las cuatro "
        "oscilaciones: el error de prueba apenas mejora respecto de una sola neurona. El salto "
        "cualitativo ocurre con **8** neuronas (el error cae dos ordenes de magnitud), y 16 "
        "mejora un poco mas en la mediana. En teoria cuatro jorobas podrian construirse con "
        "unas 8 sigmoides (dos por joroba), lo que coincide con lo observado. Con redes pequenas "
        "ademas es mas facil quedar atrapado en un minimo local: la variabilidad entre semillas "
        "(ver `09_ocultas_corridas.csv`) es la misma manifestacion vista en el XOR."
    )

    reporte.seccion("4.2 Razon de aprendizaje y momento", nivel=3)
    curvas, filas = {}, []
    for g, a in ((0.02, 0.5), (0.1, 0.5), (0.3, 0.5), (0.1, 0.0), (0.1, 0.9)):
        r = red_1h1(OCULTAS, razon_aprendizaje=g, momento=a).entrenar(Xe, de, Xp, dp)
        e = r.curva_error()
        filas.append({"gamma": g, "alpha": a, "paso_efectivo γ/(1-α)": g / (1 - a),
                      "E_final": e[-1] if np.isfinite(e[-1]) else float("nan"),
                      "E_prueba_final": r.curva_error(prueba=True)[-1],
                      "epoca_E<1e-3": int(np.argmax(e < 1e-3)) + 1 if np.any(e < 1e-3) else None,
                      "parada": r.motivo_parada})
        if a == 0.5:
            curvas[f"γ = {g}"] = e
        print(filas[-1])
    df_g = pd.DataFrame(filas)
    reporte.tabla(df_g, f"Razon de aprendizaje y momento (1-{OCULTAS}-1)", nombre_csv="09_gamma_momento")
    fig, _ = vz.curvas_comparadas(curvas, titulo="Error segun la razon de aprendizaje (α = 0.5)",
                                  etiqueta_y="E", escala_log=True)
    reporte.figura(vz.guardar(fig, FIGURAS / "09_gamma_curvas.png"),
                   "Pasos pequenos alargan las mesetas; pasos grandes producen oscilaciones.")
    reporte.texto(
        "El descenso por el gradiente sustituye la minimizacion exacta de E por pasos "
        "Δw = −γ ∂E/∂w. Si γ es pequeno cada paso es fiable pero la red tarda en salir de las "
        "mesetas; si es grande avanza deprisa pero el paso puede saltar por encima del valle y "
        "el error oscila (en modo estocastico, cada patron empuja los pesos en una direccion "
        "distinta). El momento suaviza esas oscilaciones y acelera en las mesetas: con α = 0.9 "
        "el paso efectivo se multiplica por 10, lo que ayuda o perjudica segun que γ lo "
        "acompane. La combinacion elegida (γ = 0.1, α = 0.5, paso efectivo 0.2) es el mejor "
        "compromiso observado."
    )

    # ------------------------------------------------------------------
    # 5. Parada temprana con datos ruidosos
    # ------------------------------------------------------------------
    reporte.seccion("5. Criterio de parada con datos ruidosos")
    ruidoso, _ = ds.aproximacion_funcion(n_entrenamiento=20, ruido=0.25, semilla=3)
    # validacion: 19 puntos ruidosos intercalados entre los de entrenamiento
    x_val = (np.linspace(-np.pi, np.pi, 20)[:-1] + np.pi / 19)[:, None]
    d_val = ds.funcion_objetivo(x_val[:, 0]) + 0.25 * np.random.default_rng(4).normal(size=x_val.shape[0])
    validacion = ds.Conjunto(X=x_val, d=d_val, nombre="validacion")

    # modo por lotes: la curva de validacion es suave y el minimo se localiza bien
    parametros_ruido = dict(modo="lote", razon_aprendizaje=0.5, momento=0.9,
                            max_epocas=PARADA_EPOCAS, error_objetivo=0.0)
    sin_parada = red_1h1(16, **parametros_ruido).entrenar(ruidoso.X, ruidoso.d, validacion.X, validacion.d)
    con_parada = red_1h1(16, paciencia=PACIENCIA, **parametros_ruido).entrenar(
        ruidoso.X, ruidoso.d, validacion.X, validacion.d)
    filas = []
    for nombre, r in (("sin parada temprana", sin_parada),
                      (f"con parada temprana (paciencia {PACIENCIA})", con_parada)):
        filas.append({"criterio": nombre, "epocas": r.epocas_usadas,
                      "E_entrenamiento": r.error(ruidoso.X, ruidoso.d),
                      "E_validacion": r.error(validacion.X, validacion.d),
                      "E_prueba (f real)": r.error(Xp, dp)})
    df_parada = pd.DataFrame(filas)
    print(df_parada.to_string(index=False))
    reporte.texto(
        "Con datos reales las salidas deseadas traen ruido. Se entrena una red grande (1-16-1, "
        "49 pesos) sobre solo 20 puntos con ruido gaussiano de desviacion 0.25, en modo por lotes "
        "(γ = 0.5, α = 0.9) para que las curvas sean suaves. Se reserva un **conjunto de "
        "validacion** (otros 19 puntos ruidosos, intercalados) para decidir cuando parar, y el "
        "conjunto de prueba sin ruido mide la calidad real frente a f."
    )
    reporte.tabla(df_parada, "Parada por numero de epocas frente a parada temprana",
                  nombre_csv="09_parada_temprana")
    vz.estilo()
    fig, ejes = vz.plt.subplots(1, 2, figsize=(12.0, 4.4))
    ejes[0].plot(sin_parada.curva_error(), color=vz.AZUL, lw=2.0, label="entrenamiento")
    ejes[0].plot(sin_parada.curva_error(prueba=True), color=vz.NARANJA, lw=2.0, ls="--", label="validacion")
    ejes[0].axvline(con_parada.epocas_usadas - PACIENCIA, color=vz.AQUA, lw=1.6, ls="-.",
                    label="mejor epoca de validacion")
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("epoca")
    ejes[0].set_ylabel("E")
    ejes[0].set_title("Entrenamiento y validacion (datos ruidosos)")
    ejes[0].legend(loc="best")
    ejes[1].plot(x_fino[:, 0], ds.funcion_objetivo(x_fino[:, 0]), color=vz.TINTA_2, ls="--", lw=1.6, label="f real")
    ejes[1].plot(x_fino[:, 0], sin_parada.salida(x_fino), color=vz.NARANJA, lw=1.8, label="sin parada temprana")
    ejes[1].plot(x_fino[:, 0], con_parada.salida(x_fino), color=vz.AZUL, lw=2.0, label="con parada temprana")
    ejes[1].scatter(ruidoso.X[:, 0], ruidoso.d, s=24, c=vz.TINTA_2, marker="s", zorder=4, label="datos ruidosos")
    ejes[1].set_xlabel("x")
    ejes[1].set_title("Aproximaciones obtenidas")
    ejes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "09_parada_temprana.png"),
                   "El error de entrenamiento sigue bajando mientras el de validacion se estanca o "
                   "sube: la red empieza a ajustar el ruido.")
    mejora = df_parada["E_prueba (f real)"].iloc[0] > df_parada["E_prueba (f real)"].iloc[1]
    reporte.texto(
        "Sin parada temprana la red sigue reduciendo el error de entrenamiento a costa de "
        "curvarse para pasar cerca de los puntos ruidosos (**sobreajuste**). La parada temprana "
        f"detiene el entrenamiento cuando el error de validacion lleva {PACIENCIA} epocas sin mejorar y "
        "restaura los mejores pesos. "
        + ("En esta corrida la parada temprana obtuvo menor error frente a la funcion real, "
           "con menos epocas." if mejora else
           "En esta corrida el sobreajuste fue leve y ambas redes quedaron cerca; la parada "
           "temprana obtuvo un resultado comparable con menos epocas.")
        + " El minimo del error de entrenamiento no es el objetivo: el objetivo es el error "
        "sobre datos nuevos, y solo puede estimarse con datos que no se usaron para ajustar los pesos."
    )

    # ------------------------------------------------------------------
    # 6. Extrapolacion
    # ------------------------------------------------------------------
    reporte.seccion("6. Limitacion: la red no extrapola")
    x_ext = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 600)[:, None]
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(8.4, 4.4))
    ax.axvspan(-np.pi, np.pi, color="#eef4fd", zorder=0, label="intervalo de entrenamiento")
    ax.plot(x_ext[:, 0], ds.funcion_objetivo(x_ext[:, 0]), color=vz.TINTA_2, ls="--", lw=1.6, label="f real")
    ax.plot(x_ext[:, 0], red.salida(x_ext), color=vz.AZUL, lw=2.0, label=f"red 1-{OCULTAS}-1")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Fuera del intervalo de entrenamiento")
    ax.legend(loc="lower left", fontsize=8)
    reporte.figura(vz.guardar(fig, FIGURAS / "09_extrapolacion.png"),
                   "Fuera de [−π, π] todas las sigmoides se saturan y la salida tiende a una constante.")
    reporte.texto(
        "La red aproxima f solo **donde tuvo datos**. Fuera del intervalo cada sigmoide esta "
        "saturada en 0 o en 1 y la salida se vuelve constante: la red no ha aprendido que f es "
        "periodica, solo su forma en [−π, π]. Es una limitacion general del aprendizaje a partir "
        "de ejemplos: la red interpola, no extrapola."
    )

    # ------------------------------------------------------------------
    # 7. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("7. Conclusiones")
    r2p = metricas.loc[1, "R2"]
    reporte.lista([
        f"Una red 1-{OCULTAS}-1 con sigmoides ocultas y salida lineal aproxima f(x) = sin x + "
        f"0.5 sin 3x con R² = {r2p:.5f} sobre 200 puntos de prueba no vistos.",
        "La salida se construye como suma de sigmoides desplazadas y escaladas: es la idea del "
        "teorema de aproximacion universal.",
        "El numero de neuronas ocultas fija la complejidad alcanzable: con 1-4 neuronas solo se "
        "captura la tendencia; hacen falta unas 8 para reproducir el armonico sin 3x.",
        "γ y α controlan la velocidad y estabilidad del descenso por el gradiente; la curva de "
        "error muestra mesetas que dependen de ambos.",
        "Con datos ruidosos, el criterio de parada debe apoyarse en un conjunto de validacion: "
        "seguir bajando el error de entrenamiento lleva al sobreajuste.",
        "La red interpola dentro del dominio de entrenamiento pero no extrapola fuera de el.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme escrito en {ruta}")


if __name__ == "__main__":
    main()
