"""
Experimento 12 --- Imagenes reales: de la foto a la retina de 7x6.

Extension de las actividades 3 y 5.  Los experimentos 08-11 trabajan con
patrones ya codificados.  Aqui se parte de **imagenes**: letras A, B, C y D
escritas con tipografias reales, con distorsiones de camara (rotacion, escala,
desplazamiento, desenfoque, iluminacion desigual, bajo contraste y ruido de
sensor), y se estudia:

1. El preprocesamiento que convierte una imagen cualquiera en 42 valores bipolares.
2. La red de Hopfield de la actividad 5 como reconocedor de esas imagenes.
3. Un perceptron multicapa **entrenado** con imagenes (aprendizaje supervisado).
4. Comparacion: Hopfield (Hebb y pseudoinversa), plantilla mas cercana y MLP, a
   dos resoluciones (7x6 y 14x12) y sobre una tipografia que el MLP nunca vio.
5. Como usar fotos propias.

Ejecucion
---------
    python experimentos/exp12_imagenes_reales.py
    python experimentos/exp12_imagenes_reales.py --carpeta ruta/a/mis_fotos

La carpeta de fotos propias debe tener una subcarpeta por letra (A/, B/, C/,
D/) o archivos cuyo nombre empiece por la letra (``B_movil.jpg``).  Por omision
se usa `datos/imagenes/propias/` si contiene imagenes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS, RAIZ
from ce_rna import imagenes as im
from ce_rna import visual as vz
from ce_rna.hopfield import RedHopfield
from ce_rna.perceptron_multicapa import PerceptronMulticapa
from ce_rna.reportes import Reporte, encabezado_experimento

LETRAS = list("ABCD")
FUENTE_MEMORIA = "sans_negrita"
FUENTES_ENTRENAMIENTO = ["sans_negrita", "sans", "serif_negrita", "serif", "mono"]
FUENTE_NUEVA = "mono_negrita"          # nunca se usa para entrenar el MLP
N_ENTRENAMIENTO = 40                   # imagenes distorsionadas por letra y tipografia
N_PRUEBA = 60                          # imagenes de prueba por letra
OCULTAS = 16
EPOCAS = 150
PACIENCIA = 15                         # parada temprana sobre el conjunto de validacion
DIR_DATOS = RAIZ / "datos" / "imagenes"
SEMILLA = 0
INTENSIDADES = [0.5, 1.0, 1.5, 2.0, 2.5]   # barrido de robustez (1.0 = conjuntos de prueba)
N_BARRIDO = 30                         # imagenes por letra y nivel de intensidad


def generar(fuentes, n, rng, intensidad=1.0):
    """Lista de (imagen PIL, indice de letra, fuente) con distorsiones aleatorias."""
    muestras = []
    for fuente in fuentes:
        for k, letra in enumerate(LETRAS):
            limpia = im.renderizar_letra(letra, fuente)
            for _ in range(n):
                muestras.append((im.distorsionar(limpia, rng, intensidad), k, fuente))
    return muestras


def a_matriz(muestras, forma):
    X = np.array([im.preprocesar(img, forma=forma) for img, _, _ in muestras])
    y = np.array([k for _, k, _ in muestras])
    return X, y


def uno_de_n(y, n=4):
    D = np.zeros((y.size, n))
    D[np.arange(y.size), y] = 1.0
    return D


class ReconocedorHopfield:
    """Hopfield + identificacion del estado final (sin respuesta si es espurio)."""

    def __init__(self, memorias, regla):
        self.red = RedHopfield(memorias.shape[1], regla=regla).almacenar(memorias)

    def predecir(self, X):
        salida = []
        for i, x in enumerate(X):
            res = self.red.recuperar(x, semilla=i)
            mu, tipo = self.red.identificar(res.estado)
            salida.append(mu if tipo == "patron" else -1)
        return np.array(salida)


def plantilla_mas_cercana(memorias, X):
    """Referencia sin red: la memoria con mayor solapamiento (distancia de Hamming minima)."""
    return np.argmax(X @ memorias.T, axis=1)


def matriz_confusion(y, pred):
    etiquetas = LETRAS + ["sin respuesta"]
    M = np.zeros((4, 5), dtype=int)
    for real, p in zip(y, pred):
        M[real, p if p >= 0 else 4] += 1
    return pd.DataFrame(M, index=[f"real {l}" for l in LETRAS], columns=etiquetas)


def main(carpeta_propias: Path | None = None) -> None:
    encabezado_experimento("Experimento 12 - Imagenes reales")
    rng = np.random.default_rng(SEMILLA)

    reporte = Reporte(
        titulo="Experimento 12 --- Imagenes reales: de la foto a la retina de 7x6",
        nombre_archivo="12_imagenes_reales",
        resumen=(
            "Las redes de los experimentos anteriores reciben patrones ya codificados. En la "
            "practica los datos llegan como **imagenes**: fotos o escaneos de cualquier tamano, con "
            "fondo, iluminacion irregular y el objeto en cualquier posicion. Este experimento "
            "construye el preprocesamiento que lleva una imagen a la retina de 7x6 de la actividad "
            "5, evalua la red de Hopfield sobre imagenes de letras con tipografias reales y "
            "distorsiones de camara, y la compara con un perceptron multicapa **entrenado** con "
            "cientos de imagenes. Ademas deja preparado el flujo para usar fotos propias."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Preprocesamiento
    # ------------------------------------------------------------------
    reporte.seccion("1. De la imagen al patron: preprocesamiento")
    reporte.lista([
        "**Escala de grises**: luminancia en [0, 1].",
        "**Umbral de Otsu**: separa trazo y fondo eligiendo el nivel de gris que maximiza la "
        "varianza entre las dos clases; se adapta solo a cada foto.",
        "**Polaridad**: el borde de la foto se toma como fondo; si es oscuro (tiza sobre pizarra), "
        "se invierte la imagen.",
        "**Limpieza e iluminacion**: filtro de mediana contra el ruido del sensor y division por una "
        "superficie cuadratica ajustada a los pixeles de fondo, que borra sombras y gradientes de luz.",
        "**Recorte y normalizacion**: se recorta el rectangulo que contiene el trazo y se estira "
        "hasta llenar la retina, de modo que posicion y tamano de la letra en la foto dejan de importar.",
        "**Reduccion por bloques**: cada pixel de la retina es la fraccion de trazo de su bloque; "
        "se enciende (+1) si supera 0.5 y queda en −1 en caso contrario.",
    ])
    DIR_DATOS.joinpath("ejemplos").mkdir(parents=True, exist_ok=True)
    ejemplo = im.distorsionar(im.renderizar_letra("B", FUENTE_MEMORIA), np.random.default_rng(11), 1.2)
    ejemplo.save(DIR_DATOS / "ejemplos" / "B_distorsionada.png")
    patron, pasos = im.preprocesar(ejemplo, devolver_pasos=True)
    vz.estilo()
    fig, ejes = vz.plt.subplots(1, 6, figsize=(16.5, 3.4))
    ejes[0].imshow(np.asarray(ejemplo), cmap="gray", vmin=0, vmax=255)
    ejes[0].set_title("1. imagen de entrada")
    ejes[1].imshow(pasos["limpia"], cmap="gray", vmin=0, vmax=1)
    ejes[1].set_title("2. luz corregida")
    ejes[2].hist(pasos["limpia"].ravel(), bins=60, color=vz.AZUL)
    ejes[2].axvline(pasos["umbral"], color=vz.NARANJA, lw=2, label=f"Otsu = {pasos['umbral']:.2f}")
    ejes[2].set_yscale("log")
    ejes[2].set_title("3. histograma y umbral")
    ejes[2].legend(loc="upper left", fontsize=8)
    ejes[3].imshow(pasos["trazo"], cmap="gray_r")
    ejes[3].set_title("4. trazo binarizado")
    ejes[4].imshow(pasos["recorte"], cmap="gray_r")
    ejes[4].set_title("5. recorte normalizado")
    vz.mapa_retina(patron, "6. retina 7x6 (±1)", forma=(7, 6), ax=ejes[5])
    for ax in ejes[[0, 1, 3, 4]]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "12_preprocesamiento.png"),
                   "Una 'foto' de la letra B (rotada, desenfocada, con luz desigual y ruido) "
                   "reducida a 42 pixeles bipolares.")

    # ------------------------------------------------------------------
    # 2. Datos
    # ------------------------------------------------------------------
    reporte.seccion("2. Imagenes de trabajo")
    muestras_vis = []
    for fuente in FUENTES_ENTRENAMIENTO + [FUENTE_NUEVA]:
        limpia = im.renderizar_letra("D", fuente)
        limpia.save(DIR_DATOS / "ejemplos" / f"D_{fuente}.png")
        muestras_vis.append((fuente, limpia, im.distorsionar(limpia, np.random.default_rng(5))))
    for letra in LETRAS:
        im.renderizar_letra(letra, FUENTE_MEMORIA).save(DIR_DATOS / "ejemplos" / f"{letra}_{FUENTE_MEMORIA}.png")
    fig, ejes = vz.plt.subplots(3, len(muestras_vis), figsize=(2.1 * len(muestras_vis), 6.6))
    for j, (fuente, limpia, sucia) in enumerate(muestras_vis):
        ejes[0, j].imshow(np.asarray(limpia), cmap="gray", vmin=0, vmax=255)
        ejes[0, j].set_title(fuente + (" (nueva)" if fuente == FUENTE_NUEVA else ""), fontsize=8)
        ejes[1, j].imshow(np.asarray(sucia), cmap="gray", vmin=0, vmax=255)
        vz.mapa_retina(im.preprocesar(sucia), "", forma=(7, 6), ax=ejes[2, j])
        for i in (0, 1):
            ejes[i, j].set_xticks([])
            ejes[i, j].set_yticks([])
            ejes[i, j].grid(False)
    ejes[0, 0].set_ylabel("limpia")
    ejes[1, 0].set_ylabel("distorsionada")
    ejes[2, 0].set_ylabel("retina 7x6")
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "12_tipografias.png"),
                   "La letra D en las seis tipografias usadas: limpia, con distorsiones de camara y "
                   "tras el preprocesamiento.")

    inicio = time.time()
    entrenamiento = generar(FUENTES_ENTRENAMIENTO, N_ENTRENAMIENTO, rng)
    validacion = generar(FUENTES_ENTRENAMIENTO, N_ENTRENAMIENTO // 4, rng)
    prueba_conocidas = generar(FUENTES_ENTRENAMIENTO, N_PRUEBA // len(FUENTES_ENTRENAMIENTO), rng)
    prueba_memoria = generar([FUENTE_MEMORIA], N_PRUEBA, rng)
    prueba_nueva = generar([FUENTE_NUEVA], N_PRUEBA, rng)
    print(f"imagenes generadas en {time.time() - inicio:.1f} s")
    reporte.tabla(pd.DataFrame([
        {"conjunto": "entrenamiento del MLP", "tipografias": ", ".join(FUENTES_ENTRENAMIENTO),
         "imagenes": len(entrenamiento)},
        {"conjunto": "validacion del MLP (parada temprana)", "tipografias": ", ".join(FUENTES_ENTRENAMIENTO),
         "imagenes": len(validacion)},
        {"conjunto": "prueba A: tipografia de las memorias", "tipografias": FUENTE_MEMORIA,
         "imagenes": len(prueba_memoria)},
        {"conjunto": "prueba B: tipografias del entrenamiento", "tipografias": ", ".join(FUENTES_ENTRENAMIENTO),
         "imagenes": len(prueba_conocidas)},
        {"conjunto": "prueba C: tipografia nueva", "tipografias": FUENTE_NUEVA,
         "imagenes": len(prueba_nueva)},
    ]), "Conjuntos de imagenes (distorsiones aleatorias independientes en cada imagen)",
        nombre_csv="12_conjuntos")
    reporte.texto(
        "Todas las imagenes de prueba son distorsiones nuevas, distintas de las de entrenamiento. "
        f"La tipografia {FUENTE_NUEVA} no se usa nunca para entrenar: mide si el MLP generaliza a "
        "un estilo de letra que no conoce."
    )

    # ------------------------------------------------------------------
    # 3. Hopfield con imagenes
    # ------------------------------------------------------------------
    reporte.seccion("3. La red de Hopfield sobre imagenes")
    resultados, confusiones, redes_mlp = [], {}, {}
    for forma in ((7, 6), (14, 12)):
        n = forma[0] * forma[1]
        memorias = np.array([im.preprocesar(im.renderizar_letra(l, FUENTE_MEMORIA), forma=forma) for l in LETRAS])
        solap = memorias @ memorias.T / n
        if forma == (7, 6):
            fig, _ = vz.rejilla_retinas(list(memorias), [f"memoria {l}" for l in LETRAS], forma=forma,
                                        sup_titulo=f"Memorias: letras {FUENTE_MEMORIA} preprocesadas a 7x6")
            reporte.figura(vz.guardar(fig, FIGURAS / "12_memorias.png"),
                           "Los cuatro patrones almacenados, obtenidos de imagenes limpias.")
            fig, _ = vz.mapa_calor(solap, titulo="Solapamiento entre memorias (7x6)", etiquetas_filas=LETRAS,
                                   etiquetas_columnas=LETRAS, divergente=True, figsize=(4.8, 4.0))
            reporte.figura(vz.guardar(fig, FIGURAS / "12_solapamiento.png"),
                           "B y D, reducidas a 42 pixeles, son casi la misma imagen.")
            reporte.texto(
                f"Al reducir las letras reales a 7x6 el solapamiento B-D es {solap[1, 3]:.2f}: en 42 "
                "pixeles la diferencia entre la B y la D se reduce al trazo central y a las esquinas "
                "derechas. Es el mismo problema de correlacion estudiado en el experimento 11, ahora "
                "impuesto por los datos y no por el dibujo."
            )
        fijos = [RedHopfield(n).almacenar(memorias).es_punto_fijo(p) for p in memorias]
        if forma == (7, 6):
            memorias_7x6 = memorias

        X_ent, y_ent = a_matriz(entrenamiento, forma)
        X_val, y_val = a_matriz(validacion, forma)
        conjuntos = {"A: tipografia de las memorias": a_matriz(prueba_memoria, forma),
                     "B: tipografias del entrenamiento": a_matriz(prueba_conocidas, forma),
                     "C: tipografia nueva": a_matriz(prueba_nueva, forma)}

        mlp = PerceptronMulticapa([n, OCULTAS, 4], razon_aprendizaje=0.1, momento=0.5, max_epocas=EPOCAS,
                                  error_objetivo=1e-3, escala_inicial=0.3, paciencia=PACIENCIA,
                                  semilla=SEMILLA)
        inicio = time.time()
        mlp.entrenar(X_ent, uno_de_n(y_ent), X_val, uno_de_n(y_val))
        print(f"MLP {mlp.arquitectura} entrenado en {time.time() - inicio:.1f} s ({mlp.epocas_usadas} epocas)")
        redes_mlp[forma] = mlp

        modelos = {
            "Hopfield (Hebb)": ReconocedorHopfield(memorias, "hebb").predecir,
            "Hopfield (pseudoinversa)": ReconocedorHopfield(memorias, "pseudoinversa").predecir,
            "plantilla mas cercana": lambda X, M=memorias: plantilla_mas_cercana(M, X),
            f"MLP {mlp.arquitectura}": lambda X, r=mlp: np.argmax(r.salida(X), axis=1),
        }
        for nombre_modelo, predecir in modelos.items():
            fila = {"retina": f"{forma[0]}x{forma[1]}", "modelo": nombre_modelo}
            for nombre_conj, (X, y) in conjuntos.items():
                pred = predecir(X)
                fila[f"acierto {nombre_conj[:1]} %"] = 100 * float(np.mean(pred == y))
                if nombre_conj.startswith("A") or nombre_conj.startswith("C"):
                    confusiones[(forma, nombre_modelo, nombre_conj[:1])] = matriz_confusion(y, pred)
            if nombre_modelo.startswith("Hopfield"):
                fila["sin respuesta A %"] = 100 * float(np.mean(predecir(conjuntos["A: tipografia de las memorias"][0]) < 0))
            resultados.append(fila)
            print(fila)
        if forma == (7, 6):
            reporte.tabla(pd.DataFrame({"letra": LETRAS, "punto fijo (Hebb)": fijos}),
                          "¿Son puntos fijos las memorias obtenidas de imagenes? (7x6)",
                          nombre_csv="12_puntos_fijos")

    df = pd.DataFrame(resultados)
    reporte.texto(
        "Cada imagen de prueba se preprocesa y se usa como estado inicial de la red. Se cuenta "
        "como acierto que la red termine exactamente en la memoria de la letra correcta; si "
        "termina en un estado espurio o en un inverso, la red **no da respuesta**."
    )
    conf = confusiones[((7, 6), "Hopfield (Hebb)", "A")]
    reporte.tabla(conf.reset_index().rename(columns={"index": "letra"}),
                  "Hopfield (Hebb, 7x6) sobre la tipografia de las memorias: matriz de confusion",
                  nombre_csv="12_confusion_hopfield")
    fila_h = df[(df["retina"] == "7x6") & (df["modelo"] == "Hopfield (Hebb)")].iloc[0]
    reporte.texto(
        f"Con distorsiones de camara sobre la **misma tipografia** que las memorias, la red de "
        f"Hopfield (Hebb) acierta el {fila_h['acierto A %']:.0f} % y queda sin respuesta en el "
        f"{fila_h['sin respuesta A %']:.0f} %: el preprocesamiento deja la retina muy cerca de la "
        "memoria y la dinamica recurrente corrige el resto, como con el ruido de pixel del "
        "experimento 11. Los fallos se concentran en B y D, las letras mas correlacionadas."
    )

    # desglose por tipografia (conjunto B): ¿que le cuesta a cada modelo?
    fuentes_b = np.array([f for _, _, f in prueba_conocidas])
    X_b, y_b = a_matriz(prueba_conocidas, (7, 6))
    pred_hop = ReconocedorHopfield(memorias_7x6, "hebb").predecir(X_b)
    pred_mlp = np.argmax(redes_mlp[(7, 6)].salida(X_b), axis=1)
    por_fuente = pd.DataFrame([
        {"tipografia": f,
         "Hopfield (Hebb) %": 100 * float(np.mean(pred_hop[fuentes_b == f] == y_b[fuentes_b == f])),
         "MLP 42-16-4 %": 100 * float(np.mean(pred_mlp[fuentes_b == f] == y_b[fuentes_b == f]))}
        for f in FUENTES_ENTRENAMIENTO])
    reporte.tabla(por_fuente, "Aciertos por tipografia (conjunto B, retina 7x6)",
                  nombre_csv="12_por_tipografia")
    peor = por_fuente.sort_values("Hopfield (Hebb) %").iloc[0]
    reporte.texto(
        "El problema de Hopfield no es el ruido de la camara sino el **estilo** de la letra. Con "
        f"la tipografia {peor['tipografia']} acierta solo el {peor['Hopfield (Hebb) %']:.0f} %: "
        "sus trazos finos, reducidos a 42 pixeles, quedan lejos de la memoria (escrita en "
        "negrita) y el estado inicial cae en la cuenca de otra letra o en un estado espurio. La "
        "red solo conoce una imagen por letra y trata cualquier desviacion como ruido que "
        "corregir."
    )

    # ------------------------------------------------------------------
    # 4. MLP entrenado
    # ------------------------------------------------------------------
    reporte.seccion("4. Perceptron multicapa entrenado con imagenes")
    mlp = redes_mlp[(7, 6)]
    reporte.lista([
        f"Arquitectura 42-{OCULTAS}-4: una entrada por pixel de la retina, {OCULTAS} neuronas "
        "ocultas sigmoidales y una neurona de salida por letra (codificacion *uno de n*: la "
        "salida deseada de una B es (0, 1, 0, 0)).",
        "La letra reconocida es la neurona de salida con mayor activacion.",
        f"Entrenamiento: {len(entrenamiento)} imagenes distorsionadas de "
        f"{len(FUENTES_ENTRENAMIENTO)} tipografias, retropropagacion estocastica con γ = 0.1, "
        f"α = 0.5, {EPOCAS} epocas como maximo.",
        f"**Parada temprana**: tras cada epoca se mide el error sobre {len(validacion)} imagenes de "
        f"validacion (tipografias del entrenamiento, distorsiones nuevas); si no mejora durante "
        f"{PACIENCIA} epocas se detiene el entrenamiento y se restauran los mejores pesos. "
        "La tipografia nueva no interviene en ninguna decision del entrenamiento.",
        f"Resultado: el entrenamiento termino en la epoca {mlp.epocas_usadas} "
        + ("al alcanzar el error objetivo (E ≤ 0.001)." if mlp.motivo_parada == "error_objetivo"
           else "por parada temprana." if mlp.motivo_parada == "parada_temprana"
           else "al agotar las epocas."),
    ])
    fig, _ = vz.curvas_comparadas(
        {"entrenamiento": mlp.curva_error(), "validacion": mlp.curva_error(prueba=True)},
        titulo=f"MLP {mlp.arquitectura}: evolucion del error", etiqueta_y="E", escala_log=True)
    reporte.figura(vz.guardar(fig, FIGURAS / "12_error_mlp.png"),
                   "Error de entrenamiento y de validacion por epoca.")
    conf = confusiones[((7, 6), f"MLP {mlp.arquitectura}", "C")]
    reporte.tabla(conf.reset_index().rename(columns={"index": "letra"}),
                  f"MLP {mlp.arquitectura} sobre la tipografia nueva: matriz de confusion",
                  nombre_csv="12_confusion_mlp")

    # ------------------------------------------------------------------
    # 5. Comparacion
    # ------------------------------------------------------------------
    reporte.seccion("5. Comparacion")
    reporte.tabla(df, "Porcentaje de aciertos por modelo, retina y conjunto de prueba",
                  nombre_csv="12_comparacion")
    fila_h = df[(df["retina"] == "7x6") & (df["modelo"] == "Hopfield (Hebb)")].iloc[0]
    fila_p = df[(df["retina"] == "7x6") & (df["modelo"] == "plantilla mas cercana")].iloc[0]
    fila_m = df[(df["retina"] == "7x6") & df["modelo"].str.startswith("MLP")].iloc[0]
    fila_h14 = df[(df["retina"] == "14x12") & (df["modelo"] == "Hopfield (Hebb)")].iloc[0]
    reporte.texto(
        f"Con la intensidad de distorsion de referencia, el MLP 42-{OCULTAS}-4 acierta el "
        f"{fila_m['acierto B %']:.0f} % en las tipografias del entrenamiento y el "
        f"{fila_m['acierto C %']:.0f} % en la tipografia nueva; la red de Hopfield con regla de "
        f"Hebb, el {fila_h['acierto B %']:.0f} % y el {fila_h['acierto C %']:.0f} %."
    )

    # barrido de intensidad de distorsion sobre la tipografia nueva
    rng_barrido = np.random.default_rng(SEMILLA + 1)
    # MLP con aumento de datos: las mismas tipografias, con distorsiones de intensidad 1, 1.75 y 2.5
    rng_aumento = np.random.default_rng(SEMILLA + 2)
    aumentado = [m for k in (1.0, 1.75, 2.5)
                 for m in generar(FUENTES_ENTRENAMIENTO, N_ENTRENAMIENTO // 3, rng_aumento, k)]
    validacion_aum = [m for k in (1.0, 1.75, 2.5)
                      for m in generar(FUENTES_ENTRENAMIENTO, N_ENTRENAMIENTO // 12, rng_aumento, k)]
    X_aum, y_aum = a_matriz(aumentado, (7, 6))
    X_vaum, y_vaum = a_matriz(validacion_aum, (7, 6))
    mlp_aum = PerceptronMulticapa([42, OCULTAS, 4], razon_aprendizaje=0.1, momento=0.5, max_epocas=EPOCAS,
                                  error_objetivo=1e-3, escala_inicial=0.3, paciencia=PACIENCIA,
                                  semilla=SEMILLA)
    mlp_aum.entrenar(X_aum, uno_de_n(y_aum), X_vaum, uno_de_n(y_vaum))
    print(f"MLP con aumento de datos: {len(aumentado)} imagenes, {mlp_aum.epocas_usadas} epocas "
          f"({mlp_aum.motivo_parada})")

    nombre_mlp, nombre_aum = f"MLP 42-{OCULTAS}-4", f"MLP 42-{OCULTAS}-4 + aumento de datos"
    curvas = {"Hopfield (Hebb)": [], "plantilla mas cercana": [], nombre_mlp: [], nombre_aum: []}
    hop = ReconocedorHopfield(memorias_7x6, "hebb")
    limpias_nueva = np.array([im.preprocesar(im.renderizar_letra(l, FUENTE_NUEVA)) for l in LETRAS])
    danio = []
    for intensidad in INTENSIDADES:
        X_i, y_i = a_matriz(generar([FUENTE_NUEVA], N_BARRIDO, rng_barrido, intensidad), (7, 6))
        danio.append(float(np.mean(np.sum(X_i != limpias_nueva[y_i], axis=1))))
        curvas["Hopfield (Hebb)"].append(100 * float(np.mean(hop.predecir(X_i) == y_i)))
        curvas["plantilla mas cercana"].append(100 * float(np.mean(plantilla_mas_cercana(memorias_7x6, X_i) == y_i)))
        curvas[nombre_mlp].append(100 * float(np.mean(np.argmax(mlp.salida(X_i), axis=1) == y_i)))
        curvas[nombre_aum].append(100 * float(np.mean(np.argmax(mlp_aum.salida(X_i), axis=1) == y_i)))
    barrido = pd.DataFrame({"intensidad": INTENSIDADES, **curvas,
                            "pixeles distintos de la retina limpia (media)": danio})
    print(barrido.round(1).to_string(index=False))
    fig, ax = vz.curvas_comparadas(curvas, titulo="Robustez frente a la distorsion (tipografia nueva, 7x6)",
                                   etiqueta_y="acierto %", etiqueta_x="intensidad de distorsion",
                                   escala_log=False, x=INTENSIDADES)
    ax.set_ylim(0, 102)
    reporte.figura(vz.guardar(fig, FIGURAS / "12_robustez.png"),
                   "Acierto frente a la intensidad de las distorsiones de camara (1.0 = conjuntos de prueba).")
    reporte.tabla(barrido, f"Acierto % por intensidad de distorsion ({4 * N_BARRIDO} imagenes por nivel)",
                  nombre_csv="12_robustez")
    ultimo = {k: v[-1] for k, v in curvas.items()}
    reporte.texto(
        f"Al subir la intensidad a {INTENSIDADES[-1]:.1f} (rotaciones de hasta "
        f"{10 * INTENSIDADES[-1]:.0f}°, desenfoque fuerte, poco contraste) la red de Hopfield cae "
        f"al {ultimo['Hopfield (Hebb)']:.0f} % y la plantilla mas cercana al "
        f"{ultimo['plantilla mas cercana']:.0f} %. El MLP entrenado solo con distorsiones de "
        f"intensidad 1.0 cae al {ultimo[nombre_mlp]:.0f} %: **no generaliza a variaciones que no "
        f"vio**. Entrenado con {len(aumentado)} imagenes distorsionadas con intensidades 1, 1.75 y 2.5 "
        f"(*aumento de datos*), la misma arquitectura mantiene el {ultimo[nombre_aum]:.0f} %. "
        "Esta es la leccion principal para entrenar con imagenes reales: los ejemplos de "
        "entrenamiento deben cubrir la variacion que la red encontrara despues."
    )
    reporte.texto(
        "¿Por que ni siquiera el MLP con aumento de datos llega mas alto? La ultima columna de la "
        "tabla mide cuanto se parece la retina de cada foto a la de la letra limpia: con "
        f"intensidad {INTENSIDADES[-1]:.1f} difieren en {danio[-1]:.0f} de 42 pixeles de media "
        "(tanto como dos retinas al azar, que difieren en unos 21). Con rotaciones fuertes, letras "
        "pequenas y muy desenfocadas, la reduccion a 7x6 **destruye buena parte de la informacion** "
        "antes de que llegue a la red. A partir de ahi no mejora cambiando de red sino cambiando la entrada: corregir la "
        "rotacion en el preprocesamiento, usar una retina de mas resolucion o, en ultimo termino, "
        "redes convolucionales que trabajan sobre la imagen completa."
    )
    reporte.texto("Por que difieren los modelos:")
    reporte.lista([
        "La red de Hopfield es una **memoria**, no un clasificador: conoce una imagen por letra y "
        "no puede aprender que una D rotada o de otra tipografia sigue siendo una D.",
        "El MLP aprende de **cientos de ejemplos** que variaciones no cambian la clase y cuales si "
        "(para la B y la D, el trazo central y las esquinas derechas). Es aprendizaje supervisado, "
        "y por eso generaliza a una tipografia nueva.",
        f"La plantilla mas cercana (sin dinamica) acierta mas que Hopfield "
        f"({fila_p['acierto B %']:.0f} % frente a {fila_h['acierto B %']:.0f} % en el conjunto B): "
        "la dinamica recurrente puede llevar el estado a un espurio aunque la memoria mas parecida "
        "fuera la correcta. La regla de la pseudoinversa reduce ese problema.",
        f"Subir la resolucion a 14x12 (168 neuronas) no ayuda a Hopfield con la regla de Hebb "
        f"({fila_h14['acierto B %']:.0f} % en el conjunto B): con mas pixeles las diferencias de "
        "estilo tambien pesan mas.",
    ])

    # ------------------------------------------------------------------
    # 6. Fotos propias
    # ------------------------------------------------------------------
    reporte.seccion("6. Como usar fotos propias")
    carpeta = carpeta_propias or (DIR_DATOS / "propias")
    reporte.texto(
        "1. Escribir o imprimir las letras A, B, C y D, fotografiarlas con el movil (una letra "
        "por foto, fondo liso, que la letra ocupe buena parte del encuadre).\n"
        "2. Guardarlas en `datos/imagenes/propias/A/`, `.../B/`, etc., o en una sola carpeta con "
        "nombres que empiecen por la letra (`B_1.jpg`).\n"
        "3. Ejecutar `python experimentos/exp12_imagenes_reales.py` (o con `--carpeta ruta`).\n"
        "4. Para **entrenar** con fotos propias en lugar de imagenes generadas: "
        "`X, etiquetas, _ = imagenes.cargar_carpeta(ruta)` y pasar `X` y las etiquetas en "
        "codificacion uno de n a `PerceptronMulticapa.entrenar`, igual que en la seccion 4. "
        "Conviene reservar algunas fotos para prueba y, si hay pocas, ampliar el conjunto con "
        "`imagenes.distorsionar` (aumento de datos)."
    )
    try:
        X_prop, etiquetas_prop, rutas_prop = im.cargar_carpeta(carpeta)
    except FileNotFoundError:
        reporte.texto(f"*No se encontraron fotos en `{carpeta.relative_to(RAIZ) if carpeta.is_relative_to(RAIZ) else carpeta}`; "
                      "esta seccion se completa automaticamente al anadirlas.*")
        print(f"sin fotos propias en {carpeta}")
    else:
        memorias = np.array([im.preprocesar(im.renderizar_letra(l, FUENTE_MEMORIA)) for l in LETRAS])
        hop = ReconocedorHopfield(memorias, "hebb").predecir(X_prop)
        mlp_pred = np.argmax(redes_mlp[(7, 6)].salida(X_prop), axis=1)
        tabla = pd.DataFrame({
            "archivo": [r.name for r in rutas_prop], "letra real": etiquetas_prop,
            "Hopfield": [LETRAS[p] if p >= 0 else "sin respuesta" for p in hop],
            "MLP": [LETRAS[p] for p in mlp_pred],
        })
        print(tabla.to_string(index=False))
        reporte.tabla(tabla, "Reconocimiento de las fotos propias", nombre_csv="12_fotos_propias")
        fig, _ = vz.rejilla_retinas(list(X_prop[:12]), [f"{e}: MLP → {m}" for e, m in
                                                         zip(etiquetas_prop[:12], tabla["MLP"][:12])],
                                    forma=(7, 6), n_columnas=6, sup_titulo="Fotos propias preprocesadas")
        reporte.figura(vz.guardar(fig, FIGURAS / "12_fotos_propias.png"), "Fotos propias en la retina de 7x6.")

    # ------------------------------------------------------------------
    # 7. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("7. Conclusiones")
    reporte.lista([
        "Un preprocesamiento sencillo (grises, correccion de iluminacion, Otsu, recorte robusto, "
        "reduccion por bloques) basta para llevar fotos de letras a la retina de 7x6 que usan las "
        "redes del curso. Es la pieza que mas influye en el resultado: sin corregir la luz ni "
        "filtrar las motas de ruido, el recorte se desplaza y todos los modelos empeoran.",
        "La red de Hopfield reconoce bien fotos de la misma letra que memorizo, pero falla cuando "
        "cambia el estilo (trazos finos, serifas) o la distorsion es fuerte.",
        "Para reconocer imagenes reales el modelo adecuado es un clasificador supervisado "
        "entrenado con muchos ejemplos variados: el MLP generaliza incluso a una tipografia que "
        "no vio, y con aumento de datos es el que mejor resiste las distorsiones fuertes. Su "
        "limite lo marcan los ejemplos de entrenamiento: solo es robusto frente a las "
        "variaciones que ha visto.",
        "Para fotos de mayor resolucion o problemas con mas clases (por ejemplo, digitos "
        "manuscritos), el siguiente paso natural serian redes convolucionales, que incorporan la "
        "invariancia a desplazamientos en la propia arquitectura.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme escrito en {ruta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--carpeta", type=Path, default=None,
                        help="carpeta con fotos propias (subcarpetas A/, B/, C/, D/)")
    main(parser.parse_args().carpeta)
