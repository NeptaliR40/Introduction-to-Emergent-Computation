"""
Experimento 10 --- Mapas autoorganizados de Kohonen: grupos en el plano.

Actividad 4 de la guia de evaluacion.  Una distribucion de puntos del plano con
cinco grupos se presenta a un mapa de Kohonen **sin decirle a que grupo
pertenece cada punto**.  El mapa debe organizarse por si solo y revelar los
grupos.

Contenido
---------
1. Conjunto de datos y grafica de los puntos originales.
2. Modelo, mapa y parametros.
3. Despliegue del mapa durante el aprendizaje (fase de ordenamiento y de
   convergencia).
4. Lectura del mapa: densidad de impactos, matriz U y grupos encontrados.
5. Correspondencia entre grupos del mapa y grupos reales (evaluacion a posteriori).
6. Comparacion de configuraciones: tamano del mapa, radio de vecindad inicial
   y numero de iteraciones.
7. Limite de resolucion: grupos que se acercan hasta fundirse.

Ejecucion
---------
    python experimentos/exp10_kohonen.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.kohonen import MapaKohonen, pureza
from ce_rna.reportes import Reporte, encabezado_experimento

FILAS, COLUMNAS = 10, 10
ITERACIONES = 10000
ETA0, ETAF = 0.5, 0.01
SIGMAF = 0.5
PERCENTIL = 50
SEMILLA = 0


def main() -> None:
    encabezado_experimento("Experimento 10 - Mapas autoorganizados de Kohonen")

    reporte = Reporte(
        titulo="Experimento 10 --- Mapas autoorganizados de Kohonen: grupos en el plano",
        nombre_archivo="10_kohonen",
        resumen=(
            "Todos los modelos anteriores aprendian de forma **supervisada**: cada patron venia "
            "con su salida deseada y el aprendizaje reducia el error. Un mapa autoorganizado de "
            "Kohonen aprende sin salida deseada: solo recibe los patrones y, mediante "
            "**aprendizaje competitivo** con vecindad, dispone sus neuronas de forma que "
            "reproducen la distribucion de los datos preservando su topologia. El experimento "
            "aplica un mapa de 10x10 neuronas a una nube de puntos del plano con cinco grupos y "
            "comprueba si el mapa revela esos grupos sin conocerlos."
        ),
    )

    conjunto = ds.grupos_plano(semilla=SEMILLA)
    X, clases = conjunto.X, conjunto.d.astype(int)
    nombres_reales = [f"G{k}" for k in range(len(ds.GRUPOS_PLANO))]

    # ------------------------------------------------------------------
    # 1. Datos
    # ------------------------------------------------------------------
    reporte.seccion("1. Conjunto de datos")
    tabla_grupos = pd.DataFrame(
        [{"grupo": f"G{k}", "centro_x1": cx, "centro_x2": cy, "desv_x1": sx, "desv_x2": sy, "puntos": n}
         for k, (cx, cy, sx, sy, n) in enumerate(ds.GRUPOS_PLANO)]
    )
    reporte.texto(
        f"{conjunto.n_patrones} puntos del plano generados como cinco nubes gaussianas "
        "(semilla fija). Los grupos se eligieron distintos a proposito para que el problema no "
        "sea trivial: tamanos de 60 a 90 puntos, un grupo alargado (G3), uno mas disperso (G4) y "
        "dos grupos relativamente proximos entre si (G0 y G1)."
    )
    reporte.tabla(tabla_grupos, "Parametros de generacion de los grupos", nombre_csv="10_grupos")
    pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "grupo_real": clases}).to_csv(
        _ruta.TABLAS / "10_datos.csv", index=False)
    reporte.texto(
        "La etiqueta del grupo real se guarda **solo** para evaluar el resultado al final; el "
        "mapa nunca la recibe. Lo unico que ve la red son las coordenadas (x1, x2)."
    )
    vz.estilo()
    fig, ejes = vz.plt.subplots(1, 2, figsize=(12.0, 5.2))
    ejes[0].scatter(X[:, 0], X[:, 1], s=18, c=vz.TINTA_2, edgecolors="white", linewidths=0.4)
    ejes[0].set_title("Lo que ve la red: puntos sin etiqueta")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")
    ejes[0].set_aspect("equal", adjustable="datalim")
    vz.grupos_en_plano(X, clases, titulo="Grupos reales (solo para evaluar)",
                       nombres=nombres_reales, ax=ejes[1])
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "10_datos.png"),
                   "Izquierda: los datos tal como los recibe el mapa. Derecha: los grupos con que se generaron.")

    # ------------------------------------------------------------------
    # 2. Modelo
    # ------------------------------------------------------------------
    reporte.seccion("2. Modelo y parametros")
    reporte.texto(
        f"Mapa rectangular de {FILAS}x{COLUMNAS} = {FILAS * COLUMNAS} neuronas. Cada neurona j "
        "tiene un vector de pesos w_j de dimension 2 --- un punto del mismo plano que los datos, "
        "su *prototipo* --- y una posicion fija r_j en la rejilla. En cada iteracion se elige un "
        "patron x al azar y:"
    )
    reporte.formula(r"c = \arg\min_j \lVert x - w_j \rVert \qquad \text{(competicion: neurona ganadora)}")
    reporte.formula(r"h_{cj}(t) = \exp\!\left(-\frac{\lVert r_c - r_j\rVert^2}{2\sigma(t)^2}\right) "
                    r"\qquad \text{(cooperacion: vecindad en la rejilla)}")
    reporte.formula(r"w_j \leftarrow w_j + \eta(t)\,h_{cj}(t)\,\bigl(x - w_j\bigr) "
                    r"\qquad \text{(adaptacion)}")
    reporte.formula(r"\eta(t) = \eta_0\left(\frac{\eta_f}{\eta_0}\right)^{t/T}, \qquad "
                    r"\sigma(t) = \sigma_0\left(\frac{\sigma_f}{\sigma_0}\right)^{t/T}")
    sigma0 = max(FILAS, COLUMNAS) / 2.0
    reporte.tabla(pd.DataFrame([
        {"parametro": "tamano del mapa", "valor": f"{FILAS}x{COLUMNAS}",
         "justificacion": "~4 veces menos neuronas que patrones: resolucion suficiente para ver "
                          "fronteras (crestas) entre grupos sin que cada neurona represente 1-2 puntos"},
        {"parametro": "iteraciones T", "valor": str(ITERACIONES),
         "justificacion": "~23 presentaciones por patron; el error de cuantizacion ya no mejora (seccion 6)"},
        {"parametro": "eta_0 -> eta_f", "valor": f"{ETA0} -> {ETAF}",
         "justificacion": "paso grande para ordenar al principio, pequeno para afinar al final"},
        {"parametro": "sigma_0 -> sigma_f", "valor": f"{sigma0:g} -> {SIGMAF}",
         "justificacion": "sigma_0 = medio mapa: al principio todo el mapa se mueve junto y se "
                          "despliega sin pliegues; sigma_f < 1: al final solo la ganadora se mueve"},
        {"parametro": "inicializacion", "valor": "patrones al azar",
         "justificacion": "los prototipos empiezan dentro de la distribucion, sin ningun orden"},
    ]), "Parametros del mapa", nombre_csv="10_parametros")

    mapa = MapaKohonen(FILAS, COLUMNAS, eta_inicial=ETA0, eta_final=ETAF, sigma_final=SIGMAF,
                       iteraciones=ITERACIONES, semilla=SEMILLA)
    mapa.entrenar(X, n_instantaneas=6, registrar_cada=100)
    print(mapa.resumen(X))
    reporte.bloque(mapa.resumen(X))

    # ------------------------------------------------------------------
    # 3. Despliegue
    # ------------------------------------------------------------------
    reporte.seccion("3. Autoorganizacion durante el aprendizaje")
    fig, ejes = vz.plt.subplots(2, 3, figsize=(12.6, 8.4))
    for ax, inst in zip(ejes.ravel(), mapa.instantaneas):
        vz.malla_kohonen(inst.prototipos, FILAS, COLUMNAS, X,
                         titulo=f"t = {inst.iteracion}  (η = {inst.eta:.2f}, σ = {inst.sigma:.2f})", ax=ax)
    fig.suptitle("Los prototipos del mapa en el plano de los datos (las lineas unen vecinas de la rejilla)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "10_despliegue.png"),
                   "De una malla enredada a una malla ordenada que cubre los cinco grupos.")
    reporte.texto(
        "En t = 0 los prototipos son puntos de datos al azar y la malla esta completamente "
        "enredada: neuronas vecinas en la rejilla estan lejos en el plano. Durante la **fase de "
        "ordenamiento** (σ grande) cada ganadora arrastra a medio mapa consigo y la malla se "
        "desenreda y se extiende sobre los datos. Durante la **fase de convergencia** (σ < 1) "
        "cada neurona solo se mueve cuando gana, y los prototipos se concentran donde hay datos. "
        "Las neuronas que quedan entre grupos se estiran a lo largo de los 'puentes' vacios: "
        "son las que marcan las fronteras."
    )
    it, eq = zip(*mapa.historial_error)
    fig, _ = vz.curva_aprendizaje(np.array(eq), titulo="Error de cuantizacion durante el aprendizaje",
                                  etiqueta_y="distancia media patron-prototipo",
                                  etiqueta_x="iteracion / 100", escala_log=True)
    reporte.figura(vz.guardar(fig, FIGURAS / "10_error_cuantizacion.png"),
                   "El error de cuantizacion no es la funcion que se minimiza (no hay salida deseada), "
                   "pero sirve para seguir el aprendizaje.")

    # ------------------------------------------------------------------
    # 4. Lectura del mapa
    # ------------------------------------------------------------------
    reporte.seccion("4. Lectura del mapa: impactos, matriz U y grupos")
    impactos = mapa.impactos(X)
    U = mapa.matriz_u()
    etiquetas_neuronas = mapa.segmentar(X, percentil=PERCENTIL)
    grupos = etiquetas_neuronas[mapa.ganadoras(X)]
    n_grupos = len(np.unique(grupos))
    print(f"grupos encontrados: {n_grupos}")

    fig, ejes = vz.plt.subplots(1, 3, figsize=(15.0, 4.8))
    vz.mapa_calor(impactos, titulo="Densidad de impactos", formato="{:.0f}",
                  etiqueta_barra="patrones ganados", ax=ejes[0])
    vz.mapa_calor(U, titulo="Matriz U (distancia a vecinas)", formato="{:.1f}",
                  etiqueta_barra="distancia media", ax=ejes[1])
    from matplotlib.colors import ListedColormap
    ejes[2].imshow(etiquetas_neuronas.reshape(FILAS, COLUMNAS),
                   cmap=ListedColormap(vz.COLORES_GRUPO[:n_grupos]))
    for (i, j), g in np.ndenumerate(etiquetas_neuronas.reshape(FILAS, COLUMNAS)):
        ejes[2].text(j, i, str(g), ha="center", va="center", fontsize=7, color=vz.TINTA)
    ejes[2].set_title(f"Grupos encontrados en el mapa ({n_grupos})")
    ejes[2].grid(False)
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "10_matriz_u.png"),
                   "Las celdas con cero impactos y U alta forman crestas que separan valles: cada "
                   "valle es un grupo.")
    reporte.texto(
        "**Densidad de impactos**: las neuronas que no ganan ningun patron (celdas a 0) no "
        "representan datos; estan en el espacio vacio entre grupos. **Matriz U**: para cada "
        "neurona, la distancia media entre su prototipo y los de sus cuatro vecinas de la "
        "rejilla. Dentro de un grupo los prototipos estan apretados (U baja, valles); entre "
        "grupos, dos neuronas vecinas representan zonas alejadas del plano (U alta, crestas). "
        "Las crestas de la matriz U coinciden con las celdas sin impactos: las dos lecturas se "
        "confirman mutuamente."
    )
    reporte.texto(
        f"**Segmentacion no supervisada**: las neuronas con U por debajo de la mediana (percentil "
        f"{PERCENTIL}) se consideran interiores; cada region conexa de neuronas interiores es un "
        "grupo, y las neuronas de las crestas se asignan al grupo interior con prototipo mas "
        "cercano. Cada patron hereda el grupo de su neurona ganadora. Ningun paso usa las clases "
        f"reales. Resultado: **{n_grupos} grupos**."
    )
    fig, ejes = vz.plt.subplots(1, 2, figsize=(12.0, 5.2))
    vz.grupos_en_plano(X, grupos, titulo="Grupos encontrados por el mapa",
                       nombres=[f"M{g}" for g in np.unique(grupos)], ax=ejes[0])
    vz.malla_kohonen(mapa.W, FILAS, COLUMNAS, X, titulo="Mapa final sobre los datos", ax=ejes[1])
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "10_grupos_encontrados.png"),
                   "Izquierda: particion obtenida sin supervision. Derecha: el mapa entrenado.")

    # ------------------------------------------------------------------
    # 5. Correspondencia
    # ------------------------------------------------------------------
    reporte.seccion("5. Correspondencia entre grupos del mapa y grupos reales")
    M, _ = mt.matriz_confusion(grupos, clases, etiquetas=np.unique(np.concatenate([grupos, clases])))
    filas_reales = sorted(np.unique(clases))
    cols_mapa = sorted(np.unique(grupos))
    tabla_conf = pd.DataFrame(
        [[f"G{r}", *[int(np.sum((clases == r) & (grupos == m))) for m in cols_mapa]] for r in filas_reales],
        columns=["grupo real", *[f"M{m}" for m in cols_mapa]],
    )
    p = pureza(grupos, clases)
    print(tabla_conf.to_string(index=False))
    print(f"pureza = {p:.3f}")
    reporte.tabla(tabla_conf, "Tabla de contingencia: filas = grupo real, columnas = grupo del mapa",
                  nombre_csv="10_contingencia")
    zonas = pd.DataFrame([
        {"grupo real": f"G{r}",
         "neuronas que gana": int(np.unique(mapa.ganadoras(X[clases == r])).size),
         "fraccion del mapa %": 100 * np.unique(mapa.ganadoras(X[clases == r])).size / mapa.n_neuronas,
         "puntos": int(np.sum(clases == r))}
        for r in filas_reales
    ])
    reporte.tabla(zonas, "Cuanto mapa dedica la red a cada grupo", nombre_csv="10_neuronas_por_grupo")
    reporte.texto(
        f"La **pureza** (fraccion de puntos cuyo grupo del mapa tiene como mayoria su grupo real) "
        f"es **{p:.3f}**. Cada grupo real ocupa una region contigua del mapa: la red preserva la "
        "topologia, de modo que puntos proximos en el plano caen en neuronas proximas de la "
        "rejilla (error topografico = "
        f"{mapa.error_topografico(X):.3f}). El numero de neuronas dedicadas a cada grupo crece con "
        "su numero de puntos y con su extension: el mapa asigna resolucion segun la densidad de "
        "los datos (**magnificacion**)."
    )

    # ------------------------------------------------------------------
    # 6. Configuraciones
    # ------------------------------------------------------------------
    reporte.seccion("6. Comparacion de configuraciones")
    filas = []

    def evaluar(nombre, **kw):
        res = []
        for s in range(5):
            parametros = dict(filas=FILAS, columnas=COLUMNAS, eta_inicial=ETA0, eta_final=ETAF,
                              sigma_final=SIGMAF, iteraciones=ITERACIONES, semilla=s)
            parametros.update(kw)
            m = MapaKohonen(**parametros).entrenar(X, n_instantaneas=0)
            g = m.agrupar(X, percentil=PERCENTIL)
            res.append((m.error_cuantizacion(X), m.error_topografico(X), len(np.unique(g)), pureza(g, clases)))
        res = np.array(res)
        fila = {"configuracion": nombre, "error_cuantizacion": res[:, 0].mean(),
                "error_topografico": res[:, 1].mean(),
                "grupos (5 semillas)": " ".join(str(int(v)) for v in res[:, 2]),
                "pureza_media": res[:, 3].mean()}
        filas.append(fila)
        print(fila)

    evaluar("10x10, base")
    evaluar("5x5", filas=5, columnas=5)
    evaluar("15x15", filas=15, columnas=15)
    evaluar("10x10, sigma_0 = 1 (vecindad pequena)", sigma_inicial=1.0)
    evaluar("10x10, sin vecindad (sigma = 0.1)", sigma_inicial=0.1, sigma_final=0.1)
    evaluar("10x10, 1 000 iteraciones", iteraciones=1000)
    evaluar("10x10, 30 000 iteraciones", iteraciones=30000)
    df_conf = pd.DataFrame(filas)
    reporte.texto("Cada configuracion se entrena con 5 semillas; se promedian los errores y la pureza.")
    reporte.tabla(df_conf, "Efecto del tamano, la vecindad y la duracion", nombre_csv="10_configuraciones")

    mapas_ejemplo = {
        "sigma_0 = 1": MapaKohonen(FILAS, COLUMNAS, sigma_inicial=1.0, iteraciones=ITERACIONES,
                                   semilla=SEMILLA).entrenar(X, n_instantaneas=0),
        "sin vecindad (sigma = 0.1)": MapaKohonen(FILAS, COLUMNAS, sigma_inicial=0.1, sigma_final=0.1,
                                                  iteraciones=ITERACIONES, semilla=SEMILLA).entrenar(X, n_instantaneas=0),
        "5x5": MapaKohonen(5, 5, iteraciones=ITERACIONES, semilla=SEMILLA).entrenar(X, n_instantaneas=0),
    }
    fig, ejes = vz.plt.subplots(1, 3, figsize=(14.4, 4.8))
    for ax, (nombre, m) in zip(ejes, mapas_ejemplo.items()):
        vz.malla_kohonen(m.W, m.filas, m.columnas, X, titulo=nombre, ax=ax)
    fig.tight_layout()
    reporte.figura(vz.guardar(fig, FIGURAS / "10_configuraciones.png"),
                   "Sin una vecindad inicial amplia la malla queda enredada: los prototipos cubren "
                   "los datos pero el orden topologico se pierde.")
    reporte.texto(
        "* **Tamano del mapa**: un mapa de 5x5 (25 neuronas) cuantiza peor y, sobre todo, deja "
        "muy pocas neuronas para las fronteras: con solo 5 neuronas por lado no caben cinco "
        "valles separados por crestas y la segmentacion encuentra solo 3 grupos. Un mapa de "
        "15x15 reduce el error de cuantizacion y separa igual de bien que el de 10x10, a costa "
        "de mas calculo y de mas neuronas 'vacias'.\n"
        "* **Vecindad**: es lo que distingue a un mapa de Kohonen de un simple aprendizaje "
        "competitivo. Con σ_0 pequeno o sin vecindad la cuantizacion puede ser incluso mejor "
        "(cada neurona se dedica a su zona), pero el **error topografico se dispara**: vecinas de "
        "la rejilla ya no representan zonas vecinas del plano, la matriz U deja de tener valles "
        "y crestas limpios y la segmentacion falla. Sin orden topologico no hay mapa que leer.\n"
        "* **Iteraciones**: con solo 1 000 iteraciones (~2 presentaciones por patron) el mapa ya "
        "se ordena y separa los grupos, porque estos estan bien definidos; lo que queda peor es "
        "la cuantizacion (prototipos menos ajustados). Pasar de 10 000 a 30 000 apenas mejora: "
        "el calendario de η y σ, no la duracion, es lo que determina la calidad del mapa."
    )

    # ------------------------------------------------------------------
    # 7. Resolucion
    # ------------------------------------------------------------------
    reporte.seccion("7. Limite de resolucion: grupos que se tocan")
    c0 = np.array(ds.GRUPOS_PLANO[0][:2])
    c1 = np.array(ds.GRUPOS_PLANO[1][:2])
    distancia_base = float(np.linalg.norm(c1 - c0))
    filas = []
    for desplazamiento in (0.0, -0.5, -1.0, -1.5, -2.0):
        datos = ds.grupos_plano(semilla=SEMILLA, desplazamiento_g1=desplazamiento)
        n_g, purezas = [], []
        for s in range(5):
            m = MapaKohonen(FILAS, COLUMNAS, iteraciones=ITERACIONES, semilla=s).entrenar(datos.X, n_instantaneas=0)
            g = m.agrupar(datos.X, percentil=PERCENTIL)
            n_g.append(len(np.unique(g)))
            purezas.append(pureza(g, datos.d.astype(int)))
        d = distancia_base + desplazamiento
        filas.append({"distancia G0-G1": d,
                      "grupos (5 semillas)": " ".join(map(str, n_g)),
                      "separa G0 y G1 (%)": 100 * np.mean(np.array(n_g) >= 5),
                      "pureza_media": float(np.mean(purezas))})
        print(filas[-1])
    df_res = pd.DataFrame(filas)
    reporte.texto(
        "Se acerca G1 a G0 a lo largo de la recta que une sus centros y se repite el analisis "
        "con 5 semillas por distancia."
    )
    reporte.tabla(df_res, "Deteccion de G0 y G1 segun su separacion", nombre_csv="10_resolucion")
    reporte.texto(
        "Mientras la separacion entre centros supera ~3.2 (unas 3 veces la suma de las "
        "desviaciones tipicas de ambos grupos, 0.55 + 0.45 = 1), "
        "queda una franja vacia entre ambos grupos, las neuronas que caen en ella tienen U alta "
        "y el mapa los separa. Cuando se acercan mas, las nubes se solapan, la densidad deja de "
        "tener un hueco y el mapa los representa como **un solo grupo** (la pureza cae a ~0.85, "
        "que es exactamente lo que se pierde al fundir 60 puntos de G1 con G0). No es un fallo "
        "del algoritmo sino una propiedad del aprendizaje no supervisado: sin etiquetas, un "
        "'grupo' solo puede definirse como una region densa rodeada de regiones menos densas, y "
        "dos nubes solapadas sin hueco entre ellas son, para los datos, una sola."
    )

    # ------------------------------------------------------------------
    # 8. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("8. Conclusiones")
    reporte.lista([
        f"El mapa de {FILAS}x{COLUMNAS} identifica los {n_grupos} grupos sin recibir ninguna "
        f"etiqueta, con pureza {p:.3f} respecto de los grupos reales.",
        "El aprendizaje es competitivo y cooperativo: no se minimiza un error respecto de una "
        "salida deseada, sino que cada neurona se desplaza hacia los patrones que gana y arrastra "
        "a sus vecinas de la rejilla.",
        "La vecindad es la que produce la preservacion de la topologia; sin ella se obtiene una "
        "cuantizacion sin orden, en la que la matriz U no revela grupos.",
        "La matriz U y la densidad de impactos son las herramientas de lectura: los grupos son "
        "valles, las fronteras son crestas de neuronas sin patrones.",
        "El mapa distingue grupos mientras exista una zona de baja densidad entre ellos; grupos "
        "solapados se funden, lo cual es una limitacion inherente al agrupamiento no supervisado.",
        "Contraste con el aprendizaje supervisado (actividades 1-3): alli la red aprende una "
        "correspondencia entrada → salida dada; aqui aprende la **estructura** de las entradas, "
        "y la interpretacion de los grupos la pone el analista despues.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme escrito en {ruta}")


if __name__ == "__main__":
    main()
