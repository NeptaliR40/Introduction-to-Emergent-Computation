# 05 --- Guia de uso, estructura y referencia rapida

---

## 1. Instalacion

Solo hacen falta tres paquetes, y ninguno de aprendizaje automatico:

```bash
python3 -m pip install -r requirements.txt     # numpy, pandas, matplotlib
```

No hace falta instalar el paquete `ce_rna`: los scripts de `experimentos/` anaden la
raiz del repositorio a `sys.path` mediante [`experimentos/_ruta.py`](../experimentos/_ruta.py).
Para usar `ce_rna` desde otro directorio, basta con anadir la raiz al `PYTHONPATH`.

Version de Python: **3.10 o superior** (se usan anotaciones `X | Y` y `list[T]`).

---

## 2. Como ejecutar todo

```bash
# Regenera las figuras, las tablas CSV y los doce informes (~5 min)
python experimentos/ejecutar_todo.py

# Bateria de 40 pruebas (no requiere pytest)
python pruebas/pruebas.py

# Un experimento suelto
python experimentos/exp05_adaline_decodificador.py
```

Todo es **determinista**: los generadores aleatorios llevan semilla fija, de modo que
dos ejecuciones producen exactamente los mismos numeros y las mismas figuras.

---

## 3. Estructura del repositorio

```
.
├── ce_rna/                    Paquete: los modelos, desde cero
│   ├── activaciones.py        phi(v) y sus derivadas
│   ├── datasets.py            Conjuntos de entrenamiento de las clases
│   ├── mcculloch_pitts.py     Neurona binaria de umbral fijo (1943)
│   ├── perceptron.py          Perceptron simple unicapa (1958)
│   ├── adaline.py             Neurona lineal adaptativa, regla Delta (1960)
│   ├── hebb.py                Regla de Hebb supervisada (1949)
│   ├── perceptron_multicapa.py  MLP con retropropagacion (1986)
│   ├── kohonen.py             Mapa autoorganizado, no supervisado (1982)
│   ├── hopfield.py            Memoria asociativa recurrente (1982)
│   ├── metricas.py            Exactitud, confusion, ECM, RMSE, R2, margen
│   ├── visual.py              Diagramas de red, fronteras, curvas, superficies
│   └── reportes.py            Generador de informes en Markdown
├── experimentos/              Scripts ejecutables, uno por estudio
├── pruebas/pruebas.py         40 pruebas sin dependencias externas
├── docs/                      Esta documentacion
├── clases/                    Material original de la asignatura
└── resultados/
    ├── figuras/               PNG generados
    ├── tablas/                CSV generados
    └── reportes/              Informes en Markdown generados
```

### 3.1 Principio de diseno: los informes no se escriben a mano

Cada experimento construye un objeto `Reporte`, le anade secciones, tablas y figuras,
y lo escribe al final. Los numeros de los informes salen **de la misma ejecucion** que
produce las figuras, de modo que no pueden quedar desfasados respecto del codigo. Si
un resultado cambia, el informe cambia con el.

```python
from ce_rna.reportes import Reporte

reporte = Reporte(titulo="...", nombre_archivo="08_mi_experimento", resumen="...")
reporte.seccion("1. Resultados").tabla(df, "Mi tabla", nombre_csv="08_mi_tabla")
reporte.figura(ruta_png, "Pie de figura")
reporte.escribir()
```

---

## 4. Tabla de correspondencias

| Clase | Contenido | Modulo | Experimento |
|---|---|---|---|
| `claseRN01.md` | Modelo de neurona, activaciones, paradigmas | `activaciones` | 00 |
| `claseRN02.md` | McCulloch-Pitts, AND y OR | `mcculloch_pitts` | 01 |
| `ICE-claseRN03.md` | Separabilidad lineal, perceptron, Hebb | `perceptron`, `hebb` | 02, 03, 04, 07 |
| `ICE-claseRN04.md` | ADALINE y regla Delta | `adaline` | 05, 06 |
| Clases 5-7 (guia, act. 3) | MLP, backpropagation, criterios de parada | `perceptron_multicapa` | 08, 09 |
| Guia, act. 4 | Aprendizaje no supervisado, Kohonen | `kohonen` | 10 |
| Guia, act. 5 | Memoria asociativa, Hopfield | `hopfield` | 11 |

| Experimento | Pregunta que responde |
|---|---|
| 00 | ¿Que activaciones hay y cuales son derivables? |
| 01 | ¿Puede una neurona de umbral calcular AND, OR y XOR? |
| 02 | ¿Puede una red aprender AND y OR sin que le den los pesos? |
| 03 | ¿Por que el perceptron no resuelve el XOR? |
| 04 | ¿Funciona sobre un problema de reconocimiento real, con ruido? |
| 05 | ¿Puede una red aproximar una funcion de salida real? |
| 06 | ¿Que gana y que pierde el ADALINE frente al perceptron? |
| 07 | ¿Hasta donde llega el aprendizaje de una sola pasada? |
| 08 | ¿Puede un MLP aprender el XOR, y que representa su capa oculta? |
| 09 | ¿Como aproxima un MLP una funcion, y cuando hay que parar? |
| 10 | ¿Descubre un mapa de Kohonen los grupos sin etiquetas? |
| 11 | ¿Recupera una red de Hopfield letras contaminadas? |

---

## 5. Referencia rapida de la API

### Conjuntos de datos

```python
from ce_rna import datasets as ds

ds.compuerta("AND", "bipolar")          # AND, OR, XOR, NAND, NOR / binaria, bipolar
ds.letras_xo(con_ruido=5, prob_ruido=0.1, semilla=0)
ds.letra("X")                           # vector bipolar (25,)
ds.contaminar(patron, n_pixeles=6, semilla=0)
ds.decodificador_binario(n_bits=3)
ds.pesos_optimos_decodificador(3)       # [0, 4, 2, 1]
ds.nubes_separables(n_por_clase=40, separacion=3.0, dispersion=0.8)
ds.aproximacion_funcion(n_entrenamiento=30, n_prueba=200, ruido=0.0)  # (entrenamiento, prueba)
ds.funcion_objetivo(x)                  # sin(x) + 0.5 sin(3x)
ds.grupos_plano(semilla=0, desplazamiento_g1=0.0)   # 5 grupos en el plano
ds.letras_abcd("gruesa")                # A, B, C, D de 7x6 (o "fina")
ds.letra_7x6("B")                       # vector bipolar (42,)
```

Todo conjunto es un `Conjunto` con `.X`, `.d`, `.tabla()`, `.n_patrones`,
`.n_entradas`, `.n_salidas`.

### Modelos

```python
from ce_rna.mcculloch_pitts import neurona_and, neurona_or, RedMCP, red_xor
from ce_rna.perceptron import PerceptronSimple
from ce_rna.adaline import Adaline, solucion_minimos_cuadrados
from ce_rna.hebb import RedHebb
from ce_rna.perceptron_multicapa import PerceptronMulticapa, gradiente_numerico
from ce_rna.kohonen import MapaKohonen, pureza
from ce_rna.hopfield import RedHopfield
```

`PerceptronMulticapa(capas, razon_aprendizaje, momento, activacion_salida, modo,
max_epocas, error_objetivo, paciencia)` → `.entrenar(X, d, X_prueba, d_prueba)`,
`.salida(X)`, `.predecir(X)`, `.propagar(X)`, `.gradiente(X, d)`, `.historial_df()`.

`MapaKohonen(filas, columnas, eta_inicial, sigma_inicial, iteraciones)` →
`.entrenar(X)`, `.ganadoras(X)`, `.impactos(X)`, `.matriz_u()`, `.agrupar(X)`,
`.error_cuantizacion(X)`, `.error_topografico(X)`.

`RedHopfield(n, regla="hebb" | "pseudoinversa")` → `.almacenar(P)`, `.recuperar(s0)`,
`.energia(s)`, `.es_punto_fijo(s)`, `.identificar(s)`, `.solapamientos(s)`.

Todos comparten la misma interfaz basica: `.entrenar(X, d)` (salvo McCulloch-Pitts,
que no aprende), `.predecir(X)`, `.exactitud(X, d)`, `.resumen()`.

### Metricas

```python
from ce_rna import metricas as mt

mt.exactitud(y_pred, y_real)
mt.matriz_confusion(y_pred, y_real, etiquetas=[-1, 1])
mt.ecm(y_pred, y_real)          # con el factor 1/2 de la clase
mt.rmse(y_pred, y_real)
mt.error_absoluto_medio(y_pred, y_real)
mt.r2(y_pred, y_real)
mt.margen_geometrico(w, X, d)   # distancia minima al hiperplano
```

### Figuras

```python
from ce_rna import visual as vz

vz.dibujar_red(capas, titulo, pesos=..., capas_lineales={0})
vz.dibujar_red_mcp(red)
vz.frontera_decision(w, X, d, titulo, predictor=modelo.predecir)
vz.evolucion_fronteras(trayectoria, X, d, titulo)
vz.haz_de_soluciones(soluciones, X, d, titulo)
vz.curva_aprendizaje(y, titulo, escala_log=True)
vz.curvas_comparadas({"a": y1, "b": y2}, titulo)
vz.mapa_retina(vector, titulo, divergente=True)
vz.rejilla_retinas(patrones, titulos)
vz.contorno_error(Wi, Wj, E, trayectoria=..., optimo=...)
vz.superficie_error_3d(Wi, Wj, E)
vz.dibujar_activaciones()
vz.tabla_a_figura(df, titulo)
vz.region_continua(red.salida, X, d)   # salida continua y frontera y = 0.5 (MLP)
vz.grupos_en_plano(X, grupos)          # hasta 5 grupos, color + marcador
vz.malla_kohonen(W, filas, columnas, X)
vz.mapa_calor(M, divergente=True)
vz.guardar(fig, ruta)           # crea el directorio y cierra la figura
```

Criterios de diseno de las figuras: paleta de tres tonos validada para vision con
deficiencia de color, identidad de clase codificada **tambien** por forma de marcador
(nunca solo por color), y ejes y rejilla recesivos.

---

## 6. Como anadir un experimento nuevo

1. Crear `experimentos/exp12_lo_que_sea.py` empezando por `import _ruta`.
2. Construir un `Reporte`, ejecutar el estudio, anadir tablas y figuras.
3. Guardar las figuras en `_ruta.FIGURAS` con `vz.guardar()`.
4. Llamar a `reporte.escribir()` al final.
5. Anadir el modulo a la lista `EXPERIMENTOS` de
   [`ejecutar_todo.py`](../experimentos/ejecutar_todo.py).
6. Anadir las pruebas correspondientes a `pruebas/pruebas.py` con el prefijo
   `probar_`.

Plantilla minima:

```python
import _ruta  # noqa: F401
from _ruta import FIGURAS
from ce_rna import datasets as ds, visual as vz
from ce_rna.reportes import Reporte, encabezado_experimento


def main():
    encabezado_experimento("Experimento 08 - ...")
    reporte = Reporte(titulo="...", nombre_archivo="08_...", resumen="...")
    # ... estudio ...
    reporte.escribir()


if __name__ == "__main__":
    main()
```

---

## 7. Decisiones de implementacion que conviene conocer

| Decision | Por que |
|---|---|
| El sesgo se implementa como columna `x0 = 1` | El umbral se aprende con la misma regla que los demas pesos |
| `max_epocas` en todos los modelos iterativos | El perceptron no converge en problemas no separables |
| Semillas explicitas en todo generador aleatorio | Los informes deben ser reproducibles exactamente |
| `matplotlib.use("Agg")` | Los experimentos guardan archivos, no abren ventanas |
| Los informes se generan, no se escriben | Un informe desfasado es peor que no tener informe |
| Pruebas sin `pytest` | Que funcionen con una instalacion minima de Python |
| Comentarios y nombres en espanol | El material de partida y el destinatario lo estan |

---

## 8. Problemas frecuentes

**`ModuleNotFoundError: No module named 'ce_rna'`** --- ejecutar los scripts desde la
raiz del repositorio, o asegurarse de que `import _ruta` aparece antes de importar
`ce_rna`.

**Las figuras no se ven en el informe de GitHub** --- los informes referencian las
figuras con rutas relativas (`../figuras/...`); hay que abrirlos desde el repositorio,
no copiando el `.md` a otro sitio.

**El ADALINE devuelve `inf` o `nan`** --- la razon de aprendizaje supera la cota de
estabilidad. Consultar `modelo.cota_estabilidad(X)` y usar un valor menor.

**El perceptron no converge** --- comprobar si el problema es linealmente separable.
Si lo es, revisar la codificacion de los objetivos: deben ser `±1`, no `0/1`.
