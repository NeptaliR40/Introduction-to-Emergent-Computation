"""
ce_rna --- Redes Neuronales Artificiales desde cero
===================================================

Implementacion didactica, sin librerias de aprendizaje automatico, de los
modelos vistos en la asignatura *Introduccion a la Computacion Emergente*
(Prof. Esteban Alvarez).  Las unicas dependencias son `numpy` (algebra), `pandas`
(tablas) y `matplotlib` (figuras); todo el calculo neuronal --- propagacion,
reglas de aprendizaje, criterios de parada y metricas --- esta escrito a mano.

Modulos
-------
`activaciones`     Catalogo de funciones phi(v) y sus derivadas.
`datasets`         Conjuntos de entrenamiento de las clases (compuertas,
                   letras X/O, descodificador binario-decimal).
`mcculloch_pitts`  Neurona binaria de umbral fijo y redes compuestas (1943).
`perceptron`       Perceptron simple unicapa y regla perceptronica (1958).
`adaline`          Neurona lineal adaptativa y regla Delta (1960).
`hebb`             Regla de Hebb supervisada (1949).
`perceptron_multicapa`  MLP con retropropagacion del error (1986).
`kohonen`          Mapa autoorganizado de Kohonen, no supervisado (1982).
`hopfield`         Red de Hopfield, memoria asociativa (1982).
`metricas`         Exactitud, matriz de confusion, ECM, RMSE, R2, margen.
`visual`           Diagramas de red, fronteras de decision, curvas y superficies.
`reportes`         Generacion automatica de los informes en Markdown.

Correspondencia con las clases
------------------------------
| Clase                 | Contenido                          | Modulo            |
|-----------------------|------------------------------------|-------------------|
| `claseRN01.md`        | Modelo de neurona, activaciones    | `activaciones`    |
| `claseRN02.md`        | McCulloch-Pitts, AND / OR          | `mcculloch_pitts` |
| `ICE-claseRN03.md`    | Separabilidad lineal, perceptron   | `perceptron`, `hebb` |
| `ICE-claseRN04.md`    | ADALINE y regla Delta              | `adaline`         |
| Clases 5-7 (guia, act. 3) | MLP y backpropagation          | `perceptron_multicapa` |
| Guia, actividad 4     | Mapas autoorganizados              | `kohonen`         |
| Guia, actividad 5     | Memoria asociativa                 | `hopfield`        |
"""

from . import (activaciones, adaline, datasets, hebb, hopfield, kohonen, mcculloch_pitts,
               metricas, perceptron, perceptron_multicapa, reportes, visual)

__version__ = "1.1.0"
__all__ = [
    "activaciones",
    "adaline",
    "datasets",
    "hebb",
    "hopfield",
    "kohonen",
    "mcculloch_pitts",
    "metricas",
    "perceptron",
    "perceptron_multicapa",
    "reportes",
    "visual",
]
