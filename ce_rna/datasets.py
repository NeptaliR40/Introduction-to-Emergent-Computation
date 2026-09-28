"""
Conjuntos de patrones de entrenamiento usados en los experimentos.

Todos los conjuntos provienen de los ejemplos de las clases y de la guia de evaluacion:

* Compuertas logicas AND / OR (y sus parientes XOR, NAND, NOR) -> `claseRN02.md`
  y `ICE-claseRN03.md`.
* Reconocimiento de las letras X y O con un perceptron de dos neuronas de
  salida -> `ICE-claseRN03.md`, figura "Funcionamiento de un Perceptron".
* Descodificador de binario a decimal de 3 bits -> `ICE-claseRN04.md`, ejemplo
  propuesto para el ADALINE.
* Funcion no lineal a aproximar con el perceptron multicapa -> guia, actividad 3.
* Cinco grupos de puntos en el plano para el mapa de Kohonen -> guia, actividad 4.
* Letras A, B, C, D de 7x6 pixeles para la red de Hopfield -> guia, actividad 5.

Convenciones
------------
`X` es siempre una matriz (N, m0) con un patron por fila y SIN la columna de
sesgo: cada modelo decide si antepone la entrada fija x0 = 1 (la "neurona de
inclinacion" de `ICE-claseRN03.md`).

`d` es un vector (N,) para problemas de una sola salida o una matriz (N, m1)
cuando hay varias neuronas de salida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Conjunto:
    """Contenedor simple de un conjunto de entrenamiento."""

    X: np.ndarray
    d: np.ndarray
    nombre: str
    nombres_entradas: list[str] = field(default_factory=list)
    nombres_salidas: list[str] = field(default_factory=list)
    descripcion: str = ""

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.d = np.asarray(self.d, dtype=float)
        if self.X.shape[0] != self.d.shape[0]:
            raise ValueError("X y d deben tener el mismo numero de filas")
        if not self.nombres_entradas:
            self.nombres_entradas = [f"x{i + 1}" for i in range(self.X.shape[1])]
        if not self.nombres_salidas:
            n_sal = 1 if self.d.ndim == 1 else self.d.shape[1]
            self.nombres_salidas = ["y"] if n_sal == 1 else [f"y{j + 1}" for j in range(n_sal)]

    @property
    def n_patrones(self) -> int:
        return self.X.shape[0]

    @property
    def n_entradas(self) -> int:
        return self.X.shape[1]

    @property
    def n_salidas(self) -> int:
        return 1 if self.d.ndim == 1 else self.d.shape[1]

    def tabla(self):
        """Devuelve el conjunto como `pandas.DataFrame` (tabla de verdad)."""
        import pandas as pd

        datos = {nombre: self.X[:, i] for i, nombre in enumerate(self.nombres_entradas)}
        if self.d.ndim == 1:
            datos[self.nombres_salidas[0]] = self.d
        else:
            for j, nombre in enumerate(self.nombres_salidas):
                datos[nombre] = self.d[:, j]
        return pd.DataFrame(datos)


# ---------------------------------------------------------------------------
# Compuertas logicas
# ---------------------------------------------------------------------------

#: Tabla de verdad de las compuertas en codificacion binaria {0, 1}.
#: El orden de los patrones es (0,0), (0,1), (1,0), (1,1).
_ENTRADAS_BINARIAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

_SALIDAS = {
    "AND": np.array([0, 0, 0, 1], dtype=float),
    "OR": np.array([0, 1, 1, 1], dtype=float),
    "XOR": np.array([0, 1, 1, 0], dtype=float),
    "NAND": np.array([1, 1, 1, 0], dtype=float),
    "NOR": np.array([1, 0, 0, 0], dtype=float),
}


def compuerta(nombre: str, codificacion: str = "binaria") -> Conjunto:
    """Conjunto de entrenamiento de una compuerta logica de dos entradas.

    Parameters
    ----------
    nombre:
        "AND", "OR", "XOR", "NAND" o "NOR".
    codificacion:
        ``"binaria"`` usa valores {0, 1} (la convencion de McCulloch-Pitts en
        `claseRN02.md`); ``"bipolar"`` usa {-1, +1}, que es la convencion con la
        que trabaja el perceptron bipolar de `ICE-claseRN03.md` y la regla de
        Hebb ("supongamos que xi e yi son bipolares o antisimetricas").

    Notas
    -----
    La codificacion importa mas de lo que parece.  Con entradas binarias, un
    patron (0, 0) no puede modificar ningun peso salvo el del sesgo, porque el
    incremento es proporcional a la entrada.  Con codificacion bipolar todas las
    entradas son +-1 y por tanto *todos* los pesos reciben correccion en cada
    patron: la regla de Hebb solo resuelve el AND en codificacion bipolar.
    """
    nombre = nombre.upper()
    if nombre not in _SALIDAS:
        raise KeyError(f"Compuerta desconocida '{nombre}'. Disponibles: {sorted(_SALIDAS)}")

    X = _ENTRADAS_BINARIAS.copy()
    d = _SALIDAS[nombre].copy()

    if codificacion == "bipolar":
        X = 2.0 * X - 1.0
        d = 2.0 * d - 1.0
    elif codificacion != "binaria":
        raise ValueError("codificacion debe ser 'binaria' o 'bipolar'")

    return Conjunto(
        X=X,
        d=d,
        nombre=f"{nombre} ({codificacion})",
        nombres_entradas=["x1", "x2"],
        nombres_salidas=["y"],
        descripcion=f"Tabla de verdad de la compuerta {nombre} en codificacion {codificacion}.",
    )


# ---------------------------------------------------------------------------
# Reconocimiento de las letras X y O  (ICE-claseRN03.md)
# ---------------------------------------------------------------------------

#: Retinas de 5x5 pixeles.  '#' es pixel encendido (+1), '.' apagado (-1).
_PLANTILLAS_LETRAS = {
    "X": [
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
    ],
    "O": [
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
}


def _plantilla_a_vector(filas: list[str]) -> np.ndarray:
    """Convierte una retina dibujada con '#' y '.' en un vector bipolar."""
    return np.array([1.0 if c == "#" else -1.0 for fila in filas for c in fila])


def letra(nombre: str) -> np.ndarray:
    """Devuelve la retina bipolar (25,) de la letra 'X' u 'O'."""
    return _plantilla_a_vector(_PLANTILLAS_LETRAS[nombre.upper()])


def letras_xo(con_ruido: int = 0, prob_ruido: float = 0.1, semilla: int = 0) -> Conjunto:
    """Conjunto de entrenamiento del clasificador de letras X / O.

    Reproduce la figura "Funcionamiento de un Perceptron" de
    `ICE-claseRN03.md`: la red tiene m0 = 25 entradas (una retina de 5x5) y
    m1 = 2 neuronas de salida.  La primera neurona responde +1 ante una X y -1
    en caso contrario; la segunda hace lo propio con la O.

    Parameters
    ----------
    con_ruido:
        Numero de copias ruidosas *adicionales* que se generan por letra.  Con 0
        el conjunto son los dos prototipos limpios.
    prob_ruido:
        Probabilidad de invertir cada pixel en las copias ruidosas.
    semilla:
        Semilla del generador para que el experimento sea reproducible.
    """
    rng = np.random.default_rng(semilla)
    patrones: list[np.ndarray] = []
    objetivos: list[list[float]] = []

    for indice, nombre in enumerate(("X", "O")):
        base = letra(nombre)
        objetivo = [-1.0, -1.0]
        objetivo[indice] = 1.0
        patrones.append(base)
        objetivos.append(list(objetivo))
        for _ in range(con_ruido):
            mascara = rng.random(base.size) < prob_ruido
            copia = base.copy()
            copia[mascara] *= -1.0
            patrones.append(copia)
            objetivos.append(list(objetivo))

    return Conjunto(
        X=np.array(patrones),
        d=np.array(objetivos),
        nombre="Letras X/O (retina 5x5)",
        nombres_entradas=[f"p{i}" for i in range(25)],
        nombres_salidas=["neurona_X", "neurona_O"],
        descripcion=(
            "Retinas bipolares de 5x5 pixeles. Dos neuronas de salida, una por clase, "
            "tal como en la figura 'Funcionamiento de un Perceptron' de ICE-claseRN03."
        ),
    )


def contaminar(patron: np.ndarray, n_pixeles: int, semilla: int = 0) -> np.ndarray:
    """Invierte exactamente `n_pixeles` posiciones elegidas al azar.

    Sirve para comprobar la tolerancia a "estimulos contaminados" que
    `claseRN01.md` atribuye a las redes neuronales: "es capaz de recuperar
    informacion a partir de estimulos incompletos, ruidosos o parcialmente
    erroneos".
    """
    rng = np.random.default_rng(semilla)
    copia = np.array(patron, dtype=float).copy()
    indices = rng.choice(copia.size, size=n_pixeles, replace=False)
    copia[indices] *= -1.0
    return copia


# ---------------------------------------------------------------------------
# Descodificador binario -> decimal  (ICE-claseRN04.md)
# ---------------------------------------------------------------------------

def decodificador_binario(n_bits: int = 3, bit_mas_significativo_primero: bool = True) -> Conjunto:
    """Los 2^n patrones binarios de `n_bits` con su valor decimal como objetivo.

    `ICE-claseRN04.md` propone este problema como ejemplo del ADALINE: "el
    descodificador recibe una entrada en binario y produce como salida su valor
    en decimal".  La solucion analitica existe y es lineal --- el valor decimal
    es una suma ponderada de los bits con pesos 4, 2, 1 para tres bits --- de
    modo que el ADALINE deberia poder aprenderla *exactamente* (error cuadratico
    medio -> 0), lo que convierte al problema en un banco de pruebas ideal para
    verificar la regla Delta.

    Nota sobre el enunciado: la formula que aparece en la clase, d = sum 2^i xi
    con i de 1 a n, produce pesos 2, 4, 8; aqui se usa la convencion estandar
    d = sum 2^(i-1) xi (pesos 4, 2, 1 para tres bits), que es la que hace que
    el rango de salida sea 0..2^n - 1.  El ADALINE aprende los pesos que
    correspondan a los datos, asi que el cambio de convencion solo altera la
    escala de los pesos optimos, no la naturaleza del problema.
    """
    if n_bits < 1:
        raise ValueError("n_bits debe ser >= 1")

    filas = []
    for valor in range(2 ** n_bits):
        bits = [(valor >> k) & 1 for k in range(n_bits)]
        if bit_mas_significativo_primero:
            bits = bits[::-1]
        filas.append(bits)

    X = np.array(filas, dtype=float)
    d = np.arange(2 ** n_bits, dtype=float)

    if bit_mas_significativo_primero:
        nombres = [f"b{n_bits - 1 - i}" for i in range(n_bits)]
    else:
        nombres = [f"b{i}" for i in range(n_bits)]

    return Conjunto(
        X=X,
        d=d,
        nombre=f"Descodificador binario->decimal ({n_bits} bits)",
        nombres_entradas=nombres,
        nombres_salidas=["decimal"],
        descripcion=(
            "Los 2^n patrones binarios de n bits etiquetados con su valor decimal. "
            "Problema linealmente resoluble: pesos optimos 2^(n-1), ..., 2, 1 y sesgo 0."
        ),
    )


def pesos_optimos_decodificador(n_bits: int = 3, bit_mas_significativo_primero: bool = True):
    """Solucion analitica del descodificador: (w0, w1, ..., wn) con w0 el sesgo."""
    potencias = np.array([2.0 ** k for k in range(n_bits)])
    if bit_mas_significativo_primero:
        potencias = potencias[::-1]
    return np.concatenate(([0.0], potencias))


# ---------------------------------------------------------------------------
# Nubes de puntos sinteticas
# ---------------------------------------------------------------------------

def nubes_separables(
    n_por_clase: int = 40,
    separacion: float = 3.0,
    dispersion: float = 0.8,
    semilla: int = 0,
    bipolar: bool = True,
) -> Conjunto:
    """Dos nubes gaussianas linealmente separables en el plano.

    Se usa para ilustrar la figura "Algoritmo Perceptronico: Infinitas
    Soluciones" de `ICE-claseRN03.md`: cuando un problema es linealmente
    separable no hay una solucion, sino infinitas.
    """
    rng = np.random.default_rng(semilla)
    centro = np.array([separacion / 2.0, separacion / 2.0])
    clase_pos = rng.normal(loc=centro, scale=dispersion, size=(n_por_clase, 2))
    clase_neg = rng.normal(loc=-centro, scale=dispersion, size=(n_por_clase, 2))

    X = np.vstack([clase_pos, clase_neg])
    etiqueta_neg = -1.0 if bipolar else 0.0
    d = np.concatenate([np.ones(n_por_clase), np.full(n_por_clase, etiqueta_neg)])

    return Conjunto(
        X=X,
        d=d,
        nombre="Nubes gaussianas separables",
        nombres_entradas=["x1", "x2"],
        nombres_salidas=["clase"],
        descripcion="Dos nubes gaussianas separables por una recta; infinitas soluciones.",
    )


# ---------------------------------------------------------------------------
# Perceptron multicapa: aproximacion de funciones  (Actividad 3)
# ---------------------------------------------------------------------------

def funcion_objetivo(x):
    """f(x) = sin(x) + 0.5 sin(3x) en [-pi, pi].

    Se elige porque no es lineal, no es monotona (tiene cuatro extremos en el
    intervalo) y mezcla dos frecuencias: una sola neurona sigmoidal no puede
    reproducirla, y el numero de neuronas ocultas necesario se puede estudiar.
    """
    x = np.asarray(x, dtype=float)
    return np.sin(x) + 0.5 * np.sin(3.0 * x)


def aproximacion_funcion(n_entrenamiento: int = 30, n_prueba: int = 200,
                         ruido: float = 0.0, semilla: int = 0) -> tuple[Conjunto, Conjunto]:
    """Conjuntos de entrenamiento y prueba para aproximar `funcion_objetivo`.

    El entrenamiento usa `n_entrenamiento` puntos equiespaciados (con ruido
    gaussiano opcional en la salida); la prueba usa `n_prueba` puntos distintos
    del mismo intervalo, **sin ruido**, para medir la generalizacion, es decir,
    lo que la red hace entre los puntos que vio.
    """
    rng = np.random.default_rng(semilla)
    x_ent = np.linspace(-np.pi, np.pi, n_entrenamiento)
    y_ent = funcion_objetivo(x_ent) + ruido * rng.normal(size=x_ent.size)
    # puntos de prueba desplazados para que no coincidan con los de entrenamiento
    x_pru = np.sort(rng.uniform(-np.pi, np.pi, n_prueba))
    y_pru = funcion_objetivo(x_pru)
    comun = dict(nombres_entradas=["x"], nombres_salidas=["f(x)"])
    return (
        Conjunto(X=x_ent[:, None], d=y_ent, nombre="f(x) - entrenamiento",
                 descripcion="Puntos equiespaciados en [-pi, pi]", **comun),
        Conjunto(X=x_pru[:, None], d=y_pru, nombre="f(x) - prueba",
                 descripcion="Puntos aleatorios en [-pi, pi], sin ruido", **comun),
    )


# ---------------------------------------------------------------------------
# Mapas de Kohonen: distribucion de puntos con grupos  (Actividad 4)
# ---------------------------------------------------------------------------

#: Centros y dispersiones de los cinco grupos del plano.  Tienen tamanos,
#: densidades y formas distintas a proposito: dos grupos proximos entre si
#: (G0 y G1), uno alargado (G3) y uno mas disperso (G4).
GRUPOS_PLANO = [
    # (centro x, centro y, desviacion x, desviacion y, n puntos)
    (-4.0, 3.5, 0.55, 0.55, 80),
    (-0.5, 4.8, 0.45, 0.45, 60),
    (3.5, 3.0, 0.70, 0.70, 90),
    (-3.0, -3.0, 1.30, 0.40, 90),
    (3.0, -3.0, 0.90, 0.90, 80),
]


def grupos_plano(semilla: int = 0, desplazamiento_g1: float = 0.0) -> Conjunto:
    """Puntos del plano agrupados en cinco nubes gaussianas.

    `d` contiene la etiqueta real del grupo (0..4), pero **solo** se usa para
    evaluar el agrupamiento a posteriori: el mapa de Kohonen nunca la ve.

    `desplazamiento_g1` acerca (valores negativos) o aleja el grupo G1 de G0
    a lo largo de la recta que une sus centros; sirve para medir a partir de
    que separacion el mapa deja de distinguirlos.
    """
    rng = np.random.default_rng(semilla)
    puntos, etiquetas = [], []
    c0 = np.array(GRUPOS_PLANO[0][:2])
    for k, (cx, cy, sx, sy, n) in enumerate(GRUPOS_PLANO):
        if k == 1 and desplazamiento_g1:
            u = (np.array([cx, cy]) - c0) / np.linalg.norm(np.array([cx, cy]) - c0)
            cx, cy = np.array([cx, cy]) + desplazamiento_g1 * u
        puntos.append(rng.normal(loc=(cx, cy), scale=(sx, sy), size=(n, 2)))
        etiquetas.append(np.full(n, k))
    return Conjunto(
        X=np.vstack(puntos),
        d=np.concatenate(etiquetas).astype(float),
        nombre="Cinco grupos en el plano",
        nombres_entradas=["x1", "x2"],
        nombres_salidas=["grupo_real"],
        descripcion="Cinco nubes gaussianas de distinto tamano, densidad y forma.",
    )


# ---------------------------------------------------------------------------
# Red de Hopfield: letras A, B, C, D en 7x6 pixeles  (Actividad 5)
# ---------------------------------------------------------------------------

#: Imagenes de 7 filas x 6 columnas = 42 pixeles.  '#' = negro (+1), '.' = blanco (-1).
#:
#: Se definen dos disenos de las mismas cuatro letras:
#:
#: * ``"fina"``: trazo de un pixel, el dibujo "natural".  B, C y D comparten la
#:   columna izquierda y las filas superior e inferior, y en bipolar tambien
#:   comparten la mayor parte del fondo blanco: el solapamiento B-C llega a 0.62.
#: * ``"gruesa"``: redisenado para **reducir el solapamiento** entre letras
#:   (maximo 0.33) sin perder legibilidad: B con doble trazo, C desplazada a la
#:   derecha, D con el trazo curvo engrosado.  Es el diseno que usa la red.
LETRAS_7X6 = {
    "gruesa": {
        "A": ["..##..", ".#..#.", "#....#", "######", "#....#", "#....#", "#....#"],
        "B": ["#####.", "##..##", "##..##", "#####.", "##..##", "##..##", "#####."],
        "C": ["..####", ".##...", "##....", "##....", "##....", ".##...", "..####"],
        "D": ["####..", "#..##.", "#...##", "#...##", "#...##", "#..##.", "####.."],
    },
    "fina": {
        "A": ["..##..", ".#..#.", "#....#", "#....#", "######", "#....#", "#....#"],
        "B": ["#####.", "#....#", "#....#", "#####.", "#....#", "#....#", "#####."],
        "C": [".####.", "#....#", "#.....", "#.....", "#.....", "#....#", ".####."],
        "D": ["####..", "#...#.", "#....#", "#....#", "#....#", "#...#.", "####.."],
    },
}

FORMA_7X6 = (7, 6)


def letra_7x6(nombre: str, diseno: str = "gruesa") -> np.ndarray:
    """Vector bipolar (42,) de la letra A, B, C o D, leido fila a fila."""
    return _plantilla_a_vector(LETRAS_7X6[diseno][nombre.upper()])


def letras_abcd(diseno: str = "gruesa") -> Conjunto:
    """Las cuatro letras como patrones de 42 elementos (uno por fila)."""
    nombres = list(LETRAS_7X6[diseno])
    return Conjunto(
        X=np.array([letra_7x6(n, diseno) for n in nombres]),
        d=np.arange(len(nombres), dtype=float),
        nombre=f"Letras A, B, C, D (7x6, trazo {diseno})",
        nombres_entradas=[f"p{i}" for i in range(42)],
        nombres_salidas=["indice_letra"],
        descripcion="Imagenes binarias de 7x6 pixeles codificadas en bipolar.",
    )
