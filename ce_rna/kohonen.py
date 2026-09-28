"""
Mapa autoorganizado de Kohonen (SOM, Self-Organizing Map, 1982).

Referencia: guia de evaluacion, Actividad 4.

Aprendizaje no supervisado
--------------------------
A diferencia del perceptron, el ADALINE o el MLP, aqui **no hay salida
deseada**: la red solo ve los patrones x y tiene que descubrir por si misma como
estan organizados.  No existe un error que retropropagar; el aprendizaje es
**competitivo**: las neuronas compiten por representar cada patron.

Arquitectura
------------
Una capa de entrada de dimension m (aqui m = 2: puntos del plano) totalmente
conectada a una rejilla bidimensional de F x C neuronas.  Cada neurona j tiene
un vector de pesos w_j en el MISMO espacio que los datos --- su *prototipo* --- y
una posicion fija r_j = (fila, columna) en la rejilla.

Algoritmo (en cada iteracion t, con un patron x elegido al azar)
----------------------------------------------------------------
1. Competicion: la neurona ganadora (BMU, *best matching unit*) es la de
   prototipo mas cercano,

       c = argmin_j || x - w_j ||

2. Cooperacion: la ganadora arrastra a sus vecinas **en la rejilla** con una
   intensidad que decae con la distancia topologica,

       h_cj(t) = exp( - || r_c - r_j ||^2 / (2 sigma(t)^2) )

3. Adaptacion: todos los prototipos se mueven hacia el patron,

       w_j <- w_j + eta(t) h_cj(t) (x - w_j)

La razon de aprendizaje eta(t) y el radio de vecindad sigma(t) decrecen
exponencialmente.  Al principio sigma es grande y el mapa se *ordena* (se
despliega sin pliegues); al final sigma es menor que 1 y cada neurona solo
*afina* su prototipo (fase de convergencia).

Lectura del mapa entrenado
--------------------------
* Densidad de impactos: cuantos patrones gana cada neurona.
* Matriz U (unified distance matrix): distancia media entre el prototipo de cada
  neurona y los de sus vecinas en la rejilla.  Valores bajos = interior de un
  grupo; valores altos = frontera entre grupos.  Es la forma estandar de *ver*
  los clusters sin conocer las clases.
* `segmentar`: agrupa neuronas vecinas con prototipos cercanos (componentes
  conexas del grafo de la rejilla), lo que da una particion de los datos
  obtenida de forma **totalmente no supervisada**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InstantaneaSOM:
    """Copia de los prototipos en una iteracion dada (para animar el despliegue)."""

    iteracion: int
    prototipos: np.ndarray
    eta: float
    sigma: float


class MapaKohonen:
    """Mapa autoorganizado rectangular entrenado patron a patron.

    Parameters
    ----------
    filas, columnas:
        Tamano de la rejilla de neuronas.
    dimension:
        Dimension de los patrones de entrada.
    eta_inicial, eta_final:
        Razon de aprendizaje al principio y al final del entrenamiento.
    sigma_inicial, sigma_final:
        Radio de la vecindad gaussiana (en unidades de la rejilla).  Por
        omision, sigma_inicial es la mitad del lado mayor del mapa.
    iteraciones:
        Numero total de presentaciones de patrones.
    inicializacion:
        ``"aleatoria"`` (patrones del conjunto elegidos al azar) o
        ``"rejilla"`` (malla regular sobre el rectangulo que contiene los datos).
    semilla:
        Semilla del generador.
    """

    def __init__(
        self,
        filas: int,
        columnas: int,
        dimension: int = 2,
        eta_inicial: float = 0.5,
        eta_final: float = 0.01,
        sigma_inicial: float | None = None,
        sigma_final: float = 0.5,
        iteraciones: int = 5000,
        inicializacion: str = "aleatoria",
        semilla: int = 0,
    ):
        if filas < 1 or columnas < 1:
            raise ValueError("el mapa necesita al menos una neurona")
        if inicializacion not in ("aleatoria", "rejilla"):
            raise ValueError("inicializacion debe ser 'aleatoria' o 'rejilla'")
        self.filas = int(filas)
        self.columnas = int(columnas)
        self.dimension = int(dimension)
        self.eta_inicial = float(eta_inicial)
        self.eta_final = float(eta_final)
        self.sigma_inicial = float(sigma_inicial if sigma_inicial is not None
                                   else max(filas, columnas) / 2.0)
        self.sigma_final = float(sigma_final)
        self.iteraciones = int(iteraciones)
        self.inicializacion = inicializacion
        self.rng = np.random.default_rng(semilla)

        # posicion (fila, columna) de cada neurona en la rejilla, en orden fila-mayor
        f, c = np.meshgrid(np.arange(self.filas), np.arange(self.columnas), indexing="ij")
        self.posiciones = np.column_stack([f.ravel(), c.ravel()]).astype(float)
        self.W = np.zeros((self.n_neuronas, self.dimension))
        self.instantaneas: list[InstantaneaSOM] = []
        self.historial_error: list[tuple[int, float]] = []

    @property
    def n_neuronas(self) -> int:
        return self.filas * self.columnas

    # -- calendario de parametros -------------------------------------------
    def _decaimiento(self, inicial: float, final: float, t: int) -> float:
        """Interpolacion exponencial: inicial en t = 0, final en t = T."""
        fraccion = t / max(self.iteraciones - 1, 1)
        return inicial * (final / inicial) ** fraccion

    def eta(self, t: int) -> float:
        return self._decaimiento(self.eta_inicial, self.eta_final, t)

    def sigma(self, t: int) -> float:
        return self._decaimiento(self.sigma_inicial, self.sigma_final, t)

    # -- competicion ---------------------------------------------------------
    def distancias(self, X):
        """Matriz (N, n_neuronas) de distancias euclideas patron-prototipo."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.sqrt(((X[:, None, :] - self.W[None, :, :]) ** 2).sum(axis=2))

    def ganadoras(self, X) -> np.ndarray:
        """Indice de la neurona ganadora (BMU) de cada patron."""
        return np.argmin(self.distancias(X), axis=1)

    # -- entrenamiento -------------------------------------------------------
    def _inicializar(self, X) -> None:
        if self.inicializacion == "aleatoria":
            idx = self.rng.choice(X.shape[0], size=self.n_neuronas, replace=True)
            self.W = X[idx].copy() + self.rng.normal(scale=1e-3, size=(self.n_neuronas, self.dimension))
        else:
            minimo, maximo = X.min(axis=0), X.max(axis=0)
            u = self.posiciones / np.maximum([self.filas - 1, self.columnas - 1], 1)
            self.W = minimo[:2] + u * (maximo[:2] - minimo[:2])

    def entrenar(self, X, n_instantaneas: int = 6, registrar_cada: int = 0) -> "MapaKohonen":
        """Entrena el mapa con `iteraciones` presentaciones aleatorias de patrones."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self._inicializar(X)
        momentos = set(np.linspace(0, self.iteraciones - 1, n_instantaneas).astype(int).tolist()) \
            if n_instantaneas else set()
        self.instantaneas = [InstantaneaSOM(0, self.W.copy(), self.eta(0), self.sigma(0))]
        self.historial_error = []

        for t in range(self.iteraciones):
            x = X[self.rng.integers(X.shape[0])]
            c = int(np.argmin(((self.W - x) ** 2).sum(axis=1)))           # 1. competicion
            d2 = ((self.posiciones - self.posiciones[c]) ** 2).sum(axis=1)
            s = self.sigma(t)
            h = np.exp(-d2 / (2.0 * s * s))                                 # 2. cooperacion
            self.W += self.eta(t) * h[:, None] * (x - self.W)               # 3. adaptacion

            if t in momentos and t > 0:
                self.instantaneas.append(InstantaneaSOM(t + 1, self.W.copy(), self.eta(t), s))
            if registrar_cada and (t % registrar_cada == 0 or t == self.iteraciones - 1):
                self.historial_error.append((t + 1, self.error_cuantizacion(X)))
        return self

    # -- medidas de calidad --------------------------------------------------
    def error_cuantizacion(self, X) -> float:
        """Distancia media de cada patron a su prototipo ganador."""
        return float(np.mean(np.min(self.distancias(X), axis=1)))

    def error_topografico(self, X) -> float:
        """Fraccion de patrones cuya 1a y 2a neurona mas cercana NO son vecinas.

        Mide si el mapa preserva la topologia: un valor cercano a 0 indica que
        patrones proximos en el plano caen en neuronas proximas en la rejilla.
        """
        orden = np.argsort(self.distancias(X), axis=1)[:, :2]
        salto = np.abs(self.posiciones[orden[:, 0]] - self.posiciones[orden[:, 1]]).max(axis=1)
        return float(np.mean(salto > 1))

    def impactos(self, X) -> np.ndarray:
        """Numero de patrones ganados por cada neurona, como matriz (filas, columnas)."""
        conteo = np.bincount(self.ganadoras(X), minlength=self.n_neuronas)
        return conteo.reshape(self.filas, self.columnas)

    def _vecinas(self, j: int) -> list[int]:
        f, c = divmod(j, self.columnas)
        res = []
        for df, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ff, cc = f + df, c + dc
            if 0 <= ff < self.filas and 0 <= cc < self.columnas:
                res.append(ff * self.columnas + cc)
        return res

    def matriz_u(self) -> np.ndarray:
        """Distancia media de cada prototipo a los de sus 4 vecinas de rejilla."""
        U = np.zeros(self.n_neuronas)
        for j in range(self.n_neuronas):
            U[j] = np.mean([np.linalg.norm(self.W[j] - self.W[k]) for k in self._vecinas(j)])
        return U.reshape(self.filas, self.columnas)

    def segmentar(self, X, percentil: float = 50.0, min_fraccion: float = 0.02) -> np.ndarray:
        """Particion no supervisada del mapa en grupos a partir de la matriz U.

        1. Las neuronas con U por debajo del `percentil` indicado son
           *interiores* (prototipos rodeados de prototipos cercanos): son los
           valles de la matriz U.
        2. Cada componente conexa de neuronas interiores (vecindad de 4 en la
           rejilla) es un grupo candidato.  Los candidatos que ganan menos de
           `min_fraccion` de los patrones se descartan (ruido de la frontera).
        3. Las neuronas restantes --- las crestas de la matriz U --- se asignan
           al grupo de la neurona interior con prototipo mas cercano.

        Devuelve la etiqueta de grupo (0, 1, ...) de cada neurona.  No usa las
        clases reales en ningun paso.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U = self.matriz_u().ravel()
        impactos = self.impactos(X).ravel()
        interior = U < np.percentile(U, percentil)

        etiquetas = np.full(self.n_neuronas, -1)
        grupo = 0
        for inicio in range(self.n_neuronas):
            if not interior[inicio] or etiquetas[inicio] >= 0:
                continue
            pila = [inicio]
            etiquetas[inicio] = grupo
            while pila:
                j = pila.pop()
                for k in self._vecinas(j):
                    if interior[k] and etiquetas[k] < 0:
                        etiquetas[k] = grupo
                        pila.append(k)
            grupo += 1

        # descartar valles con muy pocos patrones y renumerar
        validos = [g for g in range(grupo) if impactos[etiquetas == g].sum() >= min_fraccion * X.shape[0]]
        renumerar = {g: n for n, g in enumerate(validos)}
        etiquetas = np.array([renumerar.get(e, -1) for e in etiquetas])

        # asignar las crestas al valle mas cercano en el espacio de los datos
        con_grupo = np.where(etiquetas >= 0)[0]
        if con_grupo.size == 0:
            return np.zeros(self.n_neuronas, dtype=int)
        for j in np.where(etiquetas < 0)[0]:
            cercana = con_grupo[np.argmin(np.linalg.norm(self.W[con_grupo] - self.W[j], axis=1))]
            etiquetas[j] = etiquetas[cercana]
        return etiquetas

    def agrupar(self, X, **kwargs) -> np.ndarray:
        """Grupo asignado a cada patron: el de su neurona ganadora."""
        return self.segmentar(X, **kwargs)[self.ganadoras(X)]

    def resumen(self, X=None) -> str:
        texto = (f"SOM {self.filas}x{self.columnas} ({self.n_neuronas} neuronas) | "
                 f"eta {self.eta_inicial:g} -> {self.eta_final:g}, "
                 f"sigma {self.sigma_inicial:g} -> {self.sigma_final:g}, "
                 f"{self.iteraciones} iteraciones")
        if X is not None:
            texto += (f"\nerror de cuantizacion = {self.error_cuantizacion(X):.4f} | "
                      f"error topografico = {self.error_topografico(X):.3f}")
        return texto


def pureza(grupos_patron, clases) -> float:
    """Fraccion de patrones cuyo grupo tiene como clase mayoritaria la suya.

    Solo se usa para *evaluar* el agrupamiento a posteriori: las clases reales
    no intervienen en ningun momento del entrenamiento.
    """
    grupos_patron = np.asarray(grupos_patron)
    clases = np.asarray(clases)
    aciertos = 0
    for g in np.unique(grupos_patron):
        miembros = clases[grupos_patron == g]
        aciertos += np.bincount(miembros.astype(int)).max()
    return aciertos / clases.size
