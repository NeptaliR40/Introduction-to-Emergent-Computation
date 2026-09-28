"""
Perceptron multicapa (MLP) entrenado con retropropagacion del error.

Referencia: guia de evaluacion, Actividad 3 (clases 5, 6 y 7: arquitectura por
capas, funcion sigmoidal, funcion de error, descenso por el gradiente,
retropropagacion, razon de aprendizaje y criterios de parada).

Por que hace falta una capa oculta
----------------------------------
El perceptron simple y el ADALINE solo trazan **una** frontera lineal
w0 + w1 x1 + ... = 0.  El XOR no admite ninguna (experimento 03).  Una capa
intermedia de neuronas no lineales transforma el espacio de entrada en otro
--- el *espacio oculto* --- donde el problema si es linealmente separable, y la
neurona de salida solo tiene que trazar una recta en ese espacio nuevo.

Propagacion hacia adelante
--------------------------
Con x_0 = 1 en cada capa (sesgo), para la capa l = 1..L:

    v^(l) = W^(l) y^(l-1)          (potencial postsinaptico)
    y^(l) = phi_l(v^(l))           (salida de la capa)

con y^(0) = x.  Las capas ocultas usan la sigmoide logistica
phi(v) = 1 / (1 + e^-v), cuya derivada phi'(v) = phi(v) (1 - phi(v)) se calcula
a partir de la propia salida, sin volver a evaluar la exponencial.

Funcion de error
----------------
La misma de la clase del ADALINE, ahora sumada sobre todas las salidas k:

    E = (1/N) sum_p E^p ,     E^p = (1/2) sum_k (d_k^p - y_k^p)^2

Retropropagacion (regla Delta generalizada)
-------------------------------------------
El gradiente de E^p respecto de cada peso se obtiene aplicando la regla de la
cadena capa por capa, desde la salida hacia la entrada.  Se define el
*gradiente local* delta de cada neurona:

    capa de salida :  delta_k^(L) = (d_k - y_k) phi'(v_k^(L))
    capa oculta    :  delta_j^(l) = phi'(v_j^(l)) sum_k delta_k^(l+1) w_kj^(l+1)

y la correccion de cada peso tiene la misma forma que la regla Delta:

    Delta w_ji^(l) = gamma delta_j^(l) y_i^(l-1)   (+ alpha Delta w_ji^(l) anterior)

El termino alpha (momento) es opcional: suaviza la trayectoria del descenso y
ayuda a atravesar mesetas de la superficie de error.

Contenido del modulo
--------------------
* `PerceptronMulticapa`  : la red, la propagacion, la retropropagacion y el
  bucle de entrenamiento con tres criterios de parada.
* `gradiente_numerico`   : diferencias finitas centrales sobre E(W), usadas en
  las pruebas para verificar que la retropropagacion es exacta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activaciones import FuncionActivacion, Identidad, Sigmoide


@dataclass
class HistorialEpocaMLP:
    """Estado de la red al terminar una epoca."""

    epoca: int
    error_entrenamiento: float
    error_prueba: float | None
    norma_gradiente: float


class PerceptronMulticapa:
    """Red hacia adelante totalmente conectada entrenada con backpropagation.

    Parameters
    ----------
    capas:
        Numero de neuronas por capa, incluida la de entrada.  ``[2, 2, 1]`` es
        la arquitectura 2-2-1 de la clase para el XOR.
    razon_aprendizaje:
        gamma, el tamano del paso del descenso del gradiente.
    momento:
        alpha en [0, 1).  Con 0 se obtiene el descenso del gradiente puro.
    activacion_oculta:
        Activacion de las capas ocultas (por omision, la sigmoide logistica).
    activacion_salida:
        Activacion de la capa de salida.  Sigmoide para clasificacion con
        objetivos en [0, 1]; `Identidad` para aproximacion de funciones, donde
        la salida debe poder tomar cualquier valor real.
    modo:
        ``"estocastico"`` actualiza tras cada patron (como el procedimiento del
        ADALINE de la clase); ``"lote"`` acumula el gradiente de todo el
        conjunto y da un paso por epoca (descenso del gradiente exacto sobre E).
    max_epocas, error_objetivo, paciencia:
        Criterios de parada.  El entrenamiento termina en cuanto ocurre lo
        primero de: (a) se alcanza `max_epocas`; (b) el error de entrenamiento
        cae por debajo de `error_objetivo`; (c) si se proporciona un conjunto de
        prueba y `paciencia` > 0, el error de prueba no mejora durante
        `paciencia` epocas seguidas (*parada temprana*); se restauran entonces
        los mejores pesos vistos.
    escala_inicial:
        Los pesos iniciales se toman uniformes en [-a, a].  Pesos pequenos
        mantienen las sigmoides en su zona lineal, donde la derivada es maxima.
    barajar:
        Reordenar los patrones en cada epoca (modo estocastico).
    semilla:
        Semilla del generador (pesos iniciales y orden de presentacion).
    """

    def __init__(
        self,
        capas: list[int],
        razon_aprendizaje: float = 0.5,
        momento: float = 0.0,
        activacion_oculta: FuncionActivacion | None = None,
        activacion_salida: FuncionActivacion | None = None,
        modo: str = "estocastico",
        max_epocas: int = 5000,
        error_objetivo: float = 1e-3,
        paciencia: int = 0,
        escala_inicial: float = 1.0,
        barajar: bool = True,
        semilla: int = 0,
    ):
        if len(capas) < 2 or min(capas) < 1:
            raise ValueError("se necesitan al menos dos capas con una neurona o mas")
        if razon_aprendizaje <= 0:
            raise ValueError("la razon de aprendizaje debe ser positiva")
        if not 0.0 <= momento < 1.0:
            raise ValueError("el momento debe estar en [0, 1)")
        if modo not in ("estocastico", "lote"):
            raise ValueError("modo debe ser 'estocastico' o 'lote'")

        self.capas = [int(n) for n in capas]
        self.gamma = float(razon_aprendizaje)
        self.alpha = float(momento)
        self.phi_oculta = activacion_oculta or Sigmoide()
        self.phi_salida = activacion_salida or Sigmoide()
        self.modo = modo
        self.max_epocas = int(max_epocas)
        self.error_objetivo = float(error_objetivo)
        self.paciencia = int(paciencia)
        self.barajar = bool(barajar)
        self.rng = np.random.default_rng(semilla)

        # W[l] tiene forma (n_{l+1}, n_l + 1): la columna 0 es el sesgo.
        self.W = [
            self.rng.uniform(-escala_inicial, escala_inicial, size=(n_sig, n_ant + 1))
            for n_ant, n_sig in zip(self.capas[:-1], self.capas[1:])
        ]
        self.W_inicial = [w.copy() for w in self.W]
        self._dW_anterior = [np.zeros_like(w) for w in self.W]

        self.historial: list[HistorialEpocaMLP] = []
        self.epocas_usadas = 0
        self.motivo_parada = ""

    # -- propiedades ---------------------------------------------------------
    @property
    def n_pesos(self) -> int:
        return int(sum(w.size for w in self.W))

    @property
    def arquitectura(self) -> str:
        return "-".join(str(n) for n in self.capas)

    def _phi(self, capa: int) -> FuncionActivacion:
        return self.phi_salida if capa == len(self.W) - 1 else self.phi_oculta

    # -- propagacion hacia adelante -----------------------------------------
    def propagar(self, X):
        """Propaga un lote de patrones y devuelve las salidas de todas las capas.

        Devuelve una lista ``[y0, y1, ..., yL]`` con ``y0 = X``; cada ``y_l`` es
        una matriz (N, n_l).  Las salidas intermedias hacen falta en la
        retropropagacion y para dibujar el espacio oculto.
        """
        y = np.atleast_2d(np.asarray(X, dtype=float))
        salidas = [y]
        for l, W in enumerate(self.W):
            v = np.hstack([np.ones((y.shape[0], 1)), y]) @ W.T
            y = self._phi(l)(v)
            salidas.append(y)
        return salidas

    def salida(self, X):
        """Salida real de la red.  Vector (N,) si hay una sola neurona de salida."""
        y = self.propagar(X)[-1]
        return y[:, 0] if y.shape[1] == 1 else y

    def predecir(self, X, umbral: float = 0.5):
        """Clasificacion binaria {0, 1} umbralizando la salida."""
        return np.where(self.salida(X) >= umbral, 1.0, 0.0)

    # -- funcion de error ----------------------------------------------------
    def error(self, X, d) -> float:
        """E = (1/N) sum_p (1/2) sum_k (d_k - y_k)^2."""
        y = self.propagar(X)[-1]
        D = np.asarray(d, dtype=float).reshape(y.shape)
        return float(np.mean(0.5 * np.sum((D - y) ** 2, axis=1)))

    # -- retropropagacion ----------------------------------------------------
    def _derivada(self, capa: int, y):
        """phi'(v) expresada en funcion de la salida y = phi(v)."""
        phi = self._phi(capa)
        if isinstance(phi, Sigmoide):
            return y * (1.0 - y)
        if isinstance(phi, Identidad):
            return np.ones_like(y)
        # caso general (sigmoide bipolar tanh(v/2)): phi' = (1 - y^2) / 2
        return 0.5 * (1.0 - y ** 2)

    def gradiente(self, X, d):
        """Gradiente medio dE/dW[l] sobre el lote, por retropropagacion.

        Devuelve una lista con la misma forma que `self.W`.
        """
        salidas = self.propagar(X)
        y_L = salidas[-1]
        D = np.asarray(d, dtype=float).reshape(y_L.shape)
        N = y_L.shape[0]

        # gradiente local de la capa de salida: delta = (d - y) phi'(v)
        delta = (D - y_L) * self._derivada(len(self.W) - 1, y_L)
        gradientes = [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            y_prev = np.hstack([np.ones((N, 1)), salidas[l]])
            # dE/dW = -(1/N) sum_p delta^T y_prev   (el signo menos: E decrece)
            gradientes[l] = -(delta.T @ y_prev) / N
            if l > 0:
                # se retropropaga por los pesos SIN la columna de sesgo
                delta = (delta @ self.W[l][:, 1:]) * self._derivada(l - 1, salidas[l])
        return gradientes

    def _paso(self, X, d) -> float:
        """Un paso de descenso del gradiente con momento.  Devuelve ||grad||."""
        gradientes = self.gradiente(X, d)
        norma = 0.0
        for l, g in enumerate(gradientes):
            dW = -self.gamma * g + self.alpha * self._dW_anterior[l]
            self.W[l] += dW
            self._dW_anterior[l] = dW
            norma += float(np.sum(g ** 2))
        return float(np.sqrt(norma))

    # -- entrenamiento -------------------------------------------------------
    def entrenar(self, X, d, X_prueba=None, d_prueba=None) -> "PerceptronMulticapa":
        """Entrena la red hasta que se cumpla un criterio de parada."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = np.asarray(d, dtype=float)
        D = d.reshape(X.shape[0], -1)
        hay_prueba = X_prueba is not None and d_prueba is not None

        mejor_error_prueba = np.inf
        mejores_pesos = None
        sin_mejora = 0
        self.motivo_parada = "max_epocas"

        for epoca in range(1, self.max_epocas + 1):
            if self.modo == "lote":
                norma = self._paso(X, D)
            else:
                orden = self.rng.permutation(X.shape[0]) if self.barajar else np.arange(X.shape[0])
                normas = [self._paso(X[p:p + 1], D[p:p + 1]) for p in orden]
                norma = float(np.mean(normas))

            e_ent = self.error(X, D)
            e_pru = self.error(X_prueba, d_prueba) if hay_prueba else None
            self.historial.append(HistorialEpocaMLP(epoca, e_ent, e_pru, norma))
            self.epocas_usadas = epoca

            if not np.isfinite(e_ent):
                self.motivo_parada = "divergencia"
                break
            if e_ent <= self.error_objetivo:
                self.motivo_parada = "error_objetivo"
                break
            if hay_prueba and self.paciencia > 0:
                if e_pru < mejor_error_prueba - 1e-12:
                    mejor_error_prueba = e_pru
                    mejores_pesos = [w.copy() for w in self.W]
                    sin_mejora = 0
                else:
                    sin_mejora += 1
                    if sin_mejora >= self.paciencia:
                        self.motivo_parada = "parada_temprana"
                        self.W = mejores_pesos
                        break
        return self

    # -- informes ------------------------------------------------------------
    def historial_df(self):
        import pandas as pd

        return pd.DataFrame([
            {"epoca": h.epoca, "error_entrenamiento": h.error_entrenamiento,
             "error_prueba": h.error_prueba, "norma_gradiente": h.norma_gradiente}
            for h in self.historial
        ])

    def curva_error(self, prueba: bool = False) -> np.ndarray:
        clave = "error_prueba" if prueba else "error_entrenamiento"
        return np.array([getattr(h, clave) for h in self.historial], dtype=float)

    def resumen(self) -> str:
        final = self.historial[-1].error_entrenamiento if self.historial else float("nan")
        return (
            f"MLP {self.arquitectura} | gamma = {self.gamma:g}, alpha = {self.alpha:g}, "
            f"modo {self.modo} | {self.n_pesos} pesos\n"
            f"epocas usadas: {self.epocas_usadas} | parada: {self.motivo_parada} | "
            f"E final = {final:.3e}"
        )


def gradiente_numerico(red: PerceptronMulticapa, X, d, h: float = 1e-6):
    """Gradiente de E(W) por diferencias finitas centrales, peso a peso."""
    gradientes = []
    for W in red.W:
        g = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            original = W[idx]
            W[idx] = original + h
            e_mas = red.error(X, d)
            W[idx] = original - h
            e_menos = red.error(X, d)
            W[idx] = original
            g[idx] = (e_mas - e_menos) / (2 * h)
        gradientes.append(g)
    return gradientes
