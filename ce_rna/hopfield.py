"""
Red de Hopfield (1982): memoria asociativa direccionable por contenido.

Referencia: guia de evaluacion, Actividad 5.

Idea
----
Una red de Hopfield no clasifica ni aproxima: **recuerda**.  Se le presentan P
patrones bipolares xi^mu en {-1, +1}^N y los almacena en una matriz de pesos
simetrica.  Despues, a partir de una version incompleta o contaminada de uno de
ellos, la dinamica de la red evoluciona hasta el patron almacenado mas parecido.

Arquitectura
------------
Una sola capa de N neuronas binarias, cada una conectada con **todas las
demas** (recurrente), sin autoconexiones:  w_ij = w_ji,  w_ii = 0.  Para una
imagen de 7x6 pixeles, N = 42 neuronas y 42*41/2 = 861 pesos distintos.

Almacenamiento (regla de Hebb, una sola pasada)
-----------------------------------------------
    w_ij = (1/N) sum_mu xi_i^mu xi_j^mu       (i != j)

No hay iteraciones ni error: los pesos se calculan de una vez.  Tambien se
ofrece la regla de la **pseudoinversa** (proyeccion), que hace que cada patron
sea exactamente un punto fijo aunque los patrones esten correlacionados.

Recuperacion (dinamica asincrona)
---------------------------------
Partiendo del estado s(0) = patron de prueba, se elige una neurona i y

    s_i <- sgn( sum_j w_ij s_j )

hasta que ningun cambio de neurona modifica el estado (punto fijo).  Con pesos
simetricos y actualizacion asincrona la **energia**

    E(s) = -(1/2) sum_ij w_ij s_i s_j

nunca aumenta, de modo que la red siempre converge a un minimo local de E.  Los
patrones almacenados son (idealmente) esos minimos: son los *atractores*.

Limitaciones que el experimento mide
------------------------------------
* Capacidad: con la regla de Hebb y patrones aleatorios, P_max ~ 0.138 N.
  Para N = 42 son unos 5-6 patrones; cuatro letras estan cerca del limite,
  y ademas las letras NO son aleatorias: B y D comparten muchos pixeles.
* Estados espurios: el inverso -xi de cada patron tambien es un atractor, y
  aparecen mezclas de patrones que no son ninguno de los almacenados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResultadoRecuperacion:
    """Resultado de una recuperacion."""

    estado: np.ndarray
    iteraciones: int                    # barridos completos de las N neuronas
    convergio: bool
    energia: list[float] = field(default_factory=list)
    cambios: int = 0                    # numero total de neuronas que cambiaron


def _signo(v, anterior):
    """sgn(v) con la convencion de conservar el estado si v == 0."""
    return np.where(v > 0, 1.0, np.where(v < 0, -1.0, anterior))


class RedHopfield:
    """Red de Hopfield discreta con neuronas bipolares.

    Parameters
    ----------
    n_neuronas:
        N, numero de neuronas (= numero de pixeles de la imagen).
    regla:
        ``"hebb"`` (producto exterior, la regla clasica) o ``"pseudoinversa"``.
    """

    def __init__(self, n_neuronas: int, regla: str = "hebb"):
        if regla not in ("hebb", "pseudoinversa"):
            raise ValueError("regla debe ser 'hebb' o 'pseudoinversa'")
        self.N = int(n_neuronas)
        self.regla = regla
        self.W = np.zeros((self.N, self.N))
        self.patrones = np.zeros((0, self.N))

    # -- almacenamiento ------------------------------------------------------
    def almacenar(self, patrones) -> "RedHopfield":
        P = np.atleast_2d(np.asarray(patrones, dtype=float))
        if P.shape[1] != self.N:
            raise ValueError(f"los patrones deben tener {self.N} elementos")
        if not np.all(np.isin(P, (-1.0, 1.0))):
            raise ValueError("los patrones deben ser bipolares (-1 / +1)")
        self.patrones = P.copy()
        if self.regla == "hebb":
            self.W = (P.T @ P) / self.N
        else:
            # W = X (X^T X)^-1 X^T con X = P^T (N x P): proyector sobre el
            # subespacio generado por los patrones.  W xi = xi exactamente.
            X = P.T
            self.W = X @ np.linalg.pinv(X.T @ X) @ X.T
        np.fill_diagonal(self.W, 0.0)
        return self

    # -- dinamica ------------------------------------------------------------
    def energia(self, s) -> float:
        s = np.asarray(s, dtype=float)
        return float(-0.5 * s @ self.W @ s)

    def recuperar(self, s0, max_iteraciones: int = 100, modo: str = "asincrono",
                  semilla: int = 0) -> ResultadoRecuperacion:
        """Evoluciona desde `s0` hasta un punto fijo (o `max_iteraciones` barridos).

        En modo asincrono cada barrido visita las N neuronas en orden aleatorio
        y actualiza una a una, usando siempre el estado mas reciente.  En modo
        sincrono se actualizan todas a la vez (puede entrar en ciclos de 2).
        """
        rng = np.random.default_rng(semilla)
        s = np.asarray(s0, dtype=float).copy()
        energias = [self.energia(s)]
        cambios = 0
        for it in range(1, max_iteraciones + 1):
            cambio_en_barrido = False
            if modo == "asincrono":
                for i in rng.permutation(self.N):
                    nuevo = _signo(self.W[i] @ s, s[i])
                    if nuevo != s[i]:
                        s[i] = nuevo
                        cambios += 1
                        cambio_en_barrido = True
            else:
                nuevo = _signo(self.W @ s, s)
                cambio_en_barrido = not np.array_equal(nuevo, s)
                cambios += int(np.sum(nuevo != s))
                s = nuevo
            energias.append(self.energia(s))
            if not cambio_en_barrido:
                return ResultadoRecuperacion(s, it, True, energias, cambios)
        return ResultadoRecuperacion(s, max_iteraciones, False, energias, cambios)

    def es_punto_fijo(self, s) -> bool:
        s = np.asarray(s, dtype=float)
        return bool(np.array_equal(_signo(self.W @ s, s), s))

    # -- interpretacion ------------------------------------------------------
    def solapamientos(self, s) -> np.ndarray:
        """m_mu = (1/N) xi^mu . s  en [-1, 1]; 1 = identico, -1 = inverso."""
        return self.patrones @ np.asarray(s, dtype=float) / self.N

    def identificar(self, s) -> tuple[int, str]:
        """Clasifica un estado final.

        Devuelve ``(mu, tipo)`` con tipo ``"patron"`` si s coincide con xi^mu,
        ``"inverso"`` si coincide con -xi^mu, o ``(-1, "espurio")`` si no es
        ninguno de los almacenados ni sus inversos.
        """
        m = self.solapamientos(s)
        for mu, valor in enumerate(m):
            if np.isclose(valor, 1.0):
                return mu, "patron"
        for mu, valor in enumerate(m):
            if np.isclose(valor, -1.0):
                return mu, "inverso"
        return -1, "espurio"

    def capacidad_teorica(self) -> float:
        """P_max ~ 0.138 N (Amit, Gutfreund y Sompolinsky, 1985) para la regla de Hebb."""
        return 0.138 * self.N

    def resumen(self) -> str:
        return (f"Hopfield N = {self.N} | regla {self.regla} | {self.patrones.shape[0]} patrones "
                f"almacenados | capacidad teorica (Hebb) ~ {self.capacidad_teorica():.1f}")
