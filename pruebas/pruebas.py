"""
Bateria de pruebas del paquete `ce_rna`.

Se ejecuta sin dependencias externas (no hace falta `pytest`):

    python pruebas/pruebas.py

Cada prueba es una funcion `probar_*` que lanza `AssertionError` si algo no
cuadra.  El runner las descubre por nombre, las ejecuta todas e informa del
resultado.  Las pruebas verifican propiedades **matematicas** de los modelos, no
solo que el codigo no reviente:

* que las compuertas MCP reproduzcan sus tablas de verdad,
* que el perceptron converja en problemas separables y no lo haga en el XOR,
* que la regla Delta alcance la solucion de minimos cuadrados,
* que el gradiente implementado coincida con la derivada numerica del error,
* que las derivadas de las activaciones coincidan con sus diferencias finitas,
* que la retropropagacion del MLP coincida con el gradiente numerico y resuelva el XOR,
* que el mapa de Kohonen ordene la rejilla y separe grupos bien definidos,
* que la red de Hopfield almacene sus patrones como puntos fijos y nunca suba la energia.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np

from ce_rna import activaciones as act
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna.adaline import (Adaline, ecm_de_pesos, solucion_minimos_cuadrados)
from ce_rna.hebb import RedHebb
from ce_rna.hopfield import RedHopfield
from ce_rna.kohonen import MapaKohonen, pureza
from ce_rna.mcculloch_pitts import (NeuronaMCP, neurona_and, neurona_nand,
                                    neurona_nor, neurona_not, neurona_or,
                                    red_xor)
from ce_rna.perceptron import PerceptronSimple
from ce_rna.perceptron_multicapa import PerceptronMulticapa, gradiente_numerico


# ---------------------------------------------------------------------------
# Activaciones
# ---------------------------------------------------------------------------

def probar_escalon():
    phi = act.Escalon(umbral=2.0)
    assert np.array_equal(phi([1.9, 2.0, 2.1]), [0.0, 1.0, 1.0])


def probar_escalon_bipolar_con_zona():
    phi = act.EscalonBipolarConZona(theta=0.5)
    assert np.array_equal(phi([-1.0, -0.5, 0.0, 0.5, 1.0]), [-1.0, 0.0, 0.0, 0.0, 1.0])


def probar_sigmoide_estable():
    """La sigmoide no debe desbordar para argumentos extremos."""
    valores = act.Sigmoide()([-1000.0, 0.0, 1000.0])
    assert np.all(np.isfinite(valores))
    assert np.isclose(valores[1], 0.5)


def probar_derivadas_numericas():
    """Cada activacion derivable debe coincidir con su diferencia finita central."""
    v = np.linspace(-3, 3, 41)
    h = 1e-6
    for phi in (act.Identidad(), act.Sigmoide(), act.SigmoideBipolar(), act.Gaussiana(0.5, 1.2)):
        numerica = (phi(v + h) - phi(v - h)) / (2 * h)
        assert np.allclose(phi.derivada(v), numerica, atol=1e-5), phi.nombre


# ---------------------------------------------------------------------------
# McCulloch-Pitts
# ---------------------------------------------------------------------------

def probar_compuertas_mcp():
    for nombre, neurona in (("AND", neurona_and(2)), ("OR", neurona_or(2)),
                            ("NAND", neurona_nand(2)), ("NOR", neurona_nor(2))):
        esperado = ds.compuerta(nombre).d
        obtenido = neurona.activar(ds.compuerta(nombre).X)
        assert np.array_equal(obtenido, esperado), nombre


def probar_not_mcp():
    assert np.array_equal(neurona_not().activar([[0.0], [1.0]]), [1.0, 0.0])


def probar_red_xor_mcp():
    esperado = ds.compuerta("XOR").d
    assert np.array_equal(red_xor().salida(ds.compuerta("XOR").X), esperado)


def probar_ninguna_mcp_resuelve_xor():
    """Ninguna neurona de umbral clasifica los 4 patrones del XOR."""
    X, d = ds.compuerta("XOR").X, ds.compuerta("XOR").d
    for w1 in np.linspace(-4, 4, 33):
        for w2 in np.linspace(-4, 4, 33):
            for theta in np.linspace(-4, 4, 33):
                if np.array_equal(NeuronaMCP([w1, w2], theta).activar(X), d):
                    raise AssertionError(f"encontrada solucion inesperada: {w1}, {w2}, {theta}")


def probar_red_mcp_rechaza_fuentes_invalidas():
    from ce_rna.mcculloch_pitts import RedMCP

    red = RedMCP(["a", "b"])
    try:
        red.agregar("y", neurona_and(2), ["a", "inexistente"])
    except KeyError:
        return
    raise AssertionError("se acepto una fuente no declarada")


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

def probar_perceptron_converge_en_and_or():
    for nombre in ("AND", "OR", "NAND", "NOR"):
        conjunto = ds.compuerta(nombre, "bipolar")
        modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
        modelo.entrenar(conjunto.X, conjunto.d)
        assert modelo.convergio, nombre
        assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) == 1.0, nombre


def probar_perceptron_no_converge_en_xor():
    conjunto = ds.compuerta("XOR", "bipolar")
    modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=300)
    modelo.entrenar(conjunto.X, conjunto.d)
    assert not modelo.convergio
    assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) < 1.0


def probar_perceptron_no_toca_pesos_si_acierta():
    """Paso 4: si y == d para todas las salidas, los pesos no cambian."""
    conjunto = ds.compuerta("AND", "bipolar")
    modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
    modelo.entrenar(conjunto.X, conjunto.d)
    W_convergido = modelo.W.copy()
    modelo.entrenar(conjunto.X, conjunto.d)   # segundo entrenamiento desde el estado final
    assert np.array_equal(modelo.W, W_convergido)
    assert modelo.epocas_usadas == 1


def probar_perceptron_multiclase():
    conjunto = ds.letras_xo(con_ruido=4, semilla=0)
    modelo = PerceptronSimple(25, 2, razon_aprendizaje=1.0, max_epocas=200)
    modelo.entrenar(conjunto.X, conjunto.d)
    assert modelo.convergio
    assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) == 1.0


def probar_perceptron_valida_dimensiones():
    modelo = PerceptronSimple(2, 1)
    try:
        modelo.entrenar(np.zeros((4, 3)), np.zeros(4))
    except ValueError:
        return
    raise AssertionError("se acepto una matriz de entradas con dimension incorrecta")


def probar_margen_positivo_implica_clasificacion_correcta():
    conjunto = ds.compuerta("OR", "bipolar")
    modelo = PerceptronSimple(2, 1, max_epocas=100).entrenar(conjunto.X, conjunto.d)
    assert mt.margen_geometrico(modelo.W[0], conjunto.X, conjunto.d) > 0


# ---------------------------------------------------------------------------
# ADALINE
# ---------------------------------------------------------------------------

def probar_adaline_recupera_pesos_del_decodificador():
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=0.1, max_epocas=2000, tolerancia=1e-10,
                     inicializacion="aleatoria", semilla=0)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert np.allclose(modelo.w, ds.pesos_optimos_decodificador(3), atol=1e-5)
    assert modelo.ecm(conjunto.X, conjunto.d) < 1e-10


def probar_adaline_alcanza_minimos_cuadrados():
    """El descenso del gradiente converge al minimo global del paraboloide."""
    conjunto = ds.nubes_separables(n_por_clase=25, semilla=4)
    optimo = solucion_minimos_cuadrados(conjunto.X, conjunto.d)
    modelo = Adaline(2, razon_aprendizaje=0.05, modo="lote", max_epocas=20000,
                     tolerancia=1e-12, inicializacion="aleatoria", semilla=1)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert np.allclose(modelo.w, optimo, atol=1e-3)


def probar_gradiente_coincide_con_derivada_numerica():
    """El gradiente analitico -(X^T e)/N debe igualar la derivada numerica de E."""
    conjunto = ds.decodificador_binario(3)
    X, d = conjunto.X, conjunto.d
    w = np.array([0.3, -0.7, 1.4, 0.2])
    Xe = np.hstack([np.ones((X.shape[0], 1)), X])
    analitico = -(Xe.T @ (d - Xe @ w)) / X.shape[0]

    h = 1e-6
    numerico = np.empty_like(w)
    for i in range(w.size):
        mas, menos = w.copy(), w.copy()
        mas[i] += h
        menos[i] -= h
        numerico[i] = (ecm_de_pesos(mas, X, d) - ecm_de_pesos(menos, X, d)) / (2 * h)
    assert np.allclose(analitico, numerico, atol=1e-6)


def probar_adaline_diverge_con_razon_excesiva():
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=3.0, max_epocas=200, inicializacion="aleatoria", semilla=0)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert modelo.diverged


def probar_adaline_error_decrece_en_lote():
    """En modo por lotes y bajo la cota de estabilidad, E decrece monotonamente."""
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=0.2, modo="lote", max_epocas=300,
                     tolerancia=0.0, inicializacion="aleatoria", semilla=2)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    errores = np.array([h.ecm for h in modelo.historial])
    assert np.all(np.diff(errores) <= 1e-12)


def probar_cota_estabilidad():
    conjunto = ds.decodificador_binario(3)
    cota = Adaline(3).cota_estabilidad(conjunto.X)
    assert 0 < cota < np.inf


# ---------------------------------------------------------------------------
# Hebb
# ---------------------------------------------------------------------------

def probar_hebb_funciona_en_bipolar():
    for nombre in ("AND", "OR"):
        conjunto = ds.compuerta(nombre, "bipolar")
        red = RedHebb(2, 1).entrenar(conjunto.X, conjunto.d)
        assert red.exactitud(conjunto.X, conjunto.d) == 1.0, nombre


def probar_hebb_falla_en_and_binario():
    """Los aportes se cancelan y los pesos de entrada quedan en cero."""
    conjunto = ds.compuerta("AND", "binaria")
    objetivo = 2 * conjunto.d - 1
    red = RedHebb(2, 1).entrenar(conjunto.X, objetivo)
    assert np.allclose(red.W[0, 1:], 0.0)
    assert red.exactitud(conjunto.X, objetivo) < 1.0


def probar_hebb_independiente_de_la_razon():
    """La razon de aprendizaje solo escala los pesos: la frontera no cambia."""
    conjunto = ds.compuerta("AND", "bipolar")
    W1 = RedHebb(2, 1, razon_aprendizaje=1.0).entrenar(conjunto.X, conjunto.d).W
    W2 = RedHebb(2, 1, razon_aprendizaje=0.25).entrenar(conjunto.X, conjunto.d).W
    assert np.allclose(W1 * 0.25, W2)


# ---------------------------------------------------------------------------
# Metricas y datos
# ---------------------------------------------------------------------------

def probar_matriz_confusion():
    M, etiquetas = mt.matriz_confusion([1, 1, -1, -1], [1, -1, -1, -1], etiquetas=[-1, 1])
    assert M.sum() == 4
    assert np.array_equal(etiquetas, [-1, 1])
    assert M[1, 1] == 1 and M[0, 0] == 2


def probar_r2_perfecto():
    assert np.isclose(mt.r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def probar_decodificador_es_lineal():
    conjunto = ds.decodificador_binario(4)
    w = ds.pesos_optimos_decodificador(4)
    Xe = np.hstack([np.ones((conjunto.X.shape[0], 1)), conjunto.X])
    assert np.allclose(Xe @ w, conjunto.d)


def probar_letras_son_bipolares():
    conjunto = ds.letras_xo(con_ruido=3, semilla=1)
    assert set(np.unique(conjunto.X).tolist()) <= {-1.0, 1.0}
    assert conjunto.X.shape[1] == 25


def probar_conjunto_valida_longitudes():
    try:
        ds.Conjunto(X=np.zeros((3, 2)), d=np.zeros(4), nombre="malo")
    except ValueError:
        return
    raise AssertionError("se acepto un conjunto con X y d de distinta longitud")


# ---------------------------------------------------------------------------
# Perceptron multicapa
# ---------------------------------------------------------------------------

def probar_mlp_gradiente_numerico():
    """La retropropagacion debe coincidir con las diferencias finitas de E(W)."""
    X = ds.compuerta("XOR").X
    D = np.random.default_rng(0).random((4, 2))
    for salida in (act.Sigmoide(), act.Identidad()):
        red = PerceptronMulticapa([2, 3, 2], activacion_salida=salida, semilla=3)
        for analitico, numerico in zip(red.gradiente(X, D), gradiente_numerico(red, X, D)):
            assert np.allclose(analitico, numerico, atol=1e-7), salida.nombre


def probar_mlp_resuelve_xor():
    xor = ds.compuerta("XOR")
    red = PerceptronMulticapa([2, 2, 1], razon_aprendizaje=0.5, error_objetivo=0.005, semilla=0)
    red.entrenar(xor.X, xor.d)
    assert red.motivo_parada == "error_objetivo"
    assert np.array_equal(red.predecir(xor.X), xor.d)


def probar_mlp_error_decrece_en_lote():
    """Con un paso pequeno en modo lote, el descenso por el gradiente no sube E."""
    xor = ds.compuerta("XOR")
    red = PerceptronMulticapa([2, 3, 1], razon_aprendizaje=0.1, modo="lote",
                              max_epocas=200, error_objetivo=0.0, semilla=1)
    e = red.entrenar(xor.X, xor.d).curva_error()
    assert np.all(np.diff(e) <= 1e-12)


def probar_mlp_parada_temprana_restaura():
    ent, pru = ds.aproximacion_funcion(n_entrenamiento=10, n_prueba=20, ruido=0.3)
    red = PerceptronMulticapa([1, 6, 1], activacion_salida=act.Identidad(), razon_aprendizaje=0.05,
                              max_epocas=3000, error_objetivo=0.0, paciencia=50, semilla=0)
    red.entrenar(ent.X, ent.d, pru.X, pru.d)
    if red.motivo_parada == "parada_temprana":
        assert np.isclose(red.error(pru.X, pru.d), np.nanmin(red.curva_error(prueba=True)))


# ---------------------------------------------------------------------------
# Kohonen
# ---------------------------------------------------------------------------

def probar_kohonen_separa_grupos():
    datos = ds.grupos_plano()
    mapa = MapaKohonen(10, 10, iteraciones=6000, semilla=0).entrenar(datos.X, n_instantaneas=0)
    grupos = mapa.agrupar(datos.X)
    assert len(np.unique(grupos)) == 5
    assert pureza(grupos, datos.d) > 0.95
    assert mapa.error_topografico(datos.X) < 0.05


def probar_kohonen_reduce_cuantizacion():
    datos = ds.grupos_plano()
    mapa = MapaKohonen(6, 6, iteraciones=2000, inicializacion="rejilla", semilla=1)
    mapa._inicializar(datos.X)
    antes = mapa.error_cuantizacion(datos.X)
    mapa.entrenar(datos.X, n_instantaneas=0)
    assert mapa.error_cuantizacion(datos.X) < antes


# ---------------------------------------------------------------------------
# Hopfield
# ---------------------------------------------------------------------------

def probar_hopfield_pesos_simetricos():
    red = RedHopfield(42).almacenar(ds.letras_abcd().X)
    assert np.allclose(red.W, red.W.T)
    assert np.allclose(np.diag(red.W), 0.0)


def probar_hopfield_letras_son_puntos_fijos():
    letras = ds.letras_abcd().X
    for regla in ("hebb", "pseudoinversa"):
        red = RedHopfield(42, regla=regla).almacenar(letras)
        assert all(red.es_punto_fijo(p) for p in letras), regla


def probar_hopfield_trazo_fino_falla_con_hebb():
    """Las letras de trazo fino estan demasiado correlacionadas para la regla de Hebb."""
    finas = ds.letras_abcd("fina").X
    assert not all(RedHopfield(42).almacenar(finas).es_punto_fijo(p) for p in finas)
    assert all(RedHopfield(42, "pseudoinversa").almacenar(finas).es_punto_fijo(p) for p in finas)


def probar_hopfield_energia_no_aumenta():
    letras = ds.letras_abcd().X
    red = RedHopfield(42).almacenar(letras)
    for semilla in range(20):
        res = red.recuperar(ds.contaminar(letras[semilla % 4], 10, semilla=semilla), semilla=semilla)
        assert res.convergio
        assert np.all(np.diff(res.energia) <= 1e-12)


def probar_hopfield_recupera_con_ruido():
    letras = ds.letras_abcd().X
    red = RedHopfield(42).almacenar(letras)
    for mu in range(4):
        res = red.recuperar(ds.contaminar(letras[mu], 3, semilla=mu), semilla=mu)
        assert red.identificar(res.estado) == (mu, "patron")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    pruebas = sorted(
        ((nombre, funcion) for nombre, funcion in globals().items()
         if nombre.startswith("probar_") and callable(funcion)),
        key=lambda par: par[0],
    )
    fallos = []
    print(f"Ejecutando {len(pruebas)} pruebas del paquete ce_rna\n")
    for nombre, funcion in pruebas:
        try:
            funcion()
            print(f"  [OK]     {nombre}")
        except AssertionError as error:
            fallos.append((nombre, f"AssertionError: {error}"))
            print(f"  [FALLO]  {nombre}: {error}")
        except Exception as error:  # pragma: no cover
            fallos.append((nombre, f"{type(error).__name__}: {error}"))
            print(f"  [ERROR]  {nombre}: {type(error).__name__}: {error}")

    print(f"\n{len(pruebas) - len(fallos)} de {len(pruebas)} pruebas superadas")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
