# 07 --- Mapas autoorganizados de Kohonen

> Modulo: [`ce_rna/kohonen.py`](../ce_rna/kohonen.py)
> Experimento: [10](../experimentos/exp10_kohonen.py) ·
> Informe: [10](../resultados/reportes/10_kohonen.md)
> Fuente: guia de evaluacion, Actividad 4

---

## 1. Aprendizaje no supervisado

En el aprendizaje supervisado cada patron trae su salida deseada y el aprendizaje reduce
el error. En el **no supervisado** solo hay patrones: la red debe descubrir por si misma
como estan organizados. No hay error que retropropagar.

| | Supervisado (perceptron, ADALINE, MLP) | No supervisado (Kohonen) |
|---|---|---|
| Datos | pares (x, d) | solo x |
| Que aprende | correspondencia x → d | estructura de la distribucion de x |
| Senal de aprendizaje | error d − y | competencia entre neuronas |
| Evaluacion | error sobre datos de prueba | calidad de cuantizacion, topologia; interpretacion |

---

## 2. Arquitectura

- Capa de entrada de dimension m (aqui 2).
- Rejilla bidimensional de F x C neuronas, cada una con un **prototipo** w_j en el
  espacio de los datos y una **posicion fija** r_j en la rejilla.

Hay dos espacios: el de los datos (donde viven los prototipos) y el de la rejilla
(donde se mide la vecindad). El mapa es util porque aprende a hacerlos corresponder.

---

## 3. Algoritmo

Para cada iteracion t con un patron x al azar:

1. **Competicion**: c = argmin_j ‖x − w_j‖ (neurona ganadora).
2. **Cooperacion**: h_cj = exp(−‖r_c − r_j‖² / 2σ(t)²).
3. **Adaptacion**: w_j ← w_j + η(t) h_cj (x − w_j).

η(t) y σ(t) decaen exponencialmente. Dos fases:

- **Ordenamiento** (σ grande): el mapa se despliega sin pliegues.
- **Convergencia** (σ < 1): cada prototipo se ajusta a su region.

Sin vecindad (σ → 0) el algoritmo se reduce al aprendizaje competitivo simple
(equivalente a k-medias en linea): cuantiza bien pero pierde el orden topologico.

---

## 4. Lectura del mapa

| Herramienta | Que muestra |
|---|---|
| `impactos(X)` | patrones ganados por cada neurona; ceros = zonas vacias |
| `matriz_u()` | distancia media de cada prototipo a sus vecinas: valles = grupos, crestas = fronteras |
| `segmentar(X)` | grupos = componentes conexas de neuronas con U baja; crestas asignadas al valle mas cercano |
| `error_cuantizacion(X)` | distancia media patron → prototipo ganador |
| `error_topografico(X)` | fraccion de patrones cuya 1.a y 2.a neurona no son vecinas |

`pureza(grupos, clases)` compara con las clases reales **solo para evaluar**.

---

## 5. Resultados principales (experimento 10)

- Mapa 10x10 sobre 400 puntos en 5 grupos: encuentra los 5 grupos con pureza ≈ 0.99.
- Error topografico ≈ 0: la vecindad del plano se conserva en la rejilla.
- Mapa 5x5: demasiado pequeno para dibujar fronteras; solo encuentra 3 grupos.
- Sin vecindad: error topografico ≈ 0.95; la matriz U deja de revelar grupos.
- Grupos cuya separacion baja de ~3 desviaciones tipicas combinadas se funden: sin
  un hueco de baja densidad no hay frontera que detectar.

---

## 6. Uso

```python
from ce_rna import datasets as ds
from ce_rna.kohonen import MapaKohonen, pureza

datos = ds.grupos_plano()
mapa = MapaKohonen(10, 10, iteraciones=10000).entrenar(datos.X)
grupos = mapa.agrupar(datos.X)
print(mapa.resumen(datos.X), pureza(grupos, datos.d))
```
