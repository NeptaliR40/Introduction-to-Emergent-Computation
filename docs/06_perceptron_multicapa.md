# 06 --- El perceptron multicapa y la retropropagacion

> Modulo: [`ce_rna/perceptron_multicapa.py`](../ce_rna/perceptron_multicapa.py)
> Experimentos: [08 (XOR)](../experimentos/exp08_mlp_xor.py) ·
> [09 (aproximacion de funciones)](../experimentos/exp09_mlp_aproximacion.py)
> Informes: [08](../resultados/reportes/08_mlp_xor.md) ·
> [09](../resultados/reportes/09_mlp_aproximacion.md)
> Fuente: guia de evaluacion, Actividad 3 (clases 5, 6 y 7)

---

## 1. El problema que resuelve

El perceptron simple y el ADALINE trazan **una sola** frontera lineal. El experimento 03
demostro que el XOR no admite ninguna, y el experimento 01 lo resolvio componiendo tres
neuronas de McCulloch-Pitts, pero con los pesos **puestos a mano**. Faltaba un algoritmo
que ajustara los pesos de una red con capas intermedias.

La dificultad es la *asignacion de credito*: el error se mide en la salida, pero las
neuronas ocultas no tienen salida deseada. ¿Cuanta culpa tiene cada peso oculto? La
retropropagacion (Rumelhart, Hinton y Williams, 1986) responde con la regla de la cadena.

---

## 2. Arquitectura

```
entrada           capa oculta             salida
 x1 ──┐      ┌──  h1 = φ(w·x) ──┐
      ├──────┤                  ├──  y = φ(w·h)
 x2 ──┘      └──  h2 = φ(w·x) ──┘
```

- Capas totalmente conectadas hacia adelante, cada neurona con su peso de sesgo.
- Activacion **sigmoidal** en las capas ocultas: φ(v) = 1/(1+e^−v), derivable en todo
  punto y con φ'(v) = φ(v)(1 − φ(v)).
- Salida sigmoidal para clasificar (objetivos en [0, 1]) o **lineal** para aproximar
  funciones (salida real sin acotar).

En el codigo, `W[l]` es una matriz `(n_{l+1}, n_l + 1)` cuya columna 0 es el sesgo.

---

## 3. Funcion de error y descenso por el gradiente

$$E = \frac{1}{N}\sum_{p} E^{p}, \qquad E^{p} = \tfrac{1}{2}\sum_k \left(d_k^{p} - y_k^{p}\right)^2$$

Es el mismo error del ADALINE, sumado sobre las salidas. Cada peso se mueve en contra
de la pendiente:

$$\Delta w = -\gamma\,\frac{\partial E}{\partial w}$$

---

## 4. Retropropagacion

Definiendo el gradiente local δ de cada neurona:

| Capa | Gradiente local |
|---|---|
| Salida | δ_k = (d_k − y_k) · φ'(v_k) |
| Oculta | δ_j = φ'(v_j) · Σ_k δ_k w_kj |

y la actualizacion de cada peso tiene la forma de la regla Delta:

$$\Delta w_{ji}^{(l)}(t) = \gamma\,\delta_j^{(l)}\,y_i^{(l-1)} + \alpha\,\Delta w_{ji}^{(l)}(t-1)$$

El error se **propaga hacia atras** por los mismos pesos que usa la propagacion hacia
adelante: cada neurona oculta recibe la suma de los errores de las neuronas a las que
alimenta, ponderados por sus pesos. `gradiente()` implementa exactamente esto en forma
matricial, y la prueba `probar_mlp_gradiente_numerico` comprueba que coincide con las
diferencias finitas de E(W) con error < 1e-7.

---

## 5. Parametros de entrenamiento

| Parametro | Papel | Efecto observado (exp. 08-09) |
|---|---|---|
| γ (razon de aprendizaje) | tamano del paso | pequeno → mesetas largas; grande → oscilacion y saturacion |
| α (momento) | inercia del paso | acelera las mesetas; paso efectivo γ/(1 − α) |
| neuronas ocultas | capacidad | XOR: 2 bastan, 3+ son mas robustas; f(x): hacen falta ~8 |
| modo | estocastico / lotes | lotes necesita γ mayor; estocastico tiene ruido util |
| escala inicial | pesos iniciales en [−a, a] | pequenos: sigmoides en su zona lineal |

### Criterios de parada

1. **Error objetivo**: E ≤ ε. Clasificar bien llega antes que un error pequeno; con
   sigmoides E → 0 exige pesos → ∞, asi que ε debe ser razonable.
2. **Maximo de epocas**: garantiza terminar aunque la red este en un minimo local.
3. **Parada temprana**: con un conjunto de validacion, se detiene cuando su error no
   mejora durante `paciencia` epocas y se restauran los mejores pesos. Evita el
   sobreajuste con datos ruidosos.

---

## 6. Limitaciones

- **Minimos locales**: el descenso por el gradiente es local. Con la 2-2-1 ~10 % de las
  inicializaciones no resuelve el XOR.
- **Mesetas**: zonas donde el gradiente es casi nulo; el error parece estancado.
- **Sobreajuste**: una red grande con pocos datos ruidosos aprende el ruido.
- **No extrapola**: fuera del dominio de entrenamiento las sigmoides se saturan.

---

## 7. Uso

```python
from ce_rna import datasets as ds
from ce_rna.activaciones import Identidad
from ce_rna.perceptron_multicapa import PerceptronMulticapa

xor = ds.compuerta("XOR")
red = PerceptronMulticapa([2, 2, 1], razon_aprendizaje=0.5, error_objetivo=0.005)
red.entrenar(xor.X, xor.d)
print(red.salida(xor.X))       # ~ [0.09 0.90 0.90 0.10]

ent, pru = ds.aproximacion_funcion()
red = PerceptronMulticapa([1, 8, 1], razon_aprendizaje=0.1, momento=0.5,
                          activacion_salida=Identidad(), escala_inicial=2.0,
                          max_epocas=8000, error_objetivo=1e-5)
red.entrenar(ent.X, ent.d, pru.X, pru.d)
```
