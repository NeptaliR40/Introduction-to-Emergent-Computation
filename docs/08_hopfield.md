# 08 --- La red de Hopfield como memoria asociativa

> Modulo: [`ce_rna/hopfield.py`](../ce_rna/hopfield.py)
> Experimento: [11](../experimentos/exp11_hopfield.py) ·
> Informe: [11](../resultados/reportes/11_hopfield.md)
> Fuente: guia de evaluacion, Actividad 5

---

## 1. Memoria direccionable por contenido

Una memoria de computador se consulta por **direccion**. Una memoria asociativa se
consulta por **contenido**: se le da una version parcial o ruidosa de lo que se busca y
devuelve el recuerdo completo. Hopfield (1982) mostro que una red recurrente de neuronas
binarias con pesos simetricos se comporta asi.

---

## 2. Arquitectura

- N neuronas bipolares (±1); para imagenes de 7x6, N = 42, una por pixel.
- Conexion total y recurrente: w_ij = w_ji, w_ii = 0 → N(N−1)/2 pesos distintos.
- El **estado** de la red es la imagen.

---

## 3. Almacenamiento

**Regla de Hebb** (una sola pasada, sin iteraciones):

$$w_{ij} = \frac{1}{N}\sum_{\mu} \xi_i^{\mu}\xi_j^{\mu}, \qquad w_{ii} = 0$$

**Regla de la pseudoinversa** (proyeccion): W = Ξ(ΞᵀΞ)⁻¹Ξᵀ. Tiene en cuenta las
correlaciones entre patrones y garantiza que cada patron sea un punto fijo.

---

## 4. Recuperacion y energia

Dinamica asincrona: s_i ← sgn(Σ_j w_ij s_j), una neurona cada vez, hasta que nada cambia.

$$E(\mathbf{s}) = -\tfrac{1}{2}\sum_{i,j} w_{ij} s_i s_j$$

Con pesos simetricos y diagonal nula, cada cambio reduce E, asi que la red siempre
converge a un minimo local. Los patrones almacenados son (idealmente) esos minimos.

---

## 5. Capacidad y diafonia

El campo local sobre el patron ν es senal + diafonia:

$$h_i \approx \xi_i^{\nu} + \sum_{\mu\neq\nu} \xi_i^{\mu} m_{\mu\nu}, \qquad m_{\mu\nu} = \tfrac{1}{N}\,\xi^{\mu}\cdot\xi^{\nu}$$

- Para patrones aleatorios la capacidad es P_max ≈ 0.138 N (≈ 5.8 con N = 42).
- Para patrones **correlacionados** (letras reales) la capacidad es mucho menor: el
  dibujo natural de B, C y D (solapamiento hasta 0.62) no se puede almacenar con Hebb
  ni siquiera con 4 patrones.

---

## 6. Errores de recuperacion

| Tipo | Origen |
|---|---|
| Otra letra | la imagen contaminada cayo en la cuenca de otro patron |
| Inverso −ξ | E(−ξ) = E(ξ): la simetria de la regla crea el atractor espejo |
| Espurio | minimos que no son ningun patron, p. ej. mezclas sgn(ξ¹ + ξ² + ξ³) |

---

## 7. Resultados principales (experimento 11)

- Letras de trazo grueso (solapamiento maximo 0.33): las cuatro son atractores con Hebb.
- Con 8 de 42 pixeles invertidos la recuperacion correcta media es ~87 %.
- Letras de trazo fino: solo la A es punto fijo con Hebb; la pseudoinversa las
  almacena todas.

---

## 8. Uso

```python
from ce_rna import datasets as ds
from ce_rna.hopfield import RedHopfield

letras = ds.letras_abcd()                   # 4 x 42, bipolar
red = RedHopfield(42).almacenar(letras.X)
ruidosa = ds.contaminar(letras.X[1], 8, semilla=0)
res = red.recuperar(ruidosa)
print(red.identificar(res.estado))          # (1, 'patron')  -> B
```
