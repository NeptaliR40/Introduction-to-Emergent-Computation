# 09 --- Seminario de investigacion: redes profundas en la busqueda de particulas exoticas

> Actividad 6 de la guia de evaluacion: resumen escrito y guion de la presentacion oral.

---

## 1. Referencia completa

Baldi, P., Sadowski, P. y Whiteson, D. (2014). *Searching for exotic particles in
high-energy physics with deep learning*. **Nature Communications**, 5, 4308.
https://doi.org/10.1038/ncomms5308 · Preimpresion: arXiv:1402.4735.

Datos publicos: conjuntos HIGGS y SUSY del UCI Machine Learning Repository. Codigo:
https://github.com/uci-igb/higgs-susy.

**Pertinencia de la fuente**: revista con arbitraje del grupo Nature; los autores
combinan aprendizaje automatico (Baldi, Sadowski, UC Irvine) y fisica de particulas
experimental (Whiteson, miembro de la colaboracion ATLAS). El articulo es una referencia
muy citada del uso de redes neuronales en fisica de altas energias, y los conjuntos de
datos que publico se usan todavia como banco de pruebas.

---

## 2. Resumen

En los colisionadores de particulas, descubrir una particula nueva exige distinguir
unos pocos sucesos de **senal** entre una cantidad enorme de sucesos de **fondo** que
dejan huellas parecidas en el detector: es un problema de clasificacion binaria. La
practica habitual era combinar variables "de alto nivel" disenadas por fisicos (masas
invariantes reconstruidas, etc.) con clasificadores "poco profundos": redes con una sola
capa oculta o arboles de decision potenciados (BDT). Los autores muestran que un
perceptron multicapa **profundo**, entrenado directamente con las variables medidas
("de bajo nivel"), aprende por si mismo representaciones al menos tan utiles como las
disenadas a mano y mejora la clasificacion de forma sustancial: segun el resumen del
articulo, hasta un 8 % en la metrica de clasificacion respecto de los mejores metodos
de entonces.

---

## 3. El problema

**Caso HIGGS**: la senal es la produccion de un boson de Higgs pesado hipotetico (H⁰,
de 425 GeV) que decae en cadena a traves de un Higgs cargado (H±, 325 GeV) hasta
W∓W±bb̄; el fondo es la produccion de pares de quarks top (tt̄ → W∓W±bb̄), con el
**mismo** estado final. Ambos procesos producen un leptón, neutrinos (energia
transversal faltante) y cuatro chorros (jets), dos de ellos de quarks b.

**Caso SUSY**: la senal es la produccion de pares de charginos supersimetricos
(χ± de 200 GeV, que decaen a W y un neutralino invisible χ⁰ de 100 GeV); el fondo es la
produccion de pares WW. Estado final: dos leptones cargados y energia faltante.

La dificultad: los sucesos de senal y fondo difieren solo en correlaciones sutiles entre
muchas variables, y un pequeno aumento en la capacidad de separarlos se traduce en una
ganancia importante de **significancia estadistica** del descubrimiento.

---

## 4. Datos y metodologia

| | HIGGS | SUSY |
|---|---|---|
| Sucesos simulados | 11 millones | 5 millones |
| Variables de bajo nivel | 21 (momentos pT, η, φ del leptón y los 4 jets, etiquetas b, energia faltante) | momentos de los leptones y energia faltante |
| Variables de alto nivel | 7 masas invariantes (m_ℓν, m_jj, m_bb, m_Wbb, m_WWbb, ...) | 12 (MT2, razor, super-razor, ...) |
| Particion (HIGGS) | 2.6 M entrenamiento, 100 000 validacion, 500 000 prueba | analoga |

Los sucesos se generaron por simulacion Monte Carlo del proceso fisico y de la
respuesta del detector. Las entradas se estandarizaron (media 0, desviacion 1).

Se comparan tres metodos sobre tres conjuntos de entradas (bajo nivel, alto nivel,
todas): BDT, red neuronal poco profunda (una capa oculta) y red profunda. La metrica es
el **area bajo la curva ROC (AUC)** y la **significancia de descubrimiento** esperada.

---

## 5. El modelo

Un **perceptron multicapa** --- la misma arquitectura de la Actividad 3 de este curso,
a mayor escala:

- Red totalmente conectada hacia adelante; el articulo la describe como *"a five-layer
  neural network with 300 hidden units in each layer"*.
- Activacion **tanh** en las unidades ocultas (la sigmoide bipolar, tanh(v/2), del
  catalogo de `claseRN01.md`, reescalada) y una neurona de salida sigmoidal que da la
  probabilidad de que el suceso sea senal.
- Entrenamiento con **retropropagacion y descenso por el gradiente estocastico** en
  mini-lotes de 100 sucesos; razon de aprendizaje inicial 0.05 que decae
  multiplicativamente hasta 10⁻⁶; **momento** que crece linealmente de 0.9 a 0.99 en las
  primeras 200 epocas; penalizacion de pesos (weight decay) 10⁻⁵.
- **Parada temprana** sobre el conjunto de validacion (entre 200 y 1000 epocas). Se
  probo preentrenamiento no supervisado con autocodificadores y no mejoro
  apreciablemente. Para SUSY se uso ademas *dropout*.
- Implementacion en Theano / Pylearn2 sobre GPU.

Todos los ingredientes (capas ocultas sigmoidales, funcion de error, descenso por el
gradiente, razon de aprendizaje, momento y criterio de parada por validacion) son los
estudiados en los experimentos 08 y 09 de este repositorio.

---

## 6. Resultados

**HIGGS --- AUC** (mayor es mejor; 0.5 = azar):

| Metodo | Bajo nivel | Alto nivel | Todas |
|---|---|---|---|
| BDT | 0.73 | 0.78 | 0.81 |
| Red poco profunda | 0.733 | 0.777 | 0.816 |
| **Red profunda** | **0.880** | 0.800 | **0.885** |

**HIGGS --- significancia de descubrimiento** esperada (100 sucesos de senal frente a
1000 ± 50 de fondo): red poco profunda 2.5σ / 3.1σ / 3.7σ; red profunda 4.9σ / 3.6σ /
5.0σ (bajo nivel / alto nivel / todas). El umbral convencional de "descubrimiento" en
fisica de particulas es 5σ.

**SUSY --- AUC**: BDT 0.863, red poco profunda 0.875, red profunda con dropout 0.879
(todas las variables). La mejora existe pero es mucho menor.

Lectura de los resultados:

1. Con variables de bajo nivel, la red profunda (0.880) supera a la red poco profunda
   **con** las variables de alto nivel disenadas por fisicos (0.777): las capas ocultas
   aprenden por si solas representaciones equivalentes o mejores.
2. Anadir las variables de alto nivel a la red profunda casi no mejora (0.880 → 0.885):
   ya no aportan informacion que la red no haya extraido.
3. En SUSY las variables de alto nivel ya capturaban casi toda la informacion, y la
   ventaja de la profundidad es pequena.

---

## 7. Ventajas, limitaciones y alcance

**Ventajas**
- Elimina (o reduce) la ingenieria manual de variables, costosa y dependiente de cada
  analisis.
- Mejora medible en la significancia estadistica sin cambiar el experimento: equivale a
  acumular mas datos.
- Metodologia reproducible: datos y codigo publicos.

**Limitaciones**
- Entrenada y evaluada sobre **simulaciones**: el rendimiento en datos reales depende de
  lo bien que la simulacion reproduce el detector.
- La red es una caja negra: es dificil saber *que* variables fisicas ha construido,
  lo que complica validar sistematicos.
- Necesita millones de ejemplos y GPU; la busqueda de hiperparametros es costosa.
- Estudio restringido a dos procesos concretos; la ganancia depende del problema (grande
  en HIGGS, pequena en SUSY).

**Alcance**: fue uno de los trabajos que impulsaron la adopcion del aprendizaje profundo
en los analisis de los experimentos del LHC y popularizo los conjuntos HIGGS/SUSY como
banco de pruebas en aprendizaje automatico.

---

## 8. Relacion con el curso

| Concepto del curso | En el articulo |
|---|---|
| Separabilidad lineal (act. 1) | senal y fondo no son linealmente separables en las variables medidas |
| Regla Delta / ECM (act. 2) | descenso por el gradiente sobre una funcion de error |
| MLP + backpropagation (act. 3) | el modelo completo, con capas ocultas tanh y salida sigmoidal |
| Criterio de parada (act. 3) | parada temprana por validacion |
| Capa oculta como nuevo espacio (exp. 08) | las capas aprenden variables equivalentes a las masas invariantes |
| Aprendizaje no supervisado (act. 4) | preentrenamiento con autocodificadores (probado, sin mejora) |

---

## 9. Conclusiones

- Un perceptron multicapa profundo es una herramienta eficaz para un problema cientifico
  real: la separacion senal/fondo en fisica de particulas.
- La profundidad permite aprender directamente de las medidas brutas representaciones
  que antes se disenaban a mano.
- Pequenos aumentos del AUC se traducen en ganancias grandes de significancia (de 3.7σ a
  5.0σ en HIGGS).
- El exito depende de la calidad de la simulacion y de disponer de muchos datos; la
  interpretabilidad sigue siendo el punto debil.

---

## 10. Guion de la presentacion oral (~15 min)

| # | Diapositiva | Contenido | Min |
|---|---|---|---|
| 1 | Titulo y referencia | articulo, autores, revista, DOI | 0.5 |
| 2 | Motivacion | como se descubre una particula: senal escasa entre mucho fondo | 1.5 |
| 3 | El problema HIGGS | diagrama de la cadena de decaimiento; mismo estado final que tt̄ | 2 |
| 4 | Variables | bajo nivel (21) frente a alto nivel (7); ingenieria manual | 1.5 |
| 5 | Poco profundo vs. profundo | una capa oculta frente a cinco; enlace con XOR/espacio oculto (exp. 08) | 2 |
| 6 | Entrenamiento | backprop, SGD, momento, decaimiento, parada temprana | 2 |
| 7 | Metrica | curva ROC, AUC y significancia (σ) | 1.5 |
| 8 | Resultados | tablas de AUC y significancia; lectura 1-2-3 | 2 |
| 9 | Limitaciones | simulacion, caja negra, coste; caso SUSY | 1 |
| 10 | Conclusiones y preguntas | resumen en 3 frases | 1 |
