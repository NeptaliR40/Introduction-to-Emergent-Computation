# Experimento 00 --- Catalogo de funciones de activacion

> Informe generado automaticamente por los scripts de `experimentos/` el 2026-09-28 11:37.
> No editar a mano: se regenera con `python experimentos/ejecutar_todo.py`.

Las funciones de activacion determinan *"el nivel de activacion de la neurona en terminos de la actividad existente en sus entradas"*. Este experimento dibuja el catalogo completo de `claseRN01.md` y comprueba una propiedad que condiciona todo el resto del repositorio: cuales son derivables y cuales no.

![Las seis funciones de activacion del catalogo del curso.](../figuras/00_activaciones.png)

*Figura: Las seis funciones de activacion del catalogo del curso.*

**Propiedades verificadas numericamente**

| activacion | minimo | maximo | valores distintos | derivable | |dphi analitica - numerica| |
|---|---|---|---|---|---|
| escalon | 0 | 1 | 2 | no | no derivable |
| escalon_bipolar | -1 | 1 | 2 | no | no derivable |
| escalon_bipolar_zona | -1 | 1 | 3 | no | no derivable |
| identidad | -4 | 4 | 2001 | si | 1.40e-10 |
| sigmoide | 0.0179862 | 0.982014 | 2001 | si | 1.27e-10 |
| sigmoide_bipolar | -0.964028 | 0.964028 | 2001 | si | 8.00e-11 |
| gaussiana | 0.0285655 | 1 | 1001 | si | 1.14e-10 |

*Datos completos: [`00_activaciones.csv`](../tablas/00_activaciones.csv)*

## Consecuencia para el aprendizaje

Las tres variantes del escalon tienen derivada nula en todo punto donde esta definida. Eso significa que **el descenso del gradiente no puede aplicarse** a una neurona con activacion escalon: el gradiente del error respecto a los pesos es identicamente cero y no senala ninguna direccion de mejora. De ahi las dos familias de reglas de aprendizaje del repositorio: las **discretas de correccion de error** (perceptron, Hebb), que no derivan de ningun gradiente, y las **de gradiente** (regla Delta del ADALINE), que exigen una salida derivable y por eso aprenden sobre la salida lineal, aplicando el umbral solo despues del aprendizaje.
