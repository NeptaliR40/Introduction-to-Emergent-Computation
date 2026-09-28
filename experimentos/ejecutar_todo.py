"""
Ejecuta los trece experimentos en orden y regenera todas las figuras, tablas e
informes de `resultados/`.

    python experimentos/ejecutar_todo.py

El proceso es completamente determinista: todos los generadores aleatorios usan
semillas fijas, de modo que dos ejecuciones producen exactamente los mismos
numeros y las mismas figuras.
"""

from __future__ import annotations

import importlib
import time
import traceback

import _ruta  # noqa: F401
from _ruta import RAIZ

EXPERIMENTOS = [
    ("exp00_activaciones", "Catalogo de funciones de activacion"),
    ("exp01_mcculloch_pitts", "McCulloch-Pitts: compuertas AND y OR"),
    ("exp02_perceptron_and_or", "Perceptron simple: AND y OR"),
    ("exp03_perceptron_xor", "Perceptron simple: el limite del XOR"),
    ("exp04_perceptron_letras", "Perceptron multiclase: letras X y O"),
    ("exp05_adaline_decodificador", "ADALINE: descodificador binario-decimal"),
    ("exp06_perceptron_vs_adaline", "Perceptron frente a ADALINE"),
    ("exp07_hebb", "Regla de Hebb"),
    ("exp08_mlp_xor", "Perceptron multicapa: XOR"),
    ("exp09_mlp_aproximacion", "Perceptron multicapa: aproximacion de funciones"),
    ("exp10_kohonen", "Mapas autoorganizados de Kohonen"),
    ("exp11_hopfield", "Red de Hopfield: letras A, B, C, D"),
    ("exp12_imagenes_reales", "Imagenes reales: Hopfield frente a MLP entrenado"),
]


def main() -> int:
    inicio_total = time.time()
    fallos = []

    for modulo, descripcion in EXPERIMENTOS:
        inicio = time.time()
        try:
            importlib.import_module(modulo).main()
            print(f"\n[OK] {descripcion}  ({time.time() - inicio:.1f} s)")
        except Exception:  # pragma: no cover - diagnostico en consola
            fallos.append(modulo)
            print(f"\n[FALLO] {descripcion}")
            traceback.print_exc()

    print("\n" + "=" * 78)
    print(f"  {len(EXPERIMENTOS) - len(fallos)} de {len(EXPERIMENTOS)} experimentos completados "
          f"en {time.time() - inicio_total:.1f} s")
    if fallos:
        print(f"  Fallaron: {', '.join(fallos)}")
    else:
        print(f"  Resultados en: {RAIZ / 'resultados'}")
    print("=" * 78)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
