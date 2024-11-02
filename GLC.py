import numpy as np
import scipy.stats as stats
from hypothesisTest import chiSquareTest, kstest

def generador_glc(m, a, c, seed, N):
    # Inicializar lista para almacenar valores generados
    valores = []
    x = seed
    for _ in range(N):
        x = (a * x + c) % m
        valores.append(x / m)
    return np.array(valores)

# Definición de parámetros
param_sets = [
    {"m": 2**31 - 1, "a": 48271, "c": 0, "seed": 1, "N": 200},
    {"m": 2**25 - 1, "a": 1103515245, "c": 12345, "seed": 1, "N": 500},
    {"m": 2**14 - 1, "a": 214013, "c": 2531011, "seed": 1, "N": 1000},
]

for params in param_sets:
    muestra_glc = generador_glc(**params)
    
    # Dividir el intervalo [0, 1] en 10 bins para la prueba de Chi Cuadrado
    bins = 10
    freq_observada, _ = np.histogram(muestra_glc, bins=bins, range=(0, 1))
    freq_esperada = np.full(bins, len(muestra_glc) / bins)  # Frecuencia esperada uniforme

    # Llamar a la prueba de Chi Cuadrado
    print(f"\nParámetros: m={params['m']}, a={params['a']}, c={params['c']}, seed={params['seed']}")
    chiSquareTest(freq_observada, freq_esperada)

    # Llamar a la prueba de Kolmogorov-Smirnov con la distribución uniforme
    kstest(muestra_glc, np.random.uniform(0, 1, len(muestra_glc)).tolist())
    