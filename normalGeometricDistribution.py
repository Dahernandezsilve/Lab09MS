import numpy as np
import scipy.stats as stats
from hypothesisTest import chiSquareTest, kstest

def inverseTransformSampling(N, mu, sigma):
    # Muestra empírica usando el método de la transformada integral
    uniform_randoms = np.random.uniform(0, 1, N)
    muestra_empirica = stats.norm.ppf(uniform_randoms, loc=mu, scale=sigma)
    return muestra_empirica

# Parámetros de la distribución normal
mu = 0  # media
sigma = 1  # desviación estándar
N = 200  # tamaño de la muestra

# Muestra teórica usando scipy
muestra_teorica = stats.norm.rvs(loc=mu, scale=sigma, size=N)
muestra_empirica = inverseTransformSampling(N, mu, sigma)

# Agrupar en intervalos y calcular las frecuencias para ambas muestras
bins = 10
hist_range = (-3, 3)
freq_teorica, _ = np.histogram(muestra_teorica, bins=bins, range=hist_range)
freq_empirica, _ = np.histogram(muestra_empirica, bins=bins, range=hist_range)

# Añadir una corrección mínima para evitar ceros en las frecuencias
freq_teorica = np.where(freq_teorica == 0, 1e-10, freq_teorica)
freq_empirica = np.where(freq_empirica == 0, 1e-10, freq_empirica)

# Realizar la prueba de Chi Cuadrado con las frecuencias
chiSquareTest(freq_empirica, freq_teorica)

# La prueba de K-S puede realizarse directamente sobre las muestras
kstest(muestra_empirica, muestra_teorica)
