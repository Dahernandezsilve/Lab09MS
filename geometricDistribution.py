import numpy as np
from scipy.stats import geom
from hypothesisTest import chiSquareTest, kstest 

def inverseTransformSampling(p, N):
    uniformRandoms = np.random.uniform(0, 1, N)
    empiricalSample = np.ceil(np.log(1 - uniformRandoms) / np.log(1 - p)).astype(int)
    return empiricalSample

# Parámetros de la distribución geométrica
p = 0.5
N = 200

geometricSample = geom.rvs(p, size=N)
inverseSample = inverseTransformSampling(p, N)
valores_unicos = np.union1d(geometricSample, inverseSample)
frecuencias_teorica = [np.sum(geometricSample == val) for val in valores_unicos]
frecuencias_empirica = [np.sum(inverseSample == val) for val in valores_unicos]

print("Frecuencias teóricas:")
print([int(val) for val in frecuencias_teorica])
print("Frecuencias empíricas:")
print([int(val) for val in frecuencias_empirica])

chiSquareTest(frecuencias_empirica, frecuencias_teorica)
kstest(inverseSample, geometricSample)
