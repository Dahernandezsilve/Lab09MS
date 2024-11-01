from scipy.stats import geom
from hypothesisTest import chiSquareTest, kstest
import numpy as np

def inverseTransformSampling(p, N):
    uniformRandoms = np.random.uniform(0, 1, N)
    empiricalSample = np.ceil(np.log(1 - uniformRandoms) / np.log(1 - p)).astype(int)
    return empiricalSample

# Geometric Distribution Parameters
p = 0.5
N = 200

geometricSample = geom.rvs(p, size=N)
inverseSample = inverseTransformSampling(p, N)

# Theoretical and Empirical Samples
theoreticalSample = np.bincount(inverseSample)
empiricalSample = np.bincount(geometricSample)
theoreticalSample = theoreticalSample[:20]
empiricalSample = empiricalSample[:20]
theoreticalFrequency = theoreticalSample / 20
empiricalFrequency = empiricalSample / 20

chiSquareTest(theoreticalFrequency, empiricalFrequency)
kstest(theoreticalFrequency, empiricalFrequency)

