import numpy as np
import scipy.stats as stats
from hypothesisTest import chiSquareTest, kstest
from GLC import generador_glc

class MersenneTwister:
    def __init__(self, seed=5489):
        # Parámetros Mersenne Twister MT19937
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253
        
        self.upper_mask = 1 << (self.w - 1)
        self.lower_mask = (1 << (self.w - 1)) - 1
        
        # Inicializar el estado
        self.MT = [0] * self.n
        self.index = self.n + 1
        self.MT[0] = seed
        for i in range(1, self.n):
            self.MT[i] = (self.f * (self.MT[i-1] ^ (self.MT[i-1] >> (self.w-2))) + i) & self.d

    def twist(self):
        for i in range(self.n):
            x = (self.MT[i] & self.upper_mask) + (self.MT[(i+1) % self.n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        if self.index >= self.n:
            self.twist()
        
        y = self.MT[self.index]
        y ^= (y >> self.u)
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= (y >> self.l)
        
        self.index += 1
        return y & self.d

    def random(self):
        #Genera un número aleatorio en el intervalo [0, 1)
        return self.extract_number() / ((1 << self.w) - 1)

def generar_muestra_mt(seed, N):
    mt = MersenneTwister(seed)
    return np.array([mt.random() for _ in range(N)])

# Parámetros para las pruebas
param_sets_mt = [
    {"seed": 1, "N": 200},
    {"seed": 1, "N": 500},
    {"seed": 1, "N": 1000}
]

# Realizar pruebas para Mersenne Twister
print("\nPruebas para Mersenne Twister:")
for params in param_sets_mt:
    muestra_mt = generar_muestra_mt(**params)
    
    bins = 10
    freq_observada, _ = np.histogram(muestra_mt, bins=bins, range=(0, 1))
    freq_esperada = np.full(bins, len(muestra_mt) / bins)
    
    print(f"\nParámetros: seed={params['seed']}, N={params['N']}")
    chiSquareTest(freq_observada, freq_esperada)
    kstest(muestra_mt, np.random.uniform(0, 1, len(muestra_mt)).tolist())

# Comparación con el GLC
def comparar_generadores(N, seed):
    # Generar muestras
    mt = generar_muestra_mt(seed, N)
    glc = generador_glc(2**31 - 1, 48271, 0, seed, N)
    
    # Calcular estadísticas básicas
    print(f"\nComparación estadística para N={N}:")
    print("Mersenne Twister:")
    print(f"Media: {np.mean(mt):.4f}")
    print(f"Varianza: {np.var(mt):.4f}")
    print(f"Autocorrelación lag-1: {np.corrcoef(mt[:-1], mt[1:])[0,1]:.4f}")
    
    print("\nGLC:")
    print(f"Media: {np.mean(glc):.4f}")
    print(f"Varianza: {np.var(glc):.4f}")
    print(f"Autocorrelación lag-1: {np.corrcoef(glc[:-1], glc[1:])[0,1]:.4f}")
    
    return mt, glc

# Realizar comparación para diferentes tamaños de muestra
for N in [200, 500, 1000]:
    mt_sample, glc_sample = comparar_generadores(N, 12345)