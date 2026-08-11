#Ejercicio 2: Generador Mersenne Twister para Unif(0,1)
#Comparado contra la muestra teorica de scipy.stats 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Implementación del generador Mersenne Twister (MT19937)
## basado en el código de: https://github.com/yinengy/Mersenne-Twister-in-Python/blob/master/MT19937.py :D
# ---------------------------------------------------------------------------
# coeficientes estandar
(w, n, m, r) = (32, 624, 397, 31)
a = 0x9908B0DF
(u, d) = (11, 0xFFFFFFFF)
(s, b) = (7, 0x9D2C5680)
(t, c) = (15, 0xEFC60000)
l = 18
f = 1812433253


class MersenneTwister:
    def __init__(self, seed):
        self.MT = [0] * n
        self.index = n + 1
        self.lower_mask = 0x7FFFFFFF          # (1 << r) - 1
        self.upper_mask = 0x80000000          # bits altos de lower_mask
        self._seed(seed)

    def _seed(self, seed):
        self.MT[0] = seed & 0xFFFFFFFF
        for i in range(1, n):
            prev = self.MT[i - 1]
            self.MT[i] = (f * (prev ^ (prev >> (w - 2))) + i) & 0xFFFFFFFF
        self.index = n

    def _twist(self):
        for i in range(n):
            x = (self.MT[i] & self.upper_mask) + (self.MT[(i + 1) % n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= a
            self.MT[i] = self.MT[(i + m) % n] ^ xA
        self.index = 0

    def extract_number(self):
        if self.index >= n:
            self._twist()
        y = self.MT[self.index]
        y ^= (y >> u) & d
        y ^= (y << s) & b
        y ^= (y << t) & c
        y ^= (y >> l)
        self.index += 1
        return y & 0xFFFFFFFF

    def random(self):
        """Regresa un float pseudoaleatorio en [0, 1)"""
        return self.extract_number() / 4294967296.0  # / 2**32

    def uniform_sample(self, size):
        return np.array([self.random() for _ in range(size)])


# Estadística y funciones auxiliares
def resumen(nombre, x):
    print(f"--- {nombre} ---")
    print(f"  n         = {len(x)}")
    print(f"  media     = {np.mean(x):.6f}   (teorico: 0.500000)")
    print(f"  varianza  = {np.var(x, ddof=1):.6f}   (teorico: {1/12:.6f})")
    print(f"  minimo    = {np.min(x):.6f}")
    print(f"  maximo    = {np.max(x):.6f}")
    print(f"  skewness  = {stats.skew(x):.6f}   (teorico: 0)")
    print(f"  kurtosis  = {stats.kurtosis(x):.6f}   (teorico: -1.2)")
    print()


if __name__ == "__main__":
    N = 10_000          # tamaño de muestra
    SEED = 20260810

    mt = MersenneTwister(SEED)
    muestra_mt = mt.uniform_sample(N)

    rng_teorico = stats.uniform(loc=0, scale=1)
    muestra_teorica = rng_teorico.rvs(size=N, random_state=SEED)

    print("=" * 60)
    print("ESTADISTICOS DESCRIPTIVOS")
    print("=" * 60)
    resumen("Mersenne Twister (propio)", muestra_mt)
    resumen("Muestra teórica (scipy.stats.uniform)", muestra_teorica)


    # histogramas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(muestra_mt, bins=30, color="#4C72B0", edgecolor="white", density=True)
    axes[0].axhline(1.0, color="red", linestyle="--", label="densidad teorica")
    axes[0].set_title("Histograma - Mersenne Twister")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("densidad")
    axes[0].legend()

    axes[1].hist(muestra_teorica, bins=30, color="#55A868", edgecolor="white", density=True)
    axes[1].axhline(1.0, color="red", linestyle="--", label="densidad teorica")
    axes[1].set_title("Histograma - Muestra teorica scipy")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("densidad")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("./ex2_histogramas.png", dpi=150)
    plt.close()



    # pruebas de hipotesis para verificar correcta uniformidad 
    alpha = 0.05

    print("=" * 60)
    print("Prueba de hipótesis - Chi Cuadrado ")
    print("=" * 60)
    # H0: la muestra proviene de una Unif(0,1)
    # H1: la muestra no proviene de una Unif(0,1)
    k = 20  # numero de intervalos (clases)
    bordes = np.linspace(0, 1, k + 1)
    frec_obs, _ = np.histogram(muestra_mt, bins=bordes)
    frec_esp = np.full(k, N / k)

    chi2_stat, chi2_p = stats.chisquare(f_obs=frec_obs, f_exp=frec_esp)
    print(f"  H0: la muestra proviene de Unif(0,1)")
    print(f"  k (clases)      = {k}")
    print(f"  estadistico X^2 = {chi2_stat:.4f}")
    print(f"  p-value         = {chi2_p:.4f}")
    if chi2_p > alpha:
        print(f"  Conclusion: p-value > alpha={alpha} ;  no se rechaza H0.")
        print("  La muestra generada por el MT es consistente con Unif(0,1).")
    else:
        print(f"  Conclusion: p-value <= alpha={alpha} ; sí se rechaza H0.")
        print("  La muestra generada por el MT no es consistente con una Unif(0,1).")
    print()

    print("=" * 60)
    print("Prueba de hipótesis - Kolmogorov-Smirnov")
    print("=" * 60)
    # H0: la muestra proviene de una Unif(0,1)
    # H1: la muestra no proviene de una Unif(0,1)
    ks_stat, ks_p = stats.kstest(muestra_mt, "uniform")
    print(f"  H0: la muestra proviene de una Unif(0,1)")
    print(f"  estadistico D = {ks_stat:.4f}")
    print(f"  p-value       = {ks_p:.4f}")
    if ks_p > alpha:
        print(f"  Conclusion: p-value > alpha={alpha} ; no se rechaza H0.")
        print("  La muestra generada por el MT es consistente con Unif(0,1).")
    else:
        print(f"  Conclusion: p-value <= alpha={alpha} ; sí se rechaza H0.")
        print("  La muestra generada por el MT no es consistente con una Unif(0,1).")
    print()

    #comparación adicional: MT vs muestra teórica (dos muestras)
    print("=" * 60)
    print("KS de dos muestras: MT vs muestra teórica de scipy")
    print("=" * 60)
    ks2_stat, ks2_p = stats.ks_2samp(muestra_mt, muestra_teorica)
    print(f"  estadistico D = {ks2_stat:.4f}")
    print(f"  p-value       = {ks2_p:.4f}")
    if ks2_p > alpha:
        print("  No hay evidencia de diferencia entre ambas muestras.")
    else:
        print("  Sí hay evidencia de diferencia entre ambas muestras.")

    print("\nListo. Histogramas guardados en ex2_histogramas.png")
    