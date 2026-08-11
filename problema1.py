# Problema 1: Generador Pseudoaleatorio Uniforme 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy import stats

# Implementación del generador LCG
class LCG:
    def __init__(self, a: int, c: int, m: int, seed: int = 1):
        assert 0 <= a < m and 0 <= c < m, "Se requiere 0 <= a, c < m"
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def sample_ints(self, n: int) -> np.ndarray:
        #Muestra uniforme finita x1, x2, ..., xn en {0, 1, ..., m-1}.
        return np.array([self.next_int() for _ in range(n)])

    def sample_uniform(self, n: int) -> np.ndarray:
        #Muestra de una U(0,1): u_i = x_i / m."""
        return self.sample_ints(n) / self.m


#Funciones de análisis: estadísticos, histogramas, pruebas
def resumen_estadistico(u: np.ndarray, nombre: str):
    print(f"\n Estadísticos descriptivos: {nombre} ")
    print(f"N          = {len(u)}")
    print(f"Media      = {u.mean():.5f}   (teórica: 0.5)")
    print(f"Varianza   = {u.var():.5f}   (teórica: {1/12:.5f})")
    print(f"Mínimo     = {u.min():.5f}")
    print(f"Máximo     = {u.max():.5f}")
    print(f"Asimetría  = {stats.skew(u):.5f}   (teórica: 0)")
    print(f"Curtosis   = {stats.kurtosis(u):.5f}   (teórica exceso: -1.2)")


def graficar(u: np.ndarray, nombre: str, archivo_hist: str, archivo_lag: str):
    # Histograma
    plt.figure(figsize=(6, 4))
    plt.hist(u, bins=30, density=True, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axhline(1.0, color="red", linestyle="--", label="Densidad teórica U(0,1)")
    plt.title(f"Histograma - {nombre}")
    plt.xlabel("u")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo_hist, dpi=120)
    plt.close()

    # Lag plot (u_n vs u_{n+1}) -> revela patrones/estructura reticular
    plt.figure(figsize=(5, 5))
    plt.scatter(u[:-1], u[1:], s=3, alpha=0.5, color="darkorange")
    plt.title(f"Diagrama de dispersión (u_n vs u_n+1) - {nombre}")
    plt.xlabel("u_n")
    plt.ylabel("u_n+1")
    plt.tight_layout()
    plt.savefig(archivo_lag, dpi=120)
    plt.close()


def graficar_3d(u: np.ndarray, nombre: str, archivo: str, n_puntos: int = 3000):
    #Grafica tripletas consecutivas (u_n, u_n+1, u_n+2) en 3D.
    x, y, z = u[:-2], u[1:-1], u[2:]
    x, y, z = x[:n_puntos], y[:n_puntos], z[:n_puntos]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=3, alpha=0.5, color="mediumseagreen")
    ax.set_title(f"Tripletas (u_n, u_n+1, u_n+2) - {nombre}")
    ax.set_xlabel("u_n")
    ax.set_ylabel("u_n+1")
    ax.set_zlabel("u_n+2")
    # Ángulo de vista que suele revelar los planos de RANDU
    ax.view_init(elev=20, azim=15)
    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    plt.close()


def prueba_kolmogorov_smirnov(u: np.ndarray, nombre: str, alpha: float = 0.05):
    D, p_value = stats.kstest(u, "uniform")
    print(f"\n--- Prueba Kolmogorov-Smirnov: {nombre} ---")
    print(f"Estadístico D = {D:.5f}")
    print(f"p-valor       = {p_value:.5f}")
    if p_value < alpha:
        print(f"Conclusión: se RECHAZA H0 (alpha={alpha}). La muestra NO parece U(0,1).")
    else:
        print(f"Conclusión: NO se rechaza H0 (alpha={alpha}). Consistente con U(0,1).")
    return D, p_value


def prueba_chi_cuadrado(u: np.ndarray, nombre: str, k: int = 20, alpha: float = 0.05):
    n = len(u)
    obs, edges = np.histogram(u, bins=k, range=(0, 1))
    esperado = n / k
    chi2_stat = np.sum((obs - esperado) ** 2 / esperado)
    df = k - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"\n*. Prueba Chi-cuadrado (k={k} intervalos): {nombre} .*")
    print(f"Estadístico chi2 = {chi2_stat:.5f}")
    print(f"Grados libertad  = {df}")
    print(f"p-valor          = {p_value:.5f}")
    if p_value < alpha:
        print(f"Conclusión: se RECHAZA H0 (alpha={alpha}). La muestra NO parece U(0,1).")
    else:
        print(f"Conclusión: NO se rechaza H0 (alpha={alpha}). Consistente con U(0,1).")
    return chi2_stat, p_value


def analizar_generador(a, c, m, seed, N, nombre, prefijo_archivo):

    print(f"-CONJUNTO: {nombre}  ->  a={a}, c={c}, m={m}")
  
    gen = LCG(a=a, c=c, m=m, seed=seed)
    u = gen.sample_uniform(N)

    resumen_estadistico(u, nombre)
    graficar(u, nombre,
             archivo_hist=f"{prefijo_archivo}_hist.png",
             archivo_lag=f"{prefijo_archivo}_lag.png")
    graficar_3d(u, nombre, archivo=f"{prefijo_archivo}_3d.png")
    prueba_kolmogorov_smirnov(u, nombre)
    prueba_chi_cuadrado(u, nombre)

    return u

#s utiliza para exportar las funciones para el ejercicio 3
if __name__ == "__main__":
    # Ejecución del laboratorio
    N = 10_000  # tamaño de muestra

    # Conjunto de parámetros 1: generador "bueno" (Numerical Recipes)
    u1 = analizar_generador(
        a=1664525, c=1013904223, m=2**32, seed=42,
        N=N, nombre="Conjunto 1 (Numerical Recipes)",
        prefijo_archivo="ejercicio1Figs/lcg_conjunto1",
    )

    # Conjunto de parámetros 2: generador "malo" (RANDU)
    u2 = analizar_generador(
        a=65539, c=0, m=2**31, seed=42,
        N=N, nombre="Conjunto 2 (RANDU)",
        prefijo_archivo="ejercicio1Figs/lcg_conjunto2",
    )

