import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(2026)
N = 1000
mu, sigma = 5, 2  # parámetros de la normal


def F_inv(u):
    u = np.asarray(u)
    return stats.norm.ppf(u, mu, sigma)


# Muestra teórica
sample_t = stats.norm.rvs(mu, sigma, size=N)

# Muestra empírica
U = np.random.uniform(0, 1, N)
sample_e = F_inv(U)

# Prueba de Kolmogorov-Smirnov
ks_stat, ks_pvalue = stats.ks_2samp(sample_e, sample_t)
print(f"KS: estadístico={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

# Prueba de Chi-cuadrado
k = 10
cuantiles = np.linspace(0, 1, k + 1)
bordes = stats.norm.ppf(cuantiles, mu, sigma)
bordes[0], bordes[-1] = -np.inf, np.inf

obs, _ = np.histogram(sample_e, bins=bordes)
expected = np.full(k, N / k)

chi2_stat, chi2_pvalue = stats.chisquare(obs, expected)
print(f"Chi2: estadístico={chi2_stat:.4f}, p-value={chi2_pvalue:.4f}")

alpha = 0.05
print("No se rechaza H0" if chi2_pvalue > alpha and ks_pvalue > alpha else "Se rechaza H0")

# Visualización
plt.figure(figsize=(7, 5))
bins_hist = 30
plt.hist(sample_e, bins=bins_hist, density=True, alpha=0.6,
         label="Empírica ($F^{-1}$)", color="steelblue")
plt.hist(sample_t, bins=bins_hist, density=True, alpha=0.6,
         label="Teórica (scipy.stats)", color="darkorange")

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
plt.plot(x, stats.norm.pdf(x, mu, sigma), "k-", linewidth=2, label="PDF teórica")

plt.xlabel("x")
plt.ylabel("Densidad")
plt.title(f"Normal($\\mu$={mu}, $\\sigma$={sigma})")
plt.legend()
plt.tight_layout()
plt.show()
