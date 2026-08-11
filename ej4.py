import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(2026)
N = 1000
p = 0.3  # parámetro de la geométrica


def F_inv(u):
    u = np.asarray(u)

    return np.ceil(np.log(1-u) / np.log(1-p))


# Muestra teórica
sample_t = stats.geom.rvs(p, size=N)

# Muestra empírica
U = np.random.uniform(0, 1, N)
sample_e = F_inv(U).astype(int)

# Prueba de Kolmogorov-Smirnov
ks_stat, ks_pvalue = stats.ks_2samp(sample_e, sample_t)
print(f"KS: estadístico={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

# Prueba de Chi-cuadrado
k = 10
values, count = np.unique(np.clip(sample_e, 1, k), return_counts=True)
obs = np.zeros(k)
for v, c in zip(values, count):
    obs[int(v)-1] = c

probs = stats.geom.pmf(np.arange(1, k+1), p)
probs[-1] = 1 - stats.geom.cdf(k-1, p)
expected = probs * N

chi2_stat, chi2_pvalue = stats.chisquare(obs, expected)
print(f"Chi2: estadístico={chi2_stat:.4f}, p-value={chi2_pvalue:.4f}")

alpha = 0.05
print("No se rechaza H0" if chi2_pvalue > alpha and ks_pvalue > alpha else "Se rechaza H0")

# Visualización
bins_x = np.arange(1, k + 1)

values_t, count_t = np.unique(np.clip(sample_t, 1, k), return_counts=True)
obs_t = np.zeros(k)
for v, c in zip(values_t, count_t):
    obs_t[int(v) - 1] = c

width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(bins_x - width/2, obs / N, width=width,
        label="Empírica ($F^{-1}$)", alpha=0.85, color="steelblue")
plt.bar(bins_x + width/2, obs_t / N, width=width,
        label="Teórica (scipy.stats)", alpha=0.85, color="darkorange")
plt.plot(bins_x, probs, "ko-", markersize=5, label="PMF teórica")
plt.xlabel("x")
plt.ylabel("Frecuencia relativa")
plt.title(f"Geom(p={p})")
plt.xticks(bins_x)
plt.legend()
plt.tight_layout()
plt.show()
