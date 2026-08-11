#Ejercicio 3: Tests NIST SP 800-22 (bateria sp800-22r1a con libreria "nistrng")
import numpy as np
import pandas as pd
from nistrng import (
    SP800_22R1A_BATTERY,
    check_eligibility_all_battery,
    run_all_battery,
)
from problema2 import MersenneTwister  


# 1. Generar 1,000,000 de bits con el Mersenne Twister
NUM_BITS = 1_000_000
SEED = 20260804

mt = MersenneTwister(SEED)

n_ints = -(-NUM_BITS // 32)  # techo(NUM_BITS/32)
enteros = [mt.extract_number() for _ in range(n_ints)] # cada llamada a extract_number() entrega 32 bits utilizables


# convertir enteros de 32 bits a su binario y concatenar
bits_str = "".join(format(x, "032b") for x in enteros)[:NUM_BITS]
bits_mt = np.array([int(b) for b in bits_str], dtype=int)

print(f"Bits generados con Mersenne Twister: {len(bits_mt):,}")
print(f"Proporcion de unos: {bits_mt.mean():.5f}  (esperado ~0.5)\n")


# 2. Ejecutar la bateria NIST SP800-22 (sub-conjunto elegible sp800-22r1a)
def correr_bateria_nist(bits: np.ndarray, nombre_generador: str) -> pd.DataFrame:
    pruebas_elegibles = check_eligibility_all_battery(bits, SP800_22R1A_BATTERY)
    print(f"[{nombre_generador}] Tests elegibles segun longitud de secuencia: "
          f"{len(pruebas_elegibles)} / {len(SP800_22R1A_BATTERY)}")

    resultados = run_all_battery(bits, pruebas_elegibles, False)

    filas = []
    for resultado, tiempo_ms in resultados:
        filas.append({
            "Generador": nombre_generador,
            "Test": resultado.name,
            "p-value": round(resultado.score, 6),
            "Pasa (alpha=0.01)": "Si" if resultado.passed else "No",
            "Tiempo (ms)": round(tiempo_ms, 2),
        })
    return pd.DataFrame(filas)


print("=" * 70)
print("EJECUTANDO BATERIA NIST SP800-22 SOBRE MERSENNE TWISTER")
print("=" * 70)
tabla_mt = correr_bateria_nist(bits_mt, "Mersenne Twister")

print("\nResultados:")
print(tabla_mt.to_string(index=False))

n_pasa = (tabla_mt["Pasa (alpha=0.01)"] == "Si").sum()
n_total = len(tabla_mt)
print(f"\nResumen: el Mersenne Twister paso {n_pasa} de {n_total} tests "
      f"({100*n_pasa/n_total:.1f}%).")

tabla_mt.to_csv("./ex3_resultados_nist_mt.csv", index=False)
print("\nTabla guardada en ex3_resultados_nist_mt.csv")
