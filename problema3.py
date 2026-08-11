##ejercicio 3 batería de evaluación
import numpy as np
import pandas as pd
from nistrng import (
    SP800_22R1A_BATTERY,
    check_eligibility_all_battery,
    run_all_battery,
)

from problema1 import LCG               # generador del ejercicio 1
from problema2 import MersenneTwister    # generador del ejercicio 2 

# 1. Generar 1,000,000 de bits para cada generador
NUM_BITS = 1_000_000
SEED = 20260804


def enteros_a_bits(enteros, bit_length, num_bits):
    """Convierte una lista de enteros a su representacion binaria de
    `bit_length` bits cada uno, concatena y trunca a `num_bits`."""
    bits_str = "".join(format(x, f"0{bit_length}b") for x in enteros)[:num_bits]
    return np.array([int(b) for b in bits_str], dtype=int)


def reporte_bits(nombre, bits):
    print(f"Bits generados con {nombre}: {len(bits):,}")
    print(f"Proporcion de unos: {bits.mean():.5f}  (esperado ~0.5)\n")


#  Mersenne Twister 
mt = MersenneTwister(SEED)
n_ints_mt = -(-NUM_BITS // 32)  # techo(NUM_BITS/32)
enteros_mt = [mt.extract_number() for _ in range(n_ints_mt)]
bits_mt = enteros_a_bits(enteros_mt, bit_length=32, num_bits=NUM_BITS)
reporte_bits("Mersenne Twister", bits_mt)

# LCG Conjunto 1: Numerical Recipes (a=1664525, c=1013904223, m=2**32) 
lcg1 = LCG(a=1664525, c=1013904223, m=2**32, seed=SEED)
n_ints_lcg1 = -(-NUM_BITS // 32)  # m=2**32 -> 32 bits por entero
enteros_lcg1 = lcg1.sample_ints(n_ints_lcg1).tolist()
bits_lcg1 = enteros_a_bits(enteros_lcg1, bit_length=32, num_bits=NUM_BITS)
reporte_bits("LCG Conjunto 1 (Numerical Recipes)", bits_lcg1)

#LCG Conjunto 2: RANDU (a=65539, c=0, m=2**31) 
lcg2 = LCG(a=65539, c=0, m=2**31, seed=SEED)
n_ints_lcg2 = -(-NUM_BITS // 31)  # m=2**31 -> 31 bits por entero
enteros_lcg2 = lcg2.sample_ints(n_ints_lcg2).tolist()
bits_lcg2 = enteros_a_bits(enteros_lcg2, bit_length=31, num_bits=NUM_BITS)
reporte_bits("LCG Conjunto 2 (RANDU)", bits_lcg2)



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
print("EJECUTANDO BATERIA NIST SP800-22 SOBRE LOS TRES GENERADORES")
print("=" * 70)
tabla_mt = correr_bateria_nist(bits_mt, "Mersenne Twister")
tabla_lcg1 = correr_bateria_nist(bits_lcg1, "LCG Conjunto 1 (Numerical Recipes)")
tabla_lcg2 = correr_bateria_nist(bits_lcg2, "LCG Conjunto 2 (RANDU)")

tabla_final = pd.concat([tabla_lcg1, tabla_lcg2, tabla_mt], ignore_index=True)

print("\nResultados:")
print(tabla_final.to_string(index=False))

print("\n" + "=" * 70)
print("RESUMEN COMPARATIVO (porcentaje de tests superados por generador)")
print("=" * 70)
resumen_generadores = (
    tabla_final.groupby("Generador")["Pasa (alpha=0.01)"]
    .apply(lambda col: (col == "Si").sum())
    .rename("Tests superados")
    .to_frame()
)
resumen_generadores["Total tests"] = tabla_final.groupby("Generador").size()
resumen_generadores["% superado"] = (
    100 * resumen_generadores["Tests superados"] / resumen_generadores["Total tests"]
).round(1)
resumen_generadores = resumen_generadores.sort_values("% superado", ascending=False)
print(resumen_generadores.to_string())

mejor = resumen_generadores.index[0]
print(f"\nConclusión: el generador con mejor desempeño en la batería NIST "
      f"SP800-22 es '{mejor}'.")

tabla_final.to_csv("./ex3_resultados_nist.csv", index=False)
