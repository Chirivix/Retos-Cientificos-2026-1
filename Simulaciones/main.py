"""
main.py — Flujo principal de análisis INERLAB-R
================================================
Ejecuta el pipeline completo:
  1. Genera (o carga) datos de Tracker
  2. Calcula α por regresión lineal de ω(t)
  3. Calcula I_exp con corrección por fricción e inercia del aparato
  4. Calcula I_teo (Exp. 1: M·R²  /  Exp. 2: M·(R1²+R2²))
  5. Propaga incertidumbres
  6. Aplica prueba t-Student por radio
  7. Genera las cuatro gráficas
  8. Exporta CSV, TXT y JSON

INSTRUCCIONES DE USO:
─────────────────────
  A) Con datos reales de Tracker:
     1. Exporte los CSV de Tracker (una medición por archivo).
     2. En la sección "CONFIGURACIÓN" edite:
        - RUTAS_EXP1: lista de listas de rutas CSV por radio
        - RUTAS_EXP2: análogo para el Experimento 2
        - Parámetros físicos (M, m, r, g, m_friccion, I_aparato)
        - USE_DATOS_EJEMPLO = False
     3. Ejecute:  python main.py

  B) Con datos de ejemplo (prueba sin Tracker):
     - Deje USE_DATOS_EJEMPLO = True  →  se generan CSVs sintéticos.

Autores: Haniel Chaves, Felipe Chirivi, Tatiana Valero
Universidad Industrial de Santander — Física 1, 2026
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Importar módulos del proyecto ──────────────────────────────────────────────
from modelo import calcular_inercia_exp, descontar_inercia_aparato
from teorico import (
    inercia_masa_puntual,
    inercia_dos_masas,
    error_porcentual,
)
from datos import (
    generar_csv_ejemplo,
    cargar_tracker,
    extraer_series_temporales,
    calcular_alpha_regresion,
    cargar_multiples_mediciones,
    alpha_promedio,
)
from incertidumbre import (
    propagar_incertidumbre,
    prueba_t_student,
    resumen_incertidumbres,
)
from graficas import (
    grafica_I_vs_R2,
    grafica_paridad,
    grafica_omega_t,
    grafica_error_porcentual,
)
from reporte import exportar_csv, exportar_txt, exportar_json


# ══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURACIÓN DEL EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

# ── Modo de ejecución ─────────────────────────────────────────────────────────
USE_DATOS_EJEMPLO = True          # True → genera CSVs sintéticos para prueba
                                  # False → carga CSVs reales de Tracker

# ── Parámetros físicos del sistema ────────────────────────────────────────────
G          = 9.780       # Aceleración gravitacional en Bucaramanga [m/s²]
M_PUNTUAL  = 0.200       # Masa puntual del carro [kg]
M_COLGANTE = 0.050       # Masa de aceleración colgante [kg]
R_POLEA    = 0.015       # Radio de la polea [m]
M_FRICCION = 0.003       # Masa de fricción determinada empíricamente [kg]
I_APARATO  = 2.5e-5      # Inercia del aparato sin masa (medida aparte) [kg·m²]

# ── Incertidumbres instrumentales ─────────────────────────────────────────────
SIGMA_M    = 0.0001      # Incertidumbre de masas (balanza 0.1 g → σ = 0.1/√3 g) [kg]
SIGMA_R    = 0.0005      # Incertidumbre del calibrador [m]
SIGMA_A    = 0.001       # Incertidumbre de aceleración lineal [m/s²]

# ── Experimento 1: radios evaluados ──────────────────────────────────────────
RADIOS_EXP1 = [0.05, 0.10, 0.15, 0.20]   # [m]

# ── Experimento 2: pares (R1, R2) con R1+R2 = constante ──────────────────────
PARES_EXP2 = [
    (0.05, 0.15),
    (0.07, 0.13),
    (0.10, 0.10),
    (0.12, 0.08),
]

# ── Rutas a los CSVs reales de Tracker (solo si USE_DATOS_EJEMPLO = False) ───
# Cada elemento es una lista de mediciones independientes para ese radio.
# Ejemplo: RUTAS_EXP1[0] → 3 repeticiones para R = 5 cm
RUTAS_EXP1 = [
    ["datos/exp1_R5_med1.csv", "datos/exp1_R5_med2.csv", "datos/exp1_R5_med3.csv"],
    ["datos/exp1_R10_med1.csv", "datos/exp1_R10_med2.csv", "datos/exp1_R10_med3.csv"],
    ["datos/exp1_R15_med1.csv", "datos/exp1_R15_med2.csv", "datos/exp1_R15_med3.csv"],
    ["datos/exp1_R20_med1.csv", "datos/exp1_R20_med2.csv", "datos/exp1_R20_med3.csv"],
]
RUTAS_EXP2 = [
    ["datos/exp2_par1_med1.csv", "datos/exp2_par1_med2.csv"],
    ["datos/exp2_par2_med1.csv", "datos/exp2_par2_med2.csv"],
    ["datos/exp2_par3_med1.csv", "datos/exp2_par3_med2.csv"],
    ["datos/exp2_par4_med1.csv", "datos/exp2_par4_med2.csv"],
]

# ── Directorio de salida ──────────────────────────────────────────────────────
DIR_SALIDA = Path("resultados")
DIR_SALIDA.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#                               FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def preparar_datos_ejemplo() -> tuple[list, list]:
    """
    Genera CSVs sintéticos para ambos experimentos.

    Retorna: (rutas_exp1, rutas_exp2) — misma estructura que RUTAS_EXP1/2.
    """
    from datos import generar_csv_ejemplo

    dir_ej = Path("datos_ejemplo")

    # Alphas esperadas según I = M*R²  →  α = r*m_ef*(g-a) / I
    # (aproximación: a≈0 para estimar α_referencia)
    m_ef = M_COLGANTE - M_FRICCION

    rutas_exp1 = []
    for R in RADIOS_EXP1:
        I_ref = M_PUNTUAL * R**2 + I_APARATO
        # τ = r * m_ef * g  (ignorando a para la estimación)
        alpha_ref = R_POLEA * m_ef * G / I_ref
        rutas_radio = []
        for rep in range(3):
            ruta = dir_ej / f"exp1_R{int(R*100):02d}_med{rep+1}.csv"
            generar_csv_ejemplo(
                ruta, alpha_real=alpha_ref,
                t_max=3.0, n_puntos=60, ruido=0.04 + 0.01 * rep
            )
            rutas_radio.append(str(ruta))
        rutas_exp1.append(rutas_radio)

    rutas_exp2 = []
    for R1, R2 in PARES_EXP2:
        I_ref = M_PUNTUAL * (R1**2 + R2**2) + I_APARATO
        alpha_ref = R_POLEA * m_ef * G / I_ref
        rutas_par = []
        for rep in range(3):
            ruta = dir_ej / f"exp2_R1_{int(R1*100):02d}_R2_{int(R2*100):02d}_med{rep+1}.csv"
            generar_csv_ejemplo(
                ruta, alpha_real=alpha_ref,
                t_max=3.0, n_puntos=60, ruido=0.04 + 0.01 * rep
            )
            rutas_par.append(str(ruta))
        rutas_exp2.append(rutas_par)

    return rutas_exp1, rutas_exp2


def procesar_radio(rutas_radio: list[str]) -> dict:
    """
    Procesa todas las mediciones de un radio.

    Estrategia de cálculo de `a` y `sigma_a` (en orden de preferencia):
      1. Ajuste cuadrático de θ(t)  →  a = α·r directamente desde la regresión;
         sigma_a proviene de la covarianza del ajuste  (MEJOR — método directo).
      2. Regresión lineal de ω(t)   →  alpha_mean * R_POLEA;
         sigma_a = sigma_alpha * R_POLEA  (fallback si no hay columna θ).

    En cualquier caso, sigma_a NO es un valor fijo instrumental, sino que
    refleja la dispersión real de los datos de video.

    Parámetros
    ----------
    rutas_radio : list[str] — Archivos CSV de las repeticiones (mínimo 3)

    Retorna
    -------
    dict con: alpha_mean, sigma_alpha, a, sigma_a, mediciones, metodo_a
    """
    if len(rutas_radio) < 3:
        print(
            f"  ⚠️  Solo {len(rutas_radio)} repetición(es) para este radio. "
            "Se recomiendan ≥ 3 para una prueba t-Student confiable."
        )

    mediciones = cargar_multiples_mediciones(rutas_radio)
    if not mediciones:
        raise RuntimeError("No se procesaron mediciones válidas.")

    # ── Intentar método 1: ajuste cuadrático de θ(t) ─────────────────────────
    metodo_a = "omega_lineal"
    a_vals, sigma_a_vals, alpha_vals, sigma_alpha_vals = [], [], [], []

    for ruta in rutas_radio:
        try:
            from datos import cargar_tracker, extraer_series_temporales, calcular_a_desde_posicion
            df = cargar_tracker(ruta)
            t, omega, theta = extraer_series_temporales(df)

            if theta is not None and len(theta) >= 4:
                res_pos = calcular_a_desde_posicion(t, theta, R_POLEA)
                a_vals.append(res_pos["a"])
                sigma_a_vals.append(res_pos["sigma_a"])
                alpha_vals.append(res_pos["alpha"])
                sigma_alpha_vals.append(res_pos["sigma_alpha"])
                metodo_a = "posicion_cuadratica"
            else:
                raise ValueError("Sin columna θ")
        except Exception:
            # Fallback: usar resultado de calcular_alpha_regresion
            pass

    if metodo_a == "posicion_cuadratica" and len(a_vals) == len(rutas_radio):
        alpha_mean = float(np.mean(alpha_vals))
        sigma_alpha = float(np.std(alpha_vals, ddof=1)) if len(alpha_vals) > 1 else sigma_alpha_vals[0]
        a = float(np.mean(a_vals))
        sigma_a = float(np.mean(sigma_a_vals))          # promedio de σ_a individuales
    else:
        # Fallback: α de regresión ω(t), σ_a propagada
        alpha_mean, sigma_alpha = alpha_promedio(mediciones)
        a = alpha_mean * R_POLEA
        sigma_a = sigma_alpha * R_POLEA
        metodo_a = "omega_lineal (fallback)"

    return {
        "alpha_mean": alpha_mean,
        "sigma_alpha": sigma_alpha,
        "a": a,
        "sigma_a": sigma_a,
        "mediciones": mediciones,
        "metodo_a": metodo_a,
    }


# ══════════════════════════════════════════════════════════════════════════════
#                           FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  INERLAB-R — Sistema de Inercia Rotacional")
    print("  Universidad Industrial de Santander — Física 1, 2026")
    print("=" * 65 + "\n")

    # ── 0. Preparar datos ─────────────────────────────────────────────────────
    if USE_DATOS_EJEMPLO:
        print("[main] Modo: DATOS DE EJEMPLO (CSVs sintéticos)\n")
        rutas_exp1, rutas_exp2 = preparar_datos_ejemplo()
    else:
        print("[main] Modo: DATOS REALES de Tracker\n")
        rutas_exp1, rutas_exp2 = RUTAS_EXP1, RUTAS_EXP2

    # ═══════════════════════════════════════════════════════════════
    #  EXPERIMENTO 1 — Momento de inercia de masa puntual
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  EXPERIMENTO 1 — I = M·R²")
    print("─" * 65)

    alphas_exp1, sigma_alphas_exp1 = [], []
    I_exp_exp1, sigma_I_exp1, I_teo_exp1, errores_exp1 = [], [], [], []
    mediciones_t_exp1 = []       # para prueba t: lista de listas de I_exp
    regresiones_exp1 = []        # para graficar ω(t)

    for i, (R, rutas) in enumerate(zip(RADIOS_EXP1, rutas_exp1)):
        print(f"\n  Radio R = {R*100:.1f} cm")
        proc = procesar_radio(rutas)

        alpha = proc["alpha_mean"]
        s_alpha = proc["sigma_alpha"]
        a = proc["a"]
        sigma_a_din = proc["sigma_a"]          # ← dinámica, no fija

        alphas_exp1.append(alpha)
        sigma_alphas_exp1.append(s_alpha)
        regresiones_exp1.append(proc["mediciones"][0])  # primera rep para graficar

        # Calcular I_exp total y descontar aparato
        I_total = calcular_inercia_exp(
            m=M_COLGANTE, g=G, r=R_POLEA, a=a, alpha=alpha,
            m_friccion=M_FRICCION
        )
        I_neta = descontar_inercia_aparato(I_total, I_APARATO)

        # Propagar incertidumbre con sigma_a dinámico
        prop = propagar_incertidumbre(
            m=M_COLGANTE - M_FRICCION, sigma_m=SIGMA_M,
            r=R_POLEA, sigma_r=SIGMA_R,
            g=G, a=a, sigma_a=sigma_a_din,
            alpha=alpha, sigma_alpha=s_alpha,
        )
        s_I = prop["sigma_I"]

        # Valor teórico
        I_t = inercia_masa_puntual(M_PUNTUAL, R)
        err = error_porcentual(I_neta, I_t)

        print(f"    método a: {proc['metodo_a']}")
        print(f"    α = {alpha:.4f} ± {s_alpha:.4f} rad/s²")
        print(f"    a = {a:.5f} ± {sigma_a_din:.5f} m/s²  (σ_a dinámica)")
        print(f"    I_exp = {I_neta:.6f}  σ = {s_I:.6f} kg·m²  (σ_rel = {prop['sigma_rel_pct']:.2f}%)")
        print(f"    I_teo = {I_t:.6f} kg·m²   Error = {err:.2f}%")
        print("    " + resumen_incertidumbres(prop))

        I_exp_exp1.append(I_neta)
        sigma_I_exp1.append(s_I)
        I_teo_exp1.append(I_t)
        errores_exp1.append(err)

        # Guardar I_exp de cada repetición para prueba t
        I_por_rep = []
        for med in proc["mediciones"]:
            a_rep = med["alpha"] * R_POLEA
            I_rep_total = calcular_inercia_exp(
                m=M_COLGANTE, g=G, r=R_POLEA, a=a_rep, alpha=med["alpha"],
                m_friccion=M_FRICCION
            )
            I_por_rep.append(descontar_inercia_aparato(I_rep_total, I_APARATO))
        mediciones_t_exp1.append(I_por_rep)

    # ── Prueba t-Student por radio (Exp. 1) ──────────────────────────────────
    print("\n  ► Pruebas t-Student — Experimento 1")
    resultados_t_exp1 = []
    for i, (I_reps, I_t) in enumerate(zip(mediciones_t_exp1, I_teo_exp1)):
        if len(I_reps) >= 2:
            res_t = prueba_t_student(I_reps, I_t)
            print(f"\n  R = {RADIOS_EXP1[i]*100:.1f} cm: {res_t['conclusion']}")
            resultados_t_exp1.append(res_t)
        else:
            print(f"\n  R = {RADIOS_EXP1[i]*100:.1f} cm: Solo 1 repetición — no se aplica t")
            resultados_t_exp1.append(None)

    # ── Análisis M_exp desde pendiente de I vs R² ────────────────────────────
    print("\n  ► Extracción de M_exp desde ajuste lineal I vs R²")
    R2_arr = np.array(RADIOS_EXP1) ** 2
    I_arr  = np.array(I_exp_exp1)
    if len(R2_arr) >= 2:
        slope_m, intercept_m, r_val, _, se_slope = stats.linregress(R2_arr, I_arr)
        r2_fit = r_val ** 2
        error_M = abs(slope_m - M_PUNTUAL) / M_PUNTUAL * 100
        print(f"    Pendiente del ajuste = M_exp = {slope_m*1000:.2f} ± {se_slope*1000:.2f} g")
        print(f"    M real (balanza)     = {M_PUNTUAL*1000:.2f} g")
        print(f"    Diferencia           = {error_M:.2f}%   R² del ajuste = {r2_fit:.5f}")

    # ── Gráficas Experimento 1 ────────────────────────────────────────────────
    print("\n  ► Generando gráficas Experimento 1…")
    grafica_I_vs_R2(
        RADIOS_EXP1, I_exp_exp1, sigma_I_exp1, M_PUNTUAL, I_teo_exp1,
        guardar=DIR_SALIDA / "exp1_I_vs_R2.png"
    )
    grafica_paridad(
        I_exp_exp1, I_teo_exp1, sigma_I_exp1,
        etiquetas=[f"R={r*100:.0f} cm" for r in RADIOS_EXP1],
        titulo="Paridad I_exp vs I_teo — Experimento 1",
        guardar=DIR_SALIDA / "exp1_paridad.png"
    )
    grafica_error_porcentual(
        RADIOS_EXP1, errores_exp1,
        guardar=DIR_SALIDA / "exp1_errores.png"
    )
    # Gráfica ω(t) del primer radio como ejemplo
    grafica_omega_t(
        regresiones_exp1[0],
        radio_label=f"R = {RADIOS_EXP1[0]*100:.0f} cm",
        guardar=DIR_SALIDA / "exp1_omega_t_R05.png"
    )

    # ─── Exportar resultados Experimento 1 ───────────────────────────────────
    exportar_csv(
        RADIOS_EXP1, I_exp_exp1, I_teo_exp1, sigma_I_exp1, errores_exp1,
        alphas=alphas_exp1, sigma_alphas=sigma_alphas_exp1,
        experimento="Experimento 1 — Masa puntual",
        ruta=DIR_SALIDA / "exp1_resultados.csv",
    )
    res_t_clean = [r for r in resultados_t_exp1 if r is not None]
    exportar_txt(
        RADIOS_EXP1, I_exp_exp1, I_teo_exp1, sigma_I_exp1, errores_exp1,
        resultados_t=res_t_clean,
        experimento="Experimento 1 — Masa puntual",
        parametros_fisicos={
            "M_puntual [kg]": M_PUNTUAL,
            "m_colgante [kg]": M_COLGANTE,
            "r_polea [m]": R_POLEA,
            "m_friccion [kg]": M_FRICCION,
            "I_aparato [kg·m²]": I_APARATO,
            "g [m/s²]": G,
        },
        ruta=DIR_SALIDA / "exp1_reporte.txt",
    )

    # ═══════════════════════════════════════════════════════════════
    #  EXPERIMENTO 2 — Teorema de ejes paralelos
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  EXPERIMENTO 2 — Teorema de Ejes Paralelos  I = M(R1²+R2²)")
    print("─" * 65)

    alphas_exp2, sigma_alphas_exp2 = [], []
    I_exp_exp2, sigma_I_exp2, I_teo_exp2, errores_exp2 = [], [], [], []
    mediciones_t_exp2 = []

    for i, ((R1, R2), rutas) in enumerate(zip(PARES_EXP2, rutas_exp2)):
        print(f"\n  Par (R1={R1*100:.1f} cm, R2={R2*100:.1f} cm)")
        proc = procesar_radio(rutas)

        alpha = proc["alpha_mean"]
        s_alpha = proc["sigma_alpha"]
        a = proc["a"]
        sigma_a_din = proc["sigma_a"]

        alphas_exp2.append(alpha)
        sigma_alphas_exp2.append(s_alpha)

        I_total = calcular_inercia_exp(
            m=M_COLGANTE, g=G, r=R_POLEA, a=a, alpha=alpha,
            m_friccion=M_FRICCION
        )
        I_neta = descontar_inercia_aparato(I_total, I_APARATO)

        prop = propagar_incertidumbre(
            m=M_COLGANTE - M_FRICCION, sigma_m=SIGMA_M,
            r=R_POLEA, sigma_r=SIGMA_R,
            g=G, a=a, sigma_a=sigma_a_din,
            alpha=alpha, sigma_alpha=s_alpha,
        )
        s_I = prop["sigma_I"]

        I_t = inercia_dos_masas(M_PUNTUAL, R1, R2)
        err = error_porcentual(I_neta, I_t)

        print(f"    α = {alpha:.4f} ± {s_alpha:.4f} rad/s²")
        print(f"    I_exp = {I_neta:.6f}  σ = {s_I:.6f} kg·m²")
        print(f"    I_teo = {I_t:.6f} kg·m²   Error = {err:.2f}%")

        I_exp_exp2.append(I_neta)
        sigma_I_exp2.append(s_I)
        I_teo_exp2.append(I_t)
        errores_exp2.append(err)

        I_por_rep = []
        for med in proc["mediciones"]:
            a_rep = med["alpha"] * R_POLEA
            I_rep_total = calcular_inercia_exp(
                m=M_COLGANTE, g=G, r=R_POLEA, a=a_rep, alpha=med["alpha"],
                m_friccion=M_FRICCION
            )
            I_por_rep.append(descontar_inercia_aparato(I_rep_total, I_APARATO))
        mediciones_t_exp2.append(I_por_rep)

    # ── Prueba t-Student por par (Exp. 2) ────────────────────────────────────
    print("\n  ► Pruebas t-Student — Experimento 2")
    resultados_t_exp2 = []
    for i, (I_reps, I_t) in enumerate(zip(mediciones_t_exp2, I_teo_exp2)):
        if len(I_reps) >= 2:
            res_t = prueba_t_student(I_reps, I_t)
            R1, R2 = PARES_EXP2[i]
            print(f"\n  (R1={R1*100:.1f}, R2={R2*100:.1f} cm): {res_t['conclusion']}")
            resultados_t_exp2.append(res_t)

    # ─── Gráficas Experimento 2 ───────────────────────────────────────────────
    print("\n  ► Generando gráficas Experimento 2…")
    etiquetas_exp2 = [f"R1={r1*100:.0f}+R2={r2*100:.0f}" for r1, r2 in PARES_EXP2]
    grafica_paridad(
        I_exp_exp2, I_teo_exp2, sigma_I_exp2,
        etiquetas=etiquetas_exp2,
        titulo="Paridad I_exp vs I_teo — Experimento 2 (Steiner)",
        guardar=DIR_SALIDA / "exp2_paridad.png"
    )
    radios_equiv = [np.sqrt(R1**2 + R2**2) for R1, R2 in PARES_EXP2]
    grafica_error_porcentual(
        radios_equiv, errores_exp2,
        titulo="Error porcentual — Experimento 2",
        guardar=DIR_SALIDA / "exp2_errores.png"
    )

    # ─── Exportar Experimento 2 ───────────────────────────────────────────────
    exportar_csv(
        radios_equiv, I_exp_exp2, I_teo_exp2, sigma_I_exp2, errores_exp2,
        alphas=alphas_exp2, sigma_alphas=sigma_alphas_exp2,
        experimento="Experimento 2 — Teorema Steiner",
        ruta=DIR_SALIDA / "exp2_resultados.csv",
    )
    res_t2_clean = [r for r in resultados_t_exp2 if r is not None]
    exportar_txt(
        radios_equiv, I_exp_exp2, I_teo_exp2, sigma_I_exp2, errores_exp2,
        resultados_t=res_t2_clean,
        experimento="Experimento 2 — Teorema de Ejes Paralelos",
        parametros_fisicos={
            "M_por_masa [kg]": M_PUNTUAL,
            "m_colgante [kg]": M_COLGANTE,
            "r_polea [m]": R_POLEA,
            "m_friccion [kg]": M_FRICCION,
            "I_aparato [kg·m²]": I_APARATO,
            "g [m/s²]": G,
        },
        ruta=DIR_SALIDA / "exp2_reporte.txt",
    )

    # ═══════════════════════════════════════════════════════════════
    #  JSON GLOBAL
    # ═══════════════════════════════════════════════════════════════
    exportar_json(
        {
            "parametros": {
                "G": G, "M_puntual": M_PUNTUAL,
                "m_colgante": M_COLGANTE, "r_polea": R_POLEA,
                "m_friccion": M_FRICCION, "I_aparato": I_APARATO,
            },
            "experimento_1": {
                "radios": RADIOS_EXP1,
                "I_exp": I_exp_exp1,
                "I_teo": I_teo_exp1,
                "sigma_I": sigma_I_exp1,
                "errores_pct": errores_exp1,
                "alphas": alphas_exp1,
            },
            "experimento_2": {
                "pares": PARES_EXP2,
                "I_exp": I_exp_exp2,
                "I_teo": I_teo_exp2,
                "sigma_I": sigma_I_exp2,
                "errores_pct": errores_exp2,
                "alphas": alphas_exp2,
            },
        },
        ruta=DIR_SALIDA / "resultados_completos.json",
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    n_ok1 = sum(e < 10 for e in errores_exp1)
    n_ok2 = sum(e < 10 for e in errores_exp2)
    print(f"  Exp. 1 — Mediciones con error < 10%: {n_ok1}/{len(errores_exp1)}")
    print(f"  Exp. 2 — Mediciones con error < 10%: {n_ok2}/{len(errores_exp2)}")
    print(f"\n  Archivos generados en: {DIR_SALIDA.resolve()}")
    print("=" * 65 + "\n")

    # Mostrar todas las gráficas juntas (comentar si se ejecuta sin pantalla)
    plt.show()


if __name__ == "__main__":
    main()
