"""
incertidumbre.py — Propagación de incertidumbres para INERLAB-R
================================================================
Propaga la incertidumbre de I_exp = r·m·(g-a)/α mediante
derivadas parciales analíticas:

    σ_I² = (∂I/∂m · σ_m)² + (∂I/∂r · σ_r)²
          + (∂I/∂a · σ_a)² + (∂I/∂α · σ_α)²

También implementa la prueba t-Student para validar que la diferencia
entre I_exp e I_teo no es estadísticamente significativa.

Autores: Haniel Chaves, Felipe Chirivi, Tatiana Valero
Universidad Industrial de Santander — Física 1, 2026
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Derivadas parciales de I = r · m · (g - a) / α
# ──────────────────────────────────────────────────────────────────────────────
# I(m, r, a, α) = r·m·(g-a)/α
#
# ∂I/∂m =  r·(g-a)/α
# ∂I/∂r =  m·(g-a)/α
# ∂I/∂a = -r·m/α
# ∂I/∂α = -r·m·(g-a)/α²

def _dI_dm(r, g, a, alpha):
    return r * (g - a) / abs(alpha)

def _dI_dr(m, g, a, alpha):
    return m * (g - a) / abs(alpha)

def _dI_da(m, r, alpha):
    return -r * m / abs(alpha)

def _dI_dalpha(m, r, g, a, alpha):
    return -r * m * (g - a) / alpha**2


# ──────────────────────────────────────────────────────────────────────────────
# Propagación de incertidumbre combinada
# ──────────────────────────────────────────────────────────────────────────────

def propagar_incertidumbre(
    m: float,
    sigma_m: float,
    r: float,
    sigma_r: float,
    g: float,
    a: float,
    sigma_a: float,
    alpha: float,
    sigma_alpha: float,
) -> dict:
    """
    Calcula la incertidumbre combinada de I_exp usando propagación
    de errores con derivadas parciales analíticas.

    Parámetros
    ----------
    m           : float — Masa efectiva colgante [kg]
    sigma_m     : float — Incertidumbre de m [kg]  (resolución de balanza / √3)
    r           : float — Radio de la polea [m]
    sigma_r     : float — Incertidumbre de r [m]   (resolución de calibrador / √3)
    g           : float — Aceleración gravitacional [m/s²]
    a           : float — Aceleración lineal [m/s²]
    sigma_a     : float — Incertidumbre de a [m/s²]
    alpha       : float — Aceleración angular [rad/s²]
    sigma_alpha : float — Incertidumbre de α (desv. estándar de la regresión)

    Retorna
    -------
    dict con:
        sigma_I         : float — Incertidumbre combinada de I [kg·m²]
        contribuciones  : dict  — Aporte de cada variable a σ_I² [kg²·m⁴]
        terminos        : dict  — Derivadas parciales individuales
    """
    dI_dm = _dI_dm(r, g, a, alpha)
    dI_dr = _dI_dr(m, g, a, alpha)
    dI_da = _dI_da(m, r, alpha)
    dI_dalpha = _dI_dalpha(m, r, g, a, alpha)

    contrib_m = (dI_dm * sigma_m) ** 2
    contrib_r = (dI_dr * sigma_r) ** 2
    contrib_a = (dI_da * sigma_a) ** 2
    contrib_alpha = (dI_dalpha * sigma_alpha) ** 2

    sigma_I = np.sqrt(contrib_m + contrib_r + contrib_a + contrib_alpha)

    # I_central para calcular incertidumbre relativa
    I_central = (r * m * (g - a)) / abs(alpha)
    sigma_rel = (sigma_I / I_central * 100) if I_central != 0 else 0.0

    return {
        "sigma_I": sigma_I,
        "sigma_rel_pct": sigma_rel,           # ← NUEVO: incertidumbre relativa [%]
        "contribuciones": {
            "m": contrib_m,
            "r": contrib_r,
            "a": contrib_a,
            "alpha": contrib_alpha,
        },
        "terminos": {
            "dI/dm": dI_dm,
            "dI/dr": dI_dr,
            "dI/da": dI_da,
            "dI/dalpha": dI_dalpha,
        },
    }


def propagar_array(
    mediciones: list[dict],
    sigma_m: float,
    sigma_r: float,
    g: float,
) -> list[float]:
    """
    Aplica propagar_incertidumbre a una lista de mediciones.

    Cada elemento de `mediciones` debe tener las claves:
        m, r, a, alpha, sigma_a, sigma_alpha

    Parámetros
    ----------
    mediciones : list[dict] — Lista de mediciones experimentales
    sigma_m    : float      — Incertidumbre de la masa (misma para todas)
    sigma_r    : float      — Incertidumbre del radio (misma para todas)
    g          : float      — Aceleración gravitacional

    Retorna
    -------
    list[float] — Lista de σ_I para cada medición
    """
    sigmas = []
    for med in mediciones:
        res = propagar_incertidumbre(
            m=med["m"],
            sigma_m=sigma_m,
            r=med["r"],
            sigma_r=sigma_r,
            g=g,
            a=med["a"],
            sigma_a=med.get("sigma_a", 0.001),
            alpha=med["alpha"],
            sigma_alpha=med.get("sigma_alpha", 0.01),
        )
        sigmas.append(res["sigma_I"])
    return sigmas


# ──────────────────────────────────────────────────────────────────────────────
# Prueba t-Student
# ──────────────────────────────────────────────────────────────────────────────

def prueba_t_student(
    I_exp_mediciones: list[float] | np.ndarray,
    I_teo: float,
    nivel_significancia: float = 0.05,
) -> dict:
    """
    Prueba t-Student de una muestra para evaluar si la diferencia
    entre I_exp e I_teo es estadísticamente significativa.

    H₀: μ(I_exp) = I_teo
    H₁: μ(I_exp) ≠ I_teo

    Parámetros
    ----------
    I_exp_mediciones    : list[float] | np.ndarray — Mediciones independientes de I_exp
    I_teo               : float                    — Valor teórico de referencia
    nivel_significancia : float                    — α (típicamente 0.05)

    Retorna
    -------
    dict con:
        t_estadistico  : float — Estadístico t calculado
        t_critico      : float — Valor crítico de t para α/2 y n-1 grados de libertad
        p_value        : float — Valor p de la prueba bilateral
        I_exp_media    : float — Media de las mediciones
        I_exp_std      : float — Desviación estándar muestral
        n              : int   — Número de mediciones
        gl             : int   — Grados de libertad (n-1)
        hipotesis_nula : bool  — True si NO se rechaza H₀ (|t| < t_critico)
        conclusion     : str   — Interpretación textual
    """
    I_exp = np.asarray(I_exp_mediciones, dtype=float)
    n = len(I_exp)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 mediciones para la prueba t.")

    I_mean = np.mean(I_exp)
    I_std = np.std(I_exp, ddof=1)
    gl = n - 1

    t_stat = (I_mean - I_teo) / (I_std / np.sqrt(n))
    t_crit = stats.t.ppf(1 - nivel_significancia / 2, df=gl)
    p_val = 2 * stats.t.sf(abs(t_stat), df=gl)

    acepta_h0 = abs(t_stat) < t_crit

    conclusion = (
        f"Con α = {nivel_significancia}, n = {n} mediciones y {gl} grados de "
        f"libertad: |t| = {abs(t_stat):.3f} "
        + ("< " if acepta_h0 else "> ")
        + f"t_crítico = {t_crit:.3f}. "
        + (
            "NO se rechaza H₀: la diferencia no es estadísticamente significativa."
            if acepta_h0
            else "Se RECHAZA H₀: existe diferencia significativa con el valor teórico."
        )
    )

    return {
        "t_estadistico": t_stat,
        "t_critico": t_crit,
        "p_value": p_val,
        "I_exp_media": I_mean,
        "I_exp_std": I_std,
        "n": n,
        "gl": gl,
        "hipotesis_nula": acepta_h0,
        "conclusion": conclusion,
    }


def resumen_incertidumbres(resultado_propagacion: dict) -> str:
    """
    Genera un resumen legible de las contribuciones a la incertidumbre.

    Parámetros
    ----------
    resultado_propagacion : dict — Salida de propagar_incertidumbre

    Retorna
    -------
    str — Resumen formateado
    """
    contrib = resultado_propagacion["contribuciones"]
    total = sum(contrib.values())
    sigma_rel = resultado_propagacion.get("sigma_rel_pct", 0.0)
    lineas = [
        f"  σ_I = {resultado_propagacion['sigma_I']:.6f} kg·m²  "
        f"(σ_rel = {sigma_rel:.2f}%)",
        "  Contribuciones relativas a σ_I²:",
    ]
    for var, val in contrib.items():
        pct = (val / total * 100) if total > 0 else 0
        lineas.append(f"    {var:>5s}: {pct:6.2f}%")
    return "\n".join(lineas)
