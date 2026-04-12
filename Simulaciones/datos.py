"""
datos.py — Carga y procesamiento de datos exportados desde Tracker
===================================================================
Tracker exporta archivos .csv con columnas típicas:
    t [s], x [m], y [m], θ [rad], ω [rad/s], ...

Este módulo:
  1. Carga el CSV (tolerando encabezados en inglés y español de Tracker).
  2. Extrae las columnas de tiempo t y velocidad angular ω.
  3. Calcula α mediante regresión lineal de ω(t).
  4. Permite manejar múltiples archivos de medición.

Autores: Haniel Chaves, Felipe Chirivi, Tatiana Valero
Universidad Industrial de Santander — Física 1, 2026
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Constantes de columnas esperadas (Tracker puede usar distintos nombres)
# ──────────────────────────────────────────────────────────────────────────────

_POSIBLES_COLS_T = ["t", "time", "tiempo", "t (s)", "Time (s)"]
_POSIBLES_COLS_THETA = ["θ", "theta", "angle", "ángulo", "θ (rad)", "angle (rad)"]
_POSIBLES_COLS_OMEGA = ["ω", "omega", "angular velocity", "velocidad angular",
                        "ω (rad/s)", "omega (rad/s)"]


def _encontrar_columna(df: pd.DataFrame, candidatos: list[str]) -> Optional[str]:
    """
    Busca la primera columna del DataFrame cuyo nombre coincida (parcialmente,
    sin distinguir mayúsculas) con alguno de los candidatos.
    """
    for col in df.columns:
        col_lower = col.strip().lower()
        for cand in candidatos:
            if cand.lower() in col_lower:
                return col
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Carga de un único archivo Tracker
# ──────────────────────────────────────────────────────────────────────────────

def cargar_tracker(ruta: str | Path, skiprows: int = 2) -> pd.DataFrame:
    """
    Carga un archivo CSV exportado por Tracker.

    Tracker suele incluir 1-2 filas de encabezado antes de los datos numéricos.
    Si el CSV tiene un encabezado diferente, ajuste `skiprows`.

    Parámetros
    ----------
    ruta     : str | Path — Ruta al archivo .csv
    skiprows : int        — Número de filas a omitir antes del encabezado real

    Retorna
    -------
    pd.DataFrame con las columnas detectadas.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    # Intentar leer con coma y con tabulación (Tracker puede usar ambos)
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(ruta, sep=sep, skiprows=skiprows)
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        raise ValueError(
            f"No se pudo leer '{ruta}'. Verifique el separador o skiprows."
        )

    # Limpiar nombres de columnas
    df.columns = [c.strip() for c in df.columns]
    # Eliminar filas con todos NaN
    df.dropna(how="all", inplace=True)
    # Convertir a numérico (por si hay texto residual)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(how="all", inplace=True)

    return df


def extraer_series_temporales(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extrae t, ω y (opcionalmente) θ del DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame — DataFrame cargado desde Tracker

    Retorna
    -------
    t     : np.ndarray — Tiempo [s]
    omega : np.ndarray — Velocidad angular [rad/s]
    theta : np.ndarray | None — Posición angular [rad], si existe
    """
    col_t = _encontrar_columna(df, _POSIBLES_COLS_T)
    col_w = _encontrar_columna(df, _POSIBLES_COLS_OMEGA)
    col_th = _encontrar_columna(df, _POSIBLES_COLS_THETA)

    if col_t is None:
        raise KeyError(
            "No se encontró columna de tiempo. Columnas disponibles: "
            + str(list(df.columns))
        )
    if col_w is None:
        # Intentar calcular ω a partir de θ si existe
        if col_th is not None:
            t_arr = df[col_t].to_numpy(dtype=float)
            th_arr = df[col_th].to_numpy(dtype=float)
            omega_arr = np.gradient(th_arr, t_arr)
            theta_arr = th_arr
            print(
                "[datos] Columna ω no encontrada; calculada como dθ/dt "
                "mediante diferencias finitas."
            )
            return t_arr, omega_arr, theta_arr
        raise KeyError(
            "No se encontró columna de velocidad angular ω. "
            "Columnas disponibles: " + str(list(df.columns))
        )

    t_arr = df[col_t].to_numpy(dtype=float)
    omega_arr = df[col_w].to_numpy(dtype=float)
    theta_arr = (
        df[col_th].to_numpy(dtype=float) if col_th is not None else None
    )

    # Eliminar NaN
    mask = ~np.isnan(t_arr) & ~np.isnan(omega_arr)
    return t_arr[mask], omega_arr[mask], (theta_arr[mask] if theta_arr is not None else None)


# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de α por regresión lineal de ω(t)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_alpha_regresion(
    t: np.ndarray, omega: np.ndarray
) -> dict:
    """
    Determina la aceleración angular α como la pendiente de la regresión
    lineal de ω(t) → ω = α·t + ω₀.

    Parámetros
    ----------
    t     : np.ndarray — Tiempo [s]
    omega : np.ndarray — Velocidad angular [rad/s]

    Retorna
    -------
    dict con:
        alpha       : float — Pendiente de la regresión [rad/s²]
        omega_0     : float — Intercepto [rad/s]
        r_cuadrado  : float — Coeficiente de determinación R²
        sigma_alpha : float — Incertidumbre estándar de α [rad/s²]
        t_arr       : np.ndarray — t usado en la regresión
        omega_arr   : np.ndarray — ω usado en la regresión
        omega_fit   : np.ndarray — ω predicha por la regresión
    """
    if len(t) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para la regresión.")

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, omega)

    omega_fit = slope * t + intercept

    return {
        "alpha": slope,
        "omega_0": intercept,
        "r_cuadrado": r_value**2,
        "sigma_alpha": std_err,
        "t_arr": t,
        "omega_arr": omega,
        "omega_fit": omega_fit,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Manejo de múltiples mediciones
# ──────────────────────────────────────────────────────────────────────────────

def cargar_multiples_mediciones(
    rutas: list[str | Path],
    skiprows: int = 2,
) -> list[dict]:
    """
    Carga y procesa múltiples archivos de Tracker para el mismo radio.

    Parámetros
    ----------
    rutas    : list — Lista de rutas a archivos .csv
    skiprows : int  — Filas a omitir en cada CSV

    Retorna
    -------
    list[dict] con los resultados de calcular_alpha_regresion por cada archivo.
    """
    resultados = []
    for ruta in rutas:
        try:
            df = cargar_tracker(ruta, skiprows=skiprows)
            t, omega, _ = extraer_series_temporales(df)
            res = calcular_alpha_regresion(t, omega)
            res["archivo"] = str(ruta)
            resultados.append(res)
            print(
                f"[datos] {Path(ruta).name}: α = {res['alpha']:.4f} rad/s²  "
                f"R² = {res['r_cuadrado']:.4f}"
            )
        except Exception as e:
            print(f"[datos] Error procesando '{ruta}': {e}")

    return resultados


def calcular_a_desde_posicion(
    t: np.ndarray, theta: np.ndarray, r_polea: float
) -> dict:
    """
    Calcula la aceleración lineal `a` y su incertidumbre directamente a partir
    del ajuste cuadrático de θ(t):

        θ(t) = ½·α·t² + ω₀·t + θ₀  →  a = α·r  (de la misma regresión)

    Ventaja sobre `a = alpha_mean * r`: obtiene σ_a real de los residuos del
    ajuste, en lugar de usar un valor fijo instrumental.

    Parámetros
    ----------
    t       : np.ndarray — Tiempo [s]
    theta   : np.ndarray — Posición angular θ [rad]
    r_polea : float      — Radio de la polea [m]

    Retorna
    -------
    dict con:
        a           : float — Aceleración lineal [m/s²]
        sigma_a     : float — Incertidumbre de a desde los residuos [m/s²]
        alpha       : float — Aceleración angular [rad/s²]
        sigma_alpha : float — Incertidumbre de α [rad/s²]
        r_cuadrado  : float — R² del ajuste cuadrático
        theta_fit   : np.ndarray — θ predicha por el ajuste
    """
    if len(t) < 4:
        raise ValueError("Se necesitan al menos 4 puntos para el ajuste cuadrático.")

    # Ajuste θ = c2·t² + c1·t + c0  →  α = 2·c2
    coefs, cov = np.polyfit(t, theta, 2, cov=True)
    c2, c1, c0 = coefs

    alpha = 2.0 * c2
    # Varianza de c2 → varianza de α = 4·var(c2)
    sigma_alpha = 2.0 * np.sqrt(cov[0, 0])

    a = alpha * r_polea
    sigma_a = sigma_alpha * r_polea

    theta_fit = np.polyval(coefs, t)
    ss_res = np.sum((theta - theta_fit) ** 2)
    ss_tot = np.sum((theta - np.mean(theta)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": a,
        "sigma_a": sigma_a,
        "alpha": alpha,
        "sigma_alpha": sigma_alpha,
        "r_cuadrado": r2,
        "theta_fit": theta_fit,
    }


def alpha_promedio(mediciones: list[dict]) -> tuple[float, float]:
    """
    Calcula el promedio y la desviación estándar de α
    a partir de múltiples mediciones.

    Parámetros
    ----------
    mediciones : list[dict] — Salida de cargar_multiples_mediciones

    Retorna
    -------
    (alpha_mean, alpha_std) : tuple[float, float]
    """
    alphas = np.array([m["alpha"] for m in mediciones])
    return float(np.mean(alphas)), float(np.std(alphas, ddof=1))


# ──────────────────────────────────────────────────────────────────────────────
# Generador de CSV de ejemplo (para pruebas sin datos reales de Tracker)
# ──────────────────────────────────────────────────────────────────────────────

def generar_csv_ejemplo(
    ruta_salida: str | Path,
    alpha_real: float = 2.5,
    omega_0: float = 0.0,
    t_max: float = 3.0,
    n_puntos: int = 60,
    ruido: float = 0.05,
) -> None:
    """
    Genera un CSV de ejemplo que simula la salida de Tracker.
    SOLO para desarrollo y pruebas; en el experimento real use datos de Tracker.

    Parámetros
    ----------
    ruta_salida : str | Path — Dónde guardar el CSV
    alpha_real  : float      — α real que se desea recuperar [rad/s²]
    omega_0     : float      — Velocidad angular inicial [rad/s]
    t_max       : float      — Tiempo total de grabación [s]
    n_puntos    : int        — Número de cuadros
    ruido       : float      — Amplitud del ruido gaussiano en ω [rad/s]
    """
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, t_max, n_puntos)
    omega = alpha_real * t + omega_0 + rng.normal(0, ruido, n_puntos)
    theta = 0.5 * alpha_real * t**2 + omega_0 * t

    df = pd.DataFrame({"t (s)": t, "θ (rad)": theta, "ω (rad/s)": omega})

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # Encabezado de dos líneas al estilo Tracker
    with open(ruta_salida, "w") as f:
        f.write("# Datos generados para prueba — INERLAB-R\n")
        f.write("# Formato simulado de Tracker\n")
        df.to_csv(f, index=False)

    print(f"[datos] CSV de ejemplo guardado en: {ruta_salida}")
