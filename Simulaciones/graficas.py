"""
graficas.py — Generación de gráficas para INERLAB-R
====================================================
Produce:
  Gráfica 1 — I_exp vs R²       (Experimento 1: verificación I = MR²)
  Gráfica 2 — I_exp vs I_teo    (Paridad experimental/teórico, ambos experimentos)
  Gráfica 3 — ω(t) con regresión (diagnóstico de la calidad de los datos de Tracker)

Autores: Haniel Chaves, Felipe Chirivi, Tatiana Valero
Universidad Industrial de Santander — Física 1, 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Estilo global limpio y apto para informes de laboratorio
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.framealpha": 0.8,
})

_COLORES = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica 1 — I_exp vs R²
# ──────────────────────────────────────────────────────────────────────────────

def grafica_I_vs_R2(
    radios: Sequence[float],
    I_exp: Sequence[float],
    sigma_I: Sequence[float],
    M: float,
    I_teo: Optional[Sequence[float]] = None,
    titulo: str = "Momento de Inercia vs R² — INERLAB-R",
    guardar: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grafica I_exp vs R² con barras de error y la recta teórica I = M·R².

    Parámetros
    ----------
    radios   : Sequence[float] — Radios de rotación [m]
    I_exp    : Sequence[float] — Momentos de inercia experimentales [kg·m²]
    sigma_I  : Sequence[float] — Incertidumbres de I_exp [kg·m²]
    M        : float           — Masa puntual [kg] (para la recta teórica)
    I_teo    : Sequence[float] — Valores teóricos (opcional; si None se calcula)
    titulo   : str             — Título de la figura
    guardar  : str | Path      — Ruta para guardar la imagen (None = no guardar)

    Retorna
    -------
    matplotlib.figure.Figure
    """
    R = np.asarray(radios, dtype=float)
    R2 = R**2
    Ie = np.asarray(I_exp, dtype=float)
    sI = np.asarray(sigma_I, dtype=float)

    # Recta teórica
    R2_linea = np.linspace(0, max(R2) * 1.1, 200)
    I_linea = M * R2_linea

    fig, ax = plt.subplots(figsize=(7, 5))

    # Puntos experimentales con barras de error
    ax.errorbar(
        R2, Ie, yerr=sI,
        fmt="o", color=_COLORES[0], markersize=7,
        elinewidth=1.5, capsize=4, capthick=1.5,
        label="$I_{\\mathrm{exp}}$ (experimental)",
        zorder=5,
    )

    # Recta teórica
    ax.plot(
        R2_linea, I_linea,
        color=_COLORES[1], linewidth=1.8, linestyle="--",
        label=f"$I_{{\\mathrm{{teo}}}} = M \\cdot R^2$  ($M = {M*1000:.1f}$ g)",
    )

    # Regresión lineal de los puntos experimentales → pendiente = M_exp
    M_exp_label = ""
    if len(R2) >= 2:
        slope, intercept = np.polyfit(R2, Ie, 1)
        sigma_slope = np.std(Ie - (slope * R2 + intercept)) / (
            np.sqrt(np.sum((R2 - np.mean(R2)) ** 2))
        ) if len(R2) > 2 else float("nan")
        M_exp_label = (
            f"$M_{{\\mathrm{{exp}}}} = {slope*1000:.1f}$ g "
            f"(real: $M = {M*1000:.1f}$ g)"
        )
        ax.plot(
            R2_linea, slope * R2_linea + intercept,
            color=_COLORES[2], linewidth=1.4, linestyle="-.",
            label=f"Ajuste lineal: pendiente = {slope:.4f} kg\n({M_exp_label})",
        )

    ax.set_xlabel("$R^2$ [m²]", fontsize=12)
    ax.set_ylabel("Momento de inercia $I$ [kg·m²]", fontsize=12)
    ax.set_title(titulo, fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, bbox_inches="tight")
        print(f"[graficas] Guardada: {guardar}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica 2 — I_exp vs I_teo (paridad)
# ──────────────────────────────────────────────────────────────────────────────

def grafica_paridad(
    I_exp: Sequence[float],
    I_teo: Sequence[float],
    sigma_I: Sequence[float],
    etiquetas: Optional[Sequence[str]] = None,
    titulo: str = "Paridad I_exp vs I_teo — INERLAB-R",
    guardar: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Gráfica de paridad: I_exp vs I_teo con línea de identidad (pendiente 1).

    Un punto sobre la línea indica acuerdo perfecto; la distancia a la línea
    indica el error sistemático.

    Parámetros
    ----------
    I_exp     : Sequence[float]        — Inercias experimentales [kg·m²]
    I_teo     : Sequence[float]        — Inercias teóricas [kg·m²]
    sigma_I   : Sequence[float]        — Incertidumbres de I_exp [kg·m²]
    etiquetas : Sequence[str]          — Etiquetas para cada punto (ej. radio)
    titulo    : str                    — Título de la figura
    guardar   : str | Path             — Ruta para guardar

    Retorna
    -------
    matplotlib.figure.Figure
    """
    Ie = np.asarray(I_exp, dtype=float)
    It = np.asarray(I_teo, dtype=float)
    sI = np.asarray(sigma_I, dtype=float)

    lim_min = min(np.min(It), np.min(Ie)) * 0.85
    lim_max = max(np.max(It), np.max(Ie)) * 1.15
    linea = np.array([lim_min, lim_max])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Línea de identidad
    ax.plot(linea, linea, "k--", linewidth=1.2, label="Identidad ($I_{exp} = I_{teo}$)")

    # Puntos experimentales
    for i, (xt, ye, se) in enumerate(zip(It, Ie, sI)):
        color = _COLORES[i % len(_COLORES)]
        lbl = etiquetas[i] if etiquetas else f"Punto {i+1}"
        ax.errorbar(xt, ye, yerr=se, fmt="o", color=color,
                    markersize=7, elinewidth=1.5, capsize=4,
                    label=lbl, zorder=5)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("$I_{\\mathrm{teo}}$ [kg·m²]", fontsize=12)
    ax.set_ylabel("$I_{\\mathrm{exp}}$ [kg·m²]", fontsize=12)
    ax.set_title(titulo, fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, bbox_inches="tight")
        print(f"[graficas] Guardada: {guardar}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica 3 — ω(t) con regresión lineal (diagnóstico Tracker)
# ──────────────────────────────────────────────────────────────────────────────

def grafica_omega_t(
    resultado_regresion: dict,
    radio_label: str = "",
    titulo: str = "Velocidad angular ω(t) — Regresión lineal",
    guardar: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grafica ω vs t junto a la recta de regresión que determina α.

    Parámetros
    ----------
    resultado_regresion : dict — Salida de datos.calcular_alpha_regresion
    radio_label         : str  — Etiqueta del radio (ej. "R = 10 cm")
    titulo              : str  — Título de la figura
    guardar             : str | Path — Ruta para guardar

    Retorna
    -------
    matplotlib.figure.Figure
    """
    t = resultado_regresion["t_arr"]
    omega = resultado_regresion["omega_arr"]
    omega_fit = resultado_regresion["omega_fit"]
    alpha = resultado_regresion["alpha"]
    r2 = resultado_regresion["r_cuadrado"]
    sigma_alpha = resultado_regresion["sigma_alpha"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(t, omega, color=_COLORES[0], s=25, zorder=5,
               label="Datos Tracker")
    ax.plot(t, omega_fit, color=_COLORES[1], linewidth=2,
            label=(
                f"Regresión lineal\n"
                f"$\\alpha = {alpha:.4f} \\pm {sigma_alpha:.4f}$ rad/s²\n"
                f"$R^2 = {r2:.4f}$"
            ))

    if radio_label:
        ax.set_title(f"{titulo} — {radio_label}", fontsize=12, pad=10)
    else:
        ax.set_title(titulo, fontsize=12, pad=10)

    ax.set_xlabel("Tiempo $t$ [s]", fontsize=11)
    ax.set_ylabel("Velocidad angular $\\omega$ [rad/s]", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, bbox_inches="tight")
        print(f"[graficas] Guardada: {guardar}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Gráfica 4 — Error porcentual por radio
# ──────────────────────────────────────────────────────────────────────────────

def grafica_error_porcentual(
    radios: Sequence[float],
    errores: Sequence[float],
    umbral: float = 10.0,
    titulo: str = "Error porcentual vs Radio — INERLAB-R",
    guardar: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Gráfica de barras del error porcentual para cada radio.
    Marca el umbral del 10 % establecido como criterio de éxito.

    Parámetros
    ----------
    radios   : Sequence[float] — Radios [m]
    errores  : Sequence[float] — Errores porcentuales [%]
    umbral   : float           — Línea de umbral de aceptación [%]
    titulo   : str             — Título
    guardar  : str | Path      — Ruta para guardar

    Retorna
    -------
    matplotlib.figure.Figure
    """
    R = np.asarray(radios) * 100  # convertir a cm para el eje
    E = np.asarray(errores)

    colores_barra = [
        _COLORES[0] if e < umbral else _COLORES[1] for e in E
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [f"{r:.0f}" for r in R], E,
        color=colores_barra, edgecolor="white", linewidth=0.5
    )
    ax.axhline(umbral, color="red", linestyle="--", linewidth=1.5,
               label=f"Umbral de aceptación ({umbral:.0f}%)")
    ax.set_xlabel("Radio $R$ [cm]", fontsize=11)
    ax.set_ylabel("Error porcentual [%]", fontsize=11)
    ax.set_title(titulo, fontsize=12, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, bbox_inches="tight")
        print(f"[graficas] Guardada: {guardar}")

    return fig
