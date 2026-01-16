"""Arrhenius and kinetic isotope effect plotting."""

import numpy as np
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..kinetics.rates import RateResult


def plot_arrhenius(
    rate_result: RateResult,
    show_classical: bool = True,
    show_quantum: bool = True,
    show_fit: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot Arrhenius diagram (ln(k) vs 1000/T).

    Args:
        rate_result: RateResult with temperature-dependent rates
        show_classical: Show classical (TST) rates
        show_quantum: Show quantum-corrected rates
        show_fit: Show linear Arrhenius fits
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    temps = rate_result.temperatures
    x_data = 1000.0 / temps  # 1000/T for better scale

    if show_classical and rate_result.classical_rates is not None:
        classical_rates = rate_result.classical_rates
        ln_k_classical = np.log(classical_rates)

        ax.plot(x_data, ln_k_classical, 'b-o', linewidth=2, markersize=6,
                label='Classical (TST)', **kwargs)

        if show_fit:
            _add_arrhenius_fit(ax, x_data, ln_k_classical, 'b--', 'Classical fit')

    if show_quantum and rate_result.quantum_rates is not None:
        quantum_rates = rate_result.quantum_rates
        ln_k_quantum = np.log(quantum_rates)

        ax.plot(x_data, ln_k_quantum, 'r-s', linewidth=2, markersize=6,
                label='Quantum-corrected', **kwargs)

        if show_fit:
            _add_arrhenius_fit(ax, x_data, ln_k_quantum, 'r--', 'Quantum fit')

    ax.set_xlabel('1000 / T (K⁻¹)', fontsize=12)
    ax.set_ylabel('ln(k)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Arrhenius Plot', fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add secondary x-axis showing temperature
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    temp_ticks = [500, 400, 350, 300, 250, 200]
    temp_tick_positions = [1000.0 / t for t in temp_ticks]
    ax2.set_xticks(temp_tick_positions)
    ax2.set_xticklabels([str(t) for t in temp_ticks])
    ax2.set_xlabel('Temperature (K)', fontsize=10)

    plt.tight_layout()
    return fig, ax


def plot_arrhenius_comparison(
    rate_results: List[RateResult],
    labels: Optional[List[str]] = None,
    rate_type: str = "quantum",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Arrhenius Comparison",
    colors: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Compare Arrhenius plots for different systems/methods.

    Args:
        rate_results: List of RateResult objects
        labels: Labels for each result
        rate_type: "classical", "quantum", or "both"
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        colors: Colors for each curve
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(rate_results)))

    if labels is None:
        labels = [f"System {i+1}" for i in range(len(rate_results))]

    for result, label, color in zip(rate_results, labels, colors):
        temps = result.temperatures
        x_data = 1000.0 / temps

        if rate_type in ("quantum", "both") and result.quantum_rates is not None:
            ln_k = np.log(result.quantum_rates)
            ax.plot(x_data, ln_k, color=color, linewidth=2, marker='o',
                    markersize=5, label=f"{label} (quantum)", **kwargs)

        if rate_type in ("classical", "both") and result.classical_rates is not None:
            ln_k = np.log(result.classical_rates)
            linestyle = '--' if rate_type == "both" else '-'
            ax.plot(x_data, ln_k, color=color, linewidth=2, marker='s',
                    markersize=5, linestyle=linestyle,
                    label=f"{label} (classical)" if rate_type == "both" else label,
                    **kwargs)

    ax.set_xlabel('1000 / T (K⁻¹)', fontsize=12)
    ax.set_ylabel('ln(k)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_kie(
    rate_h: RateResult,
    rate_d: RateResult,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "Kinetic Isotope Effect",
    show_classical: bool = True,
    show_quantum: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot kinetic isotope effect (KIE = k_H / k_D) vs temperature.

    Args:
        rate_h: RateResult for H isotope
        rate_d: RateResult for D isotope
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        show_classical: Show classical KIE
        show_quantum: Show quantum KIE
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Ensure same temperature grid
    temps = rate_h.temperatures

    if show_classical:
        kie_classical = rate_h.classical_rates / rate_d.classical_rates
        ax.plot(temps, kie_classical, 'b-o', linewidth=2, markersize=6,
                label='Classical KIE', **kwargs)

    if show_quantum:
        kie_quantum = rate_h.quantum_rates / rate_d.quantum_rates
        ax.plot(temps, kie_quantum, 'r-s', linewidth=2, markersize=6,
                label='Quantum KIE', **kwargs)

    # Reference line at KIE = 1
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    # Semiclassical limit (sqrt of mass ratio)
    mass_ratio = 2.014 / 1.008  # D/H mass ratio
    semiclassical_kie = np.sqrt(mass_ratio)
    ax.axhline(semiclassical_kie, color='green', linestyle=':',
              alpha=0.5, label=f'Semiclassical ({semiclassical_kie:.2f})')

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('KIE (k_H / k_D)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Log scale if KIE varies greatly
    if show_quantum:
        kie_max = np.max(kie_quantum)
        kie_min = np.min(kie_quantum)
        if kie_max / kie_min > 10:
            ax.set_yscale('log')

    plt.tight_layout()
    return fig, ax


def plot_kie_arrhenius(
    rate_h: RateResult,
    rate_d: RateResult,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "KIE Arrhenius Plot",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot KIE in Arrhenius form (ln(KIE) vs 1000/T).

    Non-linear behavior indicates significant tunneling.

    Args:
        rate_h: RateResult for H isotope
        rate_d: RateResult for D isotope
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    temps = rate_h.temperatures
    x_data = 1000.0 / temps

    # Classical KIE
    kie_classical = rate_h.classical_rates / rate_d.classical_rates
    ln_kie_classical = np.log(kie_classical)
    ax.plot(x_data, ln_kie_classical, 'b-o', linewidth=2, markersize=6,
            label='Classical', **kwargs)

    # Quantum KIE
    kie_quantum = rate_h.quantum_rates / rate_d.quantum_rates
    ln_kie_quantum = np.log(kie_quantum)
    ax.plot(x_data, ln_kie_quantum, 'r-s', linewidth=2, markersize=6,
            label='Quantum', **kwargs)

    ax.set_xlabel('1000 / T (K⁻¹)', fontsize=12)
    ax.set_ylabel('ln(KIE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add temperature scale
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    temp_ticks = [500, 400, 350, 300, 250, 200]
    temp_tick_positions = [1000.0 / t for t in temp_ticks]
    ax2.set_xticks(temp_tick_positions)
    ax2.set_xticklabels([str(t) for t in temp_ticks])
    ax2.set_xlabel('Temperature (K)', fontsize=10)

    plt.tight_layout()
    return fig, ax


def _add_arrhenius_fit(
    ax: Axes,
    x_data: np.ndarray,
    ln_k: np.ndarray,
    linestyle: str,
    label: str
) -> Tuple[float, float]:
    """
    Add linear Arrhenius fit and return Ea, A.

    Returns:
        Tuple of (Ea in kcal/mol, ln(A))
    """
    from scipy.stats import linregress

    # Linear fit: ln(k) = ln(A) - Ea/(R*T)
    # ln(k) vs 1000/T: slope = -Ea*1000/R
    slope, intercept, r_value, p_value, std_err = linregress(x_data, ln_k)

    # Ea in kcal/mol: slope = -Ea * 1000 / R
    # R = 1.987 cal/(mol*K) = 0.001987 kcal/(mol*K)
    R_kcal = 0.001987
    Ea = -slope * R_kcal  # kcal/mol

    # Plot fit line
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, linestyle, alpha=0.5,
           label=f'{label}: Ea={Ea:.1f} kcal/mol')

    return Ea, intercept
