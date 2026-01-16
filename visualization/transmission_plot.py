"""Transmission coefficient plotting."""

import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..tunneling.base import TunnelingResult
from ..core.constants import HARTREE_TO_KCAL


def plot_transmission(
    result: TunnelingResult,
    x_units: str = "ratio",
    log_scale: bool = True,
    show_classical: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    color: str = "steelblue",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot transmission coefficient T(E) vs energy.

    Args:
        result: TunnelingResult with transmission data
        x_units: X-axis units - "ratio" (E/V), "kcal/mol", or "hartree"
        log_scale: Use log scale for y-axis
        show_classical: Show classical step function
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        color: Line color
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get data
    energies = result.energies.copy()
    transmissions = result.transmissions.copy()
    barrier = result.barrier_height

    # Convert x-axis units
    if x_units == "ratio":
        x_data = energies / barrier
        x_label = "E / V (Energy / Barrier)"
    elif x_units == "kcal/mol":
        x_data = energies * HARTREE_TO_KCAL
        x_label = "Energy (kcal/mol)"
    else:
        x_data = energies
        x_label = "Energy (Hartree)"

    # Plot transmission
    ax.plot(x_data, transmissions, color=color, linewidth=2,
            label=result.method, **kwargs)

    # Classical step function
    if show_classical:
        x_sorted = np.sort(x_data)
        if x_units == "ratio":
            classical = np.where(x_sorted >= 1.0, 1.0, 0.0)
            ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
        else:
            classical = np.where(energies >= barrier, 1.0, 0.0)
            if x_units == "kcal/mol":
                ax.axvline(barrier * HARTREE_TO_KCAL, color='gray',
                          linestyle=':', alpha=0.5)
            else:
                ax.axvline(barrier, color='gray', linestyle=':', alpha=0.5)

        ax.step(x_sorted, classical, 'k--', alpha=0.5, linewidth=1.5,
               label='Classical', where='post')

    # Formatting
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-10, top=2)
    else:
        ax.set_ylim(0, 1.1)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Transmission Coefficient T(E)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Tunneling Transmission ({result.method})', fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_transmission_comparison(
    results: List[TunnelingResult],
    labels: Optional[List[str]] = None,
    x_units: str = "ratio",
    log_scale: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Transmission Coefficient Comparison",
    colors: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Compare transmission coefficients from different methods.

    Args:
        results: List of TunnelingResult objects
        labels: Labels for each result
        x_units: X-axis units
        log_scale: Use log scale
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

    # Default colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Default labels
    if labels is None:
        labels = [r.method for r in results]

    # Reference barrier (use first result)
    barrier = results[0].barrier_height

    for result, label, color in zip(results, labels, colors):
        energies = result.energies
        transmissions = result.transmissions

        if x_units == "ratio":
            x_data = energies / barrier
        elif x_units == "kcal/mol":
            x_data = energies * HARTREE_TO_KCAL
        else:
            x_data = energies

        ax.plot(x_data, transmissions, color=color, linewidth=2,
                label=label, **kwargs)

    # Add classical reference
    if x_units == "ratio":
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5,
                  label='E = V')
        x_label = "E / V (Energy / Barrier)"
    elif x_units == "kcal/mol":
        ax.axvline(barrier * HARTREE_TO_KCAL, color='gray',
                  linestyle=':', alpha=0.5)
        x_label = "Energy (kcal/mol)"
    else:
        ax.axvline(barrier, color='gray', linestyle=':', alpha=0.5)
        x_label = "Energy (Hartree)"

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-10, top=2)
    else:
        ax.set_ylim(0, 1.1)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Transmission Coefficient T(E)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_kappa_vs_temperature(
    temperatures: np.ndarray,
    kappas: np.ndarray,
    method: str = "WKB",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    color: str = "steelblue",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot tunneling correction factor kappa vs temperature.

    Args:
        temperatures: Array of temperatures in Kelvin
        kappas: Array of kappa values
        method: Method name for label
        ax: Existing axes
        figsize: Figure size
        color: Line color
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(temperatures, kappas, color=color, linewidth=2,
            marker='o', markersize=4, label=method, **kwargs)

    # Reference line at kappa = 1
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5,
              label='Classical (κ = 1)')

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Tunneling Correction Factor κ', fontsize=12)
    ax.set_title('Tunneling Enhancement vs Temperature', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Log scale if kappa varies greatly
    if np.max(kappas) / np.min(kappas) > 100:
        ax.set_yscale('log')

    plt.tight_layout()
    return fig, ax


def plot_transmission_energy_resolved(
    result: TunnelingResult,
    temperature: float = 300.0,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot T(E) weighted by Boltzmann distribution.

    Shows which energies contribute most to tunneling at given T.

    Args:
        result: TunnelingResult object
        temperature: Temperature in Kelvin
        ax: Existing axes
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    from ..core.constants import BOLTZMANN_HARTREE

    energies = result.energies
    transmissions = result.transmissions
    barrier = result.barrier_height

    # Boltzmann weight
    beta = 1.0 / (BOLTZMANN_HARTREE * temperature)
    boltzmann = np.exp(-beta * energies)
    boltzmann /= np.sum(boltzmann)  # Normalize

    # Weighted transmission
    weighted = transmissions * boltzmann
    weighted /= np.max(weighted)  # Normalize to 1

    x_data = energies / barrier

    ax.fill_between(x_data, weighted, alpha=0.3, color='steelblue',
                   label=f'Weighted T(E) at {temperature} K')
    ax.plot(x_data, weighted, color='steelblue', linewidth=2)

    # Mark barrier
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='E = V')

    ax.set_xlabel('E / V (Energy / Barrier)', fontsize=12)
    ax.set_ylabel('Boltzmann-weighted T(E) (normalized)', fontsize=12)
    ax.set_title(f'Energy-resolved Tunneling at T = {temperature} K', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)

    plt.tight_layout()
    return fig, ax
