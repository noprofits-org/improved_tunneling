"""Potential energy surface plotting."""

import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..pes.scan import PESScanResult
from ..core.constants import HARTREE_TO_KCAL, HARTREE_TO_CM


def plot_pes(
    pes_result: PESScanResult,
    units: str = "kcal/mol",
    show_barrier: bool = True,
    show_minima: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    color: str = "steelblue",
    marker: str = "o",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot potential energy surface from scan result.

    Args:
        pes_result: PES scan result object
        units: Energy units - "kcal/mol", "kJ/mol", "cm-1", "eV", or "hartree"
        show_barrier: Mark barrier height on plot
        show_minima: Mark energy minima on plot
        ax: Existing axes to plot on (creates new figure if None)
        figsize: Figure size if creating new figure
        title: Plot title
        color: Line/marker color
        marker: Marker style
        **kwargs: Additional arguments passed to plt.plot()

    Returns:
        Tuple of (Figure, Axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get data
    angles = pes_result.angles
    energies = pes_result.energies.copy()

    # Convert energy units
    energies_relative = energies - np.min(energies)
    conversion, unit_label = _get_conversion(units)
    energies_plot = energies_relative * conversion

    # Plot PES
    ax.plot(angles, energies_plot, marker=marker, color=color,
            linewidth=2, markersize=6, **kwargs)

    # Mark minima
    if show_minima:
        min_indices = _find_local_minima(energies_plot)
        for idx in min_indices:
            ax.axvline(angles[idx], color='green', linestyle='--',
                      alpha=0.5, linewidth=1)
            ax.scatter([angles[idx]], [energies_plot[idx]],
                      color='green', s=100, zorder=5, marker='v')

    # Mark barrier
    if show_barrier:
        barrier_idx = np.argmax(energies_plot)
        barrier_height = energies_plot[barrier_idx]
        ax.axhline(barrier_height, color='red', linestyle='--',
                  alpha=0.5, linewidth=1)
        ax.annotate(f'Barrier: {barrier_height:.2f} {unit_label}',
                   xy=(angles[barrier_idx], barrier_height),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='red')

    # Labels
    ax.set_xlabel('Dihedral Angle (degrees)', fontsize=12)
    ax.set_ylabel(f'Energy ({unit_label})', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        method = pes_result.method or "Unknown"
        basis = pes_result.basis or ""
        ax.set_title(f'PES Scan ({method}/{basis})', fontsize=14)

    ax.set_xlim(min(angles), max(angles))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_pes_comparison(
    pes_results: List[PESScanResult],
    labels: Optional[List[str]] = None,
    units: str = "kcal/mol",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "PES Comparison",
    colors: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Compare multiple PES scans on same plot.

    Args:
        pes_results: List of PES scan results
        labels: Labels for each PES (default: method/basis)
        units: Energy units
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        colors: List of colors for each PES
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
        colors = plt.cm.tab10(np.linspace(0, 1, len(pes_results)))

    # Default labels
    if labels is None:
        labels = []
        for pes in pes_results:
            method = pes.method or "Unknown"
            basis = pes.basis or ""
            labels.append(f"{method}/{basis}")

    conversion, unit_label = _get_conversion(units)

    for i, (pes, label, color) in enumerate(zip(pes_results, labels, colors)):
        angles = pes.angles
        energies = pes.energies - np.min(pes.energies)
        energies_plot = energies * conversion

        ax.plot(angles, energies_plot, label=label, color=color,
                linewidth=2, marker='o', markersize=4, **kwargs)

    ax.set_xlabel('Dihedral Angle (degrees)', fontsize=12)
    ax.set_ylabel(f'Energy ({unit_label})', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_pes_with_tunneling_region(
    pes_result: PESScanResult,
    energy_level: float,
    units: str = "kcal/mol",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot PES with tunneling region highlighted.

    Args:
        pes_result: PES scan result
        energy_level: Energy level in Hartree (relative to minimum)
        units: Energy units for display
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

    conversion, unit_label = _get_conversion(units)

    angles = pes_result.angles
    energies = pes_result.energies - np.min(pes_result.energies)
    energies_plot = energies * conversion
    energy_line = energy_level * conversion

    # Plot PES
    ax.plot(angles, energies_plot, 'b-', linewidth=2, label='PES')
    ax.axhline(energy_line, color='orange', linestyle='--',
              linewidth=1.5, label=f'E = {energy_line:.2f} {unit_label}')

    # Highlight tunneling region (where E < V)
    tunneling_mask = energies_plot > energy_line
    if np.any(tunneling_mask):
        ax.fill_between(angles, energy_line, energies_plot,
                       where=tunneling_mask, alpha=0.3, color='red',
                       label='Tunneling region')

    ax.set_xlabel('Dihedral Angle (degrees)', fontsize=12)
    ax.set_ylabel(f'Energy ({unit_label})', fontsize=12)
    ax.set_title('PES with Tunneling Region', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def _get_conversion(units: str) -> Tuple[float, str]:
    """Get conversion factor and label for energy units."""
    units_lower = units.lower().replace(" ", "").replace("-", "")

    conversions = {
        "kcal/mol": (HARTREE_TO_KCAL, "kcal/mol"),
        "kcalmol": (HARTREE_TO_KCAL, "kcal/mol"),
        "kj/mol": (HARTREE_TO_KCAL * 4.184, "kJ/mol"),
        "kjmol": (HARTREE_TO_KCAL * 4.184, "kJ/mol"),
        "cm1": (HARTREE_TO_CM, "cm⁻¹"),
        "cm-1": (HARTREE_TO_CM, "cm⁻¹"),
        "ev": (27.2114, "eV"),
        "hartree": (1.0, "Hartree"),
        "ha": (1.0, "Hartree"),
    }

    if units_lower in conversions:
        return conversions[units_lower]
    else:
        raise ValueError(f"Unknown energy unit: {units}. "
                        f"Choose from: {list(conversions.keys())}")


def _find_local_minima(energies: np.ndarray) -> List[int]:
    """Find indices of local minima in energy array."""
    minima = []
    n = len(energies)

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        if energies[i] < energies[prev_idx] and energies[i] < energies[next_idx]:
            minima.append(i)

    return minima
