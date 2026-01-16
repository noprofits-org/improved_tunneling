"""Ring-polymer instanton visualization."""

import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from ..molecule.structure import Molecule


def plot_instanton_path(
    bead_positions: np.ndarray,
    atom_indices: Optional[List[int]] = None,
    symbols: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Instanton Path",
    show_connections: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot instanton (minimum action path) along reaction coordinate.

    Args:
        bead_positions: Array of shape (n_beads, n_atoms, 3) with bead geometries
        atom_indices: Indices of atoms to highlight (e.g., transferring proton)
        symbols: Atomic symbols for labeling
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        show_connections: Draw lines connecting beads
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_beads, n_atoms, _ = bead_positions.shape
    bead_indices = np.arange(n_beads)

    # If specific atoms not specified, show center of mass
    if atom_indices is None:
        # Show x, y, z of center of mass
        com = np.mean(bead_positions, axis=1)  # (n_beads, 3)
        ax.plot(bead_indices, com[:, 0], 'r-o', label='x (COM)', markersize=4)
        ax.plot(bead_indices, com[:, 1], 'g-s', label='y (COM)', markersize=4)
        ax.plot(bead_indices, com[:, 2], 'b-^', label='z (COM)', markersize=4)
    else:
        # Show positions of selected atoms
        colors = plt.cm.tab10(np.linspace(0, 1, len(atom_indices)))
        for i, (atom_idx, color) in enumerate(zip(atom_indices, colors)):
            pos = bead_positions[:, atom_idx, :]  # (n_beads, 3)
            label = symbols[atom_idx] if symbols else f"Atom {atom_idx}"

            # Plot magnitude of displacement from first bead
            displacement = np.linalg.norm(pos - pos[0], axis=1)
            ax.plot(bead_indices, displacement, color=color, marker='o',
                    markersize=5, linewidth=2, label=label, **kwargs)

    ax.set_xlabel('Bead Index', fontsize=12)
    ax.set_ylabel('Displacement (Å)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_ring_polymer(
    bead_positions: np.ndarray,
    atom_index: int = 0,
    projection: str = "xy",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    show_springs: bool = True,
    color: str = "steelblue",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot ring-polymer beads for a single atom.

    Visualizes the quantum delocalization of an atom.

    Args:
        bead_positions: Array of shape (n_beads, n_atoms, 3)
        atom_index: Which atom to visualize
        projection: "xy", "xz", "yz", or "3d"
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        show_springs: Draw spring connections between beads
        color: Bead color
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    n_beads = bead_positions.shape[0]
    pos = bead_positions[:, atom_index, :]  # (n_beads, 3)

    # Select projection
    proj_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    labels_map = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}

    if projection == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot beads
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                  c=color, s=100, alpha=0.8, **kwargs)

        # Plot springs
        if show_springs:
            for i in range(n_beads):
                j = (i + 1) % n_beads
                ax.plot([pos[i, 0], pos[j, 0]],
                       [pos[i, 1], pos[j, 1]],
                       [pos[i, 2], pos[j, 2]],
                       'k-', alpha=0.3, linewidth=1)

        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')

    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        idx1, idx2 = proj_map[projection]
        x_data = pos[:, idx1]
        y_data = pos[:, idx2]

        # Plot springs first (so beads are on top)
        if show_springs:
            for i in range(n_beads):
                j = (i + 1) % n_beads
                ax.plot([x_data[i], x_data[j]],
                       [y_data[i], y_data[j]],
                       'k-', alpha=0.3, linewidth=1)

        # Plot beads
        ax.scatter(x_data, y_data, c=color, s=100, alpha=0.8,
                  zorder=5, **kwargs)

        # Number the beads
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                       color='white', fontweight='bold')

        xlabel, ylabel = labels_map[projection]
        ax.set_xlabel(f'{xlabel} (Å)', fontsize=12)
        ax.set_ylabel(f'{ylabel} (Å)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Ring Polymer (Atom {atom_index}, {n_beads} beads)', fontsize=14)

    plt.tight_layout()
    return fig, ax


def plot_instanton_energy(
    bead_energies: np.ndarray,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Instanton Energy Profile",
    units: str = "kcal/mol",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot energy of each bead along instanton path.

    Args:
        bead_energies: Array of energies for each bead (in Hartree)
        ax: Existing axes
        figsize: Figure size
        title: Plot title
        units: Energy units
        **kwargs: Additional plot arguments

    Returns:
        Tuple of (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    from ..core.constants import HARTREE_TO_KCAL

    n_beads = len(bead_energies)
    bead_indices = np.arange(n_beads)

    # Convert units
    if units == "kcal/mol":
        energies = (bead_energies - np.min(bead_energies)) * HARTREE_TO_KCAL
        ylabel = "Energy (kcal/mol)"
    else:
        energies = bead_energies - np.min(bead_energies)
        ylabel = "Energy (Hartree)"

    ax.plot(bead_indices, energies, 'b-o', linewidth=2, markersize=6, **kwargs)
    ax.fill_between(bead_indices, 0, energies, alpha=0.2)

    # Mark barrier (maximum)
    max_idx = np.argmax(energies)
    ax.axhline(energies[max_idx], color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'Max: {energies[max_idx]:.2f}',
               xy=(max_idx, energies[max_idx]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, color='red')

    ax.set_xlabel('Bead Index', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n_beads - 0.5)

    plt.tight_layout()
    return fig, ax


def plot_instanton_action(
    temperatures: np.ndarray,
    actions: np.ndarray,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "Instanton Action vs Temperature",
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot instanton action S vs temperature.

    The crossover temperature Tc is where S = 2π.

    Args:
        temperatures: Array of temperatures in Kelvin
        actions: Array of dimensionless actions S/ℏ
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

    ax.plot(temperatures, actions, 'b-o', linewidth=2, markersize=6, **kwargs)

    # Crossover at S = 2π
    ax.axhline(2 * np.pi, color='red', linestyle='--', alpha=0.5,
              label='Crossover (S = 2π)')

    # Find crossover temperature
    if np.any(actions > 2 * np.pi) and np.any(actions < 2 * np.pi):
        idx = np.argmin(np.abs(actions - 2 * np.pi))
        tc = temperatures[idx]
        ax.axvline(tc, color='green', linestyle=':', alpha=0.5)
        ax.annotate(f'Tc ≈ {tc:.0f} K',
                   xy=(tc, 2 * np.pi),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='green')

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Action S/ℏ', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def create_instanton_summary_figure(
    bead_positions: np.ndarray,
    bead_energies: np.ndarray,
    atom_index: int = 0,
    symbols: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 10)
) -> Figure:
    """
    Create a multi-panel summary of instanton calculation.

    Args:
        bead_positions: Array of shape (n_beads, n_atoms, 3)
        bead_energies: Array of bead energies
        atom_index: Atom to highlight in ring-polymer plot
        symbols: Atomic symbols
        figsize: Figure size

    Returns:
        Figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Energy profile
    ax1 = fig.add_subplot(221)
    plot_instanton_energy(bead_energies, ax=ax1)

    # Ring polymer (xy projection)
    ax2 = fig.add_subplot(222)
    plot_ring_polymer(bead_positions, atom_index=atom_index,
                     projection="xy", ax=ax2)

    # Path along reaction coordinate
    ax3 = fig.add_subplot(223)
    plot_instanton_path(bead_positions, atom_indices=[atom_index],
                       symbols=symbols, ax=ax3)

    # Ring polymer (xz projection)
    ax4 = fig.add_subplot(224)
    plot_ring_polymer(bead_positions, atom_index=atom_index,
                     projection="xz", ax=ax4)

    fig.suptitle('Instanton Summary', fontsize=16, y=1.02)
    plt.tight_layout()

    return fig
