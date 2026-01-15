"""Proper reduced mass calculations for internal coordinates."""

import numpy as np
from typing import List, Tuple
from .structure import Molecule
from ..core.constants import AMU_TO_KG


def calculate_torsional_reduced_mass(
    mol: Molecule,
    dihedral_atoms: List[int]
) -> float:
    """
    Calculate the reduced mass for torsional motion around a bond.

    This computes the proper moment of inertia-based reduced mass for
    internal rotation around the j-k bond axis (for dihedral i-j-k-l).

    The reduced mass for torsion is:
        I_red = I_A * I_B / (I_A + I_B)

    where I_A and I_B are the moments of inertia of the two fragments
    about the rotation axis.

    Args:
        mol: Molecule object
        dihedral_atoms: List of 4 atom indices [i, j, k, l] defining the torsion

    Returns:
        Reduced mass in AMU

    Note:
        For H2O2 with atom order [O, O, H, H] and dihedral [2, 0, 1, 3],
        the torsion is H-O-O-H, rotating one OH group relative to the other.
    """
    if len(dihedral_atoms) != 4:
        raise ValueError("dihedral_atoms must have exactly 4 indices")

    i, j, k, l = dihedral_atoms
    coords = mol.coordinates
    masses = mol.masses

    # Define the rotation axis (j-k bond)
    axis_point = coords[j]
    axis_vector = coords[k] - coords[j]
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Partition atoms into two groups:
    # Group A: atoms connected to j (including j), excluding k
    # Group B: atoms connected to k (including k), excluding j
    group_a, group_b = _partition_molecule(mol, j, k)

    # Calculate moment of inertia for each group about the axis
    I_a = _moment_of_inertia_about_axis(coords, masses, group_a, axis_point, axis_vector)
    I_b = _moment_of_inertia_about_axis(coords, masses, group_b, axis_point, axis_vector)

    # Reduced moment of inertia
    if I_a + I_b < 1e-10:
        raise ValueError("Total moment of inertia is too small")

    I_reduced = (I_a * I_b) / (I_a + I_b)

    return I_reduced


def calculate_torsional_reduced_mass_simple(
    mol: Molecule,
    dihedral_atoms: List[int]
) -> float:
    """
    Simplified reduced mass for torsion (approximate).

    Uses the two terminal atoms (i and l) of the dihedral as the
    effective masses for a simple two-body reduced mass:
        mu = m_i * m_l / (m_i + m_l)

    This is a rough approximation useful for quick estimates.

    Args:
        mol: Molecule object
        dihedral_atoms: List of 4 atom indices [i, j, k, l]

    Returns:
        Reduced mass in AMU
    """
    i, j, k, l = dihedral_atoms
    m_i = mol.atoms[i].mass
    m_l = mol.atoms[l].mass
    return (m_i * m_l) / (m_i + m_l)


def _partition_molecule(
    mol: Molecule,
    j: int,
    k: int,
    bond_threshold: float = 1.8
) -> Tuple[set, set]:
    """
    Partition molecule into two groups connected to j and k respectively.

    Uses simple distance-based connectivity.

    Returns:
        Tuple of (atoms connected to j, atoms connected to k)
    """
    coords = mol.coordinates
    n_atoms = mol.num_atoms

    # Build adjacency list
    adjacency = {i: set() for i in range(n_atoms)}
    for a in range(n_atoms):
        for b in range(a + 1, n_atoms):
            dist = np.linalg.norm(coords[a] - coords[b])
            if dist < bond_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # BFS from j, not crossing the j-k bond
    group_a = set()
    to_visit = [j]
    while to_visit:
        current = to_visit.pop()
        if current in group_a:
            continue
        group_a.add(current)
        for neighbor in adjacency[current]:
            if neighbor != k and neighbor not in group_a:
                to_visit.append(neighbor)

    # BFS from k, not crossing the j-k bond
    group_b = set()
    to_visit = [k]
    while to_visit:
        current = to_visit.pop()
        if current in group_b:
            continue
        group_b.add(current)
        for neighbor in adjacency[current]:
            if neighbor != j and neighbor not in group_b:
                to_visit.append(neighbor)

    return group_a, group_b


def _moment_of_inertia_about_axis(
    coords: np.ndarray,
    masses: np.ndarray,
    atom_indices: set,
    axis_point: np.ndarray,
    axis_vector: np.ndarray
) -> float:
    """
    Calculate moment of inertia of selected atoms about an axis.

    I = sum_i m_i * r_i^2

    where r_i is the perpendicular distance from atom i to the axis.

    Args:
        coords: Nx3 coordinate array
        masses: N-element mass array
        atom_indices: Set of atom indices to include
        axis_point: A point on the rotation axis
        axis_vector: Unit vector along the rotation axis

    Returns:
        Moment of inertia in AMU*Angstrom^2
    """
    I_total = 0.0

    for idx in atom_indices:
        # Vector from axis point to atom
        r_vec = coords[idx] - axis_point

        # Component along axis
        parallel = np.dot(r_vec, axis_vector) * axis_vector

        # Perpendicular distance
        perpendicular = r_vec - parallel
        r_perp = np.linalg.norm(perpendicular)

        # Contribution to moment of inertia
        I_total += masses[idx] * r_perp**2

    return I_total


def reduced_mass_in_kg(reduced_mass_amu: float) -> float:
    """Convert reduced mass from AMU to kg."""
    return reduced_mass_amu * AMU_TO_KG


def reduced_mass_for_stretch(mass1: float, mass2: float) -> float:
    """
    Calculate reduced mass for a stretching vibration.

    Args:
        mass1, mass2: Atomic masses in AMU

    Returns:
        Reduced mass in AMU
    """
    return (mass1 * mass2) / (mass1 + mass2)
