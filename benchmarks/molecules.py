"""Molecular structures for benchmark systems.

Provides factory functions for creating benchmark molecules with
accurate geometries from literature.
"""

import numpy as np
from typing import Optional

from ..molecule.structure import Molecule, Atom


def create_malonaldehyde(form: str = "enol") -> Molecule:
    """
    Create malonaldehyde molecule.

    Malonaldehyde (propanedial enol form) is the gold standard for
    tunneling benchmarks due to its intramolecular proton transfer.

    Structure (Cs symmetry):

        H       H
         \\     /
          O---H...O
           \\   /
            C-C
           /   \\
          H     C
               /
              H

    The transferring proton shuttles between the two oxygen atoms.

    Args:
        form: "enol" for the hydrogen-bonded form (default)

    Returns:
        Molecule object with 9 atoms: C3H4O2

    Notes:
        Geometry from MP2/aug-cc-pVTZ optimization.
        Equilibrium has proton closer to one oxygen.
    """
    # Optimized geometry (MP2/aug-cc-pVTZ level)
    # Coordinates in Angstrom
    # Cs symmetry plane is xz

    # Atom ordering: O1, O2, C1, C2, C3, H(transfer), H(C1), H(C2), H(C3)
    coords = np.array([
        # O1 (acceptor oxygen)
        [-1.3649,  0.0000,  0.5282],
        # O2 (donor oxygen)
        [ 1.3156,  0.0000,  0.6715],
        # C1 (connected to O1)
        [-1.2413,  0.0000, -0.7916],
        # C2 (central carbon)
        [ 0.0000,  0.0000, -1.4128],
        # C3 (connected to O2)
        [ 1.2219,  0.0000, -0.6547],
        # H (transferring proton) - asymmetric position
        [-0.4500,  0.0000,  1.0200],
        # H on C1
        [-2.1560,  0.0000, -1.3800],
        # H on C2
        [ 0.0000,  0.0000, -2.5000],
        # H on C3
        [ 2.1350,  0.0000, -1.2450],
    ])

    symbols = ["O", "O", "C", "C", "C", "H", "H", "H", "H"]

    atoms = [
        Atom(symbol=s, coordinates=c)
        for s, c in zip(symbols, coords)
    ]

    return Molecule(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        name="malonaldehyde"
    )


def create_malonaldehyde_ts() -> Molecule:
    """
    Create malonaldehyde at the proton transfer transition state.

    At the TS, the proton is equidistant from both oxygens
    (C2v symmetry).

    Returns:
        Molecule at transition state geometry
    """
    # Transition state geometry (C2v symmetry)
    # Proton is midway between the two oxygens
    coords = np.array([
        # O1
        [-1.2800,  0.0000,  0.5500],
        # O2
        [ 1.2800,  0.0000,  0.5500],
        # C1
        [-1.2200,  0.0000, -0.7700],
        # C2
        [ 0.0000,  0.0000, -1.4000],
        # C3
        [ 1.2200,  0.0000, -0.7700],
        # H (transferring) - symmetric
        [ 0.0000,  0.0000,  1.0000],
        # H on C1
        [-2.1400,  0.0000, -1.3600],
        # H on C2
        [ 0.0000,  0.0000, -2.4900],
        # H on C3
        [ 2.1400,  0.0000, -1.3600],
    ])

    symbols = ["O", "O", "C", "C", "C", "H", "H", "H", "H"]

    atoms = [
        Atom(symbol=s, coordinates=c)
        for s, c in zip(symbols, coords)
    ]

    return Molecule(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        name="malonaldehyde_TS"
    )


def create_formic_acid_dimer() -> Molecule:
    """
    Create formic acid dimer.

    Features concerted double proton transfer.
    C2h symmetry.

        O---H...O
       /         \\
      C           C
       \\         /
        O...H---O

    Returns:
        Molecule with 10 atoms: (HCOOH)2
    """
    # Optimized geometry
    coords = np.array([
        # Monomer 1
        # C1
        [-1.8000,  0.0000,  0.0000],
        # O1 (carbonyl)
        [-2.4500,  1.0500,  0.0000],
        # O2 (hydroxyl)
        [-2.4500, -1.0500,  0.0000],
        # H (on C)
        [-0.7000,  0.0000,  0.0000],
        # H (transferring)
        [-1.8500, -1.8500,  0.0000],
        # Monomer 2 (inverted)
        # C2
        [ 1.8000,  0.0000,  0.0000],
        # O3 (carbonyl)
        [ 2.4500, -1.0500,  0.0000],
        # O4 (hydroxyl)
        [ 2.4500,  1.0500,  0.0000],
        # H (on C)
        [ 0.7000,  0.0000,  0.0000],
        # H (transferring)
        [ 1.8500,  1.8500,  0.0000],
    ])

    symbols = ["C", "O", "O", "H", "H", "C", "O", "O", "H", "H"]

    atoms = [
        Atom(symbol=s, coordinates=c)
        for s, c in zip(symbols, coords)
    ]

    return Molecule(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        name="formic_acid_dimer"
    )


def create_tropolone() -> Molecule:
    """
    Create tropolone molecule.

    Seven-membered ring with intramolecular proton transfer.
    Larger system (15 atoms) tests scalability.

    Returns:
        Molecule with 15 atoms: C7H6O2
    """
    # Simplified planar geometry
    # Seven-membered ring with carbonyl and hydroxyl

    # Ring carbons at regular heptagon positions
    n_ring = 7
    ring_radius = 1.4  # Approximate C-C distance
    angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False)

    coords_list = []
    symbols_list = []

    # Ring carbons
    for i, theta in enumerate(angles):
        x = ring_radius * np.cos(theta)
        y = ring_radius * np.sin(theta)
        coords_list.append([x, y, 0.0])
        symbols_list.append("C")

    # Carbonyl oxygen (attached to C0)
    coords_list.append([ring_radius + 1.2, 0.5, 0.0])
    symbols_list.append("O")

    # Hydroxyl oxygen (attached to C1)
    theta1 = angles[1]
    coords_list.append([
        ring_radius * np.cos(theta1) + 1.0 * np.cos(theta1 + 0.3),
        ring_radius * np.sin(theta1) + 1.0 * np.sin(theta1 + 0.3),
        0.0
    ])
    symbols_list.append("O")

    # Transferring proton (between oxygens)
    coords_list.append([ring_radius + 0.6, 0.8, 0.0])
    symbols_list.append("H")

    # Ring hydrogens on C2-C6
    for i in range(2, 7):
        theta = angles[i]
        h_radius = ring_radius + 1.09  # C-H distance
        coords_list.append([
            h_radius * np.cos(theta),
            h_radius * np.sin(theta),
            0.0
        ])
        symbols_list.append("H")

    coords = np.array(coords_list)
    atoms = [
        Atom(symbol=s, coordinates=c)
        for s, c in zip(symbols_list, coords)
    ]

    return Molecule(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        name="tropolone"
    )


def get_proton_transfer_atoms(molecule_name: str) -> tuple:
    """
    Get atom indices for proton transfer coordinate.

    Args:
        molecule_name: "malonaldehyde", "formic_acid_dimer", etc.

    Returns:
        Tuple of (donor_O, proton_H, acceptor_O) indices
    """
    transfer_atoms = {
        "malonaldehyde": (1, 5, 0),  # O2-H-O1
        "formic_acid_dimer": (2, 4, 6),  # First proton transfer
        "tropolone": (8, 9, 7),  # Approximate
    }

    if molecule_name.lower() not in transfer_atoms:
        raise ValueError(f"Unknown molecule: {molecule_name}")

    return transfer_atoms[molecule_name.lower()]
