"""Geometry calculations: distances, angles, dihedrals."""

import numpy as np
from typing import Tuple, Optional, Union
from .structure import Molecule, Atom


def calculate_distance(
    mol_or_coords: Union[Molecule, np.ndarray],
    i: int,
    j: int
) -> float:
    """
    Calculate distance between two atoms.

    Args:
        mol_or_coords: Molecule object or Nx3 coordinate array
        i, j: Atom indices

    Returns:
        Distance in Angstroms
    """
    if isinstance(mol_or_coords, Molecule):
        coords = mol_or_coords.coordinates
    else:
        coords = mol_or_coords
    return float(np.linalg.norm(coords[i] - coords[j]))


def calculate_angle(
    mol_or_coords: Union[Molecule, np.ndarray],
    i: int,
    j: int,
    k: int,
    degrees: bool = True
) -> float:
    """
    Calculate angle i-j-k (angle at atom j).

    Args:
        mol_or_coords: Molecule object or Nx3 coordinate array
        i, j, k: Atom indices (angle vertex at j)
        degrees: If True, return angle in degrees; otherwise radians

    Returns:
        Angle in degrees or radians
    """
    if isinstance(mol_or_coords, Molecule):
        coords = mol_or_coords.coordinates
    else:
        coords = mol_or_coords

    # Vectors from j to i and j to k
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]

    # Normalize
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Angle from dot product
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if degrees:
        return float(np.degrees(angle))
    return float(angle)


def calculate_dihedral(
    mol_or_coords: Union[Molecule, np.ndarray],
    i: int,
    j: int,
    k: int,
    l: int,
    degrees: bool = True
) -> float:
    """
    Calculate dihedral angle i-j-k-l.

    The dihedral angle is the angle between the i-j-k plane and the j-k-l plane.

    Args:
        mol_or_coords: Molecule object or Nx3 coordinate array
        i, j, k, l: Atom indices defining the dihedral
        degrees: If True, return angle in degrees; otherwise radians

    Returns:
        Dihedral angle in degrees or radians, in range [-180, 180]
    """
    if isinstance(mol_or_coords, Molecule):
        coords = mol_or_coords.coordinates
    else:
        coords = mol_or_coords

    # Vectors along the bonds
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]

    # Normal vectors to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        # Degenerate case (atoms collinear)
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Unit vector along b2
    b2_unit = b2 / np.linalg.norm(b2)

    # Calculate dihedral using atan2 for proper quadrant
    m1 = np.cross(n1, b2_unit)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    angle = np.arctan2(y, x)

    if degrees:
        return float(np.degrees(angle))
    return float(angle)


def set_dihedral(
    mol: Molecule,
    i: int,
    j: int,
    k: int,
    l: int,
    target_angle: float,
    degrees: bool = True
) -> Molecule:
    """
    Return a new molecule with dihedral i-j-k-l set to target angle.

    Rotates the l-side of the molecule around the j-k bond axis.
    Only atom l is moved; for full fragment rotation, use set_dihedral_fragment.

    Args:
        mol: Input molecule
        i, j, k, l: Atom indices defining the dihedral
        target_angle: Desired dihedral angle
        degrees: If True, angle is in degrees; otherwise radians

    Returns:
        New Molecule with adjusted dihedral
    """
    if degrees:
        target_rad = np.radians(target_angle)
    else:
        target_rad = target_angle

    # Current dihedral
    current = calculate_dihedral(mol, i, j, k, l, degrees=False)

    # Rotation needed
    rotation_angle = target_rad - current

    # Create copy and rotate
    new_mol = mol.copy()
    coords = new_mol.coordinates

    # Axis of rotation is j-k bond
    axis = coords[k] - coords[j]
    axis = axis / np.linalg.norm(axis)

    # Point of rotation is atom k
    pivot = coords[k].copy()

    # Rotate only atom l around the j-k axis
    coords[l] = _rotate_point_around_axis(coords[l], pivot, axis, rotation_angle)
    new_mol.coordinates = coords

    return new_mol


def set_dihedral_fragment(
    mol: Molecule,
    i: int,
    j: int,
    k: int,
    l: int,
    target_angle: float,
    degrees: bool = True
) -> Molecule:
    """
    Return a new molecule with dihedral i-j-k-l set to target angle.

    Rotates all atoms on the l-side of the j-k bond around the j-k axis.

    Args:
        mol: Input molecule
        i, j, k, l: Atom indices defining the dihedral
        target_angle: Desired dihedral angle
        degrees: If True, angle is in degrees; otherwise radians

    Returns:
        New Molecule with adjusted dihedral
    """
    if degrees:
        target_rad = np.radians(target_angle)
    else:
        target_rad = target_angle

    # Current dihedral
    current = calculate_dihedral(mol, i, j, k, l, degrees=False)
    rotation_angle = target_rad - current

    # Create copy
    new_mol = mol.copy()
    coords = new_mol.coordinates

    # Axis of rotation is j-k bond
    axis = coords[k] - coords[j]
    axis = axis / np.linalg.norm(axis)
    pivot = coords[k].copy()

    # Find atoms on the l-side of the j-k bond
    l_side_atoms = _find_connected_atoms(mol, k, exclude=j)

    # Rotate all l-side atoms
    for atom_idx in l_side_atoms:
        coords[atom_idx] = _rotate_point_around_axis(
            coords[atom_idx], pivot, axis, rotation_angle
        )

    new_mol.coordinates = coords
    return new_mol


def _rotate_point_around_axis(
    point: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Rotate a point around an axis passing through a pivot point.

    Uses Rodrigues' rotation formula.

    Args:
        point: 3D point to rotate
        pivot: Point on the rotation axis
        axis: Unit vector of rotation axis
        angle: Rotation angle in radians

    Returns:
        Rotated point
    """
    # Translate to pivot
    p = point - pivot

    # Rodrigues' formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotated = (p * cos_a +
               np.cross(axis, p) * sin_a +
               axis * np.dot(axis, p) * (1 - cos_a))

    # Translate back
    return rotated + pivot


def _find_connected_atoms(
    mol: Molecule,
    start: int,
    exclude: int,
    bond_threshold: float = 1.8
) -> set:
    """
    Find all atoms connected to 'start' without going through 'exclude'.

    Uses simple distance-based connectivity (bond_threshold in Angstroms).
    """
    coords = mol.coordinates
    n_atoms = mol.num_atoms

    # Build connectivity
    connected = set([start])
    to_visit = [start]

    while to_visit:
        current = to_visit.pop()
        for other in range(n_atoms):
            if other in connected or other == exclude:
                continue
            dist = np.linalg.norm(coords[current] - coords[other])
            if dist < bond_threshold:
                connected.add(other)
                to_visit.append(other)

    return connected
