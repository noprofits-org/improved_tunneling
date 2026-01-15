"""Molecular structure representation with Atom and Molecule dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from copy import deepcopy

from ..core.constants import ATOMIC_MASSES


@dataclass
class Atom:
    """
    Representation of a single atom.

    Attributes:
        symbol: Element symbol (e.g., "H", "O", "C")
        coordinates: 3D position in Angstroms as numpy array
        mass: Atomic mass in AMU (auto-determined from symbol if not provided)
    """
    symbol: str
    coordinates: np.ndarray
    mass: Optional[float] = None

    def __post_init__(self):
        """Ensure coordinates is numpy array and set default mass."""
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates, dtype=float)
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must have shape (3,), got {self.coordinates.shape}")
        if self.mass is None:
            self.mass = ATOMIC_MASSES.get(self.symbol, 1.0)

    @classmethod
    def from_symbol(cls, symbol: str, x: float, y: float, z: float,
                    isotope_mass: Optional[float] = None) -> "Atom":
        """Create an atom from symbol and coordinates."""
        coords = np.array([x, y, z], dtype=float)
        return cls(symbol=symbol, coordinates=coords, mass=isotope_mass)

    def copy(self) -> "Atom":
        """Create a deep copy of this atom."""
        return Atom(
            symbol=self.symbol,
            coordinates=self.coordinates.copy(),
            mass=self.mass
        )

    def distance_to(self, other: "Atom") -> float:
        """Calculate distance to another atom in Angstroms."""
        return float(np.linalg.norm(self.coordinates - other.coordinates))


@dataclass
class Molecule:
    """
    Representation of a molecular structure.

    Attributes:
        atoms: List of Atom objects
        charge: Molecular charge
        multiplicity: Spin multiplicity (2S+1)
        name: Optional name/identifier
        energy: Optional associated energy in Hartree
    """
    atoms: List[Atom]
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    energy: Optional[float] = None

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)

    @property
    def symbols(self) -> List[str]:
        """List of element symbols."""
        return [atom.symbol for atom in self.atoms]

    @property
    def coordinates(self) -> np.ndarray:
        """Nx3 array of atomic coordinates in Angstroms."""
        return np.array([atom.coordinates for atom in self.atoms])

    @coordinates.setter
    def coordinates(self, coords: np.ndarray):
        """Set coordinates from Nx3 array."""
        if coords.shape != (self.num_atoms, 3):
            raise ValueError(f"Expected shape ({self.num_atoms}, 3), got {coords.shape}")
        for i, atom in enumerate(self.atoms):
            atom.coordinates = coords[i].copy()

    @property
    def masses(self) -> np.ndarray:
        """Array of atomic masses in AMU."""
        return np.array([atom.mass for atom in self.atoms])

    @property
    def total_mass(self) -> float:
        """Total molecular mass in AMU."""
        return float(np.sum(self.masses))

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass coordinates."""
        masses = self.masses
        coords = self.coordinates
        return np.sum(masses[:, np.newaxis] * coords, axis=0) / np.sum(masses)

    @property
    def formula(self) -> str:
        """Simple molecular formula."""
        from collections import Counter
        counts = Counter(self.symbols)
        parts = []
        for element in sorted(counts.keys()):
            count = counts[element]
            if count == 1:
                parts.append(element)
            else:
                parts.append(f"{element}{count}")
        return "".join(parts)

    def copy(self) -> "Molecule":
        """Create a deep copy of this molecule."""
        return Molecule(
            atoms=[atom.copy() for atom in self.atoms],
            charge=self.charge,
            multiplicity=self.multiplicity,
            name=self.name,
            energy=self.energy
        )

    def get_atom(self, index: int) -> Atom:
        """Get atom by index."""
        return self.atoms[index]

    def with_new_coordinates(self, coords: np.ndarray) -> "Molecule":
        """Return a new molecule with updated coordinates."""
        mol = self.copy()
        mol.coordinates = coords
        return mol

    def with_isotope_substitution(self, index: int, new_mass: float,
                                   new_symbol: Optional[str] = None) -> "Molecule":
        """Return a new molecule with isotope substitution at given index."""
        mol = self.copy()
        mol.atoms[index].mass = new_mass
        if new_symbol:
            mol.atoms[index].symbol = new_symbol
        return mol

    def translate(self, vector: np.ndarray) -> None:
        """Translate all atoms by a vector (in-place)."""
        vector = np.asarray(vector)
        for atom in self.atoms:
            atom.coordinates = atom.coordinates + vector

    def center_at_origin(self) -> None:
        """Move center of mass to origin (in-place)."""
        self.translate(-self.center_of_mass)

    def to_psi4_geometry(self) -> str:
        """Generate Psi4-compatible geometry string."""
        lines = [f"{self.charge} {self.multiplicity}"]
        for atom in self.atoms:
            x, y, z = atom.coordinates
            lines.append(f"{atom.symbol}  {x:15.10f}  {y:15.10f}  {z:15.10f}")
        return "\n".join(lines)

    @classmethod
    def from_arrays(cls, symbols: List[str], coordinates: np.ndarray,
                    masses: Optional[np.ndarray] = None,
                    charge: int = 0, multiplicity: int = 1,
                    name: str = "") -> "Molecule":
        """Create a molecule from arrays of symbols and coordinates."""
        if len(symbols) != len(coordinates):
            raise ValueError("Number of symbols must match number of coordinates")
        atoms = []
        for i, (sym, coord) in enumerate(zip(symbols, coordinates)):
            mass = masses[i] if masses is not None else None
            atoms.append(Atom(symbol=sym, coordinates=coord, mass=mass))
        return cls(atoms=atoms, charge=charge, multiplicity=multiplicity, name=name)

    @classmethod
    def h2o2(cls, dihedral: float = 111.5) -> "Molecule":
        """
        Create a hydrogen peroxide molecule with specified dihedral angle.

        Args:
            dihedral: H-O-O-H dihedral angle in degrees

        Returns:
            H2O2 Molecule with atoms ordered as [O, O, H, H]
        """
        # Standard H2O2 geometry parameters
        r_oo = 1.47  # O-O bond length in Angstrom
        r_oh = 0.97  # O-H bond length in Angstrom
        angle_ooh = 99.4  # O-O-H angle in degrees

        # Convert to radians
        angle_rad = np.radians(angle_ooh)
        dihedral_rad = np.radians(dihedral)

        # Place first O at origin
        o1 = np.array([0.0, 0.0, 0.0])

        # Place second O along x-axis
        o2 = np.array([r_oo, 0.0, 0.0])

        # Place first H (attached to O1)
        # H1 is in the xz plane
        h1 = np.array([
            -r_oh * np.cos(angle_rad),
            0.0,
            r_oh * np.sin(angle_rad)
        ])

        # Place second H (attached to O2)
        # H2 is rotated by dihedral angle around O-O bond
        h2_local = np.array([
            r_oh * np.cos(angle_rad),
            r_oh * np.sin(angle_rad) * np.sin(dihedral_rad),
            r_oh * np.sin(angle_rad) * np.cos(dihedral_rad)
        ])
        h2 = o2 + h2_local

        atoms = [
            Atom(symbol="O", coordinates=o1),
            Atom(symbol="O", coordinates=o2),
            Atom(symbol="H", coordinates=h1),
            Atom(symbol="H", coordinates=h2),
        ]

        return cls(atoms=atoms, name="H2O2")
