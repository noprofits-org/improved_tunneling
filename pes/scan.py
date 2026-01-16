"""Base classes for PES scanning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import logging

from ..molecule.structure import Molecule
from ..qchem.base import QChemEngine
from ..core.config import PESScanConfig
from ..core.constants import HARTREE_TO_KCAL

logger = logging.getLogger(__name__)


@dataclass
class PESScanPoint:
    """
    Single point on the potential energy surface.

    Attributes:
        angle: Dihedral angle in degrees
        energy: Electronic energy in Hartree
        molecule: Molecular geometry at this point
        converged: Whether the calculation converged
        zpe: Optional zero-point energy correction
    """
    angle: float
    energy: float
    molecule: Molecule
    converged: bool = True
    zpe: Optional[float] = None

    @property
    def energy_with_zpe(self) -> float:
        """Energy including ZPE correction if available."""
        if self.zpe is not None:
            return self.energy + self.zpe
        return self.energy


@dataclass
class PESScanResult:
    """
    Complete PES scan results.

    Attributes:
        scan_type: "rigid" or "relaxed"
        dihedral_atoms: Atom indices defining the scanned dihedral
        points: List of PESScanPoint objects
        method: QC method used
        basis: Basis set used
    """
    scan_type: str
    dihedral_atoms: List[int]
    points: List[PESScanPoint]
    method: str = ""
    basis: str = ""

    @property
    def num_points(self) -> int:
        """Number of scan points."""
        return len(self.points)

    @property
    def angles(self) -> np.ndarray:
        """Array of dihedral angles in degrees."""
        return np.array([p.angle for p in self.points])

    @property
    def energies(self) -> np.ndarray:
        """Array of energies in Hartree."""
        return np.array([p.energy for p in self.points])

    @property
    def relative_energies(self) -> np.ndarray:
        """Array of relative energies in Hartree (min = 0)."""
        e = self.energies
        return e - np.min(e)

    @property
    def relative_energies_kcal(self) -> np.ndarray:
        """Array of relative energies in kcal/mol."""
        return self.relative_energies * HARTREE_TO_KCAL

    @property
    def min_energy(self) -> float:
        """Minimum energy in Hartree."""
        return float(np.min(self.energies))

    @property
    def max_energy(self) -> float:
        """Maximum energy in Hartree."""
        return float(np.max(self.energies))

    @property
    def barrier_height(self) -> float:
        """Barrier height (max - min) in Hartree."""
        return self.max_energy - self.min_energy

    @property
    def barrier_height_kcal(self) -> float:
        """Barrier height in kcal/mol."""
        return self.barrier_height * HARTREE_TO_KCAL

    def get_minimum(self) -> PESScanPoint:
        """Get the point with minimum energy."""
        idx = int(np.argmin(self.energies))
        return self.points[idx]

    def get_maximum(self) -> PESScanPoint:
        """Get the point with maximum energy (transition state)."""
        idx = int(np.argmax(self.energies))
        return self.points[idx]

    def get_minima(self) -> List[PESScanPoint]:
        """Find all local minima."""
        minima = []
        e = self.relative_energies_kcal
        n = len(e)

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            if e[i] < e[prev_idx] and e[i] < e[next_idx]:
                minima.append(self.points[i])

        return minima

    def get_maxima(self) -> List[PESScanPoint]:
        """Find all local maxima (barriers)."""
        maxima = []
        e = self.relative_energies_kcal
        n = len(e)

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            if e[i] > e[prev_idx] and e[i] > e[next_idx]:
                maxima.append(self.points[i])

        return maxima

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scan_type": self.scan_type,
            "dihedral_atoms": self.dihedral_atoms,
            "method": self.method,
            "basis": self.basis,
            "angles": self.angles.tolist(),
            "energies": self.energies.tolist(),
            "barrier_height_kcal": self.barrier_height_kcal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PESScanResult":
        """
        Reconstruct PESScanResult from dictionary.

        Note: This creates minimal PESScanPoint objects without full
        molecule data, sufficient for tunneling calculations.
        """
        angles = data.get("angles", [])
        energies = data.get("energies", [])

        # Create minimal PESScanPoints (without full molecule geometry)
        points = []
        for angle, energy in zip(angles, energies):
            points.append(PESScanPoint(
                angle=angle,
                energy=energy,
                molecule=None,  # Not preserved in checkpoint
                converged=True
            ))

        return cls(
            scan_type=data.get("scan_type", "unknown"),
            dihedral_atoms=data.get("dihedral_atoms", []),
            points=points,
            method=data.get("method", ""),
            basis=data.get("basis", "")
        )


class PESScan(ABC):
    """
    Abstract base class for PES scanning.

    Subclasses implement rigid or relaxed scanning strategies.
    """

    def __init__(self, engine: QChemEngine, config: PESScanConfig):
        """
        Initialize PES scanner.

        Args:
            engine: Quantum chemistry engine
            config: PES scan configuration
        """
        self.engine = engine
        self.config = config
        self._results: Optional[PESScanResult] = None

    @property
    @abstractmethod
    def scan_type(self) -> str:
        """Return scan type identifier ("rigid" or "relaxed")."""
        pass

    @abstractmethod
    def run(self, molecule: Molecule) -> PESScanResult:
        """
        Execute the PES scan.

        Args:
            molecule: Starting molecular structure

        Returns:
            PESScanResult with all scan points
        """
        pass

    def _log_progress(self, current: int, total: int, angle: float, energy: float) -> None:
        """Log scan progress."""
        rel_e = (energy - self._min_energy) * HARTREE_TO_KCAL if hasattr(self, '_min_energy') else 0
        logger.info(
            f"Point {current}/{total}: angle={angle:.1f}°, "
            f"energy={energy:.8f} Ha, rel={rel_e:.2f} kcal/mol"
        )
