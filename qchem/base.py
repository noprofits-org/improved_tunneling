"""Abstract base class for quantum chemistry engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from ..molecule.structure import Molecule
from ..core.config import ComputationalConfig


@dataclass
class QChemResult:
    """
    Result container for quantum chemistry calculations.

    Attributes:
        energy: Electronic energy in Hartree
        molecule: Optimized/computed molecular structure
        converged: Whether the calculation converged
        method: QC method used
        basis: Basis set used
        extras: Additional calculation-specific data
    """
    energy: float
    molecule: Molecule
    converged: bool = True
    method: str = ""
    basis: str = ""
    extras: Dict[str, Any] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}


@dataclass
class FrequencyResult:
    """
    Result container for frequency calculations.

    Attributes:
        frequencies: Harmonic frequencies in cm^-1
        intensities: IR intensities (optional)
        zpe: Zero-point energy in Hartree
        normal_modes: Normal mode displacement vectors
        molecule: Molecular geometry used
    """
    frequencies: np.ndarray
    zpe: float
    molecule: Molecule
    intensities: Optional[np.ndarray] = None
    normal_modes: Optional[np.ndarray] = None

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return len(self.frequencies)

    @property
    def real_frequencies(self) -> np.ndarray:
        """Return only real (positive) frequencies."""
        return self.frequencies[self.frequencies > 0]

    @property
    def imaginary_frequencies(self) -> np.ndarray:
        """Return imaginary frequencies (stored as negative values)."""
        return self.frequencies[self.frequencies < 0]

    @property
    def n_imaginary(self) -> int:
        """Number of imaginary frequencies."""
        return int(np.sum(self.frequencies < 0))


class QChemEngine(ABC):
    """
    Abstract base class for quantum chemistry backends.

    Implementations should provide methods for:
    - Single-point energy calculations
    - Geometry optimization (unconstrained and constrained)
    - Frequency calculations
    """

    def __init__(self):
        self._config: Optional[ComputationalConfig] = None
        self._initialized: bool = False

    @abstractmethod
    def initialize(self, config: ComputationalConfig) -> None:
        """
        Initialize the QC engine with configuration.

        Args:
            config: Computational configuration settings
        """
        pass

    @abstractmethod
    def single_point_energy(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> QChemResult:
        """
        Calculate single-point electronic energy.

        Args:
            molecule: Molecular structure
            method: QC method (default from config)
            basis: Basis set (default from config)

        Returns:
            QChemResult with energy
        """
        pass

    @abstractmethod
    def optimize_geometry(
        self,
        molecule: Molecule,
        constraints: Optional[Dict[str, Any]] = None
    ) -> QChemResult:
        """
        Optimize molecular geometry.

        Args:
            molecule: Starting molecular structure
            constraints: Optional geometry constraints
                Format: {
                    "frozen_dihedral": {"atoms": [i, j, k, l], "value": angle},
                    "frozen_distance": {"atoms": [i, j], "value": distance},
                    ...
                }

        Returns:
            QChemResult with optimized geometry and energy
        """
        pass

    @abstractmethod
    def frequencies(
        self,
        molecule: Molecule
    ) -> FrequencyResult:
        """
        Calculate harmonic vibrational frequencies.

        Args:
            molecule: Molecular structure (should be optimized)

        Returns:
            FrequencyResult with frequencies and ZPE
        """
        pass

    def optimize_and_frequencies(
        self,
        molecule: Molecule,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[QChemResult, FrequencyResult]:
        """
        Convenience method to optimize geometry then calculate frequencies.

        Args:
            molecule: Starting structure
            constraints: Optional geometry constraints

        Returns:
            Tuple of (optimization result, frequency result)
        """
        opt_result = self.optimize_geometry(molecule, constraints)
        freq_result = self.frequencies(opt_result.molecule)
        return opt_result, freq_result

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    def _ensure_initialized(self) -> None:
        """Raise error if engine not initialized."""
        if not self._initialized:
            raise RuntimeError("QChemEngine not initialized. Call initialize() first.")
