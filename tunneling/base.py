"""Abstract base class for tunneling methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from ..pes.scan import PESScanResult
from ..pes.interpolation import create_potential_function
from ..core.constants import HARTREE_TO_JOULE, AMU_TO_KG, HBAR_SI

logger = logging.getLogger(__name__)


@dataclass
class TunnelingResult:
    """
    Results from tunneling calculation.

    Attributes:
        method: Name of tunneling method used
        energies: Array of energy levels (Hartree, relative to minimum)
        transmissions: Array of transmission coefficients T(E)
        barrier_height: Barrier height in Hartree
        reduced_mass: Reduced mass used (AMU)
        energy_ratios: Energy as fraction of barrier height
        diagnostics: Additional diagnostic information
    """
    method: str
    energies: np.ndarray
    transmissions: np.ndarray
    barrier_height: float
    reduced_mass: float
    energy_ratios: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.energy_ratios is None and self.barrier_height > 0:
            self.energy_ratios = self.energies / self.barrier_height

    @property
    def n_points(self) -> int:
        """Number of energy points."""
        return len(self.energies)

    def get_transmission_at_ratio(self, ratio: float) -> float:
        """Interpolate transmission at given energy ratio."""
        if self.energy_ratios is None:
            raise ValueError("Energy ratios not available")
        return float(np.interp(ratio, self.energy_ratios, self.transmissions))

    def get_transmission_at_energy(self, energy: float) -> float:
        """Interpolate transmission at given absolute energy."""
        return float(np.interp(energy, self.energies, self.transmissions))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "energies": self.energies.tolist(),
            "transmissions": self.transmissions.tolist(),
            "barrier_height": self.barrier_height,
            "reduced_mass": self.reduced_mass,
            "energy_ratios": self.energy_ratios.tolist() if self.energy_ratios is not None else None,
        }


class TunnelingMethod(ABC):
    """
    Abstract base class for tunneling calculation methods.

    Subclasses implement specific approximations (WKB, SCT, Eckart).
    """

    name: str = "base"

    def __init__(
        self,
        integration_tolerance: float = 1e-8,
        max_subdivisions: int = 500
    ):
        """
        Initialize tunneling method.

        Args:
            integration_tolerance: Tolerance for numerical integration
            max_subdivisions: Maximum integration subdivisions
        """
        self.tol = integration_tolerance
        self.max_subdivisions = max_subdivisions

    @abstractmethod
    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate transmission coefficient at a single energy.

        Args:
            energy: Energy level in Hartree (relative to minimum)
            potential_func: Potential V(x) in Hartree, x in radians
            reduced_mass: Reduced mass in kg
            barrier_height: Barrier height in Hartree

        Returns:
            Tuple of (transmission_coefficient, diagnostics)
        """
        pass

    def calculate_all(
        self,
        pes_result: PESScanResult,
        reduced_mass_amu: float,
        energy_points: int = 200,
        min_ratio: float = 0.3,
        max_ratio: float = 1.0
    ) -> TunnelingResult:
        """
        Calculate transmission coefficients for range of energies.

        Args:
            pes_result: PES scan result
            reduced_mass_amu: Reduced mass in AMU
            energy_points: Number of energy levels to evaluate
            min_ratio: Minimum energy as fraction of barrier
            max_ratio: Maximum energy as fraction of barrier

        Returns:
            TunnelingResult with all calculations
        """
        # Create potential function (relative energy, radians)
        potential_func = create_potential_function(
            pes_result, use_radians=True, relative=True
        )

        # Get barrier height
        barrier_height = pes_result.barrier_height  # Hartree

        if barrier_height < 1e-10:
            logger.warning("Barrier height is essentially zero")
            # Return unit transmission for all energies
            energies = np.linspace(0, 1e-6, energy_points)
            return TunnelingResult(
                method=self.name,
                energies=energies,
                transmissions=np.ones(energy_points),
                barrier_height=barrier_height,
                reduced_mass=reduced_mass_amu
            )

        # Convert reduced mass to kg
        reduced_mass_kg = reduced_mass_amu * AMU_TO_KG

        # Generate energy levels
        energy_ratios = np.linspace(min_ratio, max_ratio, energy_points)
        energies = energy_ratios * barrier_height

        # Calculate transmission at each energy
        transmissions = []
        all_diagnostics = []

        logger.info(f"Calculating {self.name} transmission for {energy_points} energies")

        for i, (e_ratio, e_hartree) in enumerate(zip(energy_ratios, energies)):
            try:
                T, diag = self.calculate_transmission(
                    e_hartree,
                    potential_func,
                    reduced_mass_kg,
                    barrier_height
                )
                # Clamp to physical range [0, 1]
                T = max(0.0, min(1.0, T))
            except Exception as ex:
                logger.warning(f"Transmission calc failed at E/Vb={e_ratio:.2f}: {ex}")
                # Use extrapolation or zero
                T = 0.0 if e_ratio < 0.5 else 1.0
                diag = {"error": str(ex)}

            transmissions.append(T)
            all_diagnostics.append(diag)

            if (i + 1) % 50 == 0:
                logger.debug(f"Progress: {i + 1}/{energy_points}")

        return TunnelingResult(
            method=self.name,
            energies=energies,
            transmissions=np.array(transmissions),
            barrier_height=barrier_height,
            reduced_mass=reduced_mass_amu,
            energy_ratios=energy_ratios,
            diagnostics={"per_point": all_diagnostics}
        )
