"""Rate calculations with classical and quantum corrections."""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
import logging

from ..tunneling.base import TunnelingResult
from ..core.constants import BOLTZMANN_SI, HARTREE_TO_JOULE, R_SI

logger = logging.getLogger(__name__)


@dataclass
class RateResult:
    """
    Results from rate calculations.

    Attributes:
        temperatures: Array of temperatures in Kelvin
        classical_rates: Classical TST rates (s^-1)
        quantum_rates: Quantum-corrected rates (s^-1)
        tunneling_corrections: Tunneling correction factors κ
        method: Tunneling method used
        barrier_height: Barrier height in kcal/mol
        prefactor: Arrhenius prefactor used (s^-1)
    """
    temperatures: np.ndarray
    classical_rates: np.ndarray
    quantum_rates: np.ndarray
    tunneling_corrections: np.ndarray
    method: str = ""
    barrier_height: float = 0.0
    prefactor: float = 1e13

    @property
    def n_temperatures(self) -> int:
        """Number of temperature points."""
        return len(self.temperatures)

    def get_rate_at_temperature(self, T: float, quantum: bool = True) -> float:
        """Interpolate rate at given temperature."""
        rates = self.quantum_rates if quantum else self.classical_rates
        return float(np.interp(T, self.temperatures, rates))

    def get_tunneling_correction_at_T(self, T: float) -> float:
        """Interpolate tunneling correction at temperature."""
        return float(np.interp(T, self.temperatures, self.tunneling_corrections))

    def calculate_kie(self, other: "RateResult") -> np.ndarray:
        """
        Calculate kinetic isotope effect (KIE).

        KIE = k_light / k_heavy

        Args:
            other: RateResult for heavier isotopologue

        Returns:
            Array of KIE values at each temperature
        """
        # Interpolate other rates to match our temperatures
        other_rates = np.interp(
            self.temperatures,
            other.temperatures,
            other.quantum_rates
        )
        return self.quantum_rates / other_rates

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "temperatures": self.temperatures.tolist(),
            "classical_rates": self.classical_rates.tolist(),
            "quantum_rates": self.quantum_rates.tolist(),
            "tunneling_corrections": self.tunneling_corrections.tolist(),
            "method": self.method,
            "barrier_height": self.barrier_height,
            "prefactor": self.prefactor,
        }


def calculate_classical_rate(
    barrier_height: float,
    temperature: float,
    prefactor: float = 1e13
) -> float:
    """
    Calculate classical Transition State Theory (TST) rate.

    k_classical = A * exp(-E_a / RT)

    Args:
        barrier_height: Activation energy in Hartree
        temperature: Temperature in Kelvin
        prefactor: Arrhenius prefactor A in s^-1

    Returns:
        Classical rate constant in s^-1
    """
    if temperature <= 0:
        return 0.0

    # Convert barrier to J/mol
    Ea_J_per_mol = barrier_height * HARTREE_TO_JOULE * 6.02214076e23

    # Arrhenius expression
    exponent = -Ea_J_per_mol / (R_SI * temperature)

    if exponent < -700:
        return 0.0

    return prefactor * np.exp(exponent)


def calculate_tunneling_correction(
    tunneling_result: TunnelingResult,
    barrier_height: float,
    temperature: float
) -> float:
    """
    Calculate Boltzmann-weighted tunneling correction factor κ.

    κ = <T(E)>_Boltzmann / T(E_b)

    where <T(E)> is the Boltzmann average of the transmission coefficient
    and T(E_b) is transmission at the barrier height.

    For accurate tunneling-corrected rates:
        k_quantum = k_classical * κ

    Args:
        tunneling_result: Tunneling calculation result
        barrier_height: Barrier height in Hartree
        temperature: Temperature in Kelvin

    Returns:
        Tunneling correction factor κ
    """
    if temperature <= 0 or barrier_height <= 0:
        return 1.0

    kT = BOLTZMANN_SI * temperature  # J
    barrier_J = barrier_height * HARTREE_TO_JOULE

    energies = tunneling_result.energies
    transmissions = tunneling_result.transmissions

    # Convert energies to Joules
    energies_J = energies * HARTREE_TO_JOULE

    # Boltzmann weights
    boltzmann_weights = np.exp(-(barrier_J - energies_J) / kT)

    # Weighted average of transmission
    # Only include energies below barrier for meaningful average
    mask = energies < barrier_height
    if not np.any(mask):
        return 1.0

    numerator = np.sum(transmissions[mask] * boltzmann_weights[mask])
    denominator = np.sum(boltzmann_weights[mask])

    if denominator < 1e-300:
        return 1.0

    kappa = numerator / denominator

    # kappa should be >= 1 (tunneling enhances rate)
    return max(1.0, kappa)


def calculate_quantum_rate(
    barrier_height: float,
    temperature: float,
    tunneling_result: TunnelingResult,
    prefactor: float = 1e13
) -> float:
    """
    Calculate quantum-corrected rate with tunneling.

    k_quantum = A * κ(T) * exp(-E_a / RT)

    where κ is the tunneling correction factor.

    Args:
        barrier_height: Activation energy in Hartree
        temperature: Temperature in Kelvin
        tunneling_result: Tunneling calculation result
        prefactor: Arrhenius prefactor in s^-1

    Returns:
        Quantum-corrected rate constant in s^-1
    """
    k_classical = calculate_classical_rate(barrier_height, temperature, prefactor)
    kappa = calculate_tunneling_correction(
        tunneling_result, barrier_height, temperature
    )
    return k_classical * kappa


def calculate_rates_vs_temperature(
    tunneling_result: TunnelingResult,
    temperatures: List[float],
    prefactor: float = 1e13
) -> RateResult:
    """
    Calculate classical and quantum rates over temperature range.

    Args:
        tunneling_result: Tunneling calculation result
        temperatures: List of temperatures in Kelvin
        prefactor: Arrhenius prefactor in s^-1

    Returns:
        RateResult with rates at all temperatures
    """
    temperatures = np.array(temperatures)
    barrier_height = tunneling_result.barrier_height

    classical_rates = []
    quantum_rates = []
    tunneling_corrections = []

    logger.info(f"Calculating rates for {len(temperatures)} temperatures")

    for T in temperatures:
        k_class = calculate_classical_rate(barrier_height, T, prefactor)
        kappa = calculate_tunneling_correction(tunneling_result, barrier_height, T)
        k_quantum = k_class * kappa

        classical_rates.append(k_class)
        quantum_rates.append(k_quantum)
        tunneling_corrections.append(kappa)

    # Convert barrier to kcal/mol for display
    from ..core.constants import HARTREE_TO_KCAL
    barrier_kcal = barrier_height * HARTREE_TO_KCAL

    return RateResult(
        temperatures=temperatures,
        classical_rates=np.array(classical_rates),
        quantum_rates=np.array(quantum_rates),
        tunneling_corrections=np.array(tunneling_corrections),
        method=tunneling_result.method,
        barrier_height=barrier_kcal,
        prefactor=prefactor
    )


def calculate_kie_vs_temperature(
    rate_light: RateResult,
    rate_heavy: RateResult
) -> Dict[str, Any]:
    """
    Calculate kinetic isotope effect (KIE) over temperature range.

    KIE = k_light / k_heavy

    Larger KIE indicates more tunneling contribution.

    Args:
        rate_light: Rate result for lighter isotopologue (e.g., H2O2)
        rate_heavy: Rate result for heavier isotopologue (e.g., D2O2)

    Returns:
        Dict with KIE analysis results
    """
    # Use common temperature range
    T_min = max(rate_light.temperatures.min(), rate_heavy.temperatures.min())
    T_max = min(rate_light.temperatures.max(), rate_heavy.temperatures.max())

    common_T = np.linspace(T_min, T_max, 50)

    light_rates = np.interp(common_T, rate_light.temperatures, rate_light.quantum_rates)
    heavy_rates = np.interp(common_T, rate_heavy.temperatures, rate_heavy.quantum_rates)

    kie = light_rates / heavy_rates

    # Classical KIE (no tunneling)
    light_classical = np.interp(common_T, rate_light.temperatures, rate_light.classical_rates)
    heavy_classical = np.interp(common_T, rate_heavy.temperatures, rate_heavy.classical_rates)
    kie_classical = light_classical / heavy_classical

    return {
        "temperatures": common_T,
        "kie_quantum": kie,
        "kie_classical": kie_classical,
        "tunneling_contribution": kie / kie_classical,
        "max_kie": float(np.max(kie)),
        "kie_at_300K": float(np.interp(300, common_T, kie)),
    }
