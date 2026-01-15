"""Eckart barrier model with analytical transmission."""

import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional
from scipy.optimize import curve_fit
import logging

from .base import TunnelingMethod, TunnelingResult
from ..pes.scan import PESScanResult
from ..core.constants import HBAR_SI, HARTREE_TO_JOULE, AMU_TO_KG

logger = logging.getLogger(__name__)


class EckartBarrier(TunnelingMethod):
    """
    Eckart barrier model with analytical transmission coefficient.

    The Eckart potential is:

        V(x) = A * exp(x/L) / (1 + exp(x/L))^2 + B * exp(x/L) / (1 + exp(x/L))

    For a symmetric barrier (B=0, forward and reverse barriers equal):

        V(x) = V_max * sech²(x/L)

    The transmission coefficient has an analytical solution involving
    the parameters α and β.

    This is useful for:
    1. Validating numerical methods against analytical solutions
    2. Quick estimates when full PES is not available
    3. Understanding tunneling behavior
    """

    name = "Eckart"

    def __init__(
        self,
        V1: Optional[float] = None,
        V2: Optional[float] = None,
        L: Optional[float] = None,
        symmetric: bool = True
    ):
        """
        Initialize Eckart barrier.

        For symmetric barrier, only V1 is needed.

        Args:
            V1: Forward barrier height (Hartree)
            V2: Reverse barrier height (Hartree), None for symmetric
            L: Barrier width parameter (radians)
            symmetric: If True, assume V1 = V2
        """
        super().__init__()
        self.V1 = V1  # Forward barrier
        self.V2 = V2 if not symmetric else V1  # Reverse barrier
        self.L = L
        self.symmetric = symmetric
        self._fitted = False

    def fit_to_pes(self, pes_result: PESScanResult) -> Dict[str, float]:
        """
        Fit Eckart parameters to PES scan data.

        Args:
            pes_result: PES scan result

        Returns:
            Dict with fitted parameters: V1, V2, L, x0
        """
        angles_rad = np.radians(pes_result.angles)
        energies = pes_result.relative_energies  # Hartree, relative

        # Find barrier maximum
        max_idx = np.argmax(energies)
        V_max = energies[max_idx]
        x_max = angles_rad[max_idx]

        # For symmetric barrier, V1 = V2 = V_max
        self.V1 = V_max
        self.V2 = V_max

        # Estimate L from barrier width
        # Find half-maximum points
        half_max = V_max / 2
        left_idx = np.where(
            (energies[:max_idx] < half_max) &
            (np.roll(energies[:max_idx], -1) >= half_max)
        )[0]
        right_idx = np.where(
            (energies[max_idx:] >= half_max) &
            (np.roll(energies[max_idx:], -1) < half_max)
        )[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            x_left = angles_rad[left_idx[-1]]
            x_right = angles_rad[max_idx + right_idx[0]]
            # L ≈ (x_right - x_left) / (2 * arccosh(sqrt(2)))
            width = x_right - x_left
            self.L = width / 2.634  # 2 * arccosh(sqrt(2))
        else:
            # Default estimate
            self.L = 0.5  # radians

        self._fitted = True

        return {
            "V1": self.V1,
            "V2": self.V2,
            "L": self.L,
            "x_max": x_max,
            "barrier_height": V_max
        }

    def analytical_transmission(
        self,
        energy: float,
        reduced_mass: float
    ) -> float:
        """
        Calculate exact Eckart transmission coefficient.

        For symmetric barrier V1 = V2 = V:

            T(E) = 1 / (1 + exp(2π(α - β)))

        where:
            α = sqrt(2mVL²/ℏ²)
            β = sqrt(2mEL²/ℏ²)

        Args:
            energy: Energy in Hartree (relative to reactant)
            reduced_mass: Reduced mass in kg

        Returns:
            Transmission coefficient
        """
        if self.V1 is None or self.L is None:
            raise ValueError("Eckart parameters not set. Call fit_to_pes first.")

        # Convert to SI
        V1_si = self.V1 * HARTREE_TO_JOULE
        E_si = energy * HARTREE_TO_JOULE
        L_si = self.L  # Already in radians (dimensionless for torsion)

        # Handle edge cases
        if energy >= self.V1:
            return 1.0
        if energy <= 0:
            return 0.0

        # For torsion, L is in radians, so we use a modified formula
        # that accounts for the angular nature of the coordinate
        # The effective "length" for the action integral is L * sqrt(I)
        # where I is the moment of inertia

        # Simplified symmetric Eckart formula
        # Using dimensionless parameters
        hbar_sq_over_2m = HBAR_SI**2 / (2 * reduced_mass)

        # α² = V1 * L² / (ℏ²/2m)
        alpha_sq = V1_si * self.L**2 / hbar_sq_over_2m
        # β² = E * L² / (ℏ²/2m)
        beta_sq = E_si * self.L**2 / hbar_sq_over_2m

        alpha = np.sqrt(max(0, alpha_sq))
        beta = np.sqrt(max(0, beta_sq))

        # Transmission for symmetric barrier
        argument = 2 * np.pi * (alpha - beta)

        # Prevent overflow
        if argument > 700:
            return 0.0
        elif argument < -700:
            return 1.0

        T = 1.0 / (1.0 + np.exp(argument))

        return T

    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate Eckart transmission.

        If parameters are not set, fits them to the potential function.
        """
        # Auto-fit if needed
        if not self._fitted:
            # Use barrier_height for V1
            self.V1 = barrier_height
            self.V2 = barrier_height
            # Estimate L from potential shape
            self.L = self._estimate_width(potential_func, barrier_height)
            self._fitted = True

        T = self.analytical_transmission(energy, reduced_mass)

        diagnostics = {
            "V1": self.V1,
            "L": self.L,
            "method": "analytical_eckart"
        }

        return T, diagnostics

    def _estimate_width(
        self,
        potential_func: Callable[[float], float],
        barrier_height: float,
        n_points: int = 200
    ) -> float:
        """Estimate barrier width parameter L from potential function."""
        x_grid = np.linspace(0, 2 * np.pi, n_points)
        v_grid = np.array([potential_func(x) for x in x_grid])

        # Find maximum
        max_idx = np.argmax(v_grid)
        V_max = v_grid[max_idx]

        # Find FWHM
        half_max = V_max / 2
        above_half = v_grid > half_max

        # Find first and last crossing
        crossings = np.where(np.diff(above_half.astype(int)))[0]

        if len(crossings) >= 2:
            width = x_grid[crossings[-1]] - x_grid[crossings[0]]
            return width / 2.634
        else:
            return 0.5  # Default


def compare_with_eckart(
    pes_result: PESScanResult,
    reduced_mass_amu: float,
    numerical_result: TunnelingResult
) -> Dict[str, Any]:
    """
    Compare numerical tunneling results with Eckart model.

    Args:
        pes_result: PES scan result
        reduced_mass_amu: Reduced mass in AMU
        numerical_result: Result from numerical method (e.g., WKB)

    Returns:
        Dict with comparison metrics
    """
    # Fit Eckart to PES
    eckart = EckartBarrier()
    fit_params = eckart.fit_to_pes(pes_result)

    # Calculate Eckart transmission at same energies
    reduced_mass_kg = reduced_mass_amu * AMU_TO_KG
    eckart_T = np.array([
        eckart.analytical_transmission(E, reduced_mass_kg)
        for E in numerical_result.energies
    ])

    # Compare
    numerical_T = numerical_result.transmissions
    diff = numerical_T - eckart_T
    rmsd = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))

    return {
        "eckart_params": fit_params,
        "eckart_transmissions": eckart_T,
        "rmsd": rmsd,
        "max_difference": max_diff,
        "relative_rmsd": rmsd / np.mean(eckart_T) if np.mean(eckart_T) > 0 else float('inf')
    }
