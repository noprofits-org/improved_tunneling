"""WKB (Wentzel-Kramers-Brillouin) tunneling approximation."""

import numpy as np
from typing import Callable, Dict, Any, Tuple
import logging

from .base import TunnelingMethod, TunnelingResult
from .integration import adaptive_action_integral, find_turning_points_robust
from ..core.constants import HBAR_SI, HARTREE_TO_JOULE

logger = logging.getLogger(__name__)


class WKBMethod(TunnelingMethod):
    """
    WKB semiclassical tunneling approximation.

    The transmission coefficient is calculated as:

        T(E) = exp(-2S/ℏ)

    where S is the action integral:

        S = ∫[x1 to x2] √(2m(V(x) - E)) dx

    integrated between the classical turning points x1 and x2.

    This is accurate for:
    - Energies well below the barrier top
    - Slowly varying potentials
    - Heavy particles (breaks down for very light particles)
    """

    name = "WKB"

    def __init__(
        self,
        integration_tolerance: float = 1e-8,
        max_subdivisions: int = 500,
        x_range: Tuple[float, float] = (0.0, 2 * np.pi)
    ):
        """
        Initialize WKB method.

        Args:
            integration_tolerance: Tolerance for action integral
            max_subdivisions: Max subdivisions for adaptive quadrature
            x_range: Range for turning point search (radians)
        """
        super().__init__(integration_tolerance, max_subdivisions)
        self.x_min, self.x_max = x_range

    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate WKB transmission coefficient.

        Args:
            energy: Energy in Hartree (relative to PES minimum)
            potential_func: V(x) in Hartree, x in radians
            reduced_mass: Reduced mass in kg
            barrier_height: Barrier height in Hartree

        Returns:
            Tuple of (transmission, diagnostics)
        """
        diagnostics = {}

        # Handle edge cases
        if energy >= barrier_height:
            # Classical transmission above barrier
            return 1.0, {"reason": "above_barrier"}

        if energy <= 0:
            return 0.0, {"reason": "zero_or_negative_energy"}

        # Convert to SI units for integration
        energy_si = energy * HARTREE_TO_JOULE
        barrier_si = barrier_height * HARTREE_TO_JOULE

        def potential_si(x):
            """Potential in Joules."""
            return potential_func(x) * HARTREE_TO_JOULE

        # Find classical turning points
        try:
            x_left, x_right = find_turning_points_robust(
                energy_si,
                potential_si,
                self.x_min,
                self.x_max
            )
            diagnostics["turning_points"] = (x_left, x_right)
        except ValueError as e:
            logger.debug(f"Turning point search failed: {e}")
            # Estimate based on barrier shape
            diagnostics["turning_point_error"] = str(e)
            # Return small transmission for low energy
            if energy < 0.5 * barrier_height:
                return 1e-10, diagnostics
            else:
                return 0.5, diagnostics

        # Calculate action integral
        action, int_diag = adaptive_action_integral(
            energy_si,
            potential_si,
            reduced_mass,
            x_left,
            x_right,
            tolerance=self.tol,
            max_subdivisions=self.max_subdivisions
        )
        diagnostics.update(int_diag)
        diagnostics["action"] = action

        # WKB transmission formula
        # T = exp(-2S/ℏ)
        exponent = -2.0 * action / HBAR_SI

        # Prevent underflow
        if exponent < -700:
            transmission = 0.0
        else:
            transmission = np.exp(exponent)

        diagnostics["exponent"] = exponent

        return transmission, diagnostics


class ImprovedWKBMethod(WKBMethod):
    """
    Improved WKB with connection formula corrections.

    Includes corrections near the classical turning points
    where standard WKB breaks down.
    """

    name = "WKB-improved"

    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate transmission with connection formula correction.

        For energies very close to the barrier, applies a correction
        factor based on the parabolic barrier approximation.
        """
        # Get standard WKB result
        T_wkb, diagnostics = super().calculate_transmission(
            energy, potential_func, reduced_mass, barrier_height
        )

        # Energy ratio
        ratio = energy / barrier_height if barrier_height > 0 else 0

        # Apply correction near barrier top (0.9 < E/Vb < 1.0)
        if 0.9 < ratio < 1.0:
            # Parabolic barrier correction
            # T_corrected = T_wkb / (1 + T_wkb) for symmetric barrier
            T_corrected = T_wkb / (1.0 + T_wkb)
            diagnostics["correction"] = "parabolic"
            diagnostics["T_uncorrected"] = T_wkb
            return T_corrected, diagnostics

        return T_wkb, diagnostics
