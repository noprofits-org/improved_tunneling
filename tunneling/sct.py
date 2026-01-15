"""Small-Curvature Tunneling (SCT) method."""

import numpy as np
from typing import Callable, Dict, Any, Tuple
import logging

from .base import TunnelingMethod
from .wkb import WKBMethod
from .integration import adaptive_action_integral, find_turning_points_robust
from ..core.constants import HBAR_SI, HARTREE_TO_JOULE

logger = logging.getLogger(__name__)


class SCTMethod(TunnelingMethod):
    """
    Small-Curvature Tunneling (SCT) method.

    SCT improves upon WKB by accounting for reaction path curvature,
    which allows for "corner-cutting" in the tunneling path.

    The SCT transmission coefficient includes a curvature correction:

        T_SCT(E) = T_WKB(E) * κ(E)

    where κ is a correction factor that depends on the reaction path
    curvature and the effective mass along the path.

    For torsional motion, the curvature effects come from coupling
    between the torsion and other vibrational modes.

    Key features:
    - More accurate for light-atom tunneling (H, D)
    - Accounts for reaction path curvature
    - Better at low energies where WKB overestimates tunneling
    """

    name = "SCT"

    def __init__(
        self,
        integration_tolerance: float = 1e-8,
        max_subdivisions: int = 500,
        curvature_factor: float = 1.0,
        include_curvature: bool = True
    ):
        """
        Initialize SCT method.

        Args:
            integration_tolerance: Integration tolerance
            max_subdivisions: Max integration subdivisions
            curvature_factor: Scaling factor for curvature correction
            include_curvature: Whether to include curvature corrections
        """
        super().__init__(integration_tolerance, max_subdivisions)
        self.curvature_factor = curvature_factor
        self.include_curvature = include_curvature
        self._wkb = WKBMethod(integration_tolerance, max_subdivisions)

    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate SCT transmission coefficient.

        Args:
            energy: Energy in Hartree (relative to minimum)
            potential_func: V(x) in Hartree, x in radians
            reduced_mass: Reduced mass in kg
            barrier_height: Barrier height in Hartree

        Returns:
            Tuple of (transmission, diagnostics)
        """
        diagnostics = {}

        # Get base WKB transmission
        T_wkb, wkb_diag = self._wkb.calculate_transmission(
            energy, potential_func, reduced_mass, barrier_height
        )
        diagnostics["T_wkb"] = T_wkb
        diagnostics["wkb_diagnostics"] = wkb_diag

        if not self.include_curvature:
            return T_wkb, diagnostics

        # Calculate curvature correction
        kappa = self._calculate_curvature_correction(
            energy, potential_func, reduced_mass, barrier_height
        )
        diagnostics["curvature_correction"] = kappa

        # SCT transmission
        T_sct = T_wkb * kappa

        # Ensure physical bounds
        T_sct = max(0.0, min(1.0, T_sct))

        return T_sct, diagnostics

    def _calculate_curvature_correction(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> float:
        """
        Calculate the curvature correction factor κ.

        For torsional motion, the curvature correction accounts for
        coupling between the torsion and other modes.

        The correction is approximately:
            κ = 1 + α * (V_max - E) / V_max

        where α is a small positive constant depending on the
        reaction path curvature.
        """
        if barrier_height <= 0:
            return 1.0

        energy_ratio = energy / barrier_height

        # For energies very close to barrier, no correction needed
        if energy_ratio > 0.95:
            return 1.0

        # Calculate effective curvature correction
        # This is a simplified model - full SCT requires
        # knowledge of the reaction path and mode frequencies

        # Estimate curvature from second derivative of potential
        curvature = self._estimate_potential_curvature(
            potential_func, barrier_height
        )

        # Correction factor (increases tunneling for high curvature)
        # The "corner-cutting" effect
        alpha = self.curvature_factor * curvature

        # For low energies, curvature allows more tunneling
        kappa = 1.0 + alpha * (1.0 - energy_ratio) * 0.1

        # Bound the correction to reasonable range
        kappa = max(0.8, min(1.5, kappa))

        return kappa

    def _estimate_potential_curvature(
        self,
        potential_func: Callable[[float], float],
        barrier_height: float,
        dx: float = 0.01
    ) -> float:
        """
        Estimate potential curvature at barrier top.

        Uses numerical second derivative.
        """
        # Find barrier maximum
        x_grid = np.linspace(0, 2 * np.pi, 100)
        v_grid = np.array([potential_func(x) for x in x_grid])
        max_idx = np.argmax(v_grid)
        x_max = x_grid[max_idx]

        # Second derivative at maximum (should be negative for barrier)
        v_plus = potential_func(x_max + dx)
        v_center = potential_func(x_max)
        v_minus = potential_func(x_max - dx)

        d2v = (v_plus - 2 * v_center + v_minus) / dx**2

        # Normalize by barrier height
        if barrier_height > 0:
            curvature = abs(d2v) / barrier_height
        else:
            curvature = 0.0

        return curvature


class LCTMethod(TunnelingMethod):
    """
    Large-Curvature Tunneling (LCT) method (simplified).

    LCT is used when reaction path curvature is very large,
    allowing significant corner-cutting.

    This is a simplified implementation that applies larger
    corrections than SCT.
    """

    name = "LCT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sct = SCTMethod(
            curvature_factor=2.0,
            include_curvature=True,
            **kwargs
        )

    def calculate_transmission(
        self,
        energy: float,
        potential_func: Callable[[float], float],
        reduced_mass: float,
        barrier_height: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate LCT transmission with enhanced curvature correction."""
        return self._sct.calculate_transmission(
            energy, potential_func, reduced_mass, barrier_height
        )


def select_tunneling_method(
    barrier_height: float,
    reduced_mass_amu: float,
    curvature_estimate: float = 1.0
) -> TunnelingMethod:
    """
    Automatically select appropriate tunneling method.

    Args:
        barrier_height: Barrier height in Hartree
        reduced_mass_amu: Reduced mass in AMU
        curvature_estimate: Estimated reaction path curvature

    Returns:
        Appropriate TunnelingMethod instance
    """
    # For light atoms (H, D) with significant curvature, use SCT
    if reduced_mass_amu < 2.0 and curvature_estimate > 0.5:
        logger.info("Selecting SCT method for light atom with curvature")
        return SCTMethod(curvature_factor=curvature_estimate)

    # For very light atoms with high curvature, use LCT
    if reduced_mass_amu < 1.5 and curvature_estimate > 1.5:
        logger.info("Selecting LCT method for very light atom with high curvature")
        return LCTMethod()

    # Default to WKB for heavier atoms or low curvature
    logger.info("Selecting WKB method")
    return WKBMethod()
