"""PES interpolation with periodic boundary conditions."""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Callable, Optional, Tuple

from .scan import PESScanResult
from ..core.constants import DEG_TO_RAD, RAD_TO_DEG


class PeriodicSpline:
    """
    Cubic spline interpolation with periodic boundary conditions.

    Designed for torsional PES where V(0°) = V(360°).
    """

    def __init__(
        self,
        angles_deg: np.ndarray,
        energies: np.ndarray,
        period: float = 360.0
    ):
        """
        Create periodic spline interpolation.

        Args:
            angles_deg: Dihedral angles in degrees
            energies: Energies in Hartree
            period: Period of the function in degrees (default 360°)
        """
        self.period = period
        self._angles_deg = np.asarray(angles_deg)
        self._energies = np.asarray(energies)

        # Normalize angles to [0, period)
        angles_norm = self._angles_deg % period

        # Sort by angle
        sort_idx = np.argsort(angles_norm)
        angles_sorted = angles_norm[sort_idx]
        energies_sorted = self._energies[sort_idx]

        # Remove duplicates (keep first occurrence)
        unique_mask = np.concatenate([[True], np.diff(angles_sorted) > 1e-10])
        angles_sorted = angles_sorted[unique_mask]
        energies_sorted = energies_sorted[unique_mask]

        # Need at least 2 points
        if len(angles_sorted) < 2:
            raise ValueError("Need at least 2 unique angle points for interpolation")

        # Add periodic copies for proper boundary handling
        # Prepend last point shifted by -period
        # Append first point shifted by +period
        angles_extended = np.concatenate([
            [angles_sorted[-1] - period],
            angles_sorted,
            [angles_sorted[0] + period]
        ])
        energies_extended = np.concatenate([
            [energies_sorted[-1]],
            energies_sorted,
            [energies_sorted[0]]
        ])

        # Create cubic spline with natural boundary conditions
        # (we handle periodicity via the extended data points)
        self._spline = CubicSpline(
            angles_extended,
            energies_extended,
            bc_type='not-a-knot'
        )

        # Store minimum energy for relative calculations
        self.min_energy = float(np.min(self._energies))

    def __call__(self, angle_deg: float) -> float:
        """Evaluate potential at given angle (degrees)."""
        angle_norm = angle_deg % self.period
        return float(self._spline(angle_norm))

    def evaluate(self, angles_deg: np.ndarray) -> np.ndarray:
        """Evaluate potential at multiple angles."""
        angles_norm = np.asarray(angles_deg) % self.period
        return self._spline(angles_norm)

    def derivative(self, angle_deg: float, order: int = 1) -> float:
        """Evaluate derivative at given angle."""
        angle_norm = angle_deg % self.period
        return float(self._spline(angle_norm, order))

    def evaluate_radians(self, angle_rad: float) -> float:
        """Evaluate potential at angle in radians."""
        return self(angle_rad * RAD_TO_DEG)

    def get_barrier_info(self) -> dict:
        """Find barrier height and positions."""
        # Evaluate on fine grid
        fine_angles = np.linspace(0, self.period, 1000)
        fine_energies = self.evaluate(fine_angles)

        min_idx = np.argmin(fine_energies)
        max_idx = np.argmax(fine_energies)

        return {
            "barrier_height": float(np.max(fine_energies) - np.min(fine_energies)),
            "min_angle": float(fine_angles[min_idx]),
            "min_energy": float(fine_energies[min_idx]),
            "max_angle": float(fine_angles[max_idx]),
            "max_energy": float(fine_energies[max_idx]),
        }


def interpolate_pes(
    pes_result: PESScanResult,
    method: str = "cubic"
) -> PeriodicSpline:
    """
    Create interpolation function from PES scan result.

    Args:
        pes_result: PES scan result
        method: Interpolation method ("cubic" or "linear")

    Returns:
        PeriodicSpline interpolation object
    """
    return PeriodicSpline(pes_result.angles, pes_result.energies)


def create_potential_function(
    pes_result: PESScanResult,
    use_radians: bool = True,
    relative: bool = True
) -> Callable[[float], float]:
    """
    Create a callable potential function from PES scan.

    Args:
        pes_result: PES scan result
        use_radians: If True, function takes angle in radians
        relative: If True, return relative energy (min = 0)

    Returns:
        Callable V(x) function
    """
    spline = PeriodicSpline(pes_result.angles, pes_result.energies)
    min_energy = spline.min_energy

    if use_radians:
        if relative:
            def potential(x):
                return spline.evaluate_radians(x) - min_energy
        else:
            def potential(x):
                return spline.evaluate_radians(x)
    else:
        if relative:
            def potential(x):
                return spline(x) - min_energy
        else:
            def potential(x):
                return spline(x)

    return potential


def find_classical_turning_points(
    potential_func: Callable[[float], float],
    energy: float,
    x_min: float = 0.0,
    x_max: float = 2 * np.pi,
    n_points: int = 1000
) -> Tuple[float, float]:
    """
    Find classical turning points where V(x) = E.

    Args:
        potential_func: Potential energy function V(x)
        energy: Energy level to find turning points for
        x_min, x_max: Search range (in same units as potential_func input)
        n_points: Number of points for initial grid search

    Returns:
        Tuple of (left_turning_point, right_turning_point)

    Raises:
        ValueError: If turning points cannot be found
    """
    # Evaluate on fine grid
    x_grid = np.linspace(x_min, x_max, n_points)
    v_grid = np.array([potential_func(x) for x in x_grid])

    # Find where V crosses E
    crossings = []
    for i in range(len(x_grid) - 1):
        v1, v2 = v_grid[i], v_grid[i + 1]
        if (v1 - energy) * (v2 - energy) < 0:
            # Linear interpolation to find crossing
            x1, x2 = x_grid[i], x_grid[i + 1]
            x_cross = x1 + (energy - v1) * (x2 - x1) / (v2 - v1)
            crossings.append(x_cross)

    if len(crossings) < 2:
        raise ValueError(
            f"Could not find two turning points for E={energy}. "
            f"Found {len(crossings)} crossings."
        )

    # Return first and last crossing (for single-well potential)
    # For double-well, you may need more sophisticated selection
    return crossings[0], crossings[-1]
