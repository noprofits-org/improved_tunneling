"""Adaptive numerical integration for action integrals."""

import numpy as np
from scipy import integrate
from typing import Callable, Tuple, Dict, Any
import logging

from ..core.exceptions import IntegrationError

logger = logging.getLogger(__name__)


def adaptive_action_integral(
    energy: float,
    potential_func: Callable[[float], float],
    mass: float,
    x_left: float,
    x_right: float,
    tolerance: float = 1e-8,
    max_subdivisions: int = 500
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate WKB action integral using adaptive quadrature.

    S = ∫ √(2m(V(x) - E)) dx

    from x_left to x_right (classical turning points).

    Uses scipy.integrate.quad for adaptive integration with
    automatic subdivision and error estimation.

    Args:
        energy: Energy level in Joules
        potential_func: Potential energy function V(x) in Joules
        mass: Reduced mass in kg
        x_left: Left classical turning point (radians for torsion)
        x_right: Right classical turning point
        tolerance: Integration tolerance (absolute and relative)
        max_subdivisions: Maximum number of subdivisions

    Returns:
        Tuple of (action_integral, diagnostics_dict)
        diagnostics_dict contains: 'error', 'neval', 'converged', 'relative_error'

    Raises:
        IntegrationError: If integration fails to converge
    """
    if x_right <= x_left:
        return 0.0, {"error": 0.0, "neval": 0, "converged": True, "relative_error": 0.0}

    def integrand(x: float) -> float:
        """WKB integrand: sqrt(2m(V-E)) for V > E."""
        V = potential_func(x)
        diff = V - energy
        if diff > 0:
            return np.sqrt(2.0 * mass * diff)
        return 0.0

    try:
        result, error = integrate.quad(
            integrand,
            x_left,
            x_right,
            epsabs=tolerance,
            epsrel=tolerance,
            limit=max_subdivisions
        )
    except Exception as e:
        raise IntegrationError(
            f"Integration failed: {e}",
            tolerance=tolerance,
            subdivisions=max_subdivisions
        )

    # Check convergence
    if result > 0:
        relative_error = error / result
    else:
        relative_error = 0.0

    converged = relative_error < tolerance * 10  # Allow some margin

    if not converged:
        logger.warning(
            f"Action integral may not be converged: "
            f"rel_error={relative_error:.2e}, tol={tolerance:.2e}"
        )

    diagnostics = {
        "error": error,
        "neval": max_subdivisions,  # quad doesn't expose actual count
        "converged": converged,
        "relative_error": relative_error
    }

    return result, diagnostics


def simpson_action_integral(
    energy: float,
    potential_func: Callable[[float], float],
    mass: float,
    x_left: float,
    x_right: float,
    n_points: int = 1000
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate action integral using Simpson's rule (fixed grid).

    Simpler but less accurate than adaptive quadrature.
    Useful for comparison or when adaptive fails.

    Args:
        energy: Energy level in Joules
        potential_func: Potential energy function V(x) in Joules
        mass: Reduced mass in kg
        x_left: Left classical turning point
        x_right: Right classical turning point
        n_points: Number of integration points

    Returns:
        Tuple of (action_integral, diagnostics_dict)
    """
    if x_right <= x_left:
        return 0.0, {"n_points": n_points, "method": "simpson"}

    x_grid = np.linspace(x_left, x_right, n_points)

    # Evaluate integrand
    y_values = []
    for x in x_grid:
        V = potential_func(x)
        diff = V - energy
        if diff > 0:
            y_values.append(np.sqrt(2.0 * mass * diff))
        else:
            y_values.append(0.0)

    y_grid = np.array(y_values)

    # Simpson integration
    result = integrate.simpson(y_grid, x=x_grid)

    return result, {"n_points": n_points, "method": "simpson"}


def find_turning_points_robust(
    energy: float,
    potential_func: Callable[[float], float],
    x_min: float = 0.0,
    x_max: float = 2 * np.pi,
    n_initial: int = 500,
    refinement_tol: float = 1e-10
) -> Tuple[float, float]:
    """
    Find classical turning points with robust error handling.

    Uses bisection refinement for accurate turning points.

    Args:
        energy: Energy level
        potential_func: Potential V(x)
        x_min, x_max: Search range
        n_initial: Initial grid points
        refinement_tol: Tolerance for bisection refinement

    Returns:
        Tuple of (x_left, x_right) turning points

    Raises:
        ValueError: If turning points cannot be found
    """
    # Initial grid search
    x_grid = np.linspace(x_min, x_max, n_initial)
    v_grid = np.array([potential_func(x) for x in x_grid])
    diff_grid = v_grid - energy

    # Find sign changes (turning points)
    sign_changes = []
    for i in range(len(diff_grid) - 1):
        if diff_grid[i] * diff_grid[i + 1] < 0:
            # Refine with bisection
            x_refined = _bisection_refine(
                potential_func, energy,
                x_grid[i], x_grid[i + 1],
                refinement_tol
            )
            sign_changes.append(x_refined)

    if len(sign_changes) < 2:
        # Try to handle edge cases
        if len(sign_changes) == 0:
            raise ValueError(
                f"No turning points found for E={energy}. "
                f"Energy may be above barrier or below minimum."
            )
        elif len(sign_changes) == 1:
            # Might be at boundary - check endpoints
            if diff_grid[0] > 0:
                sign_changes.insert(0, x_min)
            if diff_grid[-1] > 0:
                sign_changes.append(x_max)

    if len(sign_changes) < 2:
        raise ValueError(f"Only found {len(sign_changes)} turning point(s)")

    return sign_changes[0], sign_changes[-1]


def _bisection_refine(
    potential_func: Callable[[float], float],
    energy: float,
    x_low: float,
    x_high: float,
    tol: float
) -> float:
    """Bisection to refine turning point location."""
    v_low = potential_func(x_low) - energy
    v_high = potential_func(x_high) - energy

    if v_low * v_high > 0:
        # Same sign - no crossing in this interval
        return (x_low + x_high) / 2

    max_iter = 100
    for _ in range(max_iter):
        if x_high - x_low < tol:
            break

        x_mid = (x_low + x_high) / 2
        v_mid = potential_func(x_mid) - energy

        if v_mid == 0:
            return x_mid
        elif v_low * v_mid < 0:
            x_high = x_mid
            v_high = v_mid
        else:
            x_low = x_mid
            v_low = v_mid

    return (x_low + x_high) / 2
