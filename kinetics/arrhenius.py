"""Arrhenius analysis and parameter fitting."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import curve_fit
import logging

from .rates import RateResult
from ..core.constants import R_SI, HARTREE_TO_KCAL

logger = logging.getLogger(__name__)


@dataclass
class ArrheniusParameters:
    """
    Arrhenius parameters from fitting.

    k = A * exp(-Ea / RT)

    Attributes:
        prefactor: Pre-exponential factor A in s^-1
        activation_energy: Activation energy Ea in kcal/mol
        prefactor_error: Standard error of A
        activation_energy_error: Standard error of Ea
        r_squared: R² goodness of fit
    """
    prefactor: float
    activation_energy: float
    prefactor_error: float = 0.0
    activation_energy_error: float = 0.0
    r_squared: float = 1.0

    def predict_rate(self, temperature: float) -> float:
        """Predict rate at given temperature."""
        # Convert Ea from kcal/mol to J/mol
        Ea_J = self.activation_energy * 4184.0  # kcal to J
        return self.prefactor * np.exp(-Ea_J / (R_SI * temperature))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prefactor": self.prefactor,
            "activation_energy_kcal": self.activation_energy,
            "prefactor_error": self.prefactor_error,
            "activation_energy_error_kcal": self.activation_energy_error,
            "r_squared": self.r_squared,
        }


def fit_arrhenius(
    temperatures: np.ndarray,
    rates: np.ndarray,
    min_rate: float = 1e-30
) -> ArrheniusParameters:
    """
    Fit Arrhenius parameters to rate vs temperature data.

    Uses linear regression on ln(k) vs 1/T.

    ln(k) = ln(A) - Ea/(R*T)

    Args:
        temperatures: Array of temperatures in Kelvin
        rates: Array of rate constants in s^-1
        min_rate: Minimum rate to include (filter zeros)

    Returns:
        ArrheniusParameters with fitted values
    """
    # Filter out zero or very small rates
    mask = rates > min_rate
    T_filtered = temperatures[mask]
    k_filtered = rates[mask]

    if len(T_filtered) < 2:
        logger.warning("Not enough data points for Arrhenius fit")
        return ArrheniusParameters(
            prefactor=1e13,
            activation_energy=0.0,
            r_squared=0.0
        )

    # Linear fit: ln(k) = ln(A) - Ea/(R*T)
    x = 1.0 / T_filtered  # 1/T
    y = np.log(k_filtered)  # ln(k)

    # Linear regression
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    denominator = n * sum_x2 - sum_x**2
    if abs(denominator) < 1e-30:
        logger.warning("Singular matrix in Arrhenius fit")
        return ArrheniusParameters(
            prefactor=np.exp(np.mean(y)),
            activation_energy=0.0,
            r_squared=0.0
        )

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    # Extract parameters
    ln_A = intercept
    A = np.exp(ln_A)

    # slope = -Ea/R, so Ea = -slope * R
    # R in J/(mol·K), we want kcal/mol
    Ea_J_per_mol = -slope * R_SI
    Ea_kcal = Ea_J_per_mol / 4184.0

    # Calculate R²
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors
    if n > 2:
        mse = ss_res / (n - 2)
        se_slope = np.sqrt(mse * n / denominator)
        se_intercept = np.sqrt(mse * sum_x2 / denominator)

        # Convert to parameter errors
        A_error = A * se_intercept  # Approximate
        Ea_error = se_slope * R_SI / 4184.0
    else:
        A_error = 0.0
        Ea_error = 0.0

    return ArrheniusParameters(
        prefactor=A,
        activation_energy=Ea_kcal,
        prefactor_error=A_error,
        activation_energy_error=Ea_error,
        r_squared=r_squared
    )


def fit_arrhenius_nonlinear(
    temperatures: np.ndarray,
    rates: np.ndarray
) -> ArrheniusParameters:
    """
    Fit Arrhenius parameters using nonlinear least squares.

    More robust for data with high curvature.

    Args:
        temperatures: Array of temperatures in Kelvin
        rates: Array of rate constants in s^-1

    Returns:
        ArrheniusParameters with fitted values
    """
    # Filter valid data
    mask = (rates > 0) & (temperatures > 0)
    T = temperatures[mask]
    k = rates[mask]

    if len(T) < 2:
        return ArrheniusParameters(prefactor=1e13, activation_energy=0.0)

    def arrhenius(T, A, Ea):
        """Arrhenius function with Ea in kcal/mol."""
        Ea_J = Ea * 4184.0
        return A * np.exp(-Ea_J / (R_SI * T))

    # Initial guesses from linear fit
    linear_fit = fit_arrhenius(temperatures, rates)
    p0 = [linear_fit.prefactor, linear_fit.activation_energy]

    try:
        popt, pcov = curve_fit(
            arrhenius, T, k,
            p0=p0,
            bounds=([1e-10, -100], [1e30, 100]),
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        logger.warning(f"Nonlinear fit failed: {e}")
        return linear_fit

    # Calculate R²
    k_pred = arrhenius(T, *popt)
    ss_res = np.sum((k - k_pred)**2)
    ss_tot = np.sum((k - np.mean(k))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ArrheniusParameters(
        prefactor=popt[0],
        activation_energy=popt[1],
        prefactor_error=perr[0],
        activation_energy_error=perr[1],
        r_squared=r_squared
    )


def analyze_arrhenius(rate_result: RateResult) -> dict:
    """
    Perform complete Arrhenius analysis on rate data.

    Fits both classical and quantum rates and compares.

    Args:
        rate_result: RateResult with temperature-dependent rates

    Returns:
        Dict with analysis results
    """
    T = rate_result.temperatures
    k_classical = rate_result.classical_rates
    k_quantum = rate_result.quantum_rates

    # Fit classical rates
    classical_params = fit_arrhenius(T, k_classical)

    # Fit quantum rates
    quantum_params = fit_arrhenius(T, k_quantum)

    # Tunneling lowers effective activation energy
    Ea_reduction = classical_params.activation_energy - quantum_params.activation_energy

    return {
        "classical": classical_params.to_dict(),
        "quantum": quantum_params.to_dict(),
        "Ea_reduction_kcal": Ea_reduction,
        "Ea_reduction_percent": (Ea_reduction / classical_params.activation_energy * 100
                                 if classical_params.activation_energy > 0 else 0),
        "barrier_height_kcal": rate_result.barrier_height,
    }


def calculate_wigner_correction(
    imaginary_frequency: float,
    temperature: float
) -> float:
    """
    Calculate Wigner tunneling correction.

    A simple analytical correction based on barrier frequency:

        κ_Wigner = 1 + (1/24) * (hν*/kT)²

    where ν* is the imaginary frequency at the transition state.

    Args:
        imaginary_frequency: Imaginary frequency in cm^-1 (as positive number)
        temperature: Temperature in Kelvin

    Returns:
        Wigner tunneling correction factor
    """
    if temperature <= 0 or imaginary_frequency <= 0:
        return 1.0

    # h in J·s, convert frequency cm^-1 to Hz
    h = 6.62607015e-34
    c = 299792458.0  # m/s
    freq_Hz = imaginary_frequency * 100 * c  # cm^-1 to Hz

    kT = BOLTZMANN_SI * temperature
    hv = h * freq_Hz

    ratio = hv / kT
    kappa = 1.0 + (1.0 / 24.0) * ratio**2

    return kappa


BOLTZMANN_SI = 1.380649e-23  # Imported at module level but included for clarity
