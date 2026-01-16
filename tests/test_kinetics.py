"""Tests for rate calculations and Arrhenius analysis."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from improved_tunnel.kinetics.rates import (
    RateResult,
    calculate_classical_rate,
    calculate_quantum_rate,
    calculate_rates_vs_temperature,
    calculate_tunneling_correction,
    calculate_kie_vs_temperature,
)
from improved_tunnel.kinetics.arrhenius import (
    ArrheniusParameters,
    fit_arrhenius,
    fit_arrhenius_nonlinear,
    analyze_arrhenius,
    calculate_wigner_correction,
)
from improved_tunnel.tunneling.base import TunnelingResult
from improved_tunnel.core.constants import R_SI, HARTREE_TO_KCAL


class TestClassicalRate:
    """Tests for classical rate calculations."""

    def test_classical_rate_formula(self):
        """Test classical TST rate formula."""
        # k = A * exp(-Ea/RT)
        # Note: calculate_classical_rate takes barrier in Hartree
        from improved_tunnel.core.constants import KCAL_TO_HARTREE, HARTREE_TO_JOULE, AVOGADRO

        barrier_kcal = 7.0
        barrier_hartree = barrier_kcal * KCAL_TO_HARTREE
        T = 300  # K
        prefactor = 1e13  # s^-1

        k = calculate_classical_rate(barrier_hartree, T, prefactor)

        # Manual calculation: Ea in J/mol = barrier_hartree * HARTREE_TO_JOULE * AVOGADRO
        Ea_J_per_mol = barrier_hartree * HARTREE_TO_JOULE * 6.02214076e23
        k_expected = prefactor * np.exp(-Ea_J_per_mol / (R_SI * T))

        assert_allclose(k, k_expected, rtol=1e-10)

    def test_classical_rate_temperature_dependence(self):
        """Test that rate increases with temperature."""
        from improved_tunnel.core.constants import KCAL_TO_HARTREE

        barrier = 5.0 * KCAL_TO_HARTREE  # Convert to Hartree

        k_low = calculate_classical_rate(barrier, 200)
        k_high = calculate_classical_rate(barrier, 400)

        assert k_high > k_low

    def test_classical_rate_barrier_dependence(self):
        """Test that rate decreases with higher barrier."""
        from improved_tunnel.core.constants import KCAL_TO_HARTREE

        T = 300

        k_low_barrier = calculate_classical_rate(5.0 * KCAL_TO_HARTREE, T)
        k_high_barrier = calculate_classical_rate(10.0 * KCAL_TO_HARTREE, T)

        assert k_high_barrier < k_low_barrier


class TestQuantumRate:
    """Tests for quantum-corrected rate calculations."""

    def test_quantum_rate_with_tunneling(self):
        """Test quantum rate includes tunneling correction."""
        from improved_tunnel.core.constants import KCAL_TO_HARTREE

        barrier_hartree = 7.0 * KCAL_TO_HARTREE
        T = 300

        # Create mock tunneling result
        energies = barrier_hartree * np.linspace(0.1, 1.0, 20)
        transmissions = np.linspace(1e-8, 0.5, 20)

        tun_result = TunnelingResult(
            method="test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier_hartree,
            reduced_mass=1.0,
        )

        k_classical = calculate_classical_rate(barrier_hartree, T)
        k_quantum = calculate_quantum_rate(barrier_hartree, T, tun_result)

        # Quantum rate should be >= classical rate
        assert k_quantum >= k_classical

    def test_tunneling_correction_factor(self):
        """Test tunneling correction factor calculation."""
        # Create mock tunneling result
        barrier = 0.01  # Hartree
        energies = barrier * np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        # Higher transmission at higher energy
        transmissions = np.array([1e-10, 1e-6, 1e-3, 0.1, 0.5])

        result = TunnelingResult(
            method="test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier,
            reduced_mass=1.0,
        )

        kappa = calculate_tunneling_correction(result, barrier, 300)

        # Tunneling should enhance rate (kappa > 1 at reasonable T)
        assert kappa >= 1.0


class TestRateResult:
    """Tests for RateResult dataclass."""

    def test_rate_result_creation(self):
        """Test RateResult creation."""
        temps = np.array([200, 250, 300, 350, 400])
        k_class = np.array([1e6, 1e8, 1e9, 1e10, 1e11])
        k_quant = np.array([1e7, 1e9, 1e10, 1e11, 1e12])

        result = RateResult(
            temperatures=temps,
            classical_rates=k_class,
            quantum_rates=k_quant,
            tunneling_corrections=k_quant / k_class,
            barrier_height=7.0,
        )

        assert len(result.temperatures) == 5
        assert result.barrier_height == 7.0

    def test_rate_result_to_dict(self):
        """Test serialization."""
        result = RateResult(
            temperatures=np.array([300]),
            classical_rates=np.array([1e9]),
            quantum_rates=np.array([1e10]),
            tunneling_corrections=np.array([10.0]),
            barrier_height=5.0,
        )

        d = result.to_dict()
        assert "temperatures" in d
        assert "classical_rates" in d
        assert "quantum_rates" in d


class TestRatesVsTemperature:
    """Tests for temperature-dependent rate calculation."""

    def test_rates_vs_temperature(self, mock_pes_symmetric):
        """Test rate calculation across temperatures."""
        from improved_tunnel.tunneling.eckart import EckartBarrier

        # Create tunneling result
        eckart = EckartBarrier()
        eckart.fit_to_pes(mock_pes_symmetric)

        result = eckart.calculate_all(
            mock_pes_symmetric,
            reduced_mass_amu=1.0,  # Correct parameter name
            energy_points=20,
            min_ratio=0.1,
            max_ratio=1.0,
        )

        temps = np.array([200, 250, 300, 350, 400])
        rate_result = calculate_rates_vs_temperature(result, temps)

        # Check all arrays have same length
        assert len(rate_result.temperatures) == 5
        assert len(rate_result.classical_rates) == 5
        assert len(rate_result.quantum_rates) == 5

        # Rates should increase with temperature
        assert np.all(np.diff(rate_result.classical_rates) > 0)


class TestArrheniusFitting:
    """Tests for Arrhenius parameter fitting."""

    def test_fit_arrhenius_perfect_data(self):
        """Test fitting with perfect Arrhenius data."""
        A_true = 1e13
        Ea_true = 5.0  # kcal/mol

        temps = np.array([200, 250, 300, 350, 400])
        # Generate perfect Arrhenius data
        Ea_J = Ea_true * 4184
        rates = A_true * np.exp(-Ea_J / (R_SI * temps))

        params = fit_arrhenius(temps, rates)

        assert_allclose(params.prefactor, A_true, rtol=0.01)
        assert_allclose(params.activation_energy, Ea_true, rtol=0.01)
        assert params.r_squared > 0.99

    def test_fit_arrhenius_with_noise(self):
        """Test fitting with noisy data."""
        A_true = 1e12
        Ea_true = 7.0

        temps = np.linspace(200, 400, 20)
        Ea_J = Ea_true * 4184
        rates = A_true * np.exp(-Ea_J / (R_SI * temps))
        # Add 5% noise
        rates *= 1 + 0.05 * np.random.randn(len(rates))

        params = fit_arrhenius(temps, rates)

        # Should still get reasonable fit
        assert_allclose(params.prefactor, A_true, rtol=0.2)
        assert_allclose(params.activation_energy, Ea_true, rtol=0.1)
        assert params.r_squared > 0.9

    def test_fit_arrhenius_nonlinear(self):
        """Test nonlinear Arrhenius fitting."""
        A_true = 1e13
        Ea_true = 6.0

        temps = np.linspace(200, 400, 15)
        Ea_J = Ea_true * 4184
        rates = A_true * np.exp(-Ea_J / (R_SI * temps))

        params = fit_arrhenius_nonlinear(temps, rates)

        assert_allclose(params.prefactor, A_true, rtol=0.05)
        assert_allclose(params.activation_energy, Ea_true, rtol=0.05)

    def test_arrhenius_predict_rate(self):
        """Test rate prediction from Arrhenius parameters."""
        params = ArrheniusParameters(prefactor=1e13, activation_energy=5.0)

        k_300 = params.predict_rate(300)

        # Manual calculation
        Ea_J = 5.0 * 4184
        k_expected = 1e13 * np.exp(-Ea_J / (R_SI * 300))

        assert_allclose(k_300, k_expected, rtol=1e-10)


class TestArrheniusAnalysis:
    """Tests for complete Arrhenius analysis."""

    def test_analyze_arrhenius(self):
        """Test full Arrhenius analysis."""
        temps = np.array([200, 250, 300, 350, 400])

        # Classical rates with Ea = 7 kcal/mol
        Ea_class = 7.0 * 4184
        k_class = 1e13 * np.exp(-Ea_class / (R_SI * temps))

        # Quantum rates with lower effective Ea = 5 kcal/mol (tunneling)
        Ea_quant = 5.0 * 4184
        k_quant = 1e13 * np.exp(-Ea_quant / (R_SI * temps))

        result = RateResult(
            temperatures=temps,
            classical_rates=k_class,
            quantum_rates=k_quant,
            tunneling_corrections=k_quant / k_class,
            barrier_height=7.0,
        )

        analysis = analyze_arrhenius(result)

        # Classical Ea should be ~7 kcal/mol
        assert_allclose(analysis["classical"]["activation_energy_kcal"], 7.0, rtol=0.05)

        # Quantum Ea should be ~5 kcal/mol
        assert_allclose(analysis["quantum"]["activation_energy_kcal"], 5.0, rtol=0.05)

        # Tunneling should reduce Ea by ~2 kcal/mol
        assert_allclose(analysis["Ea_reduction_kcal"], 2.0, rtol=0.1)


class TestWignerCorrection:
    """Tests for Wigner tunneling correction."""

    def test_wigner_correction_zero_frequency(self):
        """Test Wigner correction with zero frequency."""
        kappa = calculate_wigner_correction(0, 300)
        assert kappa == 1.0

    def test_wigner_correction_typical_values(self):
        """Test Wigner correction for typical H-transfer frequency."""
        # Typical imaginary frequency ~1000 cm^-1
        # At 300 K, Wigner correction is modest
        kappa = calculate_wigner_correction(1000, 300)

        # Should be > 1 but not enormous
        assert 1.0 < kappa < 2.0

    def test_wigner_temperature_dependence(self):
        """Test that Wigner correction increases at low T."""
        freq = 1500  # cm^-1

        kappa_low = calculate_wigner_correction(freq, 200)
        kappa_high = calculate_wigner_correction(freq, 400)

        assert kappa_low > kappa_high


class TestTunnelingCorrectionAnalytic:
    """Analytic tests for tunneling correction factor κ."""

    def test_kappa_with_perfect_transmission(self):
        """
        Test κ with P(E) = 1 for all sub-barrier energies.

        For P(E) = 1:
            κ = (1/kT) * ∫₀^V P(E) * exp(-(E-V)/kT) dE + 1
              = (1/kT) * ∫₀^V exp(-(E-V)/kT) dE + 1

        With substitution u = V - E:
            = (1/kT) * kT * (exp(V/kT) - 1) + 1
            = exp(V/kT)

        Physical meaning: With perfect transmission, all particles can tunnel,
        so the rate enhancement is exp(V/kT) relative to classical.
        """
        from improved_tunnel.core.constants import BOLTZMANN_SI, HARTREE_TO_JOULE

        # Choose parameters
        barrier_hartree = 0.01  # ~6.3 kcal/mol
        temperature = 300  # K

        # Create fine energy grid from 0 to just below barrier
        n_points = 1000
        energies = np.linspace(0.001 * barrier_hartree, 0.999 * barrier_hartree, n_points)
        # Perfect transmission
        transmissions = np.ones(n_points)

        result = TunnelingResult(
            method="analytic_test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier_hartree,
            reduced_mass=1.0,
        )

        kappa = calculate_tunneling_correction(result, barrier_hartree, temperature)

        # Analytic result: κ = exp(V/kT)
        kT = BOLTZMANN_SI * temperature
        V_joules = barrier_hartree * HARTREE_TO_JOULE
        kappa_analytic = np.exp(V_joules / kT)

        # Should match within numerical integration tolerance
        # The discrete grid will underestimate slightly since we don't go all the way to E=0
        assert_allclose(kappa, kappa_analytic, rtol=0.05)

    def test_kappa_with_zero_transmission(self):
        """
        Test κ with P(E) = 0 for all sub-barrier energies.

        For P(E) = 0: κ = 0 + 1 = 1 (no tunneling contribution).
        """
        barrier_hartree = 0.01
        temperature = 300

        energies = np.linspace(0.1 * barrier_hartree, 0.9 * barrier_hartree, 50)
        transmissions = np.zeros(50)  # No transmission

        result = TunnelingResult(
            method="analytic_test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier_hartree,
            reduced_mass=1.0,
        )

        kappa = calculate_tunneling_correction(result, barrier_hartree, temperature)

        # Should be exactly 1 (classical limit)
        assert_allclose(kappa, 1.0, atol=1e-10)

    def test_kappa_always_at_least_one(self):
        """Test that κ >= 1 for any valid transmission."""
        barrier_hartree = 0.01
        temperature = 300

        # Realistic small transmissions
        energies = np.linspace(0.1 * barrier_hartree, 0.9 * barrier_hartree, 20)
        transmissions = 1e-8 * np.ones(20)  # Tiny but non-zero

        result = TunnelingResult(
            method="test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier_hartree,
            reduced_mass=1.0,
        )

        kappa = calculate_tunneling_correction(result, barrier_hartree, temperature)

        # Must always be >= 1
        assert kappa >= 1.0


class TestKIE:
    """Tests for kinetic isotope effect calculations."""

    def test_kie_h_vs_d(self):
        """Test KIE calculation for H vs D."""
        temps = np.array([200, 250, 300, 350, 400])

        # H rates (faster)
        k_h = 1e10 * np.exp(-5000 / temps)
        # D rates (slower, higher effective barrier)
        k_d = 1e10 * np.exp(-6000 / temps)

        result_h = RateResult(
            temperatures=temps,
            classical_rates=k_h,
            quantum_rates=k_h,
            tunneling_corrections=np.ones_like(k_h),
            barrier_height=5.0,
        )

        result_d = RateResult(
            temperatures=temps,
            classical_rates=k_d,
            quantum_rates=k_d,
            tunneling_corrections=np.ones_like(k_d),
            barrier_height=5.0,
        )

        kie_analysis = calculate_kie_vs_temperature(result_h, result_d)

        # KIE should be > 1 (H faster than D)
        assert kie_analysis["kie_at_300K"] > 1

        # KIE should be larger at lower temperatures
        assert kie_analysis["max_kie"] > kie_analysis["kie_at_300K"]
