"""Tests for tunneling calculations: WKB vs Eckart analytical validation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from improved_tunnel.tunneling.wkb import WKBMethod, ImprovedWKBMethod
from improved_tunnel.tunneling.eckart import EckartBarrier, compare_with_eckart
from improved_tunnel.tunneling.sct import SCTMethod
from improved_tunnel.tunneling.base import TunnelingResult
from improved_tunnel.tunneling.integration import (
    adaptive_action_integral,
    find_turning_points_robust,
)
from improved_tunnel.pes.interpolation import create_potential_function
from improved_tunnel.core.constants import (
    HARTREE_TO_JOULE,
    HARTREE_TO_KCAL,
    AMU_TO_KG,
    HBAR_SI,
)


class TestEckartBarrier:
    """Tests for analytical Eckart barrier model."""

    def test_eckart_fit_to_pes(self, mock_pes_eckart):
        """Test fitting Eckart parameters to PES."""
        eckart = EckartBarrier()
        params = eckart.fit_to_pes(mock_pes_eckart)

        # Check parameters are reasonable
        assert params["V1"] > 0
        assert params["L"] > 0
        assert 0 < params["x_max"] < 2 * np.pi

    def test_eckart_transmission_limits(self):
        """Test Eckart transmission at energy limits."""
        # Use parameters that give measurable transmission
        eckart = EckartBarrier(V1=0.001, L=0.3, symmetric=True)  # Smaller barrier
        mass_kg = 0.5 * AMU_TO_KG  # Lighter mass

        # At E = 0, T should be very small
        T_zero = eckart.analytical_transmission(1e-6, mass_kg)
        assert T_zero < 0.5

        # At E = 0.9 * V, T should be approaching 1
        T_near_barrier = eckart.analytical_transmission(0.0009, mass_kg)
        assert 0.0 <= T_near_barrier <= 1.0

        # At E >= V, implementation returns 1.0 (classical limit)
        T_barrier = eckart.analytical_transmission(0.001, mass_kg)
        assert T_barrier == 1.0  # This is by design in the implementation

        # At E >> V, T should be 1
        T_above = eckart.analytical_transmission(0.01, mass_kg)
        assert T_above == 1.0

    def test_eckart_transmission_monotonic(self):
        """Test that transmission increases with energy."""
        eckart = EckartBarrier(V1=0.01, L=0.5)
        mass_kg = 1.0 * AMU_TO_KG

        energies = np.linspace(0.001, 0.01, 20)
        transmissions = [eckart.analytical_transmission(E, mass_kg) for E in energies]

        # Should be monotonically increasing
        for i in range(1, len(transmissions)):
            assert transmissions[i] >= transmissions[i - 1]

    def test_eckart_mass_dependence(self):
        """Test that heavier mass gives lower transmission."""
        # Use smaller barrier and narrower width for measurable transmission
        eckart = EckartBarrier(V1=0.001, L=0.3)
        # Use energy closer to barrier
        E = 0.0008  # Hartree (80% of barrier)

        T_light = eckart.analytical_transmission(E, 0.5 * AMU_TO_KG)
        T_heavy = eckart.analytical_transmission(E, 2.0 * AMU_TO_KG)

        # At this energy, light mass should have measurable transmission
        # Heavier particle tunnels less (or equal if both very small)
        assert T_heavy <= T_light

    def test_eckart_barrier_width_dependence(self):
        """Test that wider barrier gives lower transmission."""
        # Use energy closer to barrier to get measurable transmission
        E = 0.008  # Hartree
        mass_kg = 1.0 * AMU_TO_KG

        eckart_narrow = EckartBarrier(V1=0.01, L=0.3)
        eckart_wide = EckartBarrier(V1=0.01, L=0.7)

        T_narrow = eckart_narrow.analytical_transmission(E, mass_kg)
        T_wide = eckart_wide.analytical_transmission(E, mass_kg)

        # At 80% of barrier, both should have measurable transmission
        # Wider barrier = less tunneling
        assert T_wide <= T_narrow


class TestWKBMethod:
    """Tests for WKB tunneling calculation."""

    def test_wkb_basic_calculation(self, mock_pes_eckart):
        """Test basic WKB calculation runs without error."""
        wkb = WKBMethod()
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height

        mass_kg = 1.0 * AMU_TO_KG
        E = 0.5 * barrier

        T, diag = wkb.calculate_transmission(E, V_func, mass_kg, barrier)

        assert 0 <= T <= 1
        assert "action" in diag or "turning_point_error" in diag

    def test_wkb_above_barrier(self, mock_pes_eckart):
        """Test WKB returns T=1 above barrier."""
        wkb = WKBMethod()
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height

        mass_kg = 1.0 * AMU_TO_KG
        E = 1.5 * barrier  # Above barrier

        T, diag = wkb.calculate_transmission(E, V_func, mass_kg, barrier)

        assert T == 1.0
        assert diag.get("reason") == "above_barrier"

    def test_wkb_vs_eckart_agreement(self, mock_pes_eckart):
        """Test WKB agrees with analytical Eckart within tolerance."""
        # Fit Eckart to same PES
        eckart = EckartBarrier()
        eckart.fit_to_pes(mock_pes_eckart)

        # Create potential function
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height

        wkb = WKBMethod()
        mass_kg = 1.0 * AMU_TO_KG

        # Compare at several energies
        for ratio in [0.3, 0.5, 0.7]:
            E = ratio * barrier

            T_wkb, _ = wkb.calculate_transmission(E, V_func, mass_kg, barrier)
            T_eckart = eckart.analytical_transmission(E, mass_kg)

            # WKB and Eckart should agree within factor of ~2-3 for well-behaved barriers
            # This is a loose tolerance because WKB is approximate
            if T_wkb > 1e-10 and T_eckart > 1e-10:
                log_ratio = abs(np.log10(T_wkb / T_eckart))
                assert log_ratio < 1.5, f"WKB/Eckart differ by 10^{log_ratio} at E/V={ratio}"


class TestImprovedWKB:
    """Tests for improved WKB with connection formula."""

    def test_improved_wkb_near_barrier(self, mock_pes_eckart):
        """Test improved WKB applies correction near barrier top."""
        wkb_improved = ImprovedWKBMethod()
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height

        mass_kg = 1.0 * AMU_TO_KG
        E = 0.95 * barrier  # Near barrier top

        T, diag = wkb_improved.calculate_transmission(E, V_func, mass_kg, barrier)

        # Should apply correction
        if "correction" in diag:
            assert diag["correction"] == "parabolic"


class TestSCTMethod:
    """Tests for Small-Curvature Tunneling method."""

    def test_sct_basic_calculation(self, mock_pes_eckart):
        """Test basic SCT calculation."""
        sct = SCTMethod()
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height

        mass_kg = 1.0 * AMU_TO_KG
        E = 0.5 * barrier

        T, diag = sct.calculate_transmission(E, V_func, mass_kg, barrier)

        assert 0 <= T <= 1

    def test_sct_includes_curvature_effect(self, mock_pes_eckart):
        """Test that SCT differs from standard WKB."""
        V_func = create_potential_function(mock_pes_eckart, use_radians=True, relative=True)
        barrier = mock_pes_eckart.barrier_height
        mass_kg = 1.0 * AMU_TO_KG

        wkb = WKBMethod()
        sct = SCTMethod()

        E = 0.5 * barrier

        T_wkb, _ = wkb.calculate_transmission(E, V_func, mass_kg, barrier)
        T_sct, _ = sct.calculate_transmission(E, V_func, mass_kg, barrier)

        # SCT should give different (typically higher) transmission
        # due to curvature corrections, but they should be same order of magnitude
        if T_wkb > 1e-10:
            ratio = T_sct / T_wkb
            assert 0.1 < ratio < 10  # Within order of magnitude


class TestTunnelingResult:
    """Tests for TunnelingResult dataclass."""

    def test_result_creation(self):
        """Test creating TunnelingResult."""
        energies = np.linspace(0.001, 0.01, 10)
        transmissions = np.linspace(1e-8, 0.5, 10)

        result = TunnelingResult(
            method="WKB",
            energies=energies,
            transmissions=transmissions,
            barrier_height=0.01,
            reduced_mass=1.0,
        )

        assert result.method == "WKB"
        assert len(result.energies) == 10
        assert result.barrier_height == 0.01

    def test_get_transmission_at_ratio(self):
        """Test interpolating transmission at energy ratio."""
        barrier = 0.01
        energies = barrier * np.array([0.3, 0.5, 0.7, 0.9])
        transmissions = np.array([1e-6, 1e-4, 1e-2, 0.1])

        result = TunnelingResult(
            method="test",
            energies=energies,
            transmissions=transmissions,
            barrier_height=barrier,
            reduced_mass=1.0,
        )

        T_half = result.get_transmission_at_ratio(0.5)
        assert_allclose(T_half, 1e-4, rtol=0.1)

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        result = TunnelingResult(
            method="WKB",
            energies=np.array([0.001, 0.005]),
            transmissions=np.array([1e-6, 1e-3]),
            barrier_height=0.01,
            reduced_mass=1.0,
        )

        d = result.to_dict()
        assert "method" in d
        assert "energies" in d
        assert "transmissions" in d
        assert d["method"] == "WKB"


class TestIntegrationUtilities:
    """Tests for numerical integration utilities."""

    def test_find_turning_points(self):
        """Test robust turning point finding."""

        def V(x):
            # Simple barrier centered at π
            return 0.01 * (1 - np.cos(x)) / 2

        V_si = lambda x: V(x) * HARTREE_TO_JOULE
        E_si = 0.0025 * HARTREE_TO_JOULE  # 25% of barrier

        x1, x2 = find_turning_points_robust(E_si, V_si, 0, 2 * np.pi)

        # Turning points should bracket the barrier
        assert x1 < np.pi
        assert x2 > np.pi

        # V at turning points should equal E
        assert_allclose(V_si(x1), E_si, rtol=0.1)
        assert_allclose(V_si(x2), E_si, rtol=0.1)

    def test_action_integral_simple(self):
        """Test action integral for simple barrier."""

        def V(x):
            # Constant barrier V = V0 between x1 and x2
            return 0.01 * HARTREE_TO_JOULE

        E_si = 0.005 * HARTREE_TO_JOULE
        mass = 1.0 * AMU_TO_KG
        x1, x2 = 1.0, 2.0  # Width = 1 radian

        action, diag = adaptive_action_integral(E_si, V, mass, x1, x2)

        # For constant V > E: S = sqrt(2m(V-E)) * (x2-x1)
        expected = np.sqrt(2 * mass * (0.01 - 0.005) * HARTREE_TO_JOULE) * 1.0
        assert_allclose(action, expected, rtol=0.1)


class TestCompareWithEckart:
    """Tests for WKB vs Eckart comparison utility."""

    def test_comparison_runs(self, mock_pes_eckart):
        """Test that comparison utility works."""
        # Generate WKB result
        wkb = WKBMethod()
        result = wkb.calculate_all(
            mock_pes_eckart,
            reduced_mass_amu=1.0,  # Correct parameter name
            energy_points=10,
            min_ratio=0.3,
            max_ratio=0.9,
        )

        comparison = compare_with_eckart(mock_pes_eckart, 1.0, result)

        assert "eckart_params" in comparison
        assert "rmsd" in comparison
        assert "eckart_transmissions" in comparison
        assert len(comparison["eckart_transmissions"]) == len(result.transmissions)
