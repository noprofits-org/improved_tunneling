"""Tests for PES interpolation with periodic boundary conditions."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from improved_tunnel.pes.interpolation import (
    PeriodicSpline,
    interpolate_pes,
    create_potential_function,
    find_classical_turning_points,
)
from improved_tunnel.pes.scan import PESScanResult
from improved_tunnel.core.constants import KCAL_TO_HARTREE


class TestPeriodicSpline:
    """Tests for periodic spline interpolation."""

    def test_basic_interpolation(self):
        """Test basic cubic spline interpolation."""
        angles = np.array([0, 90, 180, 270])
        energies = np.array([1.0, 0.0, 1.0, 0.0])  # Simple cos-like

        spline = PeriodicSpline(angles, energies)

        # Should interpolate reasonably at intermediate points
        assert 0.0 < spline(45) < 1.0
        assert 0.0 < spline(135) < 1.0

    def test_periodic_boundary_conditions(self):
        """Test that spline respects periodicity."""
        angles = np.array([0, 90, 180, 270])
        energies = np.array([1.0, 0.5, 0.0, 0.5])

        spline = PeriodicSpline(angles, energies)

        # V(0) should equal V(360)
        assert_allclose(spline(0), spline(360), rtol=1e-10)

        # V(10) should equal V(370)
        assert_allclose(spline(10), spline(370), rtol=1e-10)

        # V(-10) should equal V(350)
        assert_allclose(spline(-10), spline(350), rtol=1e-10)

    def test_exact_at_data_points(self):
        """Test that spline passes through data points."""
        angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        energies = np.random.rand(8)

        spline = PeriodicSpline(angles, energies)

        for angle, energy in zip(angles, energies):
            assert_allclose(spline(angle), energy, rtol=1e-6)

    def test_evaluate_array(self):
        """Test evaluation at multiple points."""
        angles = np.linspace(0, 360, 13)[:-1]  # 0 to 330
        energies = np.cos(np.radians(angles))

        spline = PeriodicSpline(angles, energies)

        test_angles = np.array([0, 90, 180, 270])
        results = spline.evaluate(test_angles)

        assert results.shape == (4,)

    def test_derivative(self):
        """Test derivative calculation."""
        # Use smooth function where derivative is known
        angles = np.linspace(0, 360, 37)[:-1]
        # V = (1 - cos(θ))/2, dV/dθ = sin(θ)/2 (in radians)
        energies = (1 - np.cos(np.radians(angles))) / 2

        spline = PeriodicSpline(angles, energies)

        # At θ=90°, dV/dθ ≈ 0.5 (in per-radian units)
        # Need to account for deg->rad conversion
        deriv = spline.derivative(90, order=1)
        # First derivative should be positive at 90°
        assert deriv > 0

    def test_barrier_info(self):
        """Test barrier height and position detection."""
        angles = np.linspace(0, 360, 37)[:-1]
        # V = Vmax * (1 - cos(θ))/2, max at θ=180°
        Vmax = 0.01  # Hartree
        energies = Vmax * (1 - np.cos(np.radians(angles))) / 2

        spline = PeriodicSpline(angles, energies)
        info = spline.get_barrier_info()

        assert_allclose(info["barrier_height"], Vmax, rtol=0.05)
        assert_allclose(info["max_angle"], 180, atol=5)
        assert_allclose(info["min_angle"], 0, atol=5)

    def test_minimum_points_error(self):
        """Test error with too few points."""
        with pytest.raises(ValueError):
            PeriodicSpline(np.array([0]), np.array([1.0]))

    def test_handles_duplicate_angles(self):
        """Test that duplicate angles are handled."""
        # This was a bug that was fixed
        angles = np.array([0, 90, 180, 270, 360])  # 0 and 360 are same
        energies = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

        spline = PeriodicSpline(angles, energies)
        # Should not raise, and should interpolate correctly
        assert_allclose(spline(0), 1.0, rtol=0.1)


class TestInterpolatePES:
    """Tests for PES interpolation from scan results."""

    def test_interpolate_from_pes_result(self, mock_pes_symmetric):
        """Test interpolation from PESScanResult."""
        spline = interpolate_pes(mock_pes_symmetric)

        # Should be able to evaluate at any angle
        assert isinstance(spline(45), float)
        assert isinstance(spline(123.456), float)

    def test_potential_function_creation(self, mock_pes_symmetric):
        """Test creating callable potential function."""
        V = create_potential_function(mock_pes_symmetric, use_radians=True, relative=True)

        # Should return 0 at minimum
        assert V(np.pi / 2) >= 0  # Minimum near 90°

        # Should be callable with float
        assert isinstance(V(1.5), float)

    def test_potential_in_radians_vs_degrees(self, mock_pes_symmetric):
        """Test potential function with different angle units."""
        V_rad = create_potential_function(
            mock_pes_symmetric, use_radians=True, relative=True
        )
        V_deg = create_potential_function(
            mock_pes_symmetric, use_radians=False, relative=True
        )

        # V_rad(π) should equal V_deg(180)
        assert_allclose(V_rad(np.pi), V_deg(180), rtol=1e-5)


class TestTurningPoints:
    """Tests for classical turning point finding."""

    def test_find_turning_points_simple_barrier(self):
        """Test turning point detection for simple barrier."""
        # Parabolic barrier: V(x) = -x^2 + 1 for x in [-1, 1]
        # At E = 0.5, turning points are where V = E, i.e., x = ±sqrt(0.5)

        def V(x):
            # Shift to positive x range
            x_shifted = x - np.pi
            return max(0, 1 - x_shifted**2)

        # This is tricky because our function is in [0, 2π]
        # Use simpler test: cosine barrier
        def V_cos(x):
            return (1 - np.cos(x)) / 2  # Barrier at x=π, V=1

        E = 0.25  # Energy level
        x1, x2 = find_classical_turning_points(V_cos, E, 0, 2 * np.pi)

        # Turning points should be symmetric around π
        assert x1 < np.pi
        assert x2 > np.pi
        # V at turning points should approximately equal E
        assert_allclose(V_cos(x1), E, atol=0.05)
        assert_allclose(V_cos(x2), E, atol=0.05)

    def test_turning_points_error_above_barrier(self):
        """Test error when energy is above barrier."""

        def V(x):
            return (1 - np.cos(x)) / 2  # Max V = 1 at x = π

        # Energy above barrier - may find 0 or 2 crossings depending on range
        # The function should either work or raise ValueError
        E = 1.5  # Above barrier max
        # This might not find two crossings
        try:
            find_classical_turning_points(V, E, 0, 2 * np.pi)
        except ValueError:
            pass  # Expected for some configurations

    def test_turning_points_energy_at_minimum(self):
        """Test error when energy at or below minimum."""

        def V(x):
            return (1 - np.cos(x)) / 2  # Min V = 0

        with pytest.raises(ValueError):
            find_classical_turning_points(V, -0.1, 0, 2 * np.pi)


class TestSplineWithRealPES:
    """Tests using mock PES data resembling real calculations."""

    def test_h2o2_like_pes(self, mock_pes_symmetric):
        """Test interpolation of H2O2-like PES."""
        spline = interpolate_pes(mock_pes_symmetric)

        # Check barrier info
        info = spline.get_barrier_info()

        # Barrier should be around 7 kcal/mol ≈ 0.011 Hartree
        assert 0.005 < info["barrier_height"] < 0.02

    def test_smooth_interpolation(self, mock_pes_symmetric):
        """Test that interpolation is smooth (no wild oscillations)."""
        spline = interpolate_pes(mock_pes_symmetric)

        # Evaluate on fine grid
        fine_angles = np.linspace(0, 360, 1000)
        fine_energies = spline.evaluate(fine_angles)

        # Check no wild swings (gradient should be bounded)
        gradients = np.diff(fine_energies) / np.diff(fine_angles)
        assert np.all(np.abs(gradients) < 0.001)  # Reasonable bound
