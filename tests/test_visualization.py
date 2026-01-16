"""Tests for visualization module."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from improved_tunnel.visualization import (
    plot_pes, plot_pes_comparison,
    plot_transmission, plot_transmission_comparison,
    plot_arrhenius, plot_kie,
    plot_instanton_path, plot_ring_polymer,
)
from improved_tunnel.pes.scan import PESScanResult, PESScanPoint
from improved_tunnel.tunneling.base import TunnelingResult
from improved_tunnel.kinetics.rates import RateResult
from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.core.constants import KCAL_TO_HARTREE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_pes_result():
    """Create mock PES scan result."""
    angles = np.linspace(0, 360, 37)
    barrier_hartree = 5.0 * KCAL_TO_HARTREE
    angles_rad = np.radians(angles)
    energies = barrier_hartree * (1 - np.cos(2 * angles_rad)) / 2

    points = [
        PESScanPoint(angle=a, energy=e, molecule=Molecule.h2o2(dihedral=a))
        for a, e in zip(angles, energies)
    ]

    return PESScanResult(
        scan_type="rigid",
        dihedral_atoms=[2, 0, 1, 3],
        points=points,
        method="MP2",
        basis="cc-pVDZ"
    )


@pytest.fixture
def mock_tunneling_result():
    """Create mock tunneling result."""
    barrier = 5.0 * KCAL_TO_HARTREE
    energies = np.linspace(0, 1.5 * barrier, 50)
    # Avoid sqrt of negative numbers
    transmissions = np.where(
        energies < barrier,
        np.exp(-2 * np.sqrt(np.maximum(0, 2 * (barrier - energies)))),
        1.0
    )

    return TunnelingResult(
        method="WKB",
        energies=energies,
        transmissions=transmissions,
        barrier_height=barrier,
        reduced_mass=1.0
    )


@pytest.fixture
def mock_rate_result():
    """Create mock rate result."""
    temperatures = np.array([200, 250, 300, 350, 400, 500])
    classical_rates = 1e12 * np.exp(-5000 / temperatures)
    tunneling_corrections = 1 + 300 / temperatures
    quantum_rates = classical_rates * tunneling_corrections

    return RateResult(
        temperatures=temperatures,
        classical_rates=classical_rates,
        quantum_rates=quantum_rates,
        tunneling_corrections=tunneling_corrections
    )


@pytest.fixture
def mock_bead_positions():
    """Create mock ring-polymer bead positions."""
    n_beads = 16
    n_atoms = 4

    # Create positions that form a path
    bead_positions = np.zeros((n_beads, n_atoms, 3))

    for i in range(n_beads):
        theta = 2 * np.pi * i / n_beads
        # Atom 0 traces a circle (like a tunneling proton)
        bead_positions[i, 0, 0] = 0.3 * np.cos(theta)
        bead_positions[i, 0, 1] = 0.3 * np.sin(theta)
        # Other atoms are mostly stationary
        bead_positions[i, 1, :] = [0, 0, 0.7]
        bead_positions[i, 2, :] = [0, 0, -0.7]
        bead_positions[i, 3, :] = [0, 0, 1.5]

    return bead_positions


# =============================================================================
# PES Plot Tests
# =============================================================================

class TestPESPlot:
    """Tests for PES plotting functions."""

    def test_plot_pes_basic(self, mock_pes_result):
        """Test basic PES plot."""
        fig, ax = plot_pes(mock_pes_result)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_pes_units(self, mock_pes_result):
        """Test PES plot with different units."""
        for units in ["kcal/mol", "kJ/mol", "cm-1", "hartree"]:
            fig, ax = plot_pes(mock_pes_result, units=units)
            assert fig is not None
            plt.close(fig)

    def test_plot_pes_no_barrier(self, mock_pes_result):
        """Test PES plot without barrier marking."""
        fig, ax = plot_pes(mock_pes_result, show_barrier=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_pes_comparison(self, mock_pes_result):
        """Test PES comparison plot."""
        fig, ax = plot_pes_comparison([mock_pes_result, mock_pes_result])
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Transmission Plot Tests
# =============================================================================

class TestTransmissionPlot:
    """Tests for transmission coefficient plotting."""

    def test_plot_transmission_basic(self, mock_tunneling_result):
        """Test basic transmission plot."""
        fig, ax = plot_transmission(mock_tunneling_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_transmission_linear(self, mock_tunneling_result):
        """Test transmission plot with linear scale."""
        fig, ax = plot_transmission(mock_tunneling_result, log_scale=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_transmission_units(self, mock_tunneling_result):
        """Test transmission plot with different x-axis units."""
        for units in ["ratio", "kcal/mol", "hartree"]:
            fig, ax = plot_transmission(mock_tunneling_result, x_units=units)
            assert fig is not None
            plt.close(fig)

    def test_plot_transmission_comparison(self, mock_tunneling_result):
        """Test transmission comparison plot."""
        fig, ax = plot_transmission_comparison(
            [mock_tunneling_result, mock_tunneling_result],
            labels=["WKB", "Eckart"]
        )
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Arrhenius Plot Tests
# =============================================================================

class TestArrheniusPlot:
    """Tests for Arrhenius plotting."""

    def test_plot_arrhenius_basic(self, mock_rate_result):
        """Test basic Arrhenius plot."""
        fig, ax = plot_arrhenius(mock_rate_result)
        assert fig is not None
        plt.close(fig)

    def test_plot_arrhenius_classical_only(self, mock_rate_result):
        """Test Arrhenius with classical rates only."""
        fig, ax = plot_arrhenius(
            mock_rate_result,
            show_quantum=False,
            show_fit=False
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_kie(self, mock_rate_result):
        """Test KIE plot."""
        # Use same result for H and D (just testing plot function)
        fig, ax = plot_kie(mock_rate_result, mock_rate_result)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Instanton Plot Tests
# =============================================================================

class TestInstantonPlot:
    """Tests for instanton visualization."""

    def test_plot_instanton_path(self, mock_bead_positions):
        """Test instanton path plot."""
        fig, ax = plot_instanton_path(mock_bead_positions)
        assert fig is not None
        plt.close(fig)

    def test_plot_instanton_path_selected_atoms(self, mock_bead_positions):
        """Test instanton path with selected atoms."""
        fig, ax = plot_instanton_path(
            mock_bead_positions,
            atom_indices=[0, 1],
            symbols=["H", "O", "O", "H"]
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_ring_polymer_xy(self, mock_bead_positions):
        """Test ring polymer plot (xy projection)."""
        fig, ax = plot_ring_polymer(mock_bead_positions, projection="xy")
        assert fig is not None
        plt.close(fig)

    def test_plot_ring_polymer_xz(self, mock_bead_positions):
        """Test ring polymer plot (xz projection)."""
        fig, ax = plot_ring_polymer(mock_bead_positions, projection="xz")
        assert fig is not None
        plt.close(fig)

    def test_plot_ring_polymer_3d(self, mock_bead_positions):
        """Test ring polymer 3D plot."""
        fig, ax = plot_ring_polymer(mock_bead_positions, projection="3d")
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization."""

    def test_all_plots_have_axes(self, mock_pes_result, mock_tunneling_result,
                                  mock_rate_result, mock_bead_positions):
        """Verify all plot functions return valid figure and axes."""
        plots = [
            plot_pes(mock_pes_result),
            plot_transmission(mock_tunneling_result),
            plot_arrhenius(mock_rate_result),
            plot_instanton_path(mock_bead_positions),
            plot_ring_polymer(mock_bead_positions),
        ]

        for fig, ax in plots:
            assert fig is not None
            assert ax is not None
            plt.close(fig)

    def test_custom_axes(self, mock_pes_result):
        """Test plotting to existing axes."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_pes(mock_pes_result, ax=ax)
        assert ax is ax2  # Should use same axes
        plt.close(fig)

    def test_figure_size(self, mock_pes_result):
        """Test custom figure size."""
        fig, ax = plot_pes(mock_pes_result, figsize=(12, 8))
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)
