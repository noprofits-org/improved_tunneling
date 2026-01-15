"""Tests for torsional reduced mass calculations."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.molecule.reduced_mass import (
    calculate_torsional_reduced_mass,
    calculate_torsional_reduced_mass_simple,
    reduced_mass_in_kg,
    reduced_mass_for_stretch,
    _partition_molecule,
    _moment_of_inertia_about_axis,
)
from improved_tunnel.core.constants import ATOMIC_MASSES, AMU_TO_KG


class TestTorsionalReducedMass:
    """Tests for torsional reduced mass calculation."""

    def test_h2o2_reduced_mass_reasonable(self, h2o2_molecule, h2o2_dihedral_atoms):
        """Test that H2O2 reduced mass is in reasonable range."""
        mu = calculate_torsional_reduced_mass(h2o2_molecule, h2o2_dihedral_atoms)
        # Should be ~0.9-1.0 AMU based on moment of inertia calculation
        assert 0.5 < mu < 2.0, f"Reduced mass {mu} outside expected range"

    def test_h2o2_vs_d2o2_reduced_mass(
        self, h2o2_molecule, d2o2_molecule, h2o2_dihedral_atoms
    ):
        """Test that D2O2 has larger reduced mass than H2O2."""
        mu_h = calculate_torsional_reduced_mass(h2o2_molecule, h2o2_dihedral_atoms)
        mu_d = calculate_torsional_reduced_mass(d2o2_molecule, h2o2_dihedral_atoms)

        # D2O2 should have roughly 2x the reduced mass
        ratio = mu_d / mu_h
        assert 1.5 < ratio < 2.5, f"D/H ratio {ratio} outside expected range"

    def test_reduced_mass_against_literature(
        self, h2o2_molecule, h2o2_dihedral_atoms, reduced_mass_h2o2
    ):
        """Test reduced mass against literature/computed value."""
        mu = calculate_torsional_reduced_mass(h2o2_molecule, h2o2_dihedral_atoms)
        # Allow 10% tolerance for different geometry/method
        assert_allclose(mu, reduced_mass_h2o2, rtol=0.1)

    def test_simple_reduced_mass(self, h2o2_molecule, h2o2_dihedral_atoms):
        """Test simplified reduced mass calculation."""
        mu_simple = calculate_torsional_reduced_mass_simple(
            h2o2_molecule, h2o2_dihedral_atoms
        )
        # For H-O-O-H with terminal H atoms: mu = m_H * m_H / (m_H + m_H) = m_H / 2
        expected = ATOMIC_MASSES["H"] / 2
        assert_allclose(mu_simple, expected, rtol=1e-5)

    def test_reduced_mass_requires_four_atoms(self, h2o2_molecule):
        """Test that dihedral must have exactly 4 atoms."""
        with pytest.raises(ValueError):
            calculate_torsional_reduced_mass(h2o2_molecule, [0, 1, 2])  # Only 3

    def test_reduced_mass_in_kg_conversion(self, h2o2_molecule, h2o2_dihedral_atoms):
        """Test conversion to kg."""
        mu_amu = calculate_torsional_reduced_mass(h2o2_molecule, h2o2_dihedral_atoms)
        mu_kg = reduced_mass_in_kg(mu_amu)
        assert_allclose(mu_kg, mu_amu * AMU_TO_KG, rtol=1e-10)


class TestPartitionMolecule:
    """Tests for molecule partitioning."""

    def test_h2o2_partition(self, h2o2_molecule):
        """Test H2O2 partitions into two OH groups."""
        group_a, group_b = _partition_molecule(h2o2_molecule, j=0, k=1)

        # Group A should have O(0) and H(2), Group B should have O(1) and H(3)
        assert 0 in group_a
        assert 2 in group_a
        assert 1 in group_b
        assert 3 in group_b

    def test_partition_no_overlap(self, h2o2_molecule):
        """Test that partition groups don't overlap except for connecting atoms."""
        group_a, group_b = _partition_molecule(h2o2_molecule, j=0, k=1)
        # Groups may share atoms j and k which are on both sides
        # But the non-central atoms should be separate
        # j=0 is in group_a, k=1 is in group_b
        assert 0 in group_a
        assert 1 in group_b

    def test_partition_covers_all_atoms(self, h2o2_molecule):
        """Test that partition covers all atoms."""
        group_a, group_b = _partition_molecule(h2o2_molecule, j=0, k=1)
        all_atoms = group_a | group_b
        assert all_atoms == set(range(h2o2_molecule.num_atoms))


class TestMomentOfInertia:
    """Tests for moment of inertia calculation."""

    def test_point_on_axis_zero_inertia(self):
        """Test that atom on rotation axis contributes zero."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        axis_point = np.array([0.0, 0.0, 0.0])
        axis_vector = np.array([1.0, 0.0, 0.0])

        # Both atoms are on the x-axis, so perpendicular distance is 0
        I = _moment_of_inertia_about_axis(
            coords, masses, {0, 1}, axis_point, axis_vector
        )
        assert_allclose(I, 0.0, atol=1e-10)

    def test_point_mass_off_axis(self):
        """Test moment of inertia for point mass perpendicular to axis."""
        # Single atom at (0, 1, 0), axis along x
        coords = np.array([[0.0, 1.0, 0.0]])
        masses = np.array([2.0])  # 2 AMU
        axis_point = np.array([0.0, 0.0, 0.0])
        axis_vector = np.array([1.0, 0.0, 0.0])

        # I = m * r^2 = 2 * 1^2 = 2
        I = _moment_of_inertia_about_axis(coords, masses, {0}, axis_point, axis_vector)
        assert_allclose(I, 2.0, rtol=1e-10)


class TestStretchReducedMass:
    """Tests for simple two-body reduced mass."""

    def test_oh_stretch_reduced_mass(self):
        """Test O-H stretch reduced mass."""
        m_o = ATOMIC_MASSES["O"]
        m_h = ATOMIC_MASSES["H"]
        mu = reduced_mass_for_stretch(m_o, m_h)

        expected = (m_o * m_h) / (m_o + m_h)
        assert_allclose(mu, expected, rtol=1e-10)
        # Should be close to H mass since O >> H
        assert_allclose(mu, m_h, rtol=0.1)

    def test_equal_masses(self):
        """Test reduced mass with equal masses."""
        mu = reduced_mass_for_stretch(10.0, 10.0)
        # mu = m/2 for equal masses
        assert_allclose(mu, 5.0, rtol=1e-10)

    def test_symmetric_function(self):
        """Test that reduced mass is symmetric."""
        mu1 = reduced_mass_for_stretch(16.0, 1.0)
        mu2 = reduced_mass_for_stretch(1.0, 16.0)
        assert_allclose(mu1, mu2, rtol=1e-10)
