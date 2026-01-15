"""Pytest fixtures for improved_tunnel tests."""

import pytest
import numpy as np
from improved_tunnel.molecule.structure import Molecule, Atom
from improved_tunnel.pes.scan import PESScanResult
from improved_tunnel.core.constants import KCAL_TO_HARTREE


@pytest.fixture
def h2o2_molecule():
    """Standard H2O2 molecule at equilibrium dihedral (111.5°)."""
    return Molecule.h2o2(dihedral=111.5)


@pytest.fixture
def h2o2_cis():
    """H2O2 at cis configuration (0°)."""
    return Molecule.h2o2(dihedral=0.0)


@pytest.fixture
def h2o2_trans():
    """H2O2 at trans configuration (180°)."""
    return Molecule.h2o2(dihedral=180.0)


@pytest.fixture
def d2o2_molecule(h2o2_molecule):
    """D2O2 (deuterated hydrogen peroxide)."""
    from improved_tunnel.molecule.isotopes import substitute_isotopes
    return substitute_isotopes(h2o2_molecule, global_substitutions={"H": "D"})


@pytest.fixture
def water_molecule():
    """Simple water molecule for testing."""
    atoms = [
        Atom(symbol="O", coordinates=np.array([0.0, 0.0, 0.0])),
        Atom(symbol="H", coordinates=np.array([0.96, 0.0, 0.0])),
        Atom(symbol="H", coordinates=np.array([-0.24, 0.93, 0.0])),
    ]
    return Molecule(atoms=atoms, name="H2O")


@pytest.fixture
def mock_pes_symmetric():
    """
    Mock PES scan result with symmetric double-well potential.

    Simulates H2O2-like potential with barriers at 0° and 180°.
    """
    from improved_tunnel.pes.scan import PESScanPoint

    angles = np.linspace(0, 360, 37)  # Every 10 degrees

    # Symmetric double-well: V = V0 * (1 - cos(2*theta)) / 2
    barrier_kcal = 7.0  # kcal/mol barrier
    barrier_hartree = barrier_kcal * KCAL_TO_HARTREE

    # Simple model: barrier at 0° and 180°, minima at ~110° and ~250°
    angles_rad = np.radians(angles)
    # V = A*(1-cos(2θ)) gives right shape
    energies = barrier_hartree * (1 - np.cos(2 * angles_rad)) / 2

    # Create PESScanPoint objects
    points = [
        PESScanPoint(angle=a, energy=e, molecule=Molecule.h2o2(dihedral=a))
        for a, e in zip(angles, energies)
    ]

    return PESScanResult(
        scan_type="mock",
        dihedral_atoms=[2, 0, 1, 3],
        points=points,
        method="MP2",
        basis="cc-pVDZ"
    )


@pytest.fixture
def mock_pes_eckart():
    """
    Mock PES that closely matches an Eckart barrier.

    Useful for validating WKB against analytical Eckart.
    """
    from improved_tunnel.pes.scan import PESScanPoint

    angles = np.linspace(0, 360, 73)  # Every 5 degrees
    angles_rad = np.radians(angles)

    # Eckart-like: single barrier centered at 180°
    # V(x) = Vmax * sech²((x - x0)/L)
    Vmax = 0.01  # Hartree (~6.3 kcal/mol)
    L = 0.5  # radians
    x0 = np.pi  # Center at 180°

    energies = Vmax / np.cosh((angles_rad - x0) / L)**2

    # Create PESScanPoint objects
    points = [
        PESScanPoint(angle=a, energy=e, molecule=Molecule.h2o2(dihedral=a))
        for a, e in zip(angles, energies)
    ]

    return PESScanResult(
        scan_type="mock",
        dihedral_atoms=[2, 0, 1, 3],
        points=points,
        method="mock",
        basis="mock"
    )


@pytest.fixture
def reduced_mass_h2o2():
    """Expected reduced mass for H2O2 torsion in AMU."""
    # Based on proper moment of inertia calculation
    # Each OH group rotating about O-O axis
    # Literature value ~0.9-1.0 AMU
    return 0.923  # Approximate from previous calculation


@pytest.fixture
def reduced_mass_d2o2():
    """Expected reduced mass for D2O2 torsion in AMU."""
    # Approximately 2x the H2O2 value
    return 1.845


@pytest.fixture
def h2o2_dihedral_atoms():
    """Atom indices for H-O-O-H dihedral."""
    # With atom order [O, O, H, H]: dihedral is H(2)-O(0)-O(1)-H(3)
    return [2, 0, 1, 3]
