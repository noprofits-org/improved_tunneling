"""Tests for ORCA quantum chemistry engine."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from improved_tunnel.qchem.orca_engine import ORCAEngine, MockORCAEngine
from improved_tunnel.qchem.base import QChemResult, FrequencyResult
from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.core.config import ComputationalConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def h2o2_molecule():
    """H2O2 molecule for testing."""
    return Molecule.h2o2()


@pytest.fixture
def water_molecule():
    """Water molecule for testing."""
    return Molecule.from_xyz_string("""3
Water molecule
O   0.000000   0.000000   0.117369
H   0.756950   0.000000  -0.469475
H  -0.756950   0.000000  -0.469475
""")


@pytest.fixture
def mock_config():
    """Mock computational configuration."""
    return ComputationalConfig(
        method="HF",
        basis="cc-pVDZ"
    )


@pytest.fixture
def mock_orca_engine(mock_config):
    """Mock ORCA engine for testing."""
    engine = MockORCAEngine()
    engine.initialize(mock_config)
    return engine


# =============================================================================
# Tests for MockORCAEngine
# =============================================================================

class TestMockORCAEngine:
    """Tests for the mock ORCA engine."""

    def test_initialization(self, mock_config):
        """Test mock engine initialization."""
        engine = MockORCAEngine()
        engine.initialize(mock_config)
        assert engine._initialized is True

    def test_single_point_energy(self, mock_orca_engine, h2o2_molecule):
        """Test single-point energy calculation."""
        result = mock_orca_engine.single_point_energy(h2o2_molecule)

        assert isinstance(result, QChemResult)
        assert result.converged is True
        assert isinstance(result.energy, float)
        assert result.energy < 0  # Energy should be negative

    def test_single_point_returns_molecule(self, mock_orca_engine, h2o2_molecule):
        """Test that single-point returns molecule with energy."""
        result = mock_orca_engine.single_point_energy(h2o2_molecule)

        assert result.molecule is not None
        assert result.molecule.energy == result.energy

    def test_optimize_geometry(self, mock_orca_engine, h2o2_molecule):
        """Test geometry optimization."""
        result = mock_orca_engine.optimize_geometry(h2o2_molecule)

        assert isinstance(result, QChemResult)
        assert result.converged is True
        assert result.molecule is not None
        # Coordinates should be slightly different (mock adds noise)
        assert not np.allclose(result.molecule.coordinates, h2o2_molecule.coordinates)

    def test_optimize_with_constraints(self, mock_orca_engine, h2o2_molecule):
        """Test geometry optimization with constraints."""
        constraints = {
            "frozen_dihedral": {
                "atoms": [0, 1, 2, 3],
                "value": 120.0
            }
        }
        result = mock_orca_engine.optimize_geometry(h2o2_molecule, constraints=constraints)

        assert result.converged is True

    def test_frequencies(self, mock_orca_engine, h2o2_molecule):
        """Test frequency calculation."""
        result = mock_orca_engine.frequencies(h2o2_molecule)

        assert isinstance(result, FrequencyResult)
        assert len(result.frequencies) > 0
        assert result.zpe > 0

    def test_gradient(self, mock_orca_engine, h2o2_molecule):
        """Test gradient calculation."""
        energy, gradient = mock_orca_engine.gradient(h2o2_molecule)

        assert isinstance(energy, float)
        assert energy < 0
        assert gradient.shape == (4, 3)  # H2O2 has 4 atoms

    def test_not_initialized_error(self, h2o2_molecule):
        """Test error when not initialized."""
        engine = MockORCAEngine()

        with pytest.raises(RuntimeError, match="not initialized"):
            engine.single_point_energy(h2o2_molecule)

    def test_custom_method_and_basis(self, mock_orca_engine, h2o2_molecule):
        """Test with custom method and basis."""
        result = mock_orca_engine.single_point_energy(
            h2o2_molecule,
            method="MP2",
            basis="aug-cc-pVDZ"
        )

        assert result.method == "MP2"
        assert result.basis == "aug-cc-pVDZ"


# =============================================================================
# Tests for ORCAEngine (unit tests that don't require ORCA)
# =============================================================================

class TestORCAEngineUnit:
    """Unit tests for ORCA engine (no ORCA required)."""

    def test_engine_creation(self):
        """Test engine creation."""
        engine = ORCAEngine()
        # Path should contain "orca" (either "orca" or full path like "/usr/bin/orca")
        assert "orca" in engine.orca_path
        assert engine.keep_files is False

    def test_engine_custom_path(self):
        """Test engine with custom ORCA path."""
        engine = ORCAEngine(orca_path="/custom/path/orca")
        assert engine.orca_path == "/custom/path/orca"

    def test_engine_keep_files(self):
        """Test keep_files option."""
        engine = ORCAEngine(keep_files=True)
        assert engine.keep_files is True

    def test_engine_scratch_dir(self):
        """Test custom scratch directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ORCAEngine(scratch_dir=tmpdir)
            assert engine.scratch_dir == tmpdir


class TestORCAInputGeneration:
    """Tests for ORCA input file generation."""

    @pytest.fixture
    def engine_with_config(self, mock_config):
        """Engine with config but not checked for ORCA."""
        engine = ORCAEngine()
        engine._config = mock_config
        engine._initialized = True  # Skip ORCA check
        return engine

    def test_create_input_single_point(self, engine_with_config, h2o2_molecule):
        """Test single-point input generation."""
        inp = engine_with_config._create_input(
            h2o2_molecule,
            method="HF",
            basis="cc-pVDZ",
            job_type="SP"
        )

        assert "! HF cc-pVDZ TightSCF" in inp
        assert "* xyz 0 1" in inp
        assert "H" in inp
        assert "O" in inp

    def test_create_input_optimization(self, engine_with_config, h2o2_molecule):
        """Test optimization input generation."""
        inp = engine_with_config._create_input(
            h2o2_molecule,
            method="B3LYP",
            basis="def2-SVP",
            job_type="OPT"
        )

        assert "Opt" in inp
        assert "B3LYP" in inp
        assert "def2-SVP" in inp

    def test_create_input_with_constraints(self, engine_with_config, h2o2_molecule):
        """Test input with geometry constraints."""
        constraints = {
            "frozen_dihedral": {
                "atoms": [0, 1, 2, 3],
                "value": 120.0
            }
        }

        inp = engine_with_config._create_input(
            h2o2_molecule,
            method="HF",
            basis="cc-pVDZ",
            job_type="OPT",
            constraints=constraints
        )

        assert "%geom" in inp
        assert "Constraints" in inp
        assert "D 0 1 2 3 C" in inp

    def test_create_input_frequency(self, engine_with_config, h2o2_molecule):
        """Test frequency calculation input."""
        inp = engine_with_config._create_input(
            h2o2_molecule,
            method="HF",
            basis="cc-pVDZ",
            job_type="FREQ"
        )

        assert "Freq" in inp

    def test_create_input_gradient(self, engine_with_config, h2o2_molecule):
        """Test gradient calculation input."""
        inp = engine_with_config._create_input(
            h2o2_molecule,
            method="HF",
            basis="cc-pVDZ",
            job_type="ENGRAD"
        )

        assert "EnGrad" in inp

    def test_method_translation(self, engine_with_config):
        """Test method name translation."""
        assert engine_with_config._translate_method("MP2") == "RI-MP2"
        assert engine_with_config._translate_method("HF") == "HF"
        assert engine_with_config._translate_method("CCSD(T)") == "CCSD(T)"
        assert engine_with_config._translate_method("B3LYP") == "B3LYP"

    def test_memory_parsing(self, engine_with_config):
        """Test memory string parsing."""
        assert engine_with_config._parse_memory("2 GB") == 2000
        assert engine_with_config._parse_memory("4GB") == 4000
        assert engine_with_config._parse_memory("500 MB") == 500


class TestORCAOutputParsing:
    """Tests for ORCA output parsing."""

    @pytest.fixture
    def engine_with_config(self, mock_config):
        """Engine with config."""
        engine = ORCAEngine()
        engine._config = mock_config
        engine._initialized = True
        return engine

    def test_parse_single_point_output(self, engine_with_config):
        """Test parsing single-point output."""
        output = """
        Some ORCA output...
        FINAL SINGLE POINT ENERGY      -151.234567890
        More output...
        ORCA TERMINATED NORMALLY
        """

        energy, converged = engine_with_config._parse_output(output)

        assert abs(energy - (-151.234567890)) < 1e-10
        assert converged is True

    def test_parse_optimization_output(self, engine_with_config):
        """Test parsing optimization output."""
        output = """
        THE OPTIMIZATION HAS CONVERGED
        FINAL SINGLE POINT ENERGY      -151.345678901
        ORCA TERMINATED NORMALLY
        """

        energy, converged = engine_with_config._parse_output(output)

        assert abs(energy - (-151.345678901)) < 1e-10
        assert converged is True

    def test_parse_failed_output(self, engine_with_config):
        """Test parsing failed calculation output."""
        output = """
        ERROR - SCF NOT CONVERGED
        """

        energy, converged = engine_with_config._parse_output(output)

        assert energy is None
        assert converged is False


# =============================================================================
# Integration tests (only run if ORCA is available)
# =============================================================================

def orca_available():
    """Check if ORCA quantum chemistry is available."""
    import shutil
    import subprocess

    orca_path = shutil.which("orca")
    if not orca_path:
        return False

    # Check if it's actually ORCA quantum chemistry (not the screen reader)
    try:
        result = subprocess.run(
            [orca_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # ORCA quantum chemistry shows "ORCA - An Ab initio" in output
        return "Ab initio" in result.stdout or "quantum chemistry" in result.stdout.lower()
    except Exception:
        return False


@pytest.mark.skipif(not orca_available(), reason="ORCA not installed")
class TestORCAIntegration:
    """Integration tests requiring ORCA installation."""

    def test_orca_single_point(self, h2o2_molecule, mock_config):
        """Test real ORCA single-point calculation."""
        engine = ORCAEngine()
        engine.initialize(mock_config)

        result = engine.single_point_energy(h2o2_molecule)

        assert result.converged is True
        assert result.energy < 0

    def test_orca_gradient(self, h2o2_molecule, mock_config):
        """Test real ORCA gradient calculation."""
        engine = ORCAEngine()
        engine.initialize(mock_config)

        energy, gradient = engine.gradient(h2o2_molecule)

        assert energy < 0
        assert gradient.shape == (4, 3)
