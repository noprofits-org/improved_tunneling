"""Integration tests for the complete tunneling workflow."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from improved_tunnel.workflow.runner import (
    TunnelingWorkflow,
    WorkflowState,
    WorkflowStep,
    run_h2o2_workflow,
)
from improved_tunnel.core.config import (
    WorkflowConfig,
    ComputationalConfig,
    PESScanConfig,
    TunnelingConfig,
    KineticsConfig,
)
from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.core.exceptions import WorkflowError


class TestWorkflowState:
    """Tests for WorkflowState serialization."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = WorkflowState()
        assert state.completed_steps == []
        assert state.optimized_molecule is None

    def test_state_save_load(self):
        """Test state serialization round-trip."""
        state = WorkflowState()
        state.completed_steps = ["OPTIMIZATION", "FREQUENCY"]
        state.timing = {"OPTIMIZATION": 10.5, "FREQUENCY": 5.2}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            state.save(filepath)
            loaded = WorkflowState.load(filepath)

            assert loaded.completed_steps == state.completed_steps
            assert loaded.timing == state.timing
        finally:
            filepath.unlink()


class TestWorkflowConfig:
    """Tests for workflow configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig()

        assert config.computational is not None
        assert config.pes_scan is not None
        assert config.tunneling is not None
        assert config.kinetics is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            computational=ComputationalConfig(method="HF", basis="6-31G"),
            pes_scan=PESScanConfig(step_size=5.0),
        )

        assert config.computational.method == "HF"
        assert config.pes_scan.step_size == 5.0

    def test_kinetics_temperature_range(self):
        """Test temperature array generation."""
        kinetics = KineticsConfig(temp_min=200, temp_max=400, temp_step=50)
        temps = kinetics.get_temperatures()

        assert temps[0] == 200
        assert temps[-1] == 400
        assert len(temps) == 5


class TestTunnelingWorkflow:
    """Tests for main workflow orchestrator."""

    def test_workflow_creation_mock(self):
        """Test workflow creation with mock engine."""
        workflow = TunnelingWorkflow(use_mock_engine=True)
        assert workflow._use_mock is True

    def test_workflow_with_mock_engine(self, h2o2_molecule):
        """Test complete workflow with mock QChem engine."""
        config = WorkflowConfig(
            computational=ComputationalConfig(method="MP2", basis="cc-pVDZ"),
            pes_scan=PESScanConfig(
                dihedral_atoms=[2, 0, 1, 3],
                start_angle=0,
                end_angle=360,
                step_size=30,  # Coarse for speed
            ),
            tunneling=TunnelingConfig(
                methods=["WKB", "Eckart"],
                energy_points=10,
            ),
            kinetics=KineticsConfig(
                temp_min=250,
                temp_max=350,
                temp_step=50,
            ),
            calculate_zpe=False,  # Skip for speed
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(h2o2_molecule)

        # Check all steps completed
        assert "OPTIMIZATION" in state.completed_steps
        assert "PES_SCAN" in state.completed_steps
        assert "TUNNELING" in state.completed_steps
        assert "KINETICS" in state.completed_steps

        # Check results exist
        assert state.pes_result is not None
        assert state.tunneling_results is not None
        assert state.rate_results is not None

        # Check timing recorded
        assert "total" in state.timing

    def test_workflow_step_selection(self, h2o2_molecule):
        """Test running specific workflow steps."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=45),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)

        # Run only optimization and PES scan
        state = workflow.run(
            h2o2_molecule, steps=[WorkflowStep.OPTIMIZATION, WorkflowStep.PES_SCAN]
        )

        assert "OPTIMIZATION" in state.completed_steps
        assert "PES_SCAN" in state.completed_steps
        assert "TUNNELING" not in state.completed_steps

    def test_workflow_checkpoint_resume(self, h2o2_molecule):
        """Test workflow checkpoint save and load."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            config = WorkflowConfig(
                pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=60),
                checkpoint_file=checkpoint_path,
                calculate_zpe=False,
            )

            # Run partial workflow
            workflow1 = TunnelingWorkflow(config, use_mock_engine=True)
            state1 = workflow1.run(
                h2o2_molecule, steps=[WorkflowStep.OPTIMIZATION, WorkflowStep.PES_SCAN]
            )

            # Verify checkpoint was saved
            assert checkpoint_path.exists()

            # Load checkpoint and verify state
            loaded_state = WorkflowState.load(checkpoint_path)
            assert "OPTIMIZATION" in loaded_state.completed_steps
            assert "PES_SCAN" in loaded_state.completed_steps
            assert loaded_state.pes_result is not None

        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def test_workflow_get_results(self, h2o2_molecule):
        """Test getting results dictionary."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=45),
            tunneling=TunnelingConfig(methods=["Eckart"], energy_points=5),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        workflow.run(h2o2_molecule)

        results = workflow.get_results()

        assert "pes" in results
        assert "tunneling" in results
        assert "rates" in results
        assert "timing" in results


class TestConvenienceFunction:
    """Tests for convenience workflow functions."""

    def test_run_h2o2_workflow_mock(self):
        """Test H2O2 convenience function with mock engine."""
        state = run_h2o2_workflow(
            method="HF", basis="6-31G", scan_type="rigid", use_mock=True
        )

        assert "PES_SCAN" in state.completed_steps
        assert state.pes_result is not None


class TestWorkflowOutput:
    """Tests for workflow output validation."""

    def test_pes_result_structure(self, h2o2_molecule):
        """Test PES result has expected structure."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(
                dihedral_atoms=[2, 0, 1, 3],
                start_angle=0,
                end_angle=360,
                step_size=30,
            ),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(
            h2o2_molecule, steps=[WorkflowStep.OPTIMIZATION, WorkflowStep.PES_SCAN]
        )

        pes = state.pes_result
        assert "angles" in pes
        assert "energies" in pes
        assert "barrier_height_kcal" in pes

    def test_tunneling_result_structure(self, h2o2_molecule):
        """Test tunneling result has expected structure."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=30),
            tunneling=TunnelingConfig(methods=["WKB", "Eckart"], energy_points=10),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(h2o2_molecule)

        assert "WKB" in state.tunneling_results
        assert "Eckart" in state.tunneling_results

        for method, result in state.tunneling_results.items():
            assert "energies" in result
            assert "transmissions" in result

    def test_rate_result_structure(self, h2o2_molecule):
        """Test rate result has expected structure."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=45),
            tunneling=TunnelingConfig(methods=["Eckart"], energy_points=10),
            kinetics=KineticsConfig(temp_min=250, temp_max=350, temp_step=50),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(h2o2_molecule)

        for method, result in state.rate_results.items():
            assert "temperatures" in result
            assert "classical_rates" in result
            assert "quantum_rates" in result
            assert "tunneling_corrections" in result


class TestWorkflowPhysicalResults:
    """Tests that workflow produces physically reasonable results."""

    def test_barrier_height_reasonable(self, h2o2_molecule):
        """Test that mock PES gives a positive barrier height."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=15),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(
            h2o2_molecule, steps=[WorkflowStep.OPTIMIZATION, WorkflowStep.PES_SCAN]
        )

        barrier = state.pes_result["barrier_height_kcal"]
        # Mock engine returns arbitrary values; just check it's positive and finite
        assert barrier > 0, "Barrier should be positive"
        assert np.isfinite(barrier), "Barrier should be finite"

    def test_transmission_physical_range(self, h2o2_molecule):
        """Test transmission coefficients are in [0, 1]."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=30),
            tunneling=TunnelingConfig(methods=["WKB", "Eckart"], energy_points=20),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(h2o2_molecule)

        for method, result in state.tunneling_results.items():
            T = np.array(result["transmissions"])
            assert np.all(T >= 0), f"{method}: negative transmission"
            assert np.all(T <= 1), f"{method}: transmission > 1"

    def test_quantum_rates_enhanced(self, h2o2_molecule):
        """Test quantum rates are >= classical rates (tunneling enhances)."""
        config = WorkflowConfig(
            pes_scan=PESScanConfig(dihedral_atoms=[2, 0, 1, 3], step_size=30),
            tunneling=TunnelingConfig(methods=["Eckart"], energy_points=15),
            kinetics=KineticsConfig(temp_min=200, temp_max=400, temp_step=50),
            calculate_zpe=False,
        )

        workflow = TunnelingWorkflow(config, use_mock_engine=True)
        state = workflow.run(h2o2_molecule)

        for method, result in state.rate_results.items():
            k_class = np.array(result["classical_rates"])
            k_quant = np.array(result["quantum_rates"])
            kappa = np.array(result["tunneling_corrections"])

            # Tunneling correction should be >= 1
            assert np.all(kappa >= 0.99), f"{method}: kappa < 1"

            # Quantum rate should be >= classical (within numerical tolerance)
            ratio = k_quant / (k_class + 1e-100)  # Avoid division by zero
            assert np.all(ratio >= 0.99), f"{method}: quantum < classical"
