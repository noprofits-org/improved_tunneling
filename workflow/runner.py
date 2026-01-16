"""Main workflow orchestrator for tunneling calculations."""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.config import WorkflowConfig, ComputationalConfig, PESScanConfig
from ..core.exceptions import WorkflowError
from ..molecule.structure import Molecule
from ..molecule.reduced_mass import calculate_torsional_reduced_mass
from ..molecule.isotopes import substitute_isotopes
from ..qchem.base import QChemEngine, FrequencyResult
from ..qchem.psi4_engine import Psi4Engine, MockPsi4Engine
from ..pes.scan import PESScanResult
from ..pes.rigid_scan import RigidPESScan
from ..pes.relaxed_scan import RelaxedPESScan
from ..tunneling.base import TunnelingResult
from ..tunneling.wkb import WKBMethod
from ..tunneling.sct import SCTMethod
from ..tunneling.eckart import EckartBarrier
from ..kinetics.rates import RateResult, calculate_rates_vs_temperature
from ..kinetics.arrhenius import analyze_arrhenius

logger = logging.getLogger(__name__)


class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    OPTIMIZATION = auto()
    FREQUENCY = auto()
    PES_SCAN = auto()
    TUNNELING = auto()
    KINETICS = auto()


@dataclass
class WorkflowState:
    """
    Serializable workflow state for checkpointing.

    Stores results from each completed step.
    """
    completed_steps: List[str] = field(default_factory=list)
    optimized_molecule: Optional[Dict] = None
    frequency_result: Optional[Dict] = None
    pes_result: Optional[Dict] = None
    tunneling_results: Optional[Dict[str, Dict]] = None
    rate_results: Optional[Dict[str, Dict]] = None
    timing: Dict[str, float] = field(default_factory=dict)
    config: Optional[Dict] = None

    def save(self, filepath: Path) -> None:
        """Save state to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info(f"Saved workflow state to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "WorkflowState":
        """Load state from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TunnelingWorkflow:
    """
    Main workflow orchestrator for quantum tunneling calculations.

    Manages the complete calculation pipeline:
    1. Geometry optimization
    2. Frequency calculation (ZPE)
    3. PES scan (rigid or relaxed)
    4. Tunneling calculations (multiple methods)
    5. Rate calculations with temperature dependence

    Supports checkpointing for long calculations.
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        use_mock_engine: bool = False
    ):
        """
        Initialize workflow.

        Args:
            config: Workflow configuration (default if None)
            use_mock_engine: Use mock QChem engine for testing
        """
        self.config = config or WorkflowConfig()
        self.state = WorkflowState()
        self._use_mock = use_mock_engine

        # Initialize QChem engine
        if use_mock_engine:
            self._engine = MockPsi4Engine()
        else:
            self._engine = Psi4Engine()

        # Results storage
        self._molecule: Optional[Molecule] = None
        self._freq_result: Optional[FrequencyResult] = None
        self._pes_result: Optional[PESScanResult] = None
        self._tunneling_results: Dict[str, TunnelingResult] = {}
        self._rate_results: Dict[str, RateResult] = {}

    def run(
        self,
        molecule: Molecule,
        steps: Optional[List[WorkflowStep]] = None,
        resume_from: Optional[Path] = None
    ) -> WorkflowState:
        """
        Execute the workflow.

        Args:
            molecule: Input molecular structure
            steps: Specific steps to run (all if None)
            resume_from: Checkpoint file to resume from

        Returns:
            Final WorkflowState with all results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Quantum Tunneling Workflow")
        logger.info("=" * 60)

        # Resume from checkpoint if specified
        if resume_from and resume_from.exists():
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.state = WorkflowState.load(resume_from)
            self._rehydrate_from_state(molecule)
        else:
            self._molecule = molecule

        # Initialize engine
        self._engine.initialize(self.config.computational)

        # Determine which steps to run
        if steps is None:
            steps = list(WorkflowStep)

        # Execute steps in order
        try:
            for step in steps:
                if step.name in self.state.completed_steps:
                    logger.info(f"Skipping {step.name} (already completed)")
                    continue

                step_start = datetime.now()
                logger.info(f"\n--- Step: {step.name} ---")

                if step == WorkflowStep.OPTIMIZATION:
                    self._run_optimization()
                elif step == WorkflowStep.FREQUENCY:
                    self._run_frequency()
                elif step == WorkflowStep.PES_SCAN:
                    self._run_pes_scan()
                elif step == WorkflowStep.TUNNELING:
                    self._run_tunneling()
                elif step == WorkflowStep.KINETICS:
                    self._run_kinetics()

                step_time = (datetime.now() - step_start).total_seconds()
                self.state.timing[step.name] = step_time
                self.state.completed_steps.append(step.name)

                logger.info(f"Completed {step.name} in {step_time:.1f}s")

                # Save checkpoint after each step
                if self.config.checkpoint_file:
                    self.state.save(self.config.checkpoint_file)

        except Exception as e:
            logger.error(f"Workflow failed at {step.name}: {e}")
            if self.config.checkpoint_file:
                self.state.save(self.config.checkpoint_file)
            raise WorkflowError(f"Workflow failed: {e}", step=step.name)

        total_time = (datetime.now() - start_time).total_seconds()
        self.state.timing["total"] = total_time

        logger.info("\n" + "=" * 60)
        logger.info(f"Workflow completed in {total_time:.1f}s")
        logger.info("=" * 60)

        return self.state

    def _run_optimization(self) -> None:
        """Run geometry optimization."""
        logger.info(f"Optimizing {self._molecule.formula}")

        result = self._engine.optimize_geometry(self._molecule)
        self._molecule = result.molecule

        self.state.optimized_molecule = {
            "symbols": self._molecule.symbols,
            "coordinates": self._molecule.coordinates.tolist(),
            "energy": result.energy,
        }

        logger.info(f"Optimized energy: {result.energy:.8f} Hartree")

    def _run_frequency(self) -> None:
        """Run frequency calculation for ZPE."""
        if not self.config.calculate_zpe:
            logger.info("Skipping frequency calculation (disabled in config)")
            return

        logger.info("Calculating harmonic frequencies")

        self._freq_result = self._engine.frequencies(self._molecule)

        self.state.frequency_result = {
            "frequencies": self._freq_result.frequencies.tolist(),
            "zpe": self._freq_result.zpe,
            "n_imaginary": self._freq_result.n_imaginary,
        }

        logger.info(f"ZPE: {self._freq_result.zpe:.6f} Hartree")
        logger.info(f"Imaginary frequencies: {self._freq_result.n_imaginary}")

    def _run_pes_scan(self) -> None:
        """Run PES dihedral scan."""
        scan_config = self.config.pes_scan

        if scan_config.scan_type == "relaxed":
            logger.info("Running RELAXED PES scan")
            scanner = RelaxedPESScan(self._engine, scan_config)
        else:
            logger.info("Running RIGID PES scan")
            scanner = RigidPESScan(self._engine, scan_config)

        self._pes_result = scanner.run(self._molecule)

        self.state.pes_result = self._pes_result.to_dict()

        logger.info(f"Barrier height: {self._pes_result.barrier_height_kcal:.2f} kcal/mol")

    def _run_tunneling(self) -> None:
        """Run tunneling calculations with configured methods."""
        if self._pes_result is None:
            raise WorkflowError("PES scan must be completed first", step="TUNNELING")

        # Calculate reduced mass
        reduced_mass = calculate_torsional_reduced_mass(
            self._molecule,
            self.config.pes_scan.dihedral_atoms
        )
        logger.info(f"Torsional reduced mass: {reduced_mass:.4f} AMU")

        tunneling_config = self.config.tunneling
        self._tunneling_results = {}

        for method_name in tunneling_config.methods:
            logger.info(f"\nRunning {method_name} tunneling calculation")

            if method_name.upper() == "WKB":
                method = WKBMethod(
                    integration_tolerance=tunneling_config.integration_tolerance,
                    max_subdivisions=tunneling_config.max_subdivisions
                )
            elif method_name.upper() == "SCT":
                method = SCTMethod(
                    integration_tolerance=tunneling_config.integration_tolerance
                )
            elif method_name.upper() == "ECKART":
                method = EckartBarrier()
                method.fit_to_pes(self._pes_result)
            else:
                logger.warning(f"Unknown tunneling method: {method_name}")
                continue

            result = method.calculate_all(
                self._pes_result,
                reduced_mass,
                energy_points=tunneling_config.energy_points,
                min_ratio=tunneling_config.min_energy_ratio,
                max_ratio=tunneling_config.max_energy_ratio
            )

            self._tunneling_results[method_name] = result
            logger.info(f"{method_name}: T(0.5*Vb) = {result.get_transmission_at_ratio(0.5):.2e}")

        # Store in state
        self.state.tunneling_results = {
            name: result.to_dict()
            for name, result in self._tunneling_results.items()
        }

    def _run_kinetics(self) -> None:
        """Run rate calculations."""
        if not self._tunneling_results:
            raise WorkflowError("Tunneling must be completed first", step="KINETICS")

        kinetics_config = self.config.kinetics
        temperatures = kinetics_config.get_temperatures()

        self._rate_results = {}

        for method_name, tun_result in self._tunneling_results.items():
            logger.info(f"\nCalculating rates for {method_name}")

            rate_result = calculate_rates_vs_temperature(
                tun_result,
                temperatures,
                prefactor=kinetics_config.prefactor
            )

            self._rate_results[method_name] = rate_result

            # Arrhenius analysis
            arrhenius = analyze_arrhenius(rate_result)
            logger.info(
                f"Effective Ea: {arrhenius['quantum']['activation_energy_kcal']:.2f} kcal/mol"
            )
            logger.info(
                f"Tunneling reduces Ea by {arrhenius['Ea_reduction_percent']:.1f}%"
            )

        # Store in state
        self.state.rate_results = {
            name: result.to_dict()
            for name, result in self._rate_results.items()
        }

    def get_results(self) -> Dict[str, Any]:
        """Get all results as a dictionary."""
        return {
            "config": asdict(self.config) if self.config else None,
            "molecule": self.state.optimized_molecule,
            "frequency": self.state.frequency_result,
            "pes": self.state.pes_result,
            "tunneling": self.state.tunneling_results,
            "rates": self.state.rate_results,
            "timing": self.state.timing,
        }

    def _rehydrate_from_state(self, molecule: Molecule) -> None:
        """
        Restore internal state from loaded WorkflowState.

        Called when resuming from checkpoint to ensure internal objects
        match the loaded state, allowing skipped steps to work correctly.
        """
        import numpy as np

        # Start with provided molecule (may be overwritten if optimized)
        self._molecule = molecule

        # Restore optimized molecule if available
        if self.state.optimized_molecule:
            opt_data = self.state.optimized_molecule
            coords = np.array(opt_data.get("coordinates", []))
            if len(coords) > 0:
                self._molecule = molecule.copy()
                self._molecule.coordinates = coords
                self._molecule.energy = opt_data.get("energy")
            logger.info("Restored optimized molecule from checkpoint")

        # Restore PES result if available
        if self.state.pes_result:
            self._pes_result = PESScanResult.from_dict(self.state.pes_result)
            logger.info(f"Restored PES result: {self._pes_result.num_points} points")

        # Restore tunneling results if available
        if self.state.tunneling_results:
            self._tunneling_results = {
                name: TunnelingResult.from_dict(data)
                for name, data in self.state.tunneling_results.items()
            }
            logger.info(f"Restored tunneling results: {list(self._tunneling_results.keys())}")

        # Note: rate_results are not needed for rehydration since
        # they are the final step and stored directly in state


def run_h2o2_workflow(
    method: str = "MP2",
    basis: str = "cc-pVDZ",
    scan_type: str = "rigid",
    use_mock: bool = False
) -> WorkflowState:
    """
    Convenience function to run H2O2 tunneling workflow.

    Args:
        method: QC method
        basis: Basis set
        scan_type: "rigid" or "relaxed"
        use_mock: Use mock engine for testing

    Returns:
        WorkflowState with results
    """
    config = WorkflowConfig(
        computational=ComputationalConfig(
            method=method,
            basis=basis
        ),
        pes_scan=PESScanConfig(
            scan_type=scan_type,
            dihedral_atoms=[2, 0, 1, 3],  # H-O-O-H
            start_angle=0.0,
            end_angle=360.0,
            step_size=10.0
        )
    )

    workflow = TunnelingWorkflow(config, use_mock_engine=use_mock)
    molecule = Molecule.h2o2()

    return workflow.run(molecule)
