#!/usr/bin/env python3
"""
Benchmark runner for quantum tunneling calculations.

Compares calculated results against experimental and high-level
computational reference values.

Usage:
    python -m improved_tunnel.benchmarks.run_benchmark --system malonaldehyde
    python -m improved_tunnel.benchmarks.run_benchmark --system h2o2 --method MP2
"""

import argparse
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .reference_data import MALONALDEHYDE_REF, H2O2_REF, BenchmarkReference
from .molecules import create_malonaldehyde, create_malonaldehyde_ts
from ..molecule.structure import Molecule
from ..core.constants import HARTREE_TO_KCAL, HARTREE_TO_CM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark calculation."""

    system: str
    method: str
    basis: str

    # Calculated values
    barrier_height_kcal: float
    tunneling_splitting_cm: Optional[float] = None

    # Comparison with reference
    barrier_error_kcal: float = 0.0
    splitting_error_cm: float = 0.0
    barrier_error_percent: float = 0.0
    splitting_error_percent: float = 0.0

    # Timing
    wall_time_seconds: float = 0.0

    def compare_to_reference(self, ref: BenchmarkReference) -> None:
        """Calculate errors compared to reference."""
        self.barrier_error_kcal = self.barrier_height_kcal - ref.barrier_height_kcal
        self.barrier_error_percent = (
            100 * self.barrier_error_kcal / ref.barrier_height_kcal
        )

        if self.tunneling_splitting_cm is not None:
            self.splitting_error_cm = (
                self.tunneling_splitting_cm - ref.experimental_splitting_cm
            )
            self.splitting_error_percent = (
                100 * self.splitting_error_cm / ref.experimental_splitting_cm
            )

    def print_report(self, ref: BenchmarkReference) -> None:
        """Print formatted benchmark report."""
        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS: {self.system}")
        print("=" * 60)
        print(f"Method: {self.method}/{self.basis}")
        print(f"Wall time: {self.wall_time_seconds:.1f} s")
        print()

        print("Barrier Height:")
        print(f"  Calculated:  {self.barrier_height_kcal:8.3f} kcal/mol")
        print(f"  Reference:   {ref.barrier_height_kcal:8.3f} kcal/mol")
        print(f"  Error:       {self.barrier_error_kcal:+8.3f} kcal/mol "
              f"({self.barrier_error_percent:+.1f}%)")

        if self.tunneling_splitting_cm is not None:
            print()
            print("Tunneling Splitting:")
            print(f"  Calculated:  {self.tunneling_splitting_cm:8.3f} cm⁻¹")
            print(f"  Experimental:{ref.experimental_splitting_cm:8.3f} cm⁻¹")
            print(f"  Error:       {self.splitting_error_cm:+8.3f} cm⁻¹ "
                  f"({self.splitting_error_percent:+.1f}%)")

        print()
        print("Reference computational values:")
        if ref.computational_refs:
            for name, value in ref.computational_refs.items():
                print(f"  {name}: {value}")

        print("=" * 60)


def run_barrier_benchmark(
    molecule: Molecule,
    ts_molecule: Molecule,
    method: str = "HF",
    basis: str = "cc-pVDZ",
    use_mock: bool = True
) -> float:
    """
    Calculate barrier height between minimum and TS.

    Args:
        molecule: Equilibrium geometry
        ts_molecule: Transition state geometry
        method: QC method
        basis: Basis set
        use_mock: Use mock engine (for testing)

    Returns:
        Barrier height in kcal/mol
    """
    from ..qchem.psi4_engine import Psi4Engine, MockPsi4Engine
    from ..core.config import ComputationalConfig

    config = ComputationalConfig(method=method, basis=basis)

    if use_mock:
        engine = MockPsi4Engine()
    else:
        engine = Psi4Engine()

    engine.initialize(config)

    # Single-point energies
    result_min = engine.single_point(molecule)
    result_ts = engine.single_point(ts_molecule)

    barrier_hartree = result_ts.energy - result_min.energy
    barrier_kcal = barrier_hartree * HARTREE_TO_KCAL

    logger.info(f"E(min) = {result_min.energy:.8f} Ha")
    logger.info(f"E(TS)  = {result_ts.energy:.8f} Ha")
    logger.info(f"Barrier = {barrier_kcal:.3f} kcal/mol")

    return barrier_kcal


def run_malonaldehyde_benchmark(
    method: str = "MP2",
    basis: str = "cc-pVDZ",
    use_mock: bool = True
) -> BenchmarkResult:
    """
    Run malonaldehyde tunneling benchmark.

    This is the gold standard test for tunneling calculations.
    Target: 21.6 cm⁻¹ experimental splitting.

    Args:
        method: QC method
        basis: Basis set
        use_mock: Use mock engine

    Returns:
        BenchmarkResult with comparison to literature
    """
    import time

    logger.info("Running malonaldehyde benchmark")
    logger.info(f"Method: {method}/{basis}")

    start_time = time.time()

    # Create molecules
    mol_eq = create_malonaldehyde()
    mol_ts = create_malonaldehyde_ts()

    # Calculate barrier
    barrier = run_barrier_benchmark(mol_eq, mol_ts, method, basis, use_mock)

    wall_time = time.time() - start_time

    # Create result
    result = BenchmarkResult(
        system="Malonaldehyde",
        method=method,
        basis=basis,
        barrier_height_kcal=barrier,
        wall_time_seconds=wall_time,
    )

    # Compare to reference
    result.compare_to_reference(MALONALDEHYDE_REF)
    result.print_report(MALONALDEHYDE_REF)

    return result


def run_h2o2_benchmark(
    method: str = "MP2",
    basis: str = "cc-pVDZ",
    use_mock: bool = True,
    include_tunneling: bool = True
) -> BenchmarkResult:
    """
    Run H2O2 torsional tunneling benchmark.

    Args:
        method: QC method
        basis: Basis set
        use_mock: Use mock engine
        include_tunneling: Calculate tunneling transmission

    Returns:
        BenchmarkResult with comparison to literature
    """
    import time
    from ..workflow.runner import TunnelingWorkflow
    from ..core.config import (
        WorkflowConfig,
        ComputationalConfig,
        PESScanConfig,
        TunnelingConfig,
    )

    logger.info("Running H2O2 benchmark")
    logger.info(f"Method: {method}/{basis}")

    start_time = time.time()

    # Configure workflow
    config = WorkflowConfig(
        computational=ComputationalConfig(method=method, basis=basis),
        pes_scan=PESScanConfig(
            scan_type="rigid",
            dihedral_atoms=[2, 0, 1, 3],
            start_angle=0,
            end_angle=360,
            step_size=10,
        ),
        tunneling=TunnelingConfig(
            methods=["WKB", "Eckart"],
            energy_points=50,
        ),
        calculate_zpe=False,
    )

    # Run workflow
    workflow = TunnelingWorkflow(config, use_mock_engine=use_mock)
    mol = Molecule.h2o2()
    state = workflow.run(mol)

    wall_time = time.time() - start_time

    # Extract barrier
    barrier = state.pes_result.get("barrier_height_kcal", 0) if state.pes_result else 0

    # Create result
    result = BenchmarkResult(
        system="H2O2",
        method=method,
        basis=basis,
        barrier_height_kcal=barrier,
        wall_time_seconds=wall_time,
    )

    # Compare to reference
    result.compare_to_reference(H2O2_REF)
    result.print_report(H2O2_REF)

    return result


def main():
    """Run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Run quantum tunneling benchmarks")
    parser.add_argument(
        "--system",
        choices=["malonaldehyde", "h2o2", "all"],
        default="h2o2",
        help="Benchmark system to run"
    )
    parser.add_argument(
        "--method",
        default="MP2",
        help="QC method (default: MP2)"
    )
    parser.add_argument(
        "--basis",
        default="cc-pVDZ",
        help="Basis set (default: cc-pVDZ)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Psi4 (requires installation)"
    )

    args = parser.parse_args()
    use_mock = not args.real

    if args.system == "malonaldehyde" or args.system == "all":
        run_malonaldehyde_benchmark(args.method, args.basis, use_mock)

    if args.system == "h2o2" or args.system == "all":
        run_h2o2_benchmark(args.method, args.basis, use_mock)


if __name__ == "__main__":
    main()
