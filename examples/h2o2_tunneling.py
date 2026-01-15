#!/usr/bin/env python3
"""
Example: H2O2 Torsional Tunneling Analysis

This script demonstrates the improved_tunnel package by calculating
quantum tunneling rates for hydrogen peroxide torsion.

Key features demonstrated:
- Relaxed PES scan (constrained optimization)
- Multiple tunneling methods (WKB, Eckart, SCT)
- Isotope effects (H vs D)
- Temperature-dependent rate calculations
- Arrhenius analysis
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from improved_tunnel.core.config import (
    WorkflowConfig,
    ComputationalConfig,
    PESScanConfig,
    TunnelingConfig,
    KineticsConfig,
    IsotopeConfig,
)
from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.molecule.isotopes import substitute_isotopes
from improved_tunnel.molecule.reduced_mass import calculate_torsional_reduced_mass
from improved_tunnel.workflow.runner import TunnelingWorkflow
from improved_tunnel.kinetics.rates import calculate_kie_vs_temperature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_h2o2_example(use_mock: bool = True):
    """
    Run complete H2O2 tunneling analysis.

    Args:
        use_mock: If True, use mock QChem engine (no Psi4 needed)
    """
    print("=" * 70)
    print("H2O2 Torsional Tunneling Analysis")
    print("=" * 70)

    # Configuration
    config = WorkflowConfig(
        computational=ComputationalConfig(
            method="MP2",
            basis="cc-pVTZ",
            num_threads=4,
            memory="4 GB",
        ),
        pes_scan=PESScanConfig(
            scan_type="relaxed",  # Use constrained optimization
            dihedral_atoms=[2, 0, 1, 3],  # H-O-O-H dihedral
            start_angle=0.0,
            end_angle=360.0,
            step_size=15.0,  # Coarser grid for demo
        ),
        tunneling=TunnelingConfig(
            methods=["WKB", "Eckart", "SCT"],
            energy_points=100,
            min_energy_ratio=0.3,
            max_energy_ratio=1.0,
        ),
        kinetics=KineticsConfig(
            temp_min=200.0,
            temp_max=400.0,
            temp_step=25.0,
            prefactor=1e13,
        ),
        calculate_zpe=True,
    )

    # Create H2O2 molecule
    h2o2 = Molecule.h2o2(dihedral=111.5)
    print(f"\nMolecule: {h2o2.formula}")
    print(f"Atoms: {h2o2.num_atoms}")
    print(f"Initial dihedral: 111.5°")

    # Calculate reduced mass
    dihedral_atoms = config.pes_scan.dihedral_atoms
    reduced_mass = calculate_torsional_reduced_mass(h2o2, dihedral_atoms)
    print(f"Torsional reduced mass: {reduced_mass:.4f} AMU")

    # Run workflow for H2O2
    print("\n" + "-" * 50)
    print("Running workflow for H2O2...")
    print("-" * 50)

    workflow_h = TunnelingWorkflow(config, use_mock_engine=use_mock)
    state_h = workflow_h.run(h2o2)

    # Print results
    print_results(state_h, "H2O2")

    # Now run for D2O2 (deuterated)
    print("\n" + "=" * 70)
    print("Running isotope study: D2O2")
    print("=" * 70)

    d2o2 = substitute_isotopes(h2o2, global_substitutions={"H": "D"})
    reduced_mass_d = calculate_torsional_reduced_mass(d2o2, dihedral_atoms)
    print(f"D2O2 torsional reduced mass: {reduced_mass_d:.4f} AMU")

    workflow_d = TunnelingWorkflow(config, use_mock_engine=use_mock)
    state_d = workflow_d.run(d2o2)

    print_results(state_d, "D2O2")

    # Calculate KIE
    print("\n" + "=" * 70)
    print("Kinetic Isotope Effect (KIE) Analysis")
    print("=" * 70)

    if workflow_h._rate_results and workflow_d._rate_results:
        for method in workflow_h._rate_results:
            if method in workflow_d._rate_results:
                rate_h = workflow_h._rate_results[method]
                rate_d = workflow_d._rate_results[method]

                kie_analysis = calculate_kie_vs_temperature(rate_h, rate_d)

                print(f"\n{method} method:")
                print(f"  KIE at 300 K: {kie_analysis['kie_at_300K']:.2f}")
                print(f"  Maximum KIE: {kie_analysis['max_kie']:.2f}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


def print_results(state, label: str):
    """Print workflow results."""
    print(f"\n--- Results for {label} ---")

    if state.pes_result:
        barrier = state.pes_result.get("barrier_height_kcal", 0)
        print(f"Barrier height: {barrier:.2f} kcal/mol")

    if state.tunneling_results:
        print("\nTunneling probabilities at E = 0.5 * Vb:")
        for method, result in state.tunneling_results.items():
            transmissions = result.get("transmissions", [])
            energy_ratios = result.get("energy_ratios", [])
            if transmissions and energy_ratios:
                # Find transmission at E/Vb = 0.5
                import numpy as np
                T_at_half = np.interp(0.5, energy_ratios, transmissions)
                print(f"  {method}: T = {T_at_half:.4e}")

    if state.rate_results:
        print("\nRate constants at 300 K:")
        for method, result in state.rate_results.items():
            temps = result.get("temperatures", [])
            q_rates = result.get("quantum_rates", [])
            c_rates = result.get("classical_rates", [])
            kappas = result.get("tunneling_corrections", [])
            if temps and q_rates:
                import numpy as np
                k_300 = np.interp(300, temps, q_rates)
                kappa_300 = np.interp(300, temps, kappas)
                print(f"  {method}: k = {k_300:.2e} s^-1 (κ = {kappa_300:.2f})")

    print(f"\nTiming:")
    for step, time in state.timing.items():
        print(f"  {step}: {time:.1f}s")


def demo_standalone_calculation():
    """
    Demonstrate standalone usage without full workflow.

    Useful for quick calculations or custom pipelines.
    """
    print("\n" + "=" * 70)
    print("Standalone Calculation Demo")
    print("=" * 70)

    from improved_tunnel.tunneling.eckart import EckartBarrier
    from improved_tunnel.core.constants import HARTREE_TO_KCAL, AMU_TO_KG
    import numpy as np

    # Create Eckart barrier with known parameters
    # Typical H2O2 torsional barrier: ~1 kcal/mol
    barrier_kcal = 1.0
    barrier_hartree = barrier_kcal / HARTREE_TO_KCAL

    eckart = EckartBarrier(
        V1=barrier_hartree,
        V2=barrier_hartree,
        L=0.5,  # radians
        symmetric=True
    )

    # Calculate transmission for H (mass ~ 1 AMU reduced mass for torsion)
    reduced_mass_h = 1.0 * AMU_TO_KG

    print(f"\nEckart barrier: {barrier_kcal:.2f} kcal/mol")
    print(f"Reduced mass: 1.0 AMU")
    print("\nTransmission coefficients:")

    for ratio in [0.3, 0.5, 0.7, 0.9, 1.0]:
        energy = ratio * barrier_hartree
        T = eckart.analytical_transmission(energy, reduced_mass_h)
        print(f"  E/Vb = {ratio:.1f}: T = {T:.4e}")

    # Compare with deuterium
    reduced_mass_d = 2.0 * AMU_TO_KG
    print(f"\nWith D (2.0 AMU):")
    for ratio in [0.3, 0.5, 0.7, 0.9, 1.0]:
        energy = ratio * barrier_hartree
        T = eckart.analytical_transmission(energy, reduced_mass_d)
        print(f"  E/Vb = {ratio:.1f}: T = {T:.4e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="H2O2 Tunneling Analysis")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Psi4 calculations (requires Psi4 installed)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run standalone demo only"
    )
    args = parser.parse_args()

    if args.demo:
        demo_standalone_calculation()
    else:
        run_h2o2_example(use_mock=not args.real)
