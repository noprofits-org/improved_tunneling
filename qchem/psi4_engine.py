"""Psi4 implementation of QChemEngine."""

import numpy as np
from typing import Dict, Any, Optional
import logging

from .base import QChemEngine, QChemResult, FrequencyResult
from ..molecule.structure import Molecule, Atom
from ..core.config import ComputationalConfig
from ..core.exceptions import QChemError, ConvergenceError

logger = logging.getLogger(__name__)


class Psi4Engine(QChemEngine):
    """
    Psi4 quantum chemistry engine.

    Provides interface to Psi4 for:
    - Single-point energy calculations (HF, DFT, MP2, CC methods)
    - Geometry optimization with constraints
    - Frequency calculations for ZPE
    """

    def __init__(self):
        super().__init__()
        self._psi4 = None

    def initialize(self, config: ComputationalConfig) -> None:
        """Initialize Psi4 with the given configuration."""
        try:
            import psi4
            self._psi4 = psi4
        except ImportError:
            raise QChemError(
                "Psi4 is not installed. Install with: conda install psi4 -c psi4",
                error_type="import"
            )

        self._config = config

        # Configure Psi4
        psi4.set_memory(config.memory)
        psi4.set_num_threads(config.num_threads)

        # Set output to null for cleaner operation
        psi4.core.set_output_file("/dev/null", False)

        # Set default options
        psi4.set_options({
            'basis': config.basis,
            'reference': config.reference,
            'scf_type': config.scf_type,
            'e_convergence': 1e-8,
            'd_convergence': 1e-8,
        })

        self._initialized = True
        logger.info(f"Psi4 initialized: {config.method}/{config.basis}")

    def single_point_energy(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> QChemResult:
        """Calculate single-point energy using Psi4."""
        self._ensure_initialized()

        method = method or self._config.method
        basis = basis or self._config.basis

        # Create Psi4 geometry
        geom_str = molecule.to_psi4_geometry()
        psi4_mol = self._psi4.geometry(geom_str)

        # Set basis if different from config
        if basis != self._config.basis:
            self._psi4.set_options({'basis': basis})

        try:
            energy = self._psi4.energy(method, molecule=psi4_mol)
        except Exception as e:
            raise QChemError(
                f"Single-point calculation failed: {e}",
                method=method,
                basis=basis,
                error_type="energy"
            )

        # Restore original basis
        if basis != self._config.basis:
            self._psi4.set_options({'basis': self._config.basis})

        result_mol = molecule.copy()
        result_mol.energy = energy

        return QChemResult(
            energy=energy,
            molecule=result_mol,
            converged=True,
            method=method,
            basis=basis
        )

    def optimize_geometry(
        self,
        molecule: Molecule,
        constraints: Optional[Dict[str, Any]] = None
    ) -> QChemResult:
        """
        Optimize geometry with optional constraints.

        Supports constraints:
        - "frozen_dihedral": {"atoms": [i,j,k,l], "value": angle_degrees}
        - "frozen_distance": {"atoms": [i,j], "value": distance_angstrom}
        - "frozen_angle": {"atoms": [i,j,k], "value": angle_degrees}
        """
        self._ensure_initialized()

        method = self._config.method
        basis = self._config.basis

        # Create Psi4 geometry
        geom_str = molecule.to_psi4_geometry()
        psi4_mol = self._psi4.geometry(geom_str)

        # Handle constraints
        if constraints:
            self._set_constraints(constraints)

        try:
            energy = self._psi4.optimize(method, molecule=psi4_mol)
        except Exception as e:
            # Clear constraints before raising
            if constraints:
                self._clear_constraints()
            raise ConvergenceError(
                f"Geometry optimization failed: {e}",
            )

        # Clear constraints after optimization
        if constraints:
            self._clear_constraints()

        # Extract optimized coordinates
        opt_coords = np.array(psi4_mol.geometry().np) * 0.529177210903  # Bohr to Angstrom

        # Create result molecule
        result_mol = molecule.copy()
        result_mol.coordinates = opt_coords
        result_mol.energy = energy

        return QChemResult(
            energy=energy,
            molecule=result_mol,
            converged=True,
            method=method,
            basis=basis
        )

    def frequencies(self, molecule: Molecule) -> FrequencyResult:
        """Calculate harmonic frequencies and ZPE."""
        self._ensure_initialized()

        method = self._config.method
        geom_str = molecule.to_psi4_geometry()
        psi4_mol = self._psi4.geometry(geom_str)

        try:
            energy, wfn = self._psi4.frequency(
                method,
                molecule=psi4_mol,
                return_wfn=True
            )
        except Exception as e:
            raise QChemError(
                f"Frequency calculation failed: {e}",
                method=method,
                error_type="frequency"
            )

        # Extract frequencies
        freq_array = np.array(wfn.frequencies().np).flatten()

        # Calculate ZPE (sum of 0.5 * hv for real frequencies)
        # Frequencies are in cm^-1, convert to Hartree
        real_freqs = freq_array[freq_array > 0]
        zpe_cm = 0.5 * np.sum(real_freqs)
        zpe_hartree = zpe_cm / 219474.6313632  # cm^-1 to Hartree

        result_mol = molecule.copy()
        result_mol.energy = energy

        return FrequencyResult(
            frequencies=freq_array,
            zpe=zpe_hartree,
            molecule=result_mol,
            intensities=None,  # Could extract from wfn if needed
            normal_modes=None  # Could extract from wfn if needed
        )

    def _set_constraints(self, constraints: Dict[str, Any]) -> None:
        """
        Set geometry constraints in Psi4 using optking.

        Psi4/optking uses:
        - frozen_*: freeze at current geometry value
        - fixed_*: constrain to a specified value

        For PES scans, we use fixed_* with target values.
        """
        options = {'geom_maxiter': 100}

        # Dihedral constraint - use fixed_dihedral with value
        if "frozen_dihedral" in constraints:
            c = constraints["frozen_dihedral"]
            atoms = c["atoms"]
            value = c.get("value")
            # Psi4 uses 1-indexed atoms
            # Format: "atom1 atom2 atom3 atom4 value"
            if value is not None:
                # Use fixed_dihedral to constrain to specific value
                fixed_str = f"{atoms[0]+1} {atoms[1]+1} {atoms[2]+1} {atoms[3]+1} {value}"
                options['optking__fixed_dihedral'] = fixed_str
            else:
                # No value specified - freeze at current geometry
                frozen_str = f"{atoms[0]+1} {atoms[1]+1} {atoms[2]+1} {atoms[3]+1}"
                options['optking__frozen_dihedral'] = frozen_str

        # Distance constraint
        if "frozen_distance" in constraints:
            c = constraints["frozen_distance"]
            atoms = c["atoms"]
            value = c.get("value")
            if value is not None:
                fixed_str = f"{atoms[0]+1} {atoms[1]+1} {value}"
                options['optking__fixed_distance'] = fixed_str
            else:
                frozen_str = f"{atoms[0]+1} {atoms[1]+1}"
                options['optking__frozen_distance'] = frozen_str

        # Angle constraint
        if "frozen_angle" in constraints:
            c = constraints["frozen_angle"]
            atoms = c["atoms"]
            value = c.get("value")
            if value is not None:
                fixed_str = f"{atoms[0]+1} {atoms[1]+1} {atoms[2]+1} {value}"
                options['optking__fixed_bend'] = fixed_str
            else:
                frozen_str = f"{atoms[0]+1} {atoms[1]+1} {atoms[2]+1}"
                options['optking__frozen_bend'] = frozen_str

        self._psi4.set_options(options)

    def _clear_constraints(self) -> None:
        """Clear all geometry constraints."""
        self._psi4.set_options({
            'optking__frozen_dihedral': "",
            'optking__frozen_distance': "",
            'optking__frozen_bend': "",
            'optking__fixed_dihedral': "",
            'optking__fixed_distance': "",
            'optking__fixed_bend': "",
        })

    def cleanup(self) -> None:
        """Clean up Psi4 resources."""
        if self._psi4:
            self._psi4.core.clean()


class MockPsi4Engine(QChemEngine):
    """
    Mock Psi4 engine for testing without Psi4 installed.

    Returns plausible values for H2O2 calculations.
    """

    def initialize(self, config: ComputationalConfig) -> None:
        self._config = config
        self._initialized = True
        logger.info("MockPsi4Engine initialized (for testing)")

    def single_point_energy(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> QChemResult:
        """Return mock energy based on molecule."""
        self._ensure_initialized()

        # Simple mock: base energy plus small variation
        base_energy = -151.0  # Approximate H2O2 energy
        noise = np.random.normal(0, 0.001)

        result_mol = molecule.copy()
        result_mol.energy = base_energy + noise

        return QChemResult(
            energy=base_energy + noise,
            molecule=result_mol,
            converged=True,
            method=method or self._config.method,
            basis=basis or self._config.basis
        )

    def optimize_geometry(
        self,
        molecule: Molecule,
        constraints: Optional[Dict[str, Any]] = None
    ) -> QChemResult:
        """Return slightly modified geometry."""
        self._ensure_initialized()

        result_mol = molecule.copy()
        # Add small random perturbation
        result_mol.coordinates = molecule.coordinates + np.random.normal(0, 0.01, molecule.coordinates.shape)
        result_mol.energy = -151.36 + np.random.normal(0, 0.001)

        return QChemResult(
            energy=result_mol.energy,
            molecule=result_mol,
            converged=True,
            method=self._config.method,
            basis=self._config.basis
        )

    def frequencies(self, molecule: Molecule) -> FrequencyResult:
        """Return mock frequencies for H2O2."""
        self._ensure_initialized()

        # Approximate H2O2 frequencies
        mock_freqs = np.array([
            3800.0, 3700.0,  # O-H stretches
            1500.0, 1400.0,  # HOO bends
            900.0,   # O-O stretch
            400.0    # torsion
        ])

        zpe = 0.5 * np.sum(mock_freqs) / 219474.6313632

        return FrequencyResult(
            frequencies=mock_freqs,
            zpe=zpe,
            molecule=molecule.copy()
        )
