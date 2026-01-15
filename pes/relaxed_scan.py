"""Relaxed PES scan implementation with constrained optimization."""

import numpy as np
import logging
from typing import Optional

from .scan import PESScan, PESScanPoint, PESScanResult
from ..molecule.structure import Molecule
from ..molecule.geometry import set_dihedral_fragment, calculate_dihedral
from ..core.config import PESScanConfig
from ..core.exceptions import ConvergenceError

logger = logging.getLogger(__name__)


class RelaxedPESScan(PESScan):
    """
    Relaxed PES scan with constrained optimization.

    At each scan point, the dihedral angle is frozen and all other
    coordinates are optimized. This gives the minimum energy path
    along the torsional coordinate.
    """

    @property
    def scan_type(self) -> str:
        return "relaxed"

    def run(self, molecule: Molecule) -> PESScanResult:
        """
        Execute relaxed dihedral scan.

        For each target angle:
        1. Set initial dihedral to target value
        2. Optimize geometry with frozen dihedral constraint
        3. Store optimized geometry and energy

        Args:
            molecule: Starting molecular structure (should be pre-optimized)

        Returns:
            PESScanResult with all scan points
        """
        config = self.config
        i, j, k, l = config.dihedral_atoms

        angles = config.angles
        n_points = len(angles)
        points = []

        logger.info(f"Starting relaxed PES scan: {n_points} points")
        logger.info(f"Dihedral atoms: {config.dihedral_atoms}")
        logger.info(f"Angle range: {config.start_angle}° to {config.end_angle}°")

        self._min_energy = float('inf')

        # Use previous geometry as starting point for next optimization
        current_mol = molecule.copy()

        for idx, angle in enumerate(angles):
            # Set dihedral to target angle as starting geometry
            mol_at_angle = set_dihedral_fragment(current_mol, i, j, k, l, angle)

            # Define constraint: freeze the dihedral at target value
            constraints = {
                "frozen_dihedral": {
                    "atoms": [i, j, k, l],
                    "value": angle
                }
            }

            try:
                # Optimize with frozen dihedral
                result = self.engine.optimize_geometry(mol_at_angle, constraints)
                converged = result.converged

                # Verify the dihedral is close to target
                actual_dihedral = calculate_dihedral(result.molecule, i, j, k, l)
                if abs(actual_dihedral - angle) > 5.0:  # Allow some tolerance
                    # Dihedral drifted, use single-point instead
                    logger.warning(
                        f"Dihedral drifted to {actual_dihedral:.1f}° "
                        f"(target: {angle:.1f}°). Using rigid point."
                    )
                    result = self.engine.single_point_energy(mol_at_angle)

            except ConvergenceError as e:
                logger.warning(f"Optimization failed at {angle}°: {e}. Using rigid point.")
                result = self.engine.single_point_energy(mol_at_angle)
                converged = False

            # Track minimum for logging
            if result.energy < self._min_energy:
                self._min_energy = result.energy

            point = PESScanPoint(
                angle=angle,
                energy=result.energy,
                molecule=result.molecule,
                converged=converged
            )
            points.append(point)

            # Use optimized geometry as starting point for next angle
            current_mol = result.molecule

            self._log_progress(idx + 1, n_points, angle, result.energy)

        # Get method/basis info from engine config
        method = self.engine._config.method if self.engine._config else ""
        basis = self.engine._config.basis if self.engine._config else ""

        self._results = PESScanResult(
            scan_type=self.scan_type,
            dihedral_atoms=config.dihedral_atoms,
            points=points,
            method=method,
            basis=basis
        )

        logger.info(f"Relaxed scan complete. Barrier height: {self._results.barrier_height_kcal:.2f} kcal/mol")

        return self._results
