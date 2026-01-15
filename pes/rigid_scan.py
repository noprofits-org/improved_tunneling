"""Rigid PES scan implementation."""

import numpy as np
import logging
from typing import Optional

from .scan import PESScan, PESScanPoint, PESScanResult
from ..molecule.structure import Molecule
from ..molecule.geometry import set_dihedral_fragment
from ..core.config import PESScanConfig

logger = logging.getLogger(__name__)


class RigidPESScan(PESScan):
    """
    Rigid PES scan: only the dihedral angle changes.

    All other coordinates remain fixed at their initial values.
    This is faster but less accurate than relaxed scans.
    """

    @property
    def scan_type(self) -> str:
        return "rigid"

    def run(self, molecule: Molecule) -> PESScanResult:
        """
        Execute rigid dihedral scan.

        For each target angle:
        1. Set dihedral to target value
        2. Calculate single-point energy
        3. Store results

        Args:
            molecule: Starting molecular structure

        Returns:
            PESScanResult with all scan points
        """
        config = self.config
        i, j, k, l = config.dihedral_atoms

        angles = config.angles
        n_points = len(angles)
        points = []

        logger.info(f"Starting rigid PES scan: {n_points} points")
        logger.info(f"Dihedral atoms: {config.dihedral_atoms}")
        logger.info(f"Angle range: {config.start_angle}° to {config.end_angle}°")

        self._min_energy = float('inf')

        for idx, angle in enumerate(angles):
            # Set dihedral angle
            mol_at_angle = set_dihedral_fragment(molecule, i, j, k, l, angle)

            # Calculate single-point energy
            result = self.engine.single_point_energy(mol_at_angle)

            # Track minimum for logging
            if result.energy < self._min_energy:
                self._min_energy = result.energy

            point = PESScanPoint(
                angle=angle,
                energy=result.energy,
                molecule=result.molecule,
                converged=result.converged
            )
            points.append(point)

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

        logger.info(f"Rigid scan complete. Barrier height: {self._results.barrier_height_kcal:.2f} kcal/mol")

        return self._results
