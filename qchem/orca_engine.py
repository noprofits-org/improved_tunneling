"""ORCA implementation of QChemEngine.

ORCA is a popular quantum chemistry package, free for academic use,
with excellent performance for coupled-cluster methods.

Reference: https://www.orcasoftware.de/
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import tempfile
import shutil
import logging
import re
import os

from .base import QChemEngine, QChemResult, FrequencyResult
from ..molecule.structure import Molecule, Atom
from ..core.config import ComputationalConfig
from ..core.exceptions import QChemError, ConvergenceError
from ..core.constants import BOHR_TO_ANGSTROM

logger = logging.getLogger(__name__)


class ORCAEngine(QChemEngine):
    """
    ORCA quantum chemistry engine.

    ORCA runs as an external program, communicating via input/output files.

    Supports:
    - Single-point energy: HF, DFT, MP2, CCSD, CCSD(T)
    - Geometry optimization with constraints
    - Frequency calculations

    Example:
        engine = ORCAEngine(orca_path="/path/to/orca")
        engine.initialize(config)
        result = engine.single_point_energy(molecule)
    """

    def __init__(
        self,
        orca_path: Optional[str] = None,
        scratch_dir: Optional[str] = None,
        keep_files: bool = False
    ):
        """
        Initialize ORCA engine.

        Args:
            orca_path: Path to ORCA executable. If None, searches PATH.
            scratch_dir: Directory for temporary files. If None, uses system temp.
            keep_files: If True, don't delete input/output files after calculation.
        """
        super().__init__()
        self.orca_path = orca_path or self._find_orca()
        self.scratch_dir = scratch_dir
        self.keep_files = keep_files
        self._work_dir = None

    def _find_orca(self) -> str:
        """Find ORCA executable in PATH."""
        orca_cmd = shutil.which("orca")
        if orca_cmd:
            return orca_cmd

        # Check common locations
        common_paths = [
            "/opt/orca/orca",
            "/usr/local/orca/orca",
            os.path.expanduser("~/orca/orca"),
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        # Return "orca" and let it fail later with a clear message
        return "orca"

    def initialize(self, config: ComputationalConfig) -> None:
        """Initialize ORCA with the given configuration."""
        # Verify ORCA is available
        if not self._check_orca():
            raise QChemError(
                f"ORCA not found at '{self.orca_path}'. "
                "Download from https://www.orcasoftware.de/",
                error_type="import"
            )

        self._config = config
        self._initialized = True
        logger.info(f"ORCA initialized: {config.method}/{config.basis}")

    def _check_orca(self) -> bool:
        """Check if ORCA is available and executable."""
        try:
            result = subprocess.run(
                [self.orca_path, "--version"],
                capture_output=True,
                timeout=10
            )
            # ORCA exits with 0 even for --version in some versions
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return False

    def single_point_energy(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> QChemResult:
        """Calculate single-point energy using ORCA."""
        self._ensure_initialized()

        method = method or self._config.method
        basis = basis or self._config.basis

        # Create input file
        inp_content = self._create_input(
            molecule=molecule,
            method=method,
            basis=basis,
            job_type="SP"
        )

        try:
            # Run ORCA
            energy, converged = self._run_orca(inp_content, "energy")

            if not converged:
                raise ConvergenceError("ORCA single-point calculation did not converge")

            result_mol = molecule.copy()
            result_mol.energy = energy

            return QChemResult(
                energy=energy,
                molecule=result_mol,
                converged=converged,
                method=method,
                basis=basis
            )
        finally:
            self._cleanup_work_dir()

    def optimize_geometry(
        self,
        molecule: Molecule,
        constraints: Optional[Dict[str, Any]] = None
    ) -> QChemResult:
        """Optimize geometry with optional constraints."""
        self._ensure_initialized()

        method = self._config.method
        basis = self._config.basis

        # Create input file
        inp_content = self._create_input(
            molecule=molecule,
            method=method,
            basis=basis,
            job_type="OPT",
            constraints=constraints
        )

        try:
            # Run ORCA
            energy, converged = self._run_orca(inp_content, "opt")

            if not converged:
                raise ConvergenceError("ORCA geometry optimization did not converge")

            # Read optimized geometry (before cleanup!)
            opt_coords = self._read_optimized_geometry()

            result_mol = molecule.copy()
            result_mol.coordinates = opt_coords
            result_mol.energy = energy

            return QChemResult(
                energy=energy,
                molecule=result_mol,
                converged=converged,
                method=method,
                basis=basis
            )
        finally:
            self._cleanup_work_dir()

    def frequencies(self, molecule: Molecule) -> FrequencyResult:
        """Calculate harmonic frequencies and ZPE."""
        self._ensure_initialized()

        method = self._config.method
        basis = self._config.basis

        # Create input file
        inp_content = self._create_input(
            molecule=molecule,
            method=method,
            basis=basis,
            job_type="FREQ"
        )

        try:
            # Run ORCA
            energy, converged = self._run_orca(inp_content, "freq")

            if not converged:
                raise ConvergenceError("ORCA frequency calculation did not converge")

            # Read frequencies from output (before cleanup!)
            freqs = self._read_frequencies()

            # Calculate ZPE
            real_freqs = freqs[freqs > 0]
            zpe_cm = 0.5 * np.sum(real_freqs)
            zpe_hartree = zpe_cm / 219474.6313632

            result_mol = molecule.copy()
            result_mol.energy = energy

            return FrequencyResult(
                frequencies=freqs,
                zpe=zpe_hartree,
                molecule=result_mol
            )
        finally:
            self._cleanup_work_dir()

    def gradient(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> tuple:
        """
        Calculate energy and gradient.

        Args:
            molecule: Molecular structure
            method: QC method
            basis: Basis set

        Returns:
            Tuple of (energy in Hartree, gradient in Hartree/Angstrom)
        """
        self._ensure_initialized()

        method = method or self._config.method
        basis = basis or self._config.basis

        # Create input file with EnGrad
        inp_content = self._create_input(
            molecule=molecule,
            method=method,
            basis=basis,
            job_type="ENGRAD"
        )

        try:
            # Run ORCA
            energy, converged = self._run_orca(inp_content, "engrad")

            if not converged:
                raise ConvergenceError("ORCA gradient calculation did not converge")

            # Read gradient from .engrad file (before cleanup!)
            gradient = self._read_gradient(len(molecule.atoms))

            return energy, gradient
        finally:
            self._cleanup_work_dir()

    def _create_input(
        self,
        molecule: Molecule,
        method: str,
        basis: str,
        job_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create ORCA input file content."""
        lines = []

        # Main input line
        method_str = self._translate_method(method)
        job_keywords = self._get_job_keywords(job_type)

        lines.append(f"! {method_str} {basis} {job_keywords}")

        # Memory and parallelization
        nprocs = getattr(self._config, 'num_threads', 1)
        if nprocs > 1:
            lines.append(f"%pal nprocs {nprocs} end")

        # ORCA's %maxcore is memory PER CORE, not total
        total_memory_mb = self._parse_memory(getattr(self._config, 'memory', '2 GB'))
        memory_per_core = max(500, total_memory_mb // max(1, nprocs))
        lines.append(f"%maxcore {memory_per_core}")

        # Geometry constraints
        if constraints and job_type == "OPT":
            lines.append("%geom")
            lines.append("  Constraints")
            if "frozen_dihedral" in constraints:
                c = constraints["frozen_dihedral"]
                atoms = c["atoms"]
                # ORCA uses 0-indexed atoms
                lines.append(f"    {{ D {atoms[0]} {atoms[1]} {atoms[2]} {atoms[3]} C }}")
            if "frozen_distance" in constraints:
                c = constraints["frozen_distance"]
                atoms = c["atoms"]
                lines.append(f"    {{ B {atoms[0]} {atoms[1]} C }}")
            if "frozen_angle" in constraints:
                c = constraints["frozen_angle"]
                atoms = c["atoms"]
                lines.append(f"    {{ A {atoms[0]} {atoms[1]} {atoms[2]} C }}")
            lines.append("  end")
            lines.append("end")

        # Coordinates
        lines.append(f"* xyz {molecule.charge} {molecule.multiplicity}")
        for atom in molecule.atoms:
            x, y, z = atom.coordinates
            lines.append(f"  {atom.symbol}  {x:15.10f}  {y:15.10f}  {z:15.10f}")
        lines.append("*")

        return "\n".join(lines)

    def _translate_method(self, method: str) -> str:
        """Translate method name to ORCA format."""
        method_upper = method.upper()

        translations = {
            "HF": "HF",
            "MP2": "RI-MP2",
            "CCSD": "CCSD",
            "CCSD(T)": "CCSD(T)",
            "B3LYP": "B3LYP",
            "PBE": "PBE",
            "PBE0": "PBE0",
            "TPSS": "TPSS",
            "M06-2X": "M062X",
            "WB97X-D3": "wB97X-D3",
        }

        return translations.get(method_upper, method_upper)

    def _get_job_keywords(self, job_type: str) -> str:
        """Get ORCA keywords for job type."""
        keywords = {
            "SP": "TightSCF",
            "OPT": "Opt TightSCF",
            "FREQ": "Freq TightSCF",
            "ENGRAD": "EnGrad TightSCF",
        }
        return keywords.get(job_type.upper(), "TightSCF")

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to MB per core."""
        memory_str = memory_str.upper().strip()
        if "GB" in memory_str:
            return int(float(memory_str.replace("GB", "").strip()) * 1000)
        elif "MB" in memory_str:
            return int(float(memory_str.replace("MB", "").strip()))
        else:
            return 2000  # Default 2GB

    def _run_orca(self, inp_content: str, job_name: str) -> tuple:
        """
        Run ORCA calculation.

        Note: Does NOT clean up work directory - caller must call
        _cleanup_work_dir() after reading any output files.

        Returns:
            Tuple of (energy, converged)
        """
        # Create working directory
        if self.scratch_dir:
            self._work_dir = Path(self.scratch_dir) / f"orca_{job_name}_{os.getpid()}"
            self._work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._work_dir = Path(tempfile.mkdtemp(prefix=f"orca_{job_name}_"))

        inp_file = self._work_dir / "input.inp"
        # ORCA writes output to <basename>.out automatically
        out_file = self._work_dir / "input.out"

        # Write input file
        inp_file.write_text(inp_content)

        # Run ORCA - it writes its own output file
        logger.debug(f"Running ORCA: {self.orca_path} {inp_file}")
        result = subprocess.run(
            [self.orca_path, str(inp_file)],
            cwd=str(self._work_dir),
            capture_output=True,
            text=True,
            timeout=self._config.timeout if hasattr(self._config, 'timeout') else 3600
        )

        # Check for errors in return code or stderr
        if result.returncode != 0:
            logger.error(f"ORCA stderr: {result.stderr}")
            self._cleanup_work_dir()
            raise QChemError(
                f"ORCA calculation failed with return code {result.returncode}",
                error_type="calculation"
            )

        # Read output from the file ORCA wrote (not from stdout)
        # ORCA writes to <input_basename>.out
        if out_file.exists():
            output = out_file.read_text()
        else:
            # Fallback to stdout if .out file doesn't exist
            logger.warning("ORCA .out file not found, using stdout")
            output = result.stdout

        # Parse output
        energy, converged = self._parse_output(output)

        return energy, converged

    def _cleanup_work_dir(self) -> None:
        """Clean up temporary work directory if not keeping files."""
        if not self.keep_files and self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None

    def _parse_output(self, output: str) -> tuple:
        """Parse ORCA output for energy and convergence."""
        energy = None
        converged = False

        # Look for final energy
        energy_patterns = [
            r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)",
            r"Total Energy\s+:\s+(-?\d+\.\d+)",
        ]

        for pattern in energy_patterns:
            match = re.search(pattern, output)
            if match:
                energy = float(match.group(1))
                break

        # Check convergence
        if "ORCA TERMINATED NORMALLY" in output:
            converged = True
        elif "THE OPTIMIZATION HAS CONVERGED" in output:
            converged = True
        elif energy is not None:
            converged = True

        return energy, converged

    def _read_optimized_geometry(self) -> np.ndarray:
        """Read optimized geometry from ORCA output."""
        xyz_file = self._work_dir / "input.xyz"

        if xyz_file.exists():
            # Read from final XYZ file
            lines = xyz_file.read_text().strip().split("\n")
            n_atoms = int(lines[0])
            coords = []
            for line in lines[2:2+n_atoms]:
                parts = line.split()
                coords.append([float(x) for x in parts[1:4]])
            return np.array(coords)

        # Fall back to parsing output
        raise QChemError("Could not read optimized geometry", error_type="parse")

    def _read_frequencies(self) -> np.ndarray:
        """Read frequencies from ORCA output."""
        out_file = self._work_dir / "input.out"
        output = out_file.read_text()

        freqs = []
        in_freq_section = False

        for line in output.split("\n"):
            if "VIBRATIONAL FREQUENCIES" in line:
                in_freq_section = True
                continue

            if in_freq_section:
                if "NORMAL MODES" in line or line.strip() == "":
                    if freqs:  # Stop after getting frequencies
                        break
                    continue

                # Parse frequency line: "   0:         0.00 cm**-1"
                match = re.match(r"\s*\d+:\s+(-?\d+\.?\d*)\s+cm", line)
                if match:
                    freq = float(match.group(1))
                    freqs.append(freq)

        return np.array(freqs)

    def _read_gradient(self, n_atoms: int) -> np.ndarray:
        """Read gradient from ORCA .engrad file."""
        engrad_file = self._work_dir / "input.engrad"

        if not engrad_file.exists():
            raise QChemError("Could not find .engrad file", error_type="parse")

        lines = engrad_file.read_text().strip().split("\n")

        # Find gradient section (after "# The current gradient in Eh/bohr")
        gradient = []
        in_gradient = False

        for line in lines:
            if "gradient" in line.lower():
                in_gradient = True
                continue
            if in_gradient:
                if line.startswith("#"):
                    if gradient:  # End of gradient section
                        break
                    continue
                try:
                    gradient.append(float(line.strip()))
                except ValueError:
                    continue

        # Reshape to (n_atoms, 3) and convert from Hartree/Bohr to Hartree/Angstrom
        gradient = np.array(gradient[:n_atoms * 3]).reshape(n_atoms, 3)
        gradient_angstrom = gradient / BOHR_TO_ANGSTROM  # Convert

        return gradient_angstrom

    def cleanup(self) -> None:
        """Clean up any remaining temporary files."""
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)


class MockORCAEngine(QChemEngine):
    """
    Mock ORCA engine for testing without ORCA installed.

    Provides the same interface as ORCAEngine but returns mock values.
    """

    def initialize(self, config: ComputationalConfig) -> None:
        self._config = config
        self._initialized = True
        logger.info("MockORCAEngine initialized (for testing)")

    def single_point_energy(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> QChemResult:
        """Return mock energy."""
        self._ensure_initialized()

        base_energy = -151.0 + np.random.normal(0, 0.001)
        result_mol = molecule.copy()
        result_mol.energy = base_energy

        return QChemResult(
            energy=base_energy,
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
        """Return mock frequencies."""
        self._ensure_initialized()

        mock_freqs = np.array([3800.0, 3700.0, 1500.0, 1400.0, 900.0, 400.0])
        zpe = 0.5 * np.sum(mock_freqs) / 219474.6313632

        return FrequencyResult(
            frequencies=mock_freqs,
            zpe=zpe,
            molecule=molecule.copy()
        )

    def gradient(
        self,
        molecule: Molecule,
        method: Optional[str] = None,
        basis: Optional[str] = None
    ) -> tuple:
        """Return mock gradient."""
        self._ensure_initialized()

        n_atoms = len(molecule.atoms)
        energy = -151.0 + np.random.normal(0, 0.001)
        gradient = np.random.normal(0, 0.01, (n_atoms, 3))

        return energy, gradient
