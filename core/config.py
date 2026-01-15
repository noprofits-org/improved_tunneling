"""Dataclass-based configuration for all simulation parameters."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict
from pathlib import Path


@dataclass(frozen=True)
class ComputationalConfig:
    """
    Immutable configuration for quantum chemistry calculations.

    Attributes:
        method: QC method for geometry optimization and PES (e.g., "MP2", "B3LYP")
        basis: Basis set (e.g., "cc-pVTZ", "6-31G*")
        reference: Reference wavefunction type ("rhf", "uhf", "rohf")
        memory: Memory allocation (e.g., "4 GB")
        num_threads: Number of CPU threads
        scf_type: SCF algorithm type ("df", "pk", "direct")
        sp_method: Optional higher-level method for single points (e.g., "CCSD(T)")
        sp_basis: Optional basis for single-point calculations
    """
    method: str = "MP2"
    basis: str = "cc-pVTZ"
    reference: str = "rhf"
    memory: str = "4 GB"
    num_threads: int = 4
    scf_type: str = "df"
    sp_method: Optional[str] = None
    sp_basis: Optional[str] = None

    def get_single_point_method(self) -> str:
        """Return the method for single-point calculations."""
        return self.sp_method if self.sp_method else self.method

    def get_single_point_basis(self) -> str:
        """Return the basis for single-point calculations."""
        return self.sp_basis if self.sp_basis else self.basis


@dataclass
class PESScanConfig:
    """
    Configuration for potential energy surface scans.

    Attributes:
        scan_type: "rigid" (frozen geometry) or "relaxed" (constrained optimization)
        dihedral_atoms: 4 atom indices defining the torsional coordinate [i, j, k, l]
        start_angle: Starting dihedral angle in degrees
        end_angle: Ending dihedral angle in degrees
        step_size: Angle increment in degrees
        optimization_convergence: Convergence threshold for relaxed scans
    """
    scan_type: Literal["rigid", "relaxed"] = "relaxed"
    dihedral_atoms: List[int] = field(default_factory=lambda: [2, 0, 1, 3])
    start_angle: float = 0.0
    end_angle: float = 360.0
    step_size: float = 10.0
    optimization_convergence: float = 1e-6

    @property
    def num_points(self) -> int:
        """Number of points in the scan."""
        return int((self.end_angle - self.start_angle) / self.step_size) + 1

    @property
    def angles(self) -> List[float]:
        """List of scan angles."""
        return [self.start_angle + i * self.step_size
                for i in range(self.num_points)]


@dataclass
class TunnelingConfig:
    """
    Configuration for tunneling calculations.

    Attributes:
        methods: List of tunneling methods to use ("WKB", "SCT", "Eckart")
        energy_points: Number of energy levels to evaluate
        min_energy_ratio: Minimum energy as fraction of barrier height
        max_energy_ratio: Maximum energy as fraction of barrier height
        integration_tolerance: Tolerance for adaptive quadrature
        max_subdivisions: Maximum subdivisions for integration
    """
    methods: List[str] = field(default_factory=lambda: ["WKB", "Eckart"])
    energy_points: int = 200
    min_energy_ratio: float = 0.3
    max_energy_ratio: float = 1.0
    integration_tolerance: float = 1e-8
    max_subdivisions: int = 500


@dataclass
class IsotopeConfig:
    """
    Configuration for isotope substitutions.

    Attributes:
        substitutions: Mapping of atom index to isotope symbol
            Example: {0: "D", 3: "D"} to replace H atoms at indices 0 and 3 with deuterium
        global_substitutions: Mapping of element symbol to isotope
            Example: {"H": "D"} to replace all H with D
    """
    substitutions: Dict[int, str] = field(default_factory=dict)
    global_substitutions: Dict[str, str] = field(default_factory=dict)


@dataclass
class KineticsConfig:
    """
    Configuration for rate calculations.

    Attributes:
        temperatures: List of temperatures in Kelvin, or None for default range
        temp_min: Minimum temperature if using range
        temp_max: Maximum temperature if using range
        temp_step: Temperature step if using range
        prefactor: Arrhenius prefactor in s^-1
    """
    temperatures: Optional[List[float]] = None
    temp_min: float = 100.0
    temp_max: float = 500.0
    temp_step: float = 20.0
    prefactor: float = 1e13

    def get_temperatures(self) -> List[float]:
        """Return list of temperatures to evaluate."""
        if self.temperatures is not None:
            return self.temperatures
        temps = []
        t = self.temp_min
        while t <= self.temp_max:
            temps.append(t)
            t += self.temp_step
        return temps


@dataclass
class WorkflowConfig:
    """
    Master configuration combining all sub-configurations.

    Attributes:
        computational: Quantum chemistry settings
        pes_scan: PES scan settings
        tunneling: Tunneling calculation settings
        isotope: Isotope substitution settings
        kinetics: Rate calculation settings
        output_dir: Directory for output files
        checkpoint_file: Optional checkpoint file for restart
        calculate_zpe: Whether to calculate zero-point energy corrections
    """
    computational: ComputationalConfig = field(default_factory=ComputationalConfig)
    pes_scan: PESScanConfig = field(default_factory=PESScanConfig)
    tunneling: TunnelingConfig = field(default_factory=TunnelingConfig)
    isotope: IsotopeConfig = field(default_factory=IsotopeConfig)
    kinetics: KineticsConfig = field(default_factory=KineticsConfig)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    checkpoint_file: Optional[Path] = None
    calculate_zpe: bool = True

    def __post_init__(self):
        """Ensure output_dir is a Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.checkpoint_file and isinstance(self.checkpoint_file, str):
            self.checkpoint_file = Path(self.checkpoint_file)
