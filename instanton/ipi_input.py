"""i-PI input file generator for instanton calculations.

Generates XML input files for various calculation types:
- Instanton optimization
- Ring-polymer molecular dynamics
- Geometry optimization

Reference: https://docs.ipi-code.org/
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..molecule.structure import Molecule
from ..core.constants import ANGSTROM_TO_BOHR


@dataclass
class IPIConfig:
    """Configuration for i-PI calculation."""

    # Calculation type
    mode: str = "instanton"  # "instanton", "md", "geop"

    # Ring-polymer settings
    n_beads: int = 32
    temperature: float = 300.0  # Kelvin

    # Socket settings
    socket_mode: str = "inet"  # "inet" or "unix"
    host: str = "localhost"
    port: int = 31415

    # Optimization settings
    optimizer: str = "lbfgs"  # "lbfgs", "bfgs", "sd"
    max_steps: int = 1000
    tolerance_energy: float = 1e-6  # Hartree
    tolerance_force: float = 1e-4  # Hartree/Bohr

    # Output settings
    output_prefix: str = "simulation"
    checkpoint_stride: int = 10
    properties_stride: int = 1

    # Instanton-specific
    hessian: str = "numeric"  # "numeric" or "powell"
    friction: float = 0.5

    # Advanced
    extra_options: Dict[str, Any] = field(default_factory=dict)


def generate_instanton_input(
    molecule: Molecule,
    config: Optional[IPIConfig] = None,
    output_file: Optional[str] = None,
    initial_path: Optional[np.ndarray] = None,
    xyz_file: Optional[str] = None
) -> str:
    """
    Generate i-PI input file for instanton optimization.

    Args:
        molecule: Molecule object (will be used for all beads if no path given)
        config: IPIConfig settings
        output_file: Path to write XML file (if None, returns string)
        initial_path: Optional array of bead positions (n_beads, n_atoms, 3) in Angstrom
        xyz_file: Path to write XYZ file (default: same as output with .xyz)

    Returns:
        XML string of i-PI input

    Example:
        from improved_tunnel.instanton import generate_instanton_input
        from improved_tunnel.molecule.structure import Molecule

        mol = Molecule.h2o2()
        xml = generate_instanton_input(mol, output_file="instanton.xml")
    """
    if config is None:
        config = IPIConfig()

    # Determine XYZ file path
    if xyz_file is None and output_file is not None:
        xyz_file = str(Path(output_file).with_suffix('.xyz'))
    elif xyz_file is None:
        xyz_file = "init.xyz"

    # Root element
    root = ET.Element("simulation", verbosity="high")

    # Output section
    output = ET.SubElement(root, "output", prefix=config.output_prefix)
    ET.SubElement(output, "properties", stride=str(config.properties_stride),
                 filename="out").text = " [ step, time, conserved, temperature, potential, kinetic_cv ] "
    ET.SubElement(output, "trajectory", stride=str(config.checkpoint_stride),
                 filename="pos", format="xyz").text = " positions{angstrom} "
    ET.SubElement(output, "checkpoint", stride=str(config.checkpoint_stride))

    # Total steps
    ET.SubElement(root, "total_steps").text = str(config.max_steps)

    # Force field (socket connection to external driver)
    ffsocket = ET.SubElement(root, "ffsocket", name="driver", mode=config.socket_mode)
    ET.SubElement(ffsocket, "address").text = config.host
    ET.SubElement(ffsocket, "port").text = str(config.port)

    # System section
    system = ET.SubElement(root, "system")

    # Initialize positions - reference external XYZ file
    init = ET.SubElement(system, "initialize", nbeads=str(config.n_beads))
    file_elem = ET.SubElement(init, "file", mode="xyz")
    file_elem.text = f" {xyz_file} "

    # Cell (large for isolated molecule)
    cell = ET.SubElement(init, "cell", mode="abc")
    cell.text = " [ 50.0, 50.0, 50.0 ] "

    # Forces
    forces = ET.SubElement(system, "forces")
    force = ET.SubElement(forces, "force", forcefield="driver")

    # Motion section (instanton optimization)
    motion = ET.SubElement(system, "motion", mode="instanton")
    inst = ET.SubElement(motion, "instanton", mode="rate")

    # Optimizer settings
    opt = ET.SubElement(inst, "optimizer", mode=config.optimizer)
    tolerances = ET.SubElement(opt, "tolerances")
    ET.SubElement(tolerances, "energy").text = f" {config.tolerance_energy} "
    ET.SubElement(tolerances, "force").text = f" {config.tolerance_force} "
    ET.SubElement(opt, "biggest_step").text = " 0.5 "

    # Hessian calculation
    hess = ET.SubElement(inst, "hessian", mode=config.hessian)
    ET.SubElement(hess, "friction").text = f" {config.friction} "

    # Spring constant (for ring-polymer)
    from ..core.constants import BOLTZMANN_HARTREE
    omega_n = config.n_beads * BOLTZMANN_HARTREE * config.temperature
    ET.SubElement(inst, "spring").text = f" {{ {omega_n:.6f} }} "

    # Ensemble (temperature)
    ensemble = ET.SubElement(system, "ensemble")
    ET.SubElement(ensemble, "temperature", units="kelvin").text = f" {config.temperature} "

    # Format XML
    xml_str = _prettify_xml(root)

    # Write files
    if output_file:
        Path(output_file).write_text(xml_str)
        # Also write XYZ file
        _write_xyz_file(molecule, xyz_file)

    return xml_str


def generate_geop_input(
    molecule: Molecule,
    config: Optional[IPIConfig] = None,
    output_file: Optional[str] = None,
    xyz_file: Optional[str] = None
) -> str:
    """
    Generate i-PI input file for geometry optimization.

    Args:
        molecule: Molecule to optimize
        config: IPIConfig settings
        output_file: Path to write XML file
        xyz_file: Path to write XYZ file (default: same as output with .xyz)

    Returns:
        XML string
    """
    if config is None:
        config = IPIConfig(mode="geop", n_beads=1)

    # Determine XYZ file path
    if xyz_file is None and output_file is not None:
        xyz_file = str(Path(output_file).with_suffix('.xyz'))
    elif xyz_file is None:
        xyz_file = "init.xyz"

    root = ET.Element("simulation", verbosity="medium")

    # Output
    output = ET.SubElement(root, "output", prefix=config.output_prefix)
    ET.SubElement(output, "properties", stride="1", filename="out").text = \
        " [ step, potential ] "
    ET.SubElement(output, "trajectory", stride="1", filename="pos",
                 format="xyz").text = " positions{angstrom} "

    ET.SubElement(root, "total_steps").text = str(config.max_steps)

    # Socket
    ffsocket = ET.SubElement(root, "ffsocket", name="driver", mode=config.socket_mode)
    ET.SubElement(ffsocket, "address").text = config.host
    ET.SubElement(ffsocket, "port").text = str(config.port)

    # System
    system = ET.SubElement(root, "system")

    # Initialize - reference external XYZ file
    init = ET.SubElement(system, "initialize", nbeads="1")
    file_elem = ET.SubElement(init, "file", mode="xyz")
    file_elem.text = f" {xyz_file} "

    cell = ET.SubElement(init, "cell", mode="abc")
    cell.text = " [ 50.0, 50.0, 50.0 ] "

    # Forces
    forces = ET.SubElement(system, "forces")
    ET.SubElement(forces, "force", forcefield="driver")

    # Motion (geometry optimization)
    motion = ET.SubElement(system, "motion", mode="minimize")
    opt = ET.SubElement(motion, "optimizer", mode=config.optimizer)
    tolerances = ET.SubElement(opt, "tolerances")
    ET.SubElement(tolerances, "energy").text = f" {config.tolerance_energy} "
    ET.SubElement(tolerances, "force").text = f" {config.tolerance_force} "

    xml_str = _prettify_xml(root)

    # Write files
    if output_file:
        Path(output_file).write_text(xml_str)
        # Also write XYZ file
        _write_xyz_file(molecule, xyz_file)

    return xml_str


def generate_neb_input(
    reactant: Molecule,
    product: Molecule,
    config: Optional[IPIConfig] = None,
    output_file: Optional[str] = None,
    n_images: int = 16
) -> str:
    """
    Generate i-PI input file for NEB (nudged elastic band) calculation.

    Args:
        reactant: Starting geometry
        product: Ending geometry
        config: IPIConfig settings
        output_file: Path to write XML file
        n_images: Number of images along path

    Returns:
        XML string
    """
    if config is None:
        config = IPIConfig(mode="neb", n_beads=n_images)

    root = ET.Element("simulation", verbosity="medium")

    # Output
    output = ET.SubElement(root, "output", prefix=config.output_prefix)
    ET.SubElement(output, "properties", stride="1", filename="out").text = \
        " [ step, potential ] "
    ET.SubElement(output, "trajectory", stride="10", filename="path",
                 format="xyz").text = " positions{angstrom} "

    ET.SubElement(root, "total_steps").text = str(config.max_steps)

    # Socket
    ffsocket = ET.SubElement(root, "ffsocket", name="driver", mode=config.socket_mode)
    ET.SubElement(ffsocket, "address").text = config.host
    ET.SubElement(ffsocket, "port").text = str(config.port)

    # System
    system = ET.SubElement(root, "system")

    # Initialize with interpolated path
    init = ET.SubElement(system, "initialize", nbeads=str(n_images))

    # Create linear interpolation between reactant and product
    init_path = _interpolate_path(reactant, product, n_images)
    beads = ET.SubElement(init, "beads", natoms=str(len(reactant.atoms)),
                         nbeads=str(n_images))
    ET.SubElement(beads, "q", shape=f"({n_images}, {len(reactant.atoms)}, 3)").text = \
        _array_to_string(init_path * ANGSTROM_TO_BOHR)

    cell = ET.SubElement(init, "cell", mode="abc")
    cell.text = " [ 50.0, 50.0, 50.0 ] "

    # Forces
    forces = ET.SubElement(system, "forces")
    ET.SubElement(forces, "force", forcefield="driver")

    # Motion (NEB)
    motion = ET.SubElement(system, "motion", mode="neb")
    neb = ET.SubElement(motion, "neb")
    ET.SubElement(neb, "spring").text = " 0.1 "
    ET.SubElement(neb, "climbing").text = " true "

    xml_str = _prettify_xml(root)

    if output_file:
        Path(output_file).write_text(xml_str)

    return xml_str


def generate_md_input(
    molecule: Molecule,
    config: Optional[IPIConfig] = None,
    output_file: Optional[str] = None,
    timestep: float = 0.5  # fs
) -> str:
    """
    Generate i-PI input file for molecular dynamics.

    Args:
        molecule: Starting geometry
        config: IPIConfig settings
        output_file: Path to write XML file
        timestep: MD timestep in femtoseconds

    Returns:
        XML string
    """
    if config is None:
        config = IPIConfig(mode="md", n_beads=1)

    root = ET.Element("simulation", verbosity="low")

    # Output
    output = ET.SubElement(root, "output", prefix=config.output_prefix)
    ET.SubElement(output, "properties", stride="10", filename="out").text = \
        " [ step, time, temperature, potential, kinetic_cv, conserved ] "
    ET.SubElement(output, "trajectory", stride="100", filename="traj",
                 format="xyz").text = " positions{angstrom} "
    ET.SubElement(output, "checkpoint", stride="1000")

    ET.SubElement(root, "total_steps").text = str(config.max_steps)

    # Socket
    ffsocket = ET.SubElement(root, "ffsocket", name="driver", mode=config.socket_mode)
    ET.SubElement(ffsocket, "address").text = config.host
    ET.SubElement(ffsocket, "port").text = str(config.port)

    # System
    system = ET.SubElement(root, "system")

    # Initialize
    init = ET.SubElement(system, "initialize", nbeads=str(config.n_beads))
    file_elem = ET.SubElement(init, "file", mode="xyz")
    file_elem.text = _molecule_to_xyz_string(molecule)

    # Initialize velocities from temperature
    vel = ET.SubElement(init, "velocities", mode="thermal",
                       units="kelvin")
    vel.text = f" {config.temperature} "

    cell = ET.SubElement(init, "cell", mode="abc")
    cell.text = " [ 50.0, 50.0, 50.0 ] "

    # Forces
    forces = ET.SubElement(system, "forces")
    ET.SubElement(forces, "force", forcefield="driver")

    # Motion (dynamics)
    motion = ET.SubElement(system, "motion", mode="dynamics")
    dyn = ET.SubElement(motion, "dynamics", mode="nvt")
    ET.SubElement(dyn, "timestep", units="femtosecond").text = f" {timestep} "

    # Thermostat
    thermo = ET.SubElement(dyn, "thermostat", mode="pile_l")
    ET.SubElement(thermo, "tau", units="femtosecond").text = " 100.0 "

    # Ensemble
    ensemble = ET.SubElement(system, "ensemble")
    ET.SubElement(ensemble, "temperature", units="kelvin").text = f" {config.temperature} "

    xml_str = _prettify_xml(root)

    if output_file:
        Path(output_file).write_text(xml_str)

    return xml_str


def create_driver_script(
    molecule: Molecule,
    method: str = "HF",
    basis: str = "cc-pVDZ",
    port: int = 31415,
    output_file: Optional[str] = None
) -> str:
    """
    Create a Python script to run the Psi4 driver.

    Args:
        molecule: Molecule object
        method: QC method
        basis: Basis set
        port: Socket port
        output_file: Path to write script

    Returns:
        Script string
    """
    symbols = [atom.symbol for atom in molecule.atoms]

    script = f'''#!/usr/bin/env python3
"""i-PI driver script for Psi4 calculations.

Run this script after starting the i-PI server:
    i-pi input.xml &
    python driver.py
"""

from improved_tunnel.instanton import Psi4IPIDriver

# Atom symbols for the molecule
symbols = {symbols!r}

# Create and run driver
driver = Psi4IPIDriver(
    host="localhost",
    port={port},
    method="{method}",
    basis="{basis}",
    charge={molecule.charge},
    multiplicity={molecule.multiplicity},
    use_mock=False  # Set True for testing without Psi4
)

print(f"Starting driver: {{driver.config.method}}/{{driver.config.basis}}")
print(f"Connecting to port {{driver.client.port}}...")

driver.run(symbols=symbols)

print("Driver finished.")
'''

    if output_file:
        Path(output_file).write_text(script)
        Path(output_file).chmod(0o755)

    return script


def _molecule_to_xyz_string(molecule: Molecule) -> str:
    """Convert molecule to XYZ format string (for embedding in XML)."""
    lines = [str(len(molecule.atoms)), molecule.name or "molecule"]
    for atom in molecule.atoms:
        x, y, z = atom.coordinates
        lines.append(f"{atom.symbol} {x:.10f} {y:.10f} {z:.10f}")
    return "\n" + "\n".join(lines) + "\n"


def _write_xyz_file(molecule: Molecule, filepath: str) -> None:
    """Write molecule to XYZ file."""
    lines = [str(len(molecule.atoms)), molecule.name or "molecule"]
    for atom in molecule.atoms:
        x, y, z = atom.coordinates
        lines.append(f"{atom.symbol} {x:.10f} {y:.10f} {z:.10f}")
    Path(filepath).write_text("\n".join(lines) + "\n")


def _interpolate_path(
    reactant: Molecule,
    product: Molecule,
    n_images: int
) -> np.ndarray:
    """Create linear interpolation between two geometries."""
    coords_r = reactant.coordinates
    coords_p = product.coordinates

    path = np.zeros((n_images, len(reactant.atoms), 3))

    for i in range(n_images):
        t = i / (n_images - 1)
        path[i] = (1 - t) * coords_r + t * coords_p

    return path


def _array_to_string(arr: np.ndarray) -> str:
    """Convert numpy array to string for XML."""
    flat = arr.flatten()
    return " [ " + ", ".join(f"{x:.10f}" for x in flat) + " ] "


def _prettify_xml(elem: ET.Element) -> str:
    """Return pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
