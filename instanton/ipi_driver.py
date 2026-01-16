"""Psi4 driver for i-PI calculations.

This module provides a driver that connects Psi4 to i-PI for
ring-polymer instanton calculations.

Usage:
    driver = Psi4IPIDriver(
        host="localhost",
        port=31415,
        method="MP2",
        basis="cc-pVDZ"
    )
    driver.run(symbols=["H", "O", "O", "H"])
"""

import logging
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass

from .ipi_client import (
    IPISocketClient,
    IPIGeometry,
    IPIForces,
    geometry_to_angstrom,
    forces_to_atomic_units,
)
from ..core.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, HARTREE_TO_KCAL
from ..molecule.structure import Molecule, Atom

logger = logging.getLogger(__name__)


@dataclass
class DriverConfig:
    """Configuration for i-PI driver."""

    method: str = "HF"
    basis: str = "cc-pVDZ"
    charge: int = 0
    multiplicity: int = 1
    memory: str = "2 GB"
    nthreads: int = 1

    # Gradient options
    dertype: str = "gradient"  # "gradient" or "energy" (numerical)

    # Convergence
    e_convergence: float = 1e-8
    d_convergence: float = 1e-8


class Psi4IPIDriver:
    """
    Psi4 driver for i-PI server.

    Acts as a client that receives geometries from i-PI,
    computes energies and gradients with Psi4, and returns forces.

    Example:
        driver = Psi4IPIDriver(
            host="localhost",
            port=31415,
            method="MP2",
            basis="cc-pVDZ"
        )

        # Run driver loop (blocking)
        driver.run(symbols=["H", "O", "O", "H"])
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 31415,
        unix_socket: Optional[str] = None,
        method: str = "HF",
        basis: str = "cc-pVDZ",
        charge: int = 0,
        multiplicity: int = 1,
        timeout: float = 600.0,
        use_mock: bool = False
    ):
        """
        Initialize i-PI driver.

        Args:
            host: i-PI server hostname
            port: i-PI server port
            unix_socket: Unix socket path (overrides host/port)
            method: QC method (HF, MP2, CCSD, etc.)
            basis: Basis set
            charge: Molecular charge
            multiplicity: Spin multiplicity
            timeout: Socket timeout in seconds
            use_mock: Use mock Psi4 engine for testing
        """
        self.client = IPISocketClient(
            host=host,
            port=port,
            unix_socket=unix_socket,
            timeout=timeout
        )

        self.config = DriverConfig(
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity
        )

        self.use_mock = use_mock
        self._symbols: Optional[List[str]] = None
        self._psi4 = None
        self._initialized = False
        self._calculation_count = 0

    def _initialize_psi4(self) -> None:
        """Initialize Psi4 with configuration."""
        if self._initialized:
            return

        if self.use_mock:
            logger.info("Using mock Psi4 engine")
            self._initialized = True
            return

        try:
            import psi4
            self._psi4 = psi4

            # Configure Psi4
            psi4.set_memory(self.config.memory)
            psi4.set_num_threads(self.config.nthreads)
            psi4.core.set_output_file("psi4_ipi.out", False)

            # Set convergence options
            psi4.set_options({
                'e_convergence': self.config.e_convergence,
                'd_convergence': self.config.d_convergence,
            })

            self._initialized = True
            logger.info(f"Psi4 initialized: {self.config.method}/{self.config.basis}")

        except ImportError:
            raise RuntimeError(
                "Psi4 not available. Install psi4 or use use_mock=True for testing."
            )

    def _create_molecule(self, positions_angstrom: np.ndarray) -> "psi4.core.Molecule":
        """Create Psi4 molecule from positions."""
        if self._symbols is None:
            raise RuntimeError("Atom symbols not set. Call run() with symbols argument.")

        if self.use_mock:
            return None

        # Build geometry string
        geom_lines = []
        for symbol, pos in zip(self._symbols, positions_angstrom):
            geom_lines.append(f"  {symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")

        geom_str = f"""
{self.config.charge} {self.config.multiplicity}
{chr(10).join(geom_lines)}
units angstrom
no_reorient
no_com
"""

        mol = self._psi4.geometry(geom_str)
        return mol

    def _compute_gradient(
        self,
        positions_angstrom: np.ndarray
    ) -> tuple:
        """
        Compute energy and gradient with Psi4.

        Args:
            positions_angstrom: Atomic positions in Angstrom

        Returns:
            Tuple of (energy in Hartree, gradient in Hartree/Angstrom)
        """
        if self.use_mock:
            return self._mock_gradient(positions_angstrom)

        mol = self._create_molecule(positions_angstrom)

        # Compute gradient
        grad_matrix, wfn = self._psi4.gradient(
            f"{self.config.method}/{self.config.basis}",
            molecule=mol,
            return_wfn=True
        )

        energy = wfn.energy()
        gradient = np.asarray(grad_matrix)  # Shape: (n_atoms, 3)

        self._calculation_count += 1
        logger.debug(f"Calculation {self._calculation_count}: E = {energy:.10f} Ha")

        return energy, gradient

    def _mock_gradient(self, positions_angstrom: np.ndarray) -> tuple:
        """Mock gradient for testing without Psi4."""
        n_atoms = len(positions_angstrom)

        # Simple Lennard-Jones-like potential for testing
        # Gives reasonable energy and gradient for any geometry
        k = 0.1  # Force constant in Ha/Angstrom^2
        r0 = 1.5  # Equilibrium distance in Angstrom

        energy = 0.0
        gradient = np.zeros_like(positions_angstrom)

        # Pairwise interactions
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rij = positions_angstrom[j] - positions_angstrom[i]
                r = np.linalg.norm(rij)
                if r > 0.1:  # Avoid singularity
                    # Harmonic-like potential around r0
                    dr = r - r0
                    energy += 0.5 * k * dr**2

                    # Gradient contribution
                    dE_dr = k * dr
                    grad_contribution = dE_dr * rij / r
                    gradient[i] -= grad_contribution
                    gradient[j] += grad_contribution

        # Add small baseline energy to make it non-zero
        energy += -0.5

        self._calculation_count += 1
        logger.debug(f"Mock calculation {self._calculation_count}: E = {energy:.6f} Ha")

        return energy, gradient

    def run(
        self,
        symbols: List[str],
        callback: Optional[Callable[[int, float], None]] = None
    ) -> None:
        """
        Run the i-PI driver loop.

        This is the main entry point. Connects to i-PI server and
        enters the calculation loop until EXIT is received.

        Args:
            symbols: List of atomic symbols (e.g., ["H", "O", "O", "H"])
            callback: Optional callback(step, energy) called after each calculation
        """
        self._symbols = symbols
        self._initialize_psi4()

        logger.info(f"Starting i-PI driver with {len(symbols)} atoms")
        logger.info(f"Method: {self.config.method}/{self.config.basis}")

        try:
            self.client.connect()
            self._driver_loop(callback)
        finally:
            self.client.close()
            logger.info(f"Driver finished. Total calculations: {self._calculation_count}")

    def _driver_loop(
        self,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> None:
        """Main driver loop for i-PI communication.

        Implements the i-PI socket protocol:
        1. Server sends STATUS to client
        2. Client responds with READY, NEEDINIT, or HAVEDATA
        3. Based on client response, server sends appropriate data
        4. Client processes and loop continues
        """
        have_data = False
        force_data = None

        while True:
            # Wait for server's STATUS query
            msg = self.client.wait_for_status()
            logger.debug(f"Received from server: {msg}")

            if msg == "STATUS":
                # Server is asking for our status
                if have_data:
                    # We have computed forces to send
                    self.client.send_status("HAVEDATA")
                else:
                    # We're ready to receive new work
                    self.client.send_status("READY")

            elif msg == "POSDATA":
                # Server is sending geometry data
                # Read the geometry (msg already consumed POSDATA header)
                geom = self._receive_geometry_data()

                # Convert to Angstrom
                positions_angstrom, cell_angstrom = geometry_to_angstrom(geom)

                # Verify atom count
                if len(positions_angstrom) != len(self._symbols):
                    raise RuntimeError(
                        f"Atom count mismatch: got {len(positions_angstrom)}, "
                        f"expected {len(self._symbols)}"
                    )

                # Compute energy and gradient
                energy, gradient = self._compute_gradient(positions_angstrom)

                # Convert forces (negative gradient) to atomic units
                forces = -gradient  # F = -dE/dr

                # Store force data for later transmission
                force_data = forces_to_atomic_units(
                    energy_hartree=energy,
                    forces_hartree_per_angstrom=forces,
                    virial_hartree=None
                )
                have_data = True

                # Call callback if provided
                if callback:
                    callback(self._calculation_count, energy)

            elif msg == "GETFORCE":
                # Server wants our computed forces
                if force_data is None:
                    raise RuntimeError("Server requested forces but none computed")

                self.client.send_forces(force_data)
                have_data = False
                force_data = None

            elif msg == "INIT":
                # Server is sending initialization data
                # Read and discard the bead ID
                bead_id_data = self.client._recv(4)
                logger.info("Received initialization data")

            elif msg == "EXIT":
                # Server signals end of calculation
                logger.info("Received EXIT signal from server")
                break

            else:
                logger.warning(f"Unknown message: {msg}")

    def _receive_geometry_data(self) -> "IPIGeometry":
        """Receive geometry data after POSDATA header was already read."""
        from .ipi_client import IPIGeometry
        import struct

        # Cell matrix (9 doubles, column-major for Fortran compatibility)
        cell_data = self.client._recv(9 * 8)
        cell = np.frombuffer(cell_data, dtype=np.float64).reshape(3, 3).T

        # Inverse cell matrix
        cell_inv_data = self.client._recv(9 * 8)
        cell_inv = np.frombuffer(cell_inv_data, dtype=np.float64).reshape(3, 3).T

        # Number of atoms
        n_atoms_data = self.client._recv(4)
        n_atoms = struct.unpack("i", n_atoms_data)[0]

        # Positions (3*N doubles)
        pos_data = self.client._recv(3 * n_atoms * 8)
        positions = np.frombuffer(pos_data, dtype=np.float64).reshape(n_atoms, 3)

        logger.debug(f"Received geometry: {n_atoms} atoms")

        return IPIGeometry(
            cell=cell,
            cell_inv=cell_inv,
            positions=positions,
            n_atoms=n_atoms
        )


def create_driver_from_molecule(
    molecule: Molecule,
    host: str = "localhost",
    port: int = 31415,
    method: str = "HF",
    basis: str = "cc-pVDZ",
    use_mock: bool = False
) -> Psi4IPIDriver:
    """
    Create i-PI driver from a Molecule object.

    Convenience function that extracts symbols, charge, and multiplicity
    from the molecule.

    Args:
        molecule: Molecule object
        host: i-PI server hostname
        port: i-PI server port
        method: QC method
        basis: Basis set
        use_mock: Use mock engine

    Returns:
        Configured Psi4IPIDriver
    """
    return Psi4IPIDriver(
        host=host,
        port=port,
        method=method,
        basis=basis,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
        use_mock=use_mock
    )


class IPIDriverManager:
    """
    Manager for running multiple i-PI drivers in parallel.

    Useful for ring-polymer calculations where multiple beads
    need to be computed simultaneously.
    """

    def __init__(
        self,
        n_beads: int,
        base_port: int = 31415,
        host: str = "localhost",
        method: str = "HF",
        basis: str = "cc-pVDZ"
    ):
        """
        Initialize driver manager.

        Args:
            n_beads: Number of ring-polymer beads
            base_port: Starting port number (incremented for each bead)
            host: i-PI server hostname
            method: QC method
            basis: Basis set
        """
        self.n_beads = n_beads
        self.drivers = []

        for i in range(n_beads):
            driver = Psi4IPIDriver(
                host=host,
                port=base_port + i,
                method=method,
                basis=basis
            )
            self.drivers.append(driver)

    def run_parallel(self, symbols: List[str]) -> None:
        """
        Run all drivers in parallel.

        Note: This is a placeholder. Full parallel implementation
        would use multiprocessing or threading.

        Args:
            symbols: Atomic symbols
        """
        import concurrent.futures

        logger.info(f"Starting {self.n_beads} parallel drivers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_beads) as executor:
            futures = [
                executor.submit(driver.run, symbols)
                for driver in self.drivers
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures)

            # Check for exceptions
            for i, future in enumerate(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Driver {i} failed: {e}")
