"""Socket client for i-PI communication.

i-PI uses a client-server architecture:
- i-PI server: Python process managing ring-polymer dynamics/optimization
- Client driver: External QC code (Psi4) computing energies and gradients

Communication via socket protocol with fixed 12-byte message headers.

Reference: https://docs.ipi-code.org/
"""

import socket
import struct
import numpy as np
import logging
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from ..core.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR

logger = logging.getLogger(__name__)

# Message header size (fixed by i-PI protocol)
HEADER_SIZE = 12

# Standard message types
MSG_STATUS = b"STATUS      "
MSG_POSDATA = b"POSDATA     "
MSG_GETFORCE = b"GETFORCE    "
MSG_FORCEREADY = b"FORCEREADY  "
MSG_INIT = b"INIT        "
MSG_EXIT = b"EXIT        "
MSG_NEEDINIT = b"NEEDINIT    "
MSG_READY = b"READY       "
MSG_HAVEDATA = b"HAVEDATA    "


@dataclass
class IPIGeometry:
    """Geometry data received from i-PI."""

    cell: np.ndarray  # 3x3 cell matrix (Bohr)
    cell_inv: np.ndarray  # 3x3 inverse cell (1/Bohr)
    positions: np.ndarray  # Nx3 atomic positions (Bohr)
    n_atoms: int


@dataclass
class IPIForces:
    """Force data to send to i-PI."""

    energy: float  # Energy (Hartree)
    forces: np.ndarray  # Nx3 forces (Hartree/Bohr)
    virial: np.ndarray  # 3x3 virial tensor (Hartree)
    extras: Optional[str] = None  # Optional JSON extras


class IPISocketClient:
    """
    Socket client for i-PI server communication.

    Handles the low-level socket protocol for exchanging
    geometries and forces with i-PI.

    Example:
        client = IPISocketClient(host="localhost", port=31415)
        client.connect()

        while True:
            status = client.poll_status()
            if status == "READY":
                geom = client.receive_geometry()
                # Calculate forces with QC code
                forces = calculate_forces(geom)
                client.send_forces(forces)
            elif status == "EXIT":
                break

        client.close()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 31415,
        unix_socket: Optional[str] = None,
        timeout: float = 600.0
    ):
        """
        Initialize i-PI socket client.

        Args:
            host: Hostname for inet socket
            port: Port for inet socket (default 31415)
            unix_socket: Path for unix domain socket (overrides host/port)
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.unix_socket = unix_socket
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to i-PI server."""
        if self.unix_socket:
            logger.info(f"Connecting to i-PI via UNIX socket: {self.unix_socket}")
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.unix_socket)
        else:
            logger.info(f"Connecting to i-PI at {self.host}:{self.port}")
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))

        self._socket.settimeout(self.timeout)
        self._connected = True
        logger.info("Connected to i-PI server")

    def close(self) -> None:
        """Close connection to i-PI server."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._connected = False
        logger.info("Disconnected from i-PI server")

    def _send(self, data: bytes) -> None:
        """Send raw bytes to server."""
        if not self._connected:
            raise RuntimeError("Not connected to i-PI server")
        self._socket.sendall(data)

    def _recv(self, size: int) -> bytes:
        """Receive exactly `size` bytes from server."""
        if not self._connected:
            raise RuntimeError("Not connected to i-PI server")

        data = b""
        remaining = size
        while remaining > 0:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
            remaining -= len(chunk)
        return data

    def _send_msg(self, msg_type: bytes) -> None:
        """Send 12-byte message header."""
        if len(msg_type) != HEADER_SIZE:
            msg_type = msg_type.ljust(HEADER_SIZE)[:HEADER_SIZE]
        self._send(msg_type)

    def _recv_msg(self) -> str:
        """Receive and decode 12-byte message header."""
        data = self._recv(HEADER_SIZE)
        return data.decode().strip()

    def wait_for_status(self) -> str:
        """
        Wait for STATUS query from server.

        In the i-PI protocol, the server sends STATUS to ask
        the client about its state.

        Returns:
            The status message received (should be "STATUS")
        """
        return self._recv_msg()

    def send_status(self, status: str) -> None:
        """
        Send client status to server.

        Args:
            status: One of "READY", "HAVEDATA", "NEEDINIT"
        """
        status_msg = status.upper().ljust(HEADER_SIZE)[:HEADER_SIZE].encode()
        self._send(status_msg)

    def poll_status(self) -> str:
        """
        Legacy method - wait for server and respond with READY.

        For backward compatibility with tests. In real usage,
        use wait_for_status() and send_status() separately.

        Returns:
            Server's next message after we respond READY
        """
        # Wait for server's STATUS query
        msg = self.wait_for_status()
        if msg == "STATUS":
            # Respond that we're ready for work
            self.send_status("READY")
            # Return the server's next message
            return self._recv_msg()
        else:
            # Server sent something other than STATUS
            return msg

    def send_init(self, bead_id: int = 0) -> None:
        """Send initialization data to server."""
        self._send_msg(MSG_INIT)
        # Send bead identifier (used for ring-polymer parallel evaluation)
        self._send(struct.pack("i", bead_id))

    def receive_geometry(self) -> IPIGeometry:
        """
        Receive geometry from i-PI server.

        Returns:
            IPIGeometry with cell, positions in Bohr
        """
        msg = self._recv_msg()
        if msg != "POSDATA":
            raise RuntimeError(f"Expected POSDATA, got {msg}")

        # Cell matrix (9 doubles, column-major for Fortran compatibility)
        cell_data = self._recv(9 * 8)
        cell = np.frombuffer(cell_data, dtype=np.float64).reshape(3, 3).T

        # Inverse cell matrix
        cell_inv_data = self._recv(9 * 8)
        cell_inv = np.frombuffer(cell_inv_data, dtype=np.float64).reshape(3, 3).T

        # Number of atoms
        n_atoms_data = self._recv(4)
        n_atoms = struct.unpack("i", n_atoms_data)[0]

        # Positions (3*N doubles)
        pos_data = self._recv(3 * n_atoms * 8)
        positions = np.frombuffer(pos_data, dtype=np.float64).reshape(n_atoms, 3)

        logger.debug(f"Received geometry: {n_atoms} atoms")

        return IPIGeometry(
            cell=cell,
            cell_inv=cell_inv,
            positions=positions,
            n_atoms=n_atoms
        )

    def send_forces(self, forces: IPIForces) -> None:
        """
        Send calculated forces back to i-PI server.

        Args:
            forces: IPIForces with energy, forces, virial
        """
        self._send_msg(MSG_FORCEREADY)

        # Energy (1 double)
        self._send(struct.pack("d", forces.energy))

        # Number of atoms
        n_atoms = len(forces.forces)
        self._send(struct.pack("i", n_atoms))

        # Forces (3N doubles, row-major)
        self._send(forces.forces.astype(np.float64).tobytes())

        # Virial tensor (9 doubles, row-major)
        self._send(forces.virial.astype(np.float64).tobytes())

        # Extras (optional JSON string)
        if forces.extras:
            extras_bytes = forces.extras.encode()
            self._send(struct.pack("i", len(extras_bytes)))
            self._send(extras_bytes)
        else:
            self._send(struct.pack("i", 0))

        logger.debug(f"Sent forces: E={forces.energy:.8f} Ha")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def geometry_to_angstrom(geom: IPIGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert i-PI geometry to Angstrom.

    Args:
        geom: IPIGeometry in Bohr

    Returns:
        Tuple of (positions in Angstrom, cell in Angstrom)
    """
    positions_angstrom = geom.positions * BOHR_TO_ANGSTROM
    cell_angstrom = geom.cell * BOHR_TO_ANGSTROM
    return positions_angstrom, cell_angstrom


def forces_to_atomic_units(
    energy_hartree: float,
    forces_hartree_per_angstrom: np.ndarray,
    virial_hartree: Optional[np.ndarray] = None
) -> IPIForces:
    """
    Convert forces to i-PI atomic units (Hartree/Bohr).

    Args:
        energy_hartree: Energy in Hartree
        forces_hartree_per_angstrom: Forces in Hartree/Angstrom
        virial_hartree: Virial tensor in Hartree (optional)

    Returns:
        IPIForces ready for i-PI
    """
    # Convert forces from Hartree/Angstrom to Hartree/Bohr
    # f [Ha/Bohr] = f [Ha/Å] * (Å/Bohr) = f [Ha/Å] / ANGSTROM_TO_BOHR
    forces_hartree_per_bohr = forces_hartree_per_angstrom / ANGSTROM_TO_BOHR

    if virial_hartree is None:
        virial_hartree = np.zeros((3, 3))

    return IPIForces(
        energy=energy_hartree,
        forces=forces_hartree_per_bohr,
        virial=virial_hartree
    )
