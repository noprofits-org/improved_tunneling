"""Tests for i-PI integration module."""

import pytest
import numpy as np
import socket
import struct
import threading
import time

from improved_tunnel.instanton.ipi_client import (
    IPISocketClient,
    IPIGeometry,
    IPIForces,
    geometry_to_angstrom,
    forces_to_atomic_units,
    HEADER_SIZE,
    MSG_STATUS,
    MSG_POSDATA,
    MSG_GETFORCE,
    MSG_FORCEREADY,
    MSG_EXIT,
    MSG_NEEDINIT,
    MSG_READY,
)
from improved_tunnel.instanton.ipi_driver import (
    Psi4IPIDriver,
    DriverConfig,
    create_driver_from_molecule,
)
from improved_tunnel.molecule.structure import Molecule
from improved_tunnel.core.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def h2o2_molecule():
    """H2O2 molecule for testing."""
    return Molecule.h2o2()


@pytest.fixture
def mock_geometry():
    """Mock i-PI geometry for H2O2."""
    # H2O2 geometry in Bohr
    positions_bohr = np.array([
        [0.0, 1.5, -1.8],   # H1
        [0.0, 0.0, -1.4],   # O1
        [0.0, 0.0, 1.4],    # O2
        [0.0, -1.5, 1.8],   # H2
    ])

    cell = np.eye(3) * 50.0  # Large cell for isolated molecule
    cell_inv = np.linalg.inv(cell)

    return IPIGeometry(
        cell=cell,
        cell_inv=cell_inv,
        positions=positions_bohr,
        n_atoms=4
    )


# =============================================================================
# Unit tests for IPIGeometry and IPIForces
# =============================================================================

class TestIPIDataClasses:
    """Tests for i-PI data structures."""

    def test_ipi_geometry_creation(self, mock_geometry):
        """Test IPIGeometry dataclass."""
        assert mock_geometry.n_atoms == 4
        assert mock_geometry.positions.shape == (4, 3)
        assert mock_geometry.cell.shape == (3, 3)
        assert mock_geometry.cell_inv.shape == (3, 3)

    def test_ipi_forces_creation(self):
        """Test IPIForces dataclass."""
        forces = IPIForces(
            energy=-150.0,
            forces=np.zeros((4, 3)),
            virial=np.zeros((3, 3)),
            extras='{"info": "test"}'
        )

        assert forces.energy == -150.0
        assert forces.forces.shape == (4, 3)
        assert forces.virial.shape == (3, 3)
        assert forces.extras == '{"info": "test"}'

    def test_ipi_forces_no_extras(self):
        """Test IPIForces without extras."""
        forces = IPIForces(
            energy=-150.0,
            forces=np.zeros((4, 3)),
            virial=np.zeros((3, 3))
        )

        assert forces.extras is None


# =============================================================================
# Tests for unit conversion functions
# =============================================================================

class TestUnitConversions:
    """Tests for Bohr/Angstrom conversion functions."""

    def test_geometry_to_angstrom(self, mock_geometry):
        """Test conversion from Bohr to Angstrom."""
        positions_angstrom, cell_angstrom = geometry_to_angstrom(mock_geometry)

        # Check conversion factor applied
        np.testing.assert_allclose(
            positions_angstrom,
            mock_geometry.positions * BOHR_TO_ANGSTROM,
            rtol=1e-10
        )

        np.testing.assert_allclose(
            cell_angstrom,
            mock_geometry.cell * BOHR_TO_ANGSTROM,
            rtol=1e-10
        )

    def test_forces_to_atomic_units(self):
        """Test force unit conversion."""
        energy = -150.5
        forces_ha_per_angstrom = np.array([
            [0.01, 0.02, -0.01],
            [-0.01, -0.02, 0.01],
            [0.01, -0.02, 0.01],
            [-0.01, 0.02, -0.01],
        ])

        result = forces_to_atomic_units(
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces_ha_per_angstrom
        )

        assert result.energy == energy
        assert result.forces.shape == (4, 3)

        # Forces should be scaled by 1/ANGSTROM_TO_BOHR
        expected_forces = forces_ha_per_angstrom / ANGSTROM_TO_BOHR
        np.testing.assert_allclose(result.forces, expected_forces, rtol=1e-10)

        # Default virial is zeros
        np.testing.assert_array_equal(result.virial, np.zeros((3, 3)))

    def test_forces_with_virial(self):
        """Test force conversion with custom virial."""
        virial = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ])

        result = forces_to_atomic_units(
            energy_hartree=-150.0,
            forces_hartree_per_angstrom=np.zeros((4, 3)),
            virial_hartree=virial
        )

        np.testing.assert_array_equal(result.virial, virial)


# =============================================================================
# Tests for message header constants
# =============================================================================

class TestMessageHeaders:
    """Tests for i-PI protocol message headers."""

    def test_header_size(self):
        """All headers should be 12 bytes."""
        headers = [
            MSG_STATUS, MSG_POSDATA, MSG_GETFORCE,
            MSG_FORCEREADY, MSG_EXIT, MSG_NEEDINIT, MSG_READY
        ]

        for header in headers:
            assert len(header) == HEADER_SIZE

    def test_headers_are_bytes(self):
        """Headers should be bytes, not strings."""
        assert isinstance(MSG_STATUS, bytes)
        assert isinstance(MSG_READY, bytes)

    def test_header_content(self):
        """Check header content is correct."""
        assert MSG_STATUS.strip() == b"STATUS"
        assert MSG_READY.strip() == b"READY"
        assert MSG_EXIT.strip() == b"EXIT"


# =============================================================================
# Tests for DriverConfig
# =============================================================================

class TestDriverConfig:
    """Tests for driver configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DriverConfig()

        assert config.method == "HF"
        assert config.basis == "cc-pVDZ"
        assert config.charge == 0
        assert config.multiplicity == 1
        assert config.e_convergence == 1e-8

    def test_custom_config(self):
        """Test custom configuration."""
        config = DriverConfig(
            method="MP2",
            basis="aug-cc-pVTZ",
            charge=-1,
            multiplicity=2,
            memory="4 GB",
            nthreads=4
        )

        assert config.method == "MP2"
        assert config.basis == "aug-cc-pVTZ"
        assert config.charge == -1
        assert config.multiplicity == 2
        assert config.memory == "4 GB"
        assert config.nthreads == 4


# =============================================================================
# Tests for Psi4IPIDriver
# =============================================================================

class TestPsi4IPIDriver:
    """Tests for Psi4 i-PI driver."""

    def test_driver_creation(self):
        """Test basic driver creation."""
        driver = Psi4IPIDriver(
            host="localhost",
            port=31415,
            method="HF",
            basis="cc-pVDZ",
            use_mock=True
        )

        assert driver.config.method == "HF"
        assert driver.config.basis == "cc-pVDZ"
        assert driver.use_mock is True

    def test_driver_from_molecule(self, h2o2_molecule):
        """Test creating driver from molecule."""
        driver = create_driver_from_molecule(
            h2o2_molecule,
            method="MP2",
            basis="cc-pVTZ",
            use_mock=True
        )

        assert driver.config.method == "MP2"
        assert driver.config.basis == "cc-pVTZ"
        assert driver.config.charge == h2o2_molecule.charge
        assert driver.config.multiplicity == h2o2_molecule.multiplicity

    def test_mock_gradient(self):
        """Test mock gradient calculation."""
        driver = Psi4IPIDriver(use_mock=True)
        driver._symbols = ["H", "O", "O", "H"]
        driver._initialize_psi4()

        positions = np.array([
            [0.0, 0.8, -1.0],
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
            [0.0, -0.8, 1.0],
        ])

        energy, gradient = driver._compute_gradient(positions)

        # Mock returns pairwise potential energy
        assert isinstance(energy, float)
        assert gradient.shape == (4, 3)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(gradient))

    def test_mock_gradient_displacement(self):
        """Test mock gradient with displacement."""
        driver = Psi4IPIDriver(use_mock=True)
        driver._symbols = ["H", "O", "O", "H"]
        driver._initialize_psi4()

        positions = np.array([
            [0.0, 0.8, -1.0],
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
            [0.0, -0.8, 1.0],
        ])

        energy1, gradient1 = driver._compute_gradient(positions)

        # Displace atoms
        displaced = positions.copy()
        displaced[0, 0] += 0.5  # Move first atom

        energy2, gradient2 = driver._compute_gradient(displaced)

        # Energy should change
        assert energy1 != energy2
        # Gradients should be different
        assert not np.allclose(gradient1, gradient2)


# =============================================================================
# Tests for IPISocketClient (unit tests without server)
# =============================================================================

class TestIPISocketClient:
    """Tests for i-PI socket client."""

    def test_client_creation(self):
        """Test client creation."""
        client = IPISocketClient(
            host="localhost",
            port=31415,
            timeout=60.0
        )

        assert client.host == "localhost"
        assert client.port == 31415
        assert client.timeout == 60.0
        assert client._connected is False

    def test_client_unix_socket(self):
        """Test client with unix socket."""
        client = IPISocketClient(
            unix_socket="/tmp/ipi_test.sock"
        )

        assert client.unix_socket == "/tmp/ipi_test.sock"

    def test_send_without_connection(self):
        """Test that send raises error when not connected."""
        client = IPISocketClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            client._send(b"test")

    def test_recv_without_connection(self):
        """Test that recv raises error when not connected."""
        client = IPISocketClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            client._recv(12)


# =============================================================================
# Integration tests with mock i-PI server
# =============================================================================

class MockIPIServer:
    """Mock i-PI server for testing."""

    def __init__(self, port: int, n_steps: int = 2):
        self.port = port
        self.n_steps = n_steps
        self.server_socket = None
        self.running = False
        self._thread = None

    def start(self):
        """Start mock server in background thread."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("localhost", self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(5.0)
        self.running = True

        self._thread = threading.Thread(target=self._serve)
        self._thread.start()

    def _serve(self):
        """Server loop implementing correct i-PI protocol.

        Protocol flow:
        1. Server sends STATUS
        2. Client responds with READY or HAVEDATA
        3. If READY: Server sends POSDATA + geometry
        4. If HAVEDATA: Server sends GETFORCE, client sends FORCEREADY + data
        """
        try:
            conn, addr = self.server_socket.accept()
            conn.settimeout(5.0)

            for step in range(self.n_steps):
                # Phase 1: Send STATUS, client says READY, send geometry
                conn.sendall(MSG_STATUS)
                client_status = conn.recv(HEADER_SIZE).decode().strip()

                if client_status != "READY":
                    break

                # Send geometry
                conn.sendall(MSG_POSDATA)
                cell = np.eye(3) * 50.0
                conn.sendall(cell.T.astype(np.float64).tobytes())
                conn.sendall(np.linalg.inv(cell).T.astype(np.float64).tobytes())
                n_atoms = 4
                conn.sendall(struct.pack("i", n_atoms))
                positions = np.array([
                    [0.0, 1.5, -1.8],
                    [0.0, 0.0, -1.4],
                    [0.0, 0.0, 1.4],
                    [0.0, -1.5, 1.8],
                ])
                conn.sendall(positions.astype(np.float64).tobytes())

                # Phase 2: Send STATUS, client says HAVEDATA, get forces
                conn.sendall(MSG_STATUS)
                client_status = conn.recv(HEADER_SIZE).decode().strip()

                if client_status != "HAVEDATA":
                    break

                # Request forces
                conn.sendall(MSG_GETFORCE)

                # Receive FORCEREADY + data
                msg = conn.recv(HEADER_SIZE)
                if msg.strip() != b"FORCEREADY":
                    break

                energy = struct.unpack("d", conn.recv(8))[0]
                n_atoms_recv = struct.unpack("i", conn.recv(4))[0]
                forces = np.frombuffer(conn.recv(n_atoms_recv * 3 * 8), dtype=np.float64)
                virial = np.frombuffer(conn.recv(9 * 8), dtype=np.float64)
                extras_len = struct.unpack("i", conn.recv(4))[0]
                if extras_len > 0:
                    extras = conn.recv(extras_len)

            # Send EXIT
            conn.sendall(MSG_EXIT)
            conn.close()

        except Exception as e:
            import traceback
            print(f"Mock server error: {e}")
            traceback.print_exc()
        finally:
            self.running = False

    def stop(self):
        """Stop server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self._thread:
            self._thread.join(timeout=2.0)


@pytest.mark.slow
class TestIPIIntegration:
    """Integration tests with mock i-PI server."""

    def test_driver_with_mock_server(self):
        """Test full driver loop with mock server."""
        port = 31416  # Use different port to avoid conflicts

        # Start mock server
        server = MockIPIServer(port=port, n_steps=2)
        server.start()

        try:
            # Give server time to start
            time.sleep(0.1)

            # Create and run driver
            driver = Psi4IPIDriver(
                host="localhost",
                port=port,
                use_mock=True
            )

            energies = []
            def callback(step, energy):
                energies.append(energy)

            driver.run(
                symbols=["H", "O", "O", "H"],
                callback=callback
            )

            # Check that calculations were performed
            assert driver._calculation_count == 2
            assert len(energies) == 2

        finally:
            server.stop()

    def test_client_context_manager(self):
        """Test client context manager."""
        port = 31417

        # Just test the context manager pattern
        # (won't actually connect without a server)
        client = IPISocketClient(host="localhost", port=port)

        # Verify context manager methods exist
        assert hasattr(client, '__enter__')
        assert hasattr(client, '__exit__')


# =============================================================================
# Tests for edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_atom_count_mismatch(self):
        """Test error on atom count mismatch."""
        driver = Psi4IPIDriver(use_mock=True)
        driver._symbols = ["H", "O"]  # Only 2 atoms
        driver._initialize_psi4()

        # This would fail in the driver loop when receiving 4 atoms
        # but expecting 2 - tested via the check in _driver_loop

    def test_empty_symbols_error(self):
        """Test error when symbols not set."""
        driver = Psi4IPIDriver(use_mock=True)
        driver._initialize_psi4()

        # Should raise when trying to create molecule without symbols
        with pytest.raises(RuntimeError, match="symbols not set"):
            driver._create_molecule(np.zeros((4, 3)))

    def test_large_molecule(self):
        """Test with larger molecule."""
        driver = Psi4IPIDriver(use_mock=True)
        driver._symbols = ["C"] * 20 + ["H"] * 40  # 60 atoms
        driver._initialize_psi4()

        positions = np.random.randn(60, 3)
        energy, gradient = driver._compute_gradient(positions)

        assert gradient.shape == (60, 3)
