"""Ring-Polymer Instanton (RPI) calculations.

This module provides interfaces for instanton-based tunneling calculations,
including integration with i-PI for ring-polymer optimization.

The Ring-Polymer Instanton method is the gold standard for tunneling rate
calculations, superseding WKB and Eckart approximations for quantitative work.

Key components:
- ipi_client: Socket client for i-PI communication
- ipi_driver: Psi4 driver for i-PI calculations
- ipi_input: i-PI input file generation

Example usage:
    from improved_tunnel.instanton import (
        generate_instanton_input,
        create_driver_script,
        Psi4IPIDriver
    )
    from improved_tunnel.molecule.structure import Molecule

    # Generate i-PI input file
    mol = Molecule.h2o2()
    generate_instanton_input(mol, output_file="instanton.xml")
    create_driver_script(mol, method="MP2", output_file="driver.py")

    # Or run driver directly
    driver = Psi4IPIDriver(method="MP2", use_mock=True)
    driver.run(symbols=["H", "O", "O", "H"])
"""

from .ipi_client import IPISocketClient
from .ipi_driver import Psi4IPIDriver
from .ipi_input import (
    IPIConfig,
    generate_instanton_input,
    generate_geop_input,
    generate_neb_input,
    generate_md_input,
    create_driver_script,
)

__all__ = [
    "IPISocketClient",
    "Psi4IPIDriver",
    "IPIConfig",
    "generate_instanton_input",
    "generate_geop_input",
    "generate_neb_input",
    "generate_md_input",
    "create_driver_script",
]
