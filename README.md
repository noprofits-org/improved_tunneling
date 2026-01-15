# improved_tunneling

A modular Python package for simulating quantum tunneling in molecular systems.

## Features

- **Multiple Tunneling Methods**: WKB approximation, Small Curvature Tunneling (SCT), and Eckart barrier models
- **Potential Energy Surface Scans**: Relaxed (constrained optimization) and rigid PES scans with spline interpolation
- **Isotope Effects**: Automatic isotope substitution (H/D/T) with proper reduced mass calculations
- **Kinetics**: Temperature-dependent rate constants, tunneling corrections, and Arrhenius analysis
- **Quantum Chemistry Integration**: Psi4 interface for ab initio calculations (with mock engine for testing)

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Psi4 (optional, for real quantum chemistry calculations)

```bash
# Clone the repository
git clone https://github.com/noprofits-org/improved_tunneling.git
cd improved_tunneling

# Install dependencies
pip install numpy scipy

# Optional: Install Psi4 for ab initio calculations
conda install psi4 -c psi4
```

## Quick Start

```python
from improved_tunnel.tunneling.eckart import EckartBarrier
from improved_tunnel.core.constants import HARTREE_TO_KCAL, AMU_TO_KG

# Create an Eckart barrier (1 kcal/mol)
barrier_kcal = 1.0
barrier_hartree = barrier_kcal / HARTREE_TO_KCAL

eckart = EckartBarrier(
    V1=barrier_hartree,
    V2=barrier_hartree,
    L=0.5,  # barrier width in radians
    symmetric=True
)

# Calculate transmission coefficient
reduced_mass = 1.0 * AMU_TO_KG  # 1 AMU
energy = 0.5 * barrier_hartree  # E/Vb = 0.5
T = eckart.analytical_transmission(energy, reduced_mass)
print(f"Transmission coefficient: {T:.4e}")
```

## Running the Example

```bash
# Run H2O2 torsional tunneling analysis (mock engine)
python examples/h2o2_tunneling.py

# Run standalone demo
python examples/h2o2_tunneling.py --demo

# Run with real Psi4 calculations (requires Psi4)
python examples/h2o2_tunneling.py --real
```

## Package Structure

```
improved_tunneling/
├── core/           # Constants, configuration, exceptions
├── molecule/       # Molecular structure, geometry, isotopes
├── pes/            # Potential energy surface scanning
├── qchem/          # Quantum chemistry engine interface (Psi4)
├── tunneling/      # Tunneling methods (WKB, SCT, Eckart)
├── kinetics/       # Rate constants and Arrhenius analysis
├── workflow/       # High-level workflow runner
└── examples/       # Example scripts
```

## Tunneling Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **WKB** | Semiclassical WKB approximation | General barriers |
| **Eckart** | Analytical Eckart barrier model | Symmetric/asymmetric barriers |
| **SCT** | Small Curvature Tunneling | Reaction path tunneling |

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
