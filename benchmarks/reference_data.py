"""Literature reference data for benchmark systems.

Sources:
- Malonaldehyde: Firth et al. JCP 1991; Tew et al. PCCP 2018; Lauvergnat 2023
- H2O2: Koput 1986; Pelz et al. 1993
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BenchmarkReference:
    """Reference data for a benchmark system."""

    name: str
    formula: str

    # Tunneling splitting (cm^-1)
    experimental_splitting_cm: float
    experimental_splitting_error: float

    # Barrier height (kcal/mol)
    barrier_height_kcal: float
    barrier_height_error: float

    # Key geometry parameters
    geometry_params: Dict[str, float]

    # Isotope effect
    deuterated_splitting_cm: Optional[float] = None

    # Computational benchmarks
    computational_refs: Optional[Dict[str, float]] = None

    # Literature sources
    sources: Optional[Dict[str, str]] = None


# =============================================================================
# MALONALDEHYDE - The gold standard for tunneling benchmarks
# =============================================================================

MALONALDEHYDE_REF = BenchmarkReference(
    name="Malonaldehyde",
    formula="C3H4O2",

    # Experimental tunneling splitting
    # From microwave spectroscopy
    experimental_splitting_cm=21.6,
    experimental_splitting_error=0.3,

    # Barrier height for intramolecular proton transfer
    # High-level ab initio: ~3.4 kcal/mol (14.4 kJ/mol)
    barrier_height_kcal=3.4,
    barrier_height_error=0.3,

    # Key structural parameters
    geometry_params={
        "O_H_distance_A": 1.69,  # O···H hydrogen bond distance
        "C_C_distance_A": 1.45,  # C-C bond in conjugated system
        "C_O_distance_A": 1.32,  # C=O/C-O averaged
        "O_O_distance_A": 2.55,  # O···O distance
    },

    # Deuterium tunneling splitting (much smaller)
    deuterated_splitting_cm=2.9,

    # Recent computational benchmarks (cm^-1)
    computational_refs={
        "Smolyak_2023": 21.7,
        "PIMD_2025": 21.1,
        "DMC_2007": 21.0,
        "MCTDH_2008": 23.8,
        "Full_dim_QD": 24.5,
        "Gauss_expansion_2023": 27.1,
    },

    sources={
        "experimental": "Firth et al., J. Chem. Phys. 94, 1812 (1991)",
        "barrier": "Tew et al., Phys. Chem. Chem. Phys. 20, 27630 (2018)",
        "smolyak": "Lauvergnat, ChemPhysChem 24, e202300501 (2023)",
        "pimd": "Markland et al., Mol. Phys. (2025)",
    }
)


# =============================================================================
# HYDROGEN PEROXIDE - Simple torsional tunneling
# =============================================================================

H2O2_REF = BenchmarkReference(
    name="Hydrogen Peroxide",
    formula="H2O2",

    # Torsional tunneling splitting
    # From high-resolution IR spectroscopy
    experimental_splitting_cm=11.4,  # Ground state splitting
    experimental_splitting_error=0.1,

    # Torsional barrier height
    # cis barrier ~1 kcal/mol, trans barrier ~7 kcal/mol
    barrier_height_kcal=1.1,  # cis barrier (lower)
    barrier_height_error=0.2,

    # Key structural parameters
    geometry_params={
        "O_O_distance_A": 1.475,
        "O_H_distance_A": 0.965,
        "H_O_O_angle_deg": 99.4,
        "dihedral_eq_deg": 111.5,  # Equilibrium dihedral
    },

    # D2O2 splitting
    deuterated_splitting_cm=8.1,

    # Computational references
    computational_refs={
        "CCSD_T_CBS": 1.05,  # Barrier in kcal/mol
        "MP2_aug_cc_pVTZ": 1.15,
    },

    sources={
        "experimental": "Koput, J. Mol. Spectrosc. 115, 438 (1986)",
        "structure": "Pelz et al., J. Mol. Spectrosc. 159, 507 (1993)",
    }
)


# =============================================================================
# Additional systems (placeholders for future implementation)
# =============================================================================

FORMIC_ACID_DIMER_REF = BenchmarkReference(
    name="Formic Acid Dimer",
    formula="(HCOOH)2",

    # Double proton transfer
    experimental_splitting_cm=0.016,  # Very small - concerted transfer
    experimental_splitting_error=0.002,

    barrier_height_kcal=8.0,
    barrier_height_error=1.0,

    geometry_params={
        "O_O_distance_A": 2.70,
        "H_bond_length_A": 1.67,
    },

    sources={
        "experimental": "Ortlieb & Havenith, J. Phys. Chem. A 111, 7355 (2007)",
    }
)


TROPOLONE_REF = BenchmarkReference(
    name="Tropolone",
    formula="C7H6O2",

    # Intramolecular proton transfer
    experimental_splitting_cm=0.99,
    experimental_splitting_error=0.02,

    barrier_height_kcal=4.0,
    barrier_height_error=0.5,

    geometry_params={
        "O_O_distance_A": 2.52,
    },

    deuterated_splitting_cm=0.051,  # Large H/D isotope effect

    sources={
        "experimental": "Redington et al., J. Chem. Phys. 88, 627 (1988)",
    }
)
