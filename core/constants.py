"""Physical constants and unit conversion factors."""

import math

# Fundamental constants (CODATA 2018 values)

# Planck constant
H_SI = 6.62607015e-34  # J·s (exact)
HBAR_SI = 1.054571817e-34  # J·s (ℏ = h/2π)
HBAR = HBAR_SI  # Alias for convenience

# Boltzmann constant
BOLTZMANN_SI = 1.380649e-23  # J/K (exact)
BOLTZMANN = BOLTZMANN_SI  # Alias for convenience
BOLTZMANN_HARTREE = 3.1668115634556e-6  # Hartree/K (kB in atomic units)

# Speed of light
C_SI = 299792458  # m/s (exact)

# Avogadro's number
AVOGADRO = 6.02214076e23  # mol^-1 (exact)

# Electron mass
ELECTRON_MASS_SI = 9.1093837015e-31  # kg

# Atomic mass unit
AMU_SI = 1.66053906660e-27  # kg
AMU_TO_KG = AMU_SI

# Unit conversions - Energy

# Hartree (atomic unit of energy)
HARTREE_TO_JOULE = 4.3597447222071e-18  # J
JOULE_TO_HARTREE = 1.0 / HARTREE_TO_JOULE

# Hartree to kcal/mol
HARTREE_TO_KCAL = 627.5094740631  # kcal/mol
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL

# Hartree to kJ/mol
HARTREE_TO_KJ = 2625.4996394799  # kJ/mol
KJ_TO_HARTREE = 1.0 / HARTREE_TO_KJ

# Hartree to eV
HARTREE_TO_EV = 27.211386245988  # eV
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Hartree to cm^-1 (wavenumbers)
HARTREE_TO_CM = 219474.6313632  # cm^-1
CM_TO_HARTREE = 1.0 / HARTREE_TO_CM

# Unit conversions - Length

# Bohr radius (atomic unit of length)
BOHR_TO_ANGSTROM = 0.529177210903  # Å
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

BOHR_TO_METER = 5.29177210903e-11  # m
METER_TO_BOHR = 1.0 / BOHR_TO_METER

# Unit conversions - Angle
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Unit conversions - Time
# Atomic unit of time
AU_TIME_TO_SEC = 2.4188843265857e-17  # s
SEC_TO_AU_TIME = 1.0 / AU_TIME_TO_SEC

# Femtoseconds
FS_TO_SEC = 1e-15
SEC_TO_FS = 1e15

# Gas constant
R_SI = 8.314462618  # J/(mol·K)
R_KCAL = R_SI / 4184.0  # kcal/(mol·K)

# Common atomic masses (in AMU)
ATOMIC_MASSES = {
    "H": 1.00782503207,
    "D": 2.01410177785,
    "T": 3.0160492777,
    "C": 12.0000000,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "F": 18.99840322,
    "S": 31.97207100,
    "Cl": 34.96885268,
}


def hartree_to_kcal(energy_hartree: float) -> float:
    """Convert energy from Hartree to kcal/mol."""
    return energy_hartree * HARTREE_TO_KCAL


def kcal_to_hartree(energy_kcal: float) -> float:
    """Convert energy from kcal/mol to Hartree."""
    return energy_kcal * KCAL_TO_HARTREE


def hartree_to_joule(energy_hartree: float) -> float:
    """Convert energy from Hartree to Joules."""
    return energy_hartree * HARTREE_TO_JOULE


def joule_to_hartree(energy_joule: float) -> float:
    """Convert energy from Joules to Hartree."""
    return energy_joule * JOULE_TO_HARTREE


def amu_to_kg(mass_amu: float) -> float:
    """Convert mass from AMU to kg."""
    return mass_amu * AMU_TO_KG


def kg_to_amu(mass_kg: float) -> float:
    """Convert mass from kg to AMU."""
    return mass_kg / AMU_TO_KG


def angstrom_to_bohr(length_angstrom: float) -> float:
    """Convert length from Angstrom to Bohr."""
    return length_angstrom * ANGSTROM_TO_BOHR


def bohr_to_angstrom(length_bohr: float) -> float:
    """Convert length from Bohr to Angstrom."""
    return length_bohr * BOHR_TO_ANGSTROM
