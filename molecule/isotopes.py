"""Isotope data and substitution support."""

from typing import Dict, Optional, List
from .structure import Molecule

# Comprehensive isotope mass data (in AMU)
# Format: "Symbol" or "Mass_Symbol" -> mass
ISOTOPE_MASSES: Dict[str, float] = {
    # Hydrogen isotopes
    "H": 1.00782503207,
    "1H": 1.00782503207,
    "D": 2.01410177785,
    "2H": 2.01410177785,
    "T": 3.0160492777,
    "3H": 3.0160492777,

    # Carbon isotopes
    "C": 12.0000000,
    "12C": 12.0000000,
    "13C": 13.00335483778,

    # Nitrogen isotopes
    "N": 14.0030740048,
    "14N": 14.0030740048,
    "15N": 15.00010889823,

    # Oxygen isotopes
    "O": 15.99491461956,
    "16O": 15.99491461956,
    "17O": 16.99913170,
    "18O": 17.9991610,

    # Fluorine
    "F": 18.99840322,
    "19F": 18.99840322,

    # Sulfur isotopes
    "S": 31.97207100,
    "32S": 31.97207100,
    "33S": 32.97145876,
    "34S": 33.96786690,

    # Chlorine isotopes
    "Cl": 34.96885268,
    "35Cl": 34.96885268,
    "37Cl": 36.96590259,

    # Bromine isotopes
    "Br": 78.9183371,
    "79Br": 78.9183371,
    "81Br": 80.9162906,

    # Phosphorus
    "P": 30.97376163,
    "31P": 30.97376163,

    # Silicon
    "Si": 27.9769265325,
    "28Si": 27.9769265325,
    "29Si": 28.976494700,
    "30Si": 29.97377017,
}

# Common isotope substitution shortcuts
ISOTOPE_SHORTCUTS: Dict[str, str] = {
    "D": "2H",
    "T": "3H",
}


def get_isotope_mass(isotope: str) -> float:
    """
    Get the mass of an isotope.

    Args:
        isotope: Isotope symbol (e.g., "H", "D", "13C", "18O")

    Returns:
        Mass in AMU

    Raises:
        ValueError: If isotope is not found in database
    """
    # Check shortcuts
    if isotope in ISOTOPE_SHORTCUTS:
        isotope = ISOTOPE_SHORTCUTS[isotope]

    if isotope not in ISOTOPE_MASSES:
        raise ValueError(f"Unknown isotope: {isotope}")

    return ISOTOPE_MASSES[isotope]


def substitute_isotopes(
    mol: Molecule,
    substitutions: Optional[Dict[int, str]] = None,
    global_substitutions: Optional[Dict[str, str]] = None
) -> Molecule:
    """
    Create a new molecule with isotope substitutions.

    Args:
        mol: Original molecule
        substitutions: Dict mapping atom index to isotope symbol
            Example: {0: "D", 3: "D"} replaces atoms 0 and 3 with deuterium
        global_substitutions: Dict mapping element symbol to isotope
            Example: {"H": "D"} replaces all H with deuterium

    Returns:
        New Molecule with updated masses

    Example:
        >>> h2o2 = Molecule.h2o2()
        >>> d2o2 = substitute_isotopes(h2o2, global_substitutions={"H": "D"})
        >>> h2o18_2 = substitute_isotopes(h2o2, global_substitutions={"O": "18O"})
    """
    new_mol = mol.copy()

    # Apply global substitutions first
    if global_substitutions:
        for i, atom in enumerate(new_mol.atoms):
            if atom.symbol in global_substitutions:
                new_isotope = global_substitutions[atom.symbol]
                new_mass = get_isotope_mass(new_isotope)
                new_mol.atoms[i].mass = new_mass

    # Apply specific substitutions (overrides global)
    if substitutions:
        for idx, isotope in substitutions.items():
            if idx < 0 or idx >= len(new_mol.atoms):
                raise IndexError(f"Atom index {idx} out of range")
            new_mass = get_isotope_mass(isotope)
            new_mol.atoms[idx].mass = new_mass

    return new_mol


def create_isotopologue_set(
    mol: Molecule,
    target_element: str,
    isotopes: List[str]
) -> Dict[str, Molecule]:
    """
    Create a set of isotopologues by substituting one element.

    Args:
        mol: Original molecule
        target_element: Element to substitute (e.g., "H")
        isotopes: List of isotopes to create (e.g., ["H", "D", "T"])

    Returns:
        Dict mapping isotope name to Molecule

    Example:
        >>> h2o2 = Molecule.h2o2()
        >>> isotopologues = create_isotopologue_set(h2o2, "H", ["H", "D"])
        >>> # Returns {"H": h2o2_with_H, "D": d2o2}
    """
    result = {}
    for isotope in isotopes:
        name = isotope if isotope != target_element else target_element
        result[name] = substitute_isotopes(
            mol,
            global_substitutions={target_element: isotope}
        )
    return result


def calculate_kinetic_isotope_effect(
    rate_light: float,
    rate_heavy: float
) -> float:
    """
    Calculate kinetic isotope effect (KIE).

    Args:
        rate_light: Rate constant for lighter isotope
        rate_heavy: Rate constant for heavier isotope

    Returns:
        KIE = k_light / k_heavy
    """
    if rate_heavy == 0:
        return float('inf')
    return rate_light / rate_heavy
