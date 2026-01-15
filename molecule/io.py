"""File I/O for molecular structures (XYZ format)."""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from .structure import Molecule, Atom
from ..core.exceptions import FileIOError


def read_xyz(filepath: Union[str, Path]) -> Molecule:
    """
    Read a molecule from an XYZ file.

    XYZ format:
        N                    (number of atoms)
        comment line         (optional title/energy)
        symbol x y z
        symbol x y z
        ...

    Args:
        filepath: Path to XYZ file

    Returns:
        Molecule object

    Raises:
        FileIOError: If file cannot be read or parsed
    """
    filepath = Path(filepath)

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        raise FileIOError(f"Cannot read file: {e}", filepath=str(filepath))

    if len(lines) < 2:
        raise FileIOError("XYZ file too short", filepath=str(filepath))

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise FileIOError(
            f"First line must be number of atoms, got: {lines[0].strip()}",
            filepath=str(filepath)
        )

    comment = lines[1].strip() if len(lines) > 1 else ""

    # Try to extract energy from comment line
    energy = _parse_energy_from_comment(comment)

    if len(lines) < n_atoms + 2:
        raise FileIOError(
            f"Expected {n_atoms} atoms but file has only {len(lines) - 2} coordinate lines",
            filepath=str(filepath)
        )

    atoms = []
    for i in range(n_atoms):
        line = lines[i + 2].strip()
        parts = line.split()
        if len(parts) < 4:
            raise FileIOError(
                f"Invalid coordinate line {i + 3}: {line}",
                filepath=str(filepath)
            )

        symbol = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise FileIOError(
                f"Invalid coordinates on line {i + 3}: {line}",
                filepath=str(filepath)
            )

        atoms.append(Atom(symbol=symbol, coordinates=np.array([x, y, z])))

    return Molecule(atoms=atoms, name=comment, energy=energy)


def write_xyz(
    mol: Molecule,
    filepath: Union[str, Path],
    comment: Optional[str] = None
) -> None:
    """
    Write a molecule to an XYZ file.

    Args:
        mol: Molecule to write
        filepath: Output file path
        comment: Optional comment line (default: molecule name or energy)

    Raises:
        FileIOError: If file cannot be written
    """
    filepath = Path(filepath)

    if comment is None:
        if mol.energy is not None:
            comment = f"Energy: {mol.energy:.10f} Hartree"
        else:
            comment = mol.name or ""

    try:
        with open(filepath, 'w') as f:
            f.write(f"{mol.num_atoms}\n")
            f.write(f"{comment}\n")
            for atom in mol.atoms:
                x, y, z = atom.coordinates
                f.write(f"{atom.symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")
    except IOError as e:
        raise FileIOError(f"Cannot write file: {e}", filepath=str(filepath))


def read_xyz_trajectory(filepath: Union[str, Path]) -> List[Molecule]:
    """
    Read multiple molecules from a concatenated XYZ trajectory file.

    Args:
        filepath: Path to XYZ trajectory file

    Returns:
        List of Molecule objects
    """
    filepath = Path(filepath)

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except IOError as e:
        raise FileIOError(f"Cannot read file: {e}", filepath=str(filepath))

    molecules = []
    lines = content.strip().split('\n')
    i = 0

    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            break

        if i + n_atoms + 2 > len(lines):
            raise FileIOError(
                f"Incomplete molecule at line {i + 1}",
                filepath=str(filepath)
            )

        comment = lines[i + 1].strip()
        energy = _parse_energy_from_comment(comment)

        atoms = []
        for j in range(n_atoms):
            line = lines[i + 2 + j].strip()
            parts = line.split()
            symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(Atom(symbol=symbol, coordinates=np.array([x, y, z])))

        molecules.append(Molecule(atoms=atoms, name=comment, energy=energy))
        i += n_atoms + 2

    return molecules


def write_xyz_trajectory(
    molecules: List[Molecule],
    filepath: Union[str, Path]
) -> None:
    """
    Write multiple molecules to a concatenated XYZ trajectory file.

    Args:
        molecules: List of Molecule objects
        filepath: Output file path
    """
    filepath = Path(filepath)

    try:
        with open(filepath, 'w') as f:
            for mol in molecules:
                comment = f"Energy: {mol.energy:.10f}" if mol.energy else mol.name
                f.write(f"{mol.num_atoms}\n")
                f.write(f"{comment}\n")
                for atom in mol.atoms:
                    x, y, z = atom.coordinates
                    f.write(f"{atom.symbol:2s}  {x:15.10f}  {y:15.10f}  {z:15.10f}\n")
    except IOError as e:
        raise FileIOError(f"Cannot write file: {e}", filepath=str(filepath))


def _parse_energy_from_comment(comment: str) -> Optional[float]:
    """Try to extract energy value from XYZ comment line."""
    comment_lower = comment.lower()

    # Try various formats
    for prefix in ["energy:", "energy=", "e=", "e:"]:
        if prefix in comment_lower:
            idx = comment_lower.index(prefix)
            remainder = comment[idx + len(prefix):].strip()
            parts = remainder.split()
            if parts:
                try:
                    return float(parts[0])
                except ValueError:
                    pass

    return None
