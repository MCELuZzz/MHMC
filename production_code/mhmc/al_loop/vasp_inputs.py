"""
Generate VASP input file directories for selected configurations.

For each selected configuration, creates a subdirectory with:
  - POSCAR : the configuration (written by ASE)
  - INCAR  : from template, with MAGMOM auto-generated from POSCAR
  - KPOINTS: from template
  - POTCAR : concatenated from per-element POTCAR library
  - metadata.json : source tag, iteration, basin_id, energy

Reference: Section 2.3 of the manuscript.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
from ase import Atoms
from ase.io import write as ase_write


# ============================================================
#  Default INCAR (single-point DFT for MLP training data)
#  Based on the user's existing Fe(110) NH3 settings.
# ============================================================

DEFAULT_INCAR_TEMPLATE = """\
SYSTEM = {system_name}

# Initialization
ISTART = 0
ISPIN  = 2

# Parallelization (user should adjust to cluster)
NPAR   = {npar}

# Electronic
ENCUT  = {encut}
PREC   = {prec}
EDIFF  = {ediff}
ALGO   = {algo}
NELM   = {nelm}
NSIM   = {nsim}
LREAL  = {lreal}
VOSKOWN= 0
GGA    = {gga}

# Ionic (single-point only)
IBRION = -1
NSW    = 0
ISIF   = 2
POTIM  = 0.2
EDIFFG = -0.01

# Smearing
ISMEAR = {ismear}
SIGMA  = {sigma}

# Symmetry
ISYM   = 0

# vdW correction
IVDW   = {ivdw}

# I/O
LWAVE  = .FALSE.
LCHARG = .FALSE.
LORBIT = 11

# Spin: per-element initial magnetic moments
MAGMOM = {magmom}
"""


# ============================================================
#  Default per-element initial magnetic moments
# ============================================================

DEFAULT_MAGMOM_BY_ELEMENT = {
    # Magnetic metals
    "Fe":  2.5,
    "Co":  1.7,
    "Ni":  0.6,
    "Mn":  3.5,
    # Non-magnetic elements (set 0 explicitly, since ISPIN=2 is on)
    "H":   0.0,
    "C":   0.0,
    "N":   0.0,
    "O":   0.0,
    "Pd":  0.0,
    "Cu":  0.0,
    "Pt":  0.0,
    "Au":  0.0,
    "Ag":  0.0,
}


# ============================================================
#  Default KPOINTS (Gamma-centered, surface slab default)
# ============================================================

DEFAULT_KPOINTS_TEMPLATE = """\
Automatic mesh
0
Gamma
{nkx} {nky} {nkz}
0 0 0
"""


# ============================================================
#  INCAR / KPOINTS generators
# ============================================================

def generate_magmom_string(
    atoms: Atoms,
    magmom_by_element: dict[str, float],
    default_for_unknown: float = 0.0,
) -> str:
    """
    Generate MAGMOM string in VASP's `n*value` format, following the
    element order in atoms.get_chemical_symbols().

    For example:
        atoms with 32 Fe, 4 N, 8 H (in that order)
        → "32*2.5 4*0.0 8*0.0"

    The element order is determined by ASE's `write_vasp` ordering,
    which groups identical species together in the order they first
    appear in atoms.
    """
    symbols = atoms.get_chemical_symbols()

    # Group consecutive identical species (mirrors VASP POSCAR convention)
    grouped: list[tuple[str, int]] = []
    for sym in symbols:
        if grouped and grouped[-1][0] == sym:
            grouped[-1] = (sym, grouped[-1][1] + 1)
        else:
            grouped.append((sym, 1))

    parts: list[str] = []
    for elem, count in grouped:
        mom = magmom_by_element.get(elem, default_for_unknown)
        parts.append(f"{count}*{mom:.1f}")
    return " ".join(parts)


def generate_incar(
    atoms: Atoms,
    output_path: Path,
    *,
    system_name: str = "MHMC-AL DFT",
    encut: float = 400.0,
    ediff: float = 1.0e-5,
    prec: str = "High",
    algo: str = "Fast",
    nelm: int = 200,
    nsim: int = 4,
    npar: int = 2,
    lreal: str = "A",
    gga: str = "PE",
    ismear: int = 1,
    sigma: float = 0.1,
    ivdw: int = 12,
    magmom_by_element: Optional[dict[str, float]] = None,
) -> None:
    """Write an INCAR file for the given configuration."""
    if magmom_by_element is None:
        magmom_by_element = DEFAULT_MAGMOM_BY_ELEMENT

    magmom = generate_magmom_string(atoms, magmom_by_element)

    content = DEFAULT_INCAR_TEMPLATE.format(
        system_name = system_name,
        encut       = f"{encut:.0f}",
        ediff       = f"{ediff:.1e}",
        prec        = prec,
        algo        = algo,
        nelm        = nelm,
        nsim        = nsim,
        npar        = npar,
        lreal       = lreal,
        gga         = gga,
        ismear      = ismear,
        sigma       = sigma,
        ivdw        = ivdw,
        magmom      = magmom,
    )

    output_path.write_text(content)


def generate_kpoints(
    output_path: Path,
    nkx: int = 3,
    nky: int = 3,
    nkz: int = 1,
) -> None:
    """Write a Gamma-centered KPOINTS file."""
    content = DEFAULT_KPOINTS_TEMPLATE.format(nkx=nkx, nky=nky, nkz=nkz)
    output_path.write_text(content)


# ============================================================
#  POTCAR concatenation
# ============================================================
def generate_potcar(
    atoms: Atoms,
    output_path: Path,
    potcar_library: Path,
) -> None:
    """
    Concatenate per-element POTCAR files from the user's library, in the
    element order of the POSCAR.

    This version is safer:
      1. follows POSCAR element order;
      2. inserts newline between POTCAR blocks;
      3. checks TITEL / End of Dataset count;
      4. catches the common bug:
         "End of Dataset PAW_PBE H ..." on the same line.
    """
    symbols = atoms.get_chemical_symbols()

    # POSCAR-equivalent element order: first appearance, no duplicates
    seen: set[str] = set()
    element_order: list[str] = []
    for sym in symbols:
        if sym not in seen:
            element_order.append(sym)
            seen.add(sym)

    # Check all POTCAR files exist
    potcar_files: list[Path] = []
    missing: list[str] = []

    for elem in element_order:
        candidate = potcar_library / elem / "POTCAR"
        if not candidate.is_file():
            missing.append(str(candidate))
        else:
            potcar_files.append(candidate)

    if missing:
        raise FileNotFoundError(
            "POTCAR files not found:\n  "
            + "\n  ".join(missing)
            + f"\n\nPOTCAR library was: {potcar_library}\n"
            + "Expected structure: <library>/<element>/POTCAR"
        )

    # Concatenate safely.
    # Important: force newline between different element POTCAR blocks.
    with output_path.open("wb") as fout:
        for potcar_file in potcar_files:
            data = potcar_file.read_bytes()

            # Remove trailing spaces/newlines, then add exactly one newline.
            # This avoids:
            #   End of Dataset PAW_PBE H ...
            data = data.rstrip() + b"\n"

            fout.write(data)

    # Basic validation
    text = output_path.read_text(errors="replace")

    titel_lines = [line for line in text.splitlines() if "TITEL" in line]
    end_lines = [line for line in text.splitlines() if "End of Dataset" in line]

    if len(titel_lines) != len(element_order):
        raise RuntimeError(
            f"Generated POTCAR may be wrong: expected {len(element_order)} TITEL lines "
            f"for elements {element_order}, but found {len(titel_lines)}.\n"
            f"TITEL lines:\n" + "\n".join(titel_lines)
        )

    if len(end_lines) != len(element_order):
        raise RuntimeError(
            f"Generated POTCAR may be wrong: expected {len(element_order)} "
            f"'End of Dataset' lines, but found {len(end_lines)}.\n"
            f"End lines:\n" + "\n".join(end_lines)
        )

    # Catch exactly your current error
    bad_end_lines = [
        line for line in end_lines
        if line.strip() != "End of Dataset"
    ]

    if bad_end_lines:
        raise RuntimeError(
            "Generated POTCAR has suspicious 'End of Dataset' lines.\n"
            "This usually means different POTCAR blocks were glued together "
            "without a newline.\n\n"
            "Bad lines:\n" + "\n".join(bad_end_lines)
        )

    print(f"[POTCAR] written: {output_path}")
    print(f"[POTCAR] element order: {' '.join(element_order)}")
    for line in titel_lines:
        print(f"[POTCAR] {line.strip()}")
# ============================================================
#  Main entry point: prepare all VASP input directories
# ============================================================

def prepare_vasp_inputs(
    selected_atoms: list[Atoms],
    output_dir: Path,
    potcar_library: Path,
    *,
    incar_params: Optional[dict] = None,
    kpoints_grid: tuple[int, int, int] = (3, 3, 1),
    magmom_by_element: Optional[dict[str, float]] = None,
    metadata_list: Optional[list[dict]] = None,
    selected_indices: Optional[list[int]] = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Prepare one VASP input directory per selected configuration.

    Parameters
    ----------
    selected_atoms : list of ase.Atoms
        Configurations to be DFT-labeled.
    output_dir : Path
        Root output directory; will contain subdirectories 0001/, 0002/, ...
    potcar_library : Path
        Root of the POTCAR library (subdirs per element).
    incar_params : dict, optional
        Overrides for INCAR generation. See generate_incar() kwargs.
    kpoints_grid : (int, int, int)
        K-point mesh. Default (3, 3, 1) for surface slabs.
    magmom_by_element : dict, optional
        Per-element initial magnetic moments. Default uses
        DEFAULT_MAGMOM_BY_ELEMENT.
    metadata_list : list of dict, optional
        Per-selection metadata (source, iteration, basin_id, energy).
        Written to <subdir>/metadata.json for traceability.
    selected_indices : list of int, optional
        Original indices in the candidate pool, written to metadata for
        traceability.
    verbose : bool

    Returns
    -------
    list of Path : paths to the created subdirectories.
    """
    output_dir = Path(output_dir)
    potcar_library = Path(potcar_library)
    output_dir.mkdir(parents=True, exist_ok=True)

    incar_params = incar_params or {}
    if magmom_by_element is None:
        magmom_by_element = DEFAULT_MAGMOM_BY_ELEMENT

    if verbose:
        print(f"\n[VASP] preparing {len(selected_atoms)} input directories "
              f"under {output_dir}")

    subdirs: list[Path] = []

    for i, atoms in enumerate(selected_atoms):
        # Zero-padded 4-digit naming: 0001, 0002, ...
        run_dir = output_dir / f"{i+1:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        subdirs.append(run_dir)

        # Reorder atoms by first-appearance element grouping.
        # This single ordering is then used consistently for POSCAR, MAGMOM, POTCAR.
        symbols = atoms.get_chemical_symbols()
        first_appearance: list[str] = []
        seen: set[str] = set()
        for sym in symbols:
            if sym not in seen:
                first_appearance.append(sym)
                seen.add(sym)
        sort_indices = []
        for elem in first_appearance:
            for j, sym in enumerate(symbols):
                if sym == elem:
                    sort_indices.append(j)
        sorted_atoms = atoms[sort_indices]

        # 1. POSCAR (with sort=False; we've already done the sort)
        ase_write(
            run_dir / "POSCAR",
            sorted_atoms,
            format="vasp",
            direct=True,
            sort=False,
            vasp5=True,
        )

        # 2. INCAR (MAGMOM follows sorted_atoms order)
        generate_incar(
            sorted_atoms,
            run_dir / "INCAR",
            magmom_by_element=magmom_by_element,
            **incar_params,
        )

        # 3. KPOINTS
        generate_kpoints(
            run_dir / "KPOINTS",
            nkx=kpoints_grid[0],
            nky=kpoints_grid[1],
            nkz=kpoints_grid[2],
        )

        # 4. POTCAR (concatenated, follows sorted_atoms order)
        generate_potcar(
            sorted_atoms,
            run_dir / "POTCAR",
            potcar_library=potcar_library,
        )

        # 5. metadata.json (for traceability)
        meta: dict = {
            "run_idx": i + 1,
            "n_atoms": len(atoms),
            "elements": sorted_atoms.get_chemical_symbols(),
        }
        if metadata_list and i < len(metadata_list):
            meta.update(metadata_list[i])
        if selected_indices and i < len(selected_indices):
            meta["original_pool_idx"] = int(selected_indices[i])

        with (run_dir / "metadata.json").open("w") as f:
            # Line 384 附近，改为：
            json.dump(meta, f, indent=2, default=lambda o: (
                int(o) if isinstance(o, np.integer)
                else float(o) if isinstance(o, np.floating)
                else bool(o) if isinstance(o, np.bool_)
                else o.tolist() if isinstance(o, np.ndarray)
                else (_ for _ in ()).throw(TypeError(f"can't serialize {type(o)}"))
            ))
            
    if verbose:
        print(f"[VASP] done. Created {len(subdirs)} run directories.")
        print(f"[VASP] First: {subdirs[0]}")
        print(f"[VASP] Last:  {subdirs[-1]}")

    return subdirs