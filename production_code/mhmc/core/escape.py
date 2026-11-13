"""
MH escape phase: short high-temperature MD followed by local relaxation.

Now optionally returns the full MD and relaxation trajectories (with
energies stored in atoms.info) so they can be archived by the dataset
collector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS


def escape_phase(
    seed: Atoms,
    calculator: Calculator,
    T_escape: float,
    md_steps: int = 500,
    md_timestep_fs: float = 1.0,
    relax_fmax: float = 0.05,
    relax_steps: int = 200,
    fixed_indices: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
    md_stride: int = 0,
    relax_stride: int = 0,
) -> tuple[Atoms, float, list[Atoms], list[Atoms]]:
    """
    Run one MH escape and return (new_min, energy, md_traj, relax_traj).

    Parameters
    ----------
    md_stride : int
        If > 0, snapshot every md_stride MD steps into md_traj.
        If 0 (default), md_traj is returned empty.
    relax_stride : int
        If > 0, snapshot every relax_stride BFGS steps into relax_traj.
        If 0 (default), relax_traj is returned empty.

    Returns
    -------
    new_min : ase.Atoms
        Relaxed minimum (energy stored in atoms.info["potential_energy"]).
    energy : float
        Potential energy at the minimum (eV).
    md_traj : list[Atoms]
        MD trajectory snapshots (energies in atoms.info; calculator
        does NOT survive).
    relax_traj : list[Atoms]
        BFGS trajectory snapshots (similarly tagged).
    """
    atoms = seed.copy()
    atoms.calc = calculator

    if fixed_indices:
        from ase.constraints import FixAtoms
        # Keep constraints inherited from the seed (e.g. Hookean surface walls)
        # instead of replacing them with FixAtoms.
        existing = list(atoms.constraints)
        if not any(isinstance(c, FixAtoms) for c in existing):
            existing.append(FixAtoms(indices=fixed_indices))
            atoms.set_constraint(existing)

    if rng is None:
        rng = np.random.default_rng()
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_escape, rng=rng)

    # --------- MD phase ---------
    md_traj: list[Atoms] = []
    dyn = VelocityVerlet(atoms, timestep=md_timestep_fs * units.fs)

    if md_stride > 0:
        # Capture frames at stride
        def _md_callback():
            try:
                E = float(atoms.get_potential_energy())
            except Exception:
                E = None
            snap = atoms.copy()
            if E is not None:
                snap.info["potential_energy"] = E
            md_traj.append(snap)

        # ASE's attach hooks are invoked every `interval` steps
        dyn.attach(_md_callback, interval=md_stride)

    dyn.run(md_steps)

    # --------- Relaxation phase ---------
    relax_traj: list[Atoms] = []
    opt = BFGS(atoms, logfile=None)

    if relax_stride > 0:
        def _relax_callback():
            try:
                E = float(atoms.get_potential_energy())
            except Exception:
                E = None
            snap = atoms.copy()
            if E is not None:
                snap.info["potential_energy"] = E
            relax_traj.append(snap)

        opt.attach(_relax_callback, interval=relax_stride)

    opt.run(fmax=relax_fmax, steps=relax_steps)

    energy = float(atoms.get_potential_energy())
    new_min = atoms.copy()
    new_min.info["potential_energy"] = energy

    return new_min, energy, md_traj, relax_traj