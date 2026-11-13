"""
Basin-confined walk phase.

Now optionally returns rejected trial structures alongside accepted
frames, so the dataset collector can capture high-energy / off-basin
configurations that are still useful for MLP training (they teach the
potential about the boundaries of physically reasonable configurations).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from .basin import adsorbate_distance, _adsorbate_indices


def walk_phase(
    m_t: Atoms,
    calculator: Calculator,
    slab_n_atoms: int,
    T_walk: float = 600.0,
    n_steps: int = 80,
    sigma: float = 0.15,
    dE_basin: float = 0.8,
    rng: Optional[np.random.Generator] = None,
    rejected_keep_ratio: float = 0.0,
) -> tuple[list[Atoms], float, float, list[Atoms]]:
    """
    Run one basin-confined walk.

    Parameters
    ----------
    rejected_keep_ratio : float in [0, 1]
        Fraction of rejected trials to retain (random sampling). 0 means
        keep none; 1 means keep all. Default 0.

    Returns
    -------
    frames : list of accepted Atoms (energy stored in atoms.info)
    rho : float, acceptance ratio
    R : float, mean adsorbate-RMSD of accepted frames vs m_t (Å)
    rejected_kept : list of randomly sampled rejected trial Atoms
        (energy stored in atoms.info)
    """
    if rng is None:
        rng = np.random.default_rng()

    ads_idx = _adsorbate_indices(m_t, slab_n_atoms)
    if len(ads_idx) == 0:
        return [], 0.0, 0.0, []

    current = m_t.copy()
    current.calc = calculator
    E_current = current.get_potential_energy()
    E_min = E_current

    kT = units.kB * T_walk

    accepted_frames: list[Atoms] = []
    rejected_kept: list[Atoms] = []
    n_accepted = 0

    for _ in range(n_steps):
        trial = current.copy()
        trial.calc = calculator

        i = ads_idx[rng.integers(len(ads_idx))]
        delta = rng.normal(0.0, sigma, size=3)
        pos = trial.get_positions()
        pos[i] += delta
        trial.set_positions(pos)

        try:
            E_trial = trial.get_potential_energy()
        except Exception:
            continue

        # Basin confinement
        if (E_trial - E_min) > dE_basin:
            # This trial is "out of basin" - sample a fraction for archive
            if rejected_keep_ratio > 0 and rng.random() < rejected_keep_ratio:
                rej = trial.copy()
                rej.info["potential_energy"] = E_trial
                rej.info["reject_reason"] = "out_of_basin"
                rejected_kept.append(rej)
            continue

        # Metropolis-style acceptance
        dE = E_trial - E_current
        if dE <= 0.0 or rng.random() < np.exp(-dE / kT):
            current = trial
            E_current = E_trial
            frame = current.copy()
            frame.info["potential_energy"] = E_trial
            accepted_frames.append(frame)
            n_accepted += 1
        else:
            # Metropolis rejected - sample a fraction for archive
            if rejected_keep_ratio > 0 and rng.random() < rejected_keep_ratio:
                rej = trial.copy()
                rej.info["potential_energy"] = E_trial
                rej.info["reject_reason"] = "metropolis"
                rejected_kept.append(rej)

    rho = n_accepted / max(n_steps, 1)

    if accepted_frames:
        R = float(np.mean([
            adsorbate_distance(f, m_t, slab_n_atoms)
            for f in accepted_frames
        ]))
    else:
        R = 0.0

    return accepted_frames, rho, R, rejected_kept