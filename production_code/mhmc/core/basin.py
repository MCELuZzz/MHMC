"""
Basin data structure, classification, and permutation-invariant distance.

This module defines what a "basin" is, how a newly relaxed minimum is
assigned to an existing or new basin, and how distances between basins
are computed in a way that respects the indistinguishability of
chemically equivalent adsorbate atoms.

Reference: Section 2.2 of the manuscript, basin classification rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment


# ============================================================
#  Permutation- and PBC-invariant distance between adsorbate sets
# ============================================================

def _adsorbate_indices(atoms: Atoms, slab_n_atoms: int) -> np.ndarray:
    """Indices of adsorbate atoms (assumed appended after slab atoms)."""
    return np.arange(slab_n_atoms, len(atoms))


def _mic_displacement(
    pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Minimum-image displacement pos_a - pos_b under PBC."""
    disp = pos_a - pos_b
    inv = np.linalg.inv(cell)
    frac = disp @ inv
    frac -= np.round(frac)
    return frac @ cell


def adsorbate_distance(
    atoms_a: Atoms,
    atoms_b: Atoms,
    slab_n_atoms: int,
) -> float:
    """
    Permutation- and PBC-invariant RMSD between two adsorbate configurations.

    For each chemical species, the optimal one-to-one assignment between
    atoms of `atoms_a` and `atoms_b` is found via the Hungarian algorithm
    on minimum-image displacements. The returned value is

        d = sqrt( (1/N_ads) * sum_k ||r_{a,k} - r_{b,pi(k)}||^2_MIC )

    Returns 0 if there are no adsorbates.
    """
    idx = _adsorbate_indices(atoms_a, slab_n_atoms)
    if len(idx) == 0:
        return 0.0

    cell = atoms_a.get_cell().array
    pa = atoms_a.get_positions()[idx]
    pb = atoms_b.get_positions()[idx]
    sa = np.array(atoms_a.get_chemical_symbols())[idx]
    sb = np.array(atoms_b.get_chemical_symbols())[idx]

    if sorted(sa.tolist()) != sorted(sb.tolist()):
        # Different composition → not the same physical state.
        return np.inf

    sq_total = 0.0
    n_total = 0
    for sym in np.unique(sa):
        mask_a = sa == sym
        mask_b = sb == sym
        if mask_a.sum() != mask_b.sum():
            return np.inf

        pa_s = pa[mask_a]
        pb_s = pb[mask_b]
        n = len(pa_s)

        # Pairwise squared distances (with MIC) between every (a,b) pair
        sq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d = _mic_displacement(pa_s[i], pb_s[j], cell)
                sq[i, j] = float(np.dot(d, d))

        # Hungarian: minimize total squared distance for this species
        row, col = linear_sum_assignment(sq)
        sq_total += float(sq[row, col].sum())
        n_total += n

    return float(np.sqrt(sq_total / n_total))


# ============================================================
#  Basin object
# ============================================================

@dataclass
class Basin:
    """A basin on the PES, identified by its representative minimum.
    
    The representative is updated whenever a lower-energy minimum is
    classified into this basin, ensuring `representative` and `energy`
    always reflect the deepest known point of the basin.
    """
    id: int
    representative: Atoms
    energy: float
    discovery_iter: int
    members: list[Atoms] = field(default_factory=list)
    n_visits: int = 0
    n_upgrades: int = 0          # how many times representative was replaced
    energy_history: list[float] = field(default_factory=list)
                                  # all energies ever assigned to this basin
                                  # (for diagnostics and 3.2 ablation analysis)

    def add_members(self, frames: list[Atoms]) -> None:
        self.members.extend(frames)

    def upgrade_representative(self, new_min: Atoms, new_energy: float) -> None:
        """Replace the basin's representative with a lower-energy minimum.
        
        Members are preserved; only the reference point changes.
        """
        self.representative = new_min.copy()
        self.energy = new_energy
        self.n_upgrades += 1

# ============================================================
#  Basin database
# ============================================================

class BasinDatabase:
    """
    Holds all known basins and assigns new minima to existing basins
    or creates new ones.

    Parameters
    ----------
    slab_n_atoms : int
        Number of slab atoms (assumed to occupy the first slab_n_atoms
        indices in every Atoms object). Adsorbate atoms occupy the rest.
    eps_E : float
        Energy tolerance (eV) for basin identity.
    eps_R : float
        Adsorbate-RMSD tolerance (Å) for basin identity.
    """

    def __init__(
        self,
        slab_n_atoms: int,
        eps_E: float = 0.05,
        eps_R: float = 0.4,
    ):
        self.slab_n_atoms = slab_n_atoms
        self.eps_E = eps_E
        self.eps_R = eps_R
        self.basins: list[Basin] = []
        self._next_id = 0

    def classify(
        self,
        new_min: Atoms,
        energy: float,
        current_iter: int,
    ) -> tuple[Basin, bool]:
        """
        Assign `new_min` to an existing basin or create a new one.

        Two minima belong to the same basin iff
            |E_i - E_j| < eps_E   AND   d_ads(m_i, m_j) < eps_R.

        When `new_min` is matched to an existing basin AND has a lower
        energy than that basin's current representative, the basin is
        UPGRADED: its `representative` and `energy` are replaced. Existing
        members are preserved. This ensures `basin.representative` always
        reflects the deepest known point of the basin, which is critical
        for downstream walk/feedback consistency.

        Returns
        -------
        basin : the matched (or newly created) Basin
        is_new : True iff a new Basin was created
        """
        # Energy pre-filter for speed
        candidates = [b for b in self.basins
                      if abs(b.energy - energy) < self.eps_E]

        for b in candidates:
            d = adsorbate_distance(new_min, b.representative, self.slab_n_atoms)
            if d < self.eps_R:
                b.n_visits += 1
                b.energy_history.append(energy)
                # Upgrade if the new minimum is deeper
                if energy < b.energy:
                    b.upgrade_representative(new_min, energy)
                return b, False

        # No match → new basin
        b = Basin(
            id=self._next_id,
            representative=new_min.copy(),
            energy=energy,
            discovery_iter=current_iter,
            n_visits=1,
            energy_history=[energy],
        )
        self._next_id += 1
        self.basins.append(b)
        return b, True

    def __len__(self) -> int:
        return len(self.basins)

    def summary(self) -> dict:
        return {
            "n_basins": len(self.basins),
            "energy_range": (
                (min(b.energy for b in self.basins),
                 max(b.energy for b in self.basins))
                if self.basins else (None, None)
            ),
            "total_members": sum(len(b.members) for b in self.basins),
        }