"""
Reference implementation: dual-channel initialization building blocks.

This file is included to make the public reproducibility repository more
executable. It demonstrates the core ideas of the two-channel initialization
strategy: Delaunay high-symmetry site identification, near-surface grid
sampling, clash filtering, and duplicate removal.

It is not the full production initialization/orchestration code used in the
internal workflow; system-specific molecule templates, production batch
management, VASP/MLIP job orchestration, and unpublished software-release
components are intentionally not included.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from scipy.spatial import cKDTree
from typing import Optional

try:
    from .delaunay_sites import find_adsorption_sites
    from .grid_sampling import GridSampler
except ImportError:  # allow running examples directly from this folder
    from delaunay_sites import find_adsorption_sites
    from grid_sampling import GridSampler


# ---------------------------------------------------------------------------
# Element-specific minimum distance (covalent-radii based)
# ---------------------------------------------------------------------------
def _min_dist(sym1: str, sym2: str, scale: float = 0.75) -> float:
    """Minimum allowed interatomic distance based on covalent radii.

    Two atoms are accepted as non-clashing if their distance exceeds
        scale * (r_cov(sym1) + r_cov(sym2)).
    Default scale=0.75 follows pymatgen / ASE conventions for "close contact".
    """
    r1 = covalent_radii[atomic_numbers[sym1]]
    r2 = covalent_radii[atomic_numbers[sym2]]
    return scale * (r1 + r2)


class DualChannelInitializer:
    """
    Generate the initial dataset of co-adsorption configurations using the
    physics-informed dual-channel strategy described in manuscript Section 2.1.
    """

    def __init__(
        self,
        slab: Atoms,
        d_top: float = 1.5,
        d_bridge: float = 1.3,
        d_hollow: float = 1.2,
        alpha: float = 0.5,
        epsilon: float = 0.15,
        z_min_offset: float = 1.2,
        z_max_offset: float = 4.0,
        grid_density: float = 0.4,
        seed: Optional[int] = None,
    ):
        self.slab = slab.copy()
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # ---- Channel 1: Delaunay-based site identification ----
        self.delaunay_sites = find_adsorption_sites(
            slab,
            d_top=d_top,
            d_bridge=d_bridge,
            d_hollow=d_hollow,
        )

        # ---- Channel 2: Z-biased 3D grid sampler ----
        self.grid_sampler = GridSampler(
            alpha=alpha,
            epsilon=epsilon,
            seed=seed,
        )
        self.grid_points = self.grid_sampler.sample_grid_points(
            slab,
            z_min_offset=z_min_offset,
            z_max_offset=z_max_offset,
            grid_density=grid_density,
        )

        # ---- Combined candidate pool ----
        self.candidate_pool = np.vstack([
            self.delaunay_sites["top"],
            self.delaunay_sites["bridge"],
            self.delaunay_sites["hollow"],
            self.grid_points,
        ])
# ---- Save channel-level statistics for diagnostics ----
        self.stats = {
            "n_top":    len(self.delaunay_sites["top"]),
            "n_bridge": len(self.delaunay_sites["bridge"]),
            "n_hollow": len(self.delaunay_sites["hollow"]),
            "n_channel1_total": (
                len(self.delaunay_sites["top"])
                + len(self.delaunay_sites["bridge"])
                + len(self.delaunay_sites["hollow"])
            ),
            "n_channel2_grid": len(self.grid_points),
            "n_candidate_pool": len(self.candidate_pool),
        }
        # ---- Persistent record of generated co-adsorption configurations,
        # using a permutation- and PBC-invariant structure hash ----
        self._struct_hash_history: set = set()

    # ------------------------------------------------------------------
    # Permutation- and PBC-invariant structure hash
    # ------------------------------------------------------------------
    def _hash_structure(
        self, positions: np.ndarray, symbols: list[str]
    ) -> tuple:
        """
        Hash a co-adsorption configuration in a way that is invariant to:
          (1) The order in which adsorbate atoms are placed (permutation
              invariance over atoms of the same element).
          (2) Rigid in-plane translation by lattice vectors (PBC invariance):
              we shift by the mean position of all adsorbates before hashing.

        Note: this hash is *not* invariant to point-group symmetries of the
        slab (e.g., mirror planes). For most catalytic applications, treating
        such mirror images as distinct is acceptable because adsorbate
        chirality on chiral or stepped surfaces matters. Adding point-group
        canonicalization is a future extension.
        """
        positions = np.asarray(positions)
        com = positions.mean(axis=0)
        rel = positions - com  # remove rigid translation

        per_atom = []
        for sym, p in zip(symbols, rel):
            cell = (
                int(np.round(p[0] / self.epsilon)),
                int(np.round(p[1] / self.epsilon)),
                int(np.round(p[2] / self.epsilon)),
            )
            per_atom.append((sym, cell))

        per_atom.sort()  # permutation invariance over identical species
        return tuple(per_atom)

    # ------------------------------------------------------------------
    # Build co-adsorption configurations
    # ------------------------------------------------------------------
    def generate_configurations(
        self,
        adsorbates: list[str],
        n_configs: int,
        clash_scale: float = 0.75,
        max_trials_factor: int = 200,
        verbose: bool = True,
    ) -> tuple[list[Atoms], dict]:
        """
        Generate `n_configs` distinct co-adsorption configurations.

        Returns
        -------
        configs : list of ase.Atoms
        diagnostics : dict with keys
            'n_target'         : requested number
            'n_generated'      : actually generated unique configurations
            'n_trials'         : total trial attempts
            'n_clash_failures' : trials rejected due to clash failures
            'n_hash_collisions': trials rejected as duplicates of prior configs
            'success_rate'     : n_generated / n_trials
        """
        slab_pos = self.slab.get_positions()
        slab_syms = self.slab.get_chemical_symbols()
        slab_tree = cKDTree(slab_pos)

        slab_clash_radii = {
            ads: max(_min_dist(ads, s, clash_scale) for s in set(slab_syms))
            for ads in set(adsorbates)
        }

        configs: list[Atoms] = []
        n_clash_failures = 0
        n_hash_collisions = 0
        trials = 0
        max_trials = n_configs * max_trials_factor

        while len(configs) < n_configs and trials < max_trials:
            trials += 1
            order = self.rng.permutation(len(self.candidate_pool))
            shuffled = self.candidate_pool[order]

            placed_pos: list[np.ndarray] = []
            placed_syms: list[str] = []
            success = True

            for ads in adsorbates:
                clash_with_slab = slab_clash_radii[ads]
                placed = False
                for pt in shuffled:
                    d_slab, _ = slab_tree.query(pt, k=1)
                    if d_slab < clash_with_slab:
                        continue
                    bad = False
                    for q, qsym in zip(placed_pos, placed_syms):
                        if np.linalg.norm(pt - q) < _min_dist(ads, qsym, clash_scale):
                            bad = True
                            break
                    if bad:
                        continue
                    placed_pos.append(pt)
                    placed_syms.append(ads)
                    placed = True
                    break

                if not placed:
                    success = False
                    break

            if not success:
                n_clash_failures += 1
                continue

            h = self._hash_structure(placed_pos, placed_syms)
            if h in self._struct_hash_history:
                n_hash_collisions += 1
                continue
            self._struct_hash_history.add(h)

            new_atoms = self.slab.copy()
            for sym, pos in zip(placed_syms, placed_pos):
                new_atoms.append(Atoms(sym, positions=[pos])[0])
            configs.append(new_atoms)

        diagnostics = {
            "n_target":          n_configs,
            "n_generated":       len(configs),
            "n_trials":          trials,
            "n_clash_failures":  n_clash_failures,
            "n_hash_collisions": n_hash_collisions,
            "success_rate":      len(configs) / max(trials, 1),
        }

        if verbose:
            print(f"  [generate_configurations] "
                  f"target={n_configs}, generated={len(configs)}, "
                  f"trials={trials}, clash_fail={n_clash_failures}, "
                  f"hash_collisions={n_hash_collisions}")

        return configs, diagnostics

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     from ase.build import bcc110
#     from ase.io import write

#     slab = bcc110("Fe", size=(4, 4, 3), vacuum=10.0)

#     initializer = DualChannelInitializer(
#         slab=slab,
#         d_top=1.5, d_bridge=1.3, d_hollow=1.2,
#         alpha=0.5, epsilon=0.15,
#         z_min_offset=1.2, z_max_offset=4.0, grid_density=0.4,
#         seed=42,
#     )

#     print(f"Channel-1 sites: top={len(initializer.delaunay_sites['top'])}, "
#           f"bridge={len(initializer.delaunay_sites['bridge'])}, "
#           f"hollow={len(initializer.delaunay_sites['hollow'])}")
#     print(f"Channel-2 grid points: {len(initializer.grid_points)}")
#     print(f"Total candidate pool size: {len(initializer.candidate_pool)}")

#     configs = initializer.generate_configurations(
#         adsorbates=["H", "H", "N", "N"],
#         n_configs=80,
#     )

#     print(f"Generated {len(configs)} unique co-adsorption configurations.")
#     write("Fe110_init_HHNN_n80.extxyz", configs)