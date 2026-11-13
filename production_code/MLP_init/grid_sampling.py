"""
Channel 2: Adaptive 3D grid sampling with Z-axis exponential bias.

This module generates candidate adsorbate positions by sampling a 3D grid above
the catalyst surface, with sampling density decaying exponentially with height.
A precision-based hashing scheme prevents redundant grid points across
sampling rounds.

Reference: Section 2.1 of the manuscript, eq. (4): W(z) = exp(-alpha (z - z_surf)).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from typing import Optional


def _get_surface_z(atoms: Atoms, top_percentile: float = 90.0) -> float:
    """Return the average z of the topmost atomic layer.

    Robust to small surface relaxations: takes the mean over atoms whose z
    falls in the top `top_percentile` percentile.
    """
    z = atoms.get_positions()[:, 2]
    cutoff = np.percentile(z, top_percentile)
    return float(np.mean(z[z >= cutoff]))


class GridSampler:
    """
    Z-axis biased 3D grid sampler with precision hashing for de-duplication.

    Parameters
    ----------
    alpha : float
        Decay rate (Å^-1) of the Z-direction acceptance probability:
            W(z) = exp(-alpha * (z - z_surf))
        Larger alpha biases sampling more strongly toward the near-surface
        region. Typical: 0.5–1.0.
    epsilon : float
        Spatial hashing precision (Å). Two grid points within `epsilon` of
        each other (in all three axes simultaneously) are treated as duplicates.
        Should be larger than DFT geometric noise (~0.01 Å) and smaller than
        physically meaningful displacement (~0.5 Å). Default 0.15 Å.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Persistent history across rounds (allows incremental sampling).
        self._grid_hash_history: set[tuple[int, int, int]] = set()

    # ----------------------------------------------------------------------
    # Hashing
    # ----------------------------------------------------------------------
    def _hash_point(self, point: np.ndarray) -> tuple[int, int, int]:
        """Discretize a 3D point onto an epsilon-resolution grid:
            H(x, y, z) = (round(x / eps), round(y / eps), round(z / eps))
        """
        return tuple(int(np.round(p / self.epsilon)) for p in point)

    # ----------------------------------------------------------------------
    # Core sampling
    # ----------------------------------------------------------------------
    def sample_grid_points(
        self,
        slab: Atoms,
        z_min_offset: float = 1.2,
        z_max_offset: float = 4.0,
        grid_density: float = 0.4,
    ) -> np.ndarray:
        """
        Generate candidate grid points above the slab surface.

        Parameters
        ----------
        slab : ase.Atoms
            Catalyst slab.
        z_min_offset, z_max_offset : float
            Minimum and maximum height (Å) above the surface mean z to sample.
            Default range [1.2, 4.0] covers typical chemisorption distances
            (1.0–2.0 Å) and physisorption / molecular precursors (2–4 Å).
        grid_density : float
            Grid spacing (Å) in all three directions. Smaller = denser.

        Returns
        -------
        accepted_points : (M, 3) array of Cartesian coordinates.
        """
        cell = slab.get_cell().array
        ax, ay = cell[0][0], cell[1][1]   # assumes orthorhombic in-plane cell
        z_surf = _get_surface_z(slab)
        z_lo = z_surf + z_min_offset
        z_hi = z_surf + z_max_offset

        # Build the regular grid
        nx = max(int(ax / grid_density), 2)
        ny = max(int(ay / grid_density), 2)
        nz = max(int((z_hi - z_lo) / grid_density), 2)
        xs = np.linspace(0.0, ax, nx, endpoint=False)
        ys = np.linspace(0.0, ay, ny, endpoint=False)
        zs = np.linspace(z_lo, z_hi, nz)

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        all_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Z-axis biased acceptance: W(z) = exp(-alpha (z - z_surf))
        weights = np.exp(-self.alpha * (all_points[:, 2] - z_surf))
        weights = np.clip(weights, 0.0, 1.0)
        rand = self.rng.random(len(all_points))
        accepted_mask = rand <= weights

        # Precision hashing: drop points that hash to a previously seen cell
        accepted: list[np.ndarray] = []
        for pt, ok in zip(all_points, accepted_mask):
            if not ok:
                continue
            h = self._hash_point(pt)
            if h in self._grid_hash_history:
                continue
            self._grid_hash_history.add(h)
            accepted.append(pt)

        return np.array(accepted) if accepted else np.zeros((0, 3))