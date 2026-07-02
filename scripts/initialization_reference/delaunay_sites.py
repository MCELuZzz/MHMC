"""
Channel 1: Physics-informed adsorption site identification via Delaunay triangulation.

This module identifies top, bridge, and hollow sites on a catalyst surface using
2D Delaunay triangulation of surface atoms, with proper periodic boundary
condition (PBC) handling and degenerate triangle filtering.

Reference: Section 2.1 of the manuscript, eqs. (1)-(3).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay
from typing import Optional


# Default adsorption distances (Å) along surface normal for common adsorbate atoms.
# These are typical values; users can override via the `d_offset` parameter.
DEFAULT_D_OFFSET = {
    "H": 1.0, "D": 1.0,
    "C": 1.5, "N": 1.5, "O": 1.3,
    "S": 1.7, "F": 1.3, "Cl": 1.7,
}


def _identify_surface_atoms(
    atoms: Atoms, z_tolerance: float = 0.5
) -> np.ndarray:
    """
    Identify the indices of atoms in the topmost surface layer.

    Uses a tolerance window below the maximum z-coordinate, so that small
    relaxations or numerical noise do not exclude valid surface atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        Catalyst slab.
    z_tolerance : float
        Atoms within `z_tolerance` Å below z_max are considered surface atoms.

    Returns
    -------
    surface_indices : np.ndarray of int
    """
    z = atoms.get_positions()[:, 2]
    z_max = z.max()
    return np.where(z >= z_max - z_tolerance)[0]


def _replicate_2d_pbc(
    points_2d: np.ndarray, cell_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replicate 2D points by 3x3 in-plane periodic images.

    This is required so that Delaunay triangulation does not miss bridge/hollow
    sites at cell boundaries. After triangulation, only sites whose centroids
    fall inside the central cell are kept.

    Parameters
    ----------
    points_2d : (N, 2) array
        Surface atom positions projected to xy-plane.
    cell_2d : (2, 2) array
        In-plane lattice vectors as rows: [[a_x, a_y], [b_x, b_y]].

    Returns
    -------
    extended_points : (9N, 2)
    parent_indices  : (9N,) int — index in the original points_2d for each replica.
    """
    a_vec, b_vec = cell_2d[0], cell_2d[1]
    extended, parent = [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = i * a_vec + j * b_vec
            extended.append(points_2d + shift)
            parent.extend(range(len(points_2d)))
    return np.vstack(extended), np.asarray(parent, dtype=int)


def _is_inside_cell_2d(point_2d: np.ndarray, cell_2d: np.ndarray) -> bool:
    """Check whether a 2D point lies inside the central cell [0, a) x [0, b)
    in fractional coordinates."""
    inv = np.linalg.inv(cell_2d.T)  # transforms (x, y) -> (frac_a, frac_b)
    frac = inv @ point_2d
    eps = 1e-9
    return (-eps <= frac[0] < 1.0 - eps) and (-eps <= frac[1] < 1.0 - eps)


def _triangle_min_angle_deg(p1, p2, p3) -> float:
    """Return the smallest interior angle of the triangle (p1, p2, p3) in degrees.

    Used to filter out degenerate (near-collinear) triangles that arise on
    quasi-square lattices (e.g., bcc(110), fcc(100)) where Delaunay arbitrarily
    splits a square into two triangles, producing physically equivalent
    pseudo-hollow sites.
    """
    def angle(a, b, c):
        # angle at vertex b
        v1 = a - b
        v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return min(angle(p1, p2, p3), angle(p2, p3, p1), angle(p3, p1, p2))


def find_adsorption_sites(
    slab: Atoms,
    d_top: float = 1.5,
    d_bridge: float = 1.3,
    d_hollow: float = 1.2,
    z_tolerance: float = 0.5,
    min_triangle_angle: float = 20.0,
    use_pbc: bool = True,
) -> dict[str, np.ndarray]:
    """
    Identify high-symmetry adsorption sites on a catalyst surface.

    Implementation of Section 2.1, eqs. (1)-(3):
        s_top    = p_i + d * n
        s_bridge = (p_i + p_j) / 2 + d' * n
        s_hollow = (p_i + p_j + p_k) / 3 + d'' * n

    Parameters
    ----------
    slab : ase.Atoms
        Catalyst slab. The surface is assumed to be normal to z (i.e., the
        topmost atomic layer is identified by largest z-coordinate).
    d_top, d_bridge, d_hollow : float
        Vertical offset (Å) above each site type along the surface normal.
    z_tolerance : float
        Atoms within this distance below z_max are treated as surface atoms.
    min_triangle_angle : float
        Triangles with any interior angle below this threshold (degrees) are
        considered degenerate and discarded. Acts as the "orthogonal angle-
        checking penalty" of the manuscript.
    use_pbc : bool
        If True, replicate surface atoms into a 3x3 supercell before
        triangulation, so that boundary bridge/hollow sites are not missed.

    Returns
    -------
    sites : dict with keys 'top', 'bridge', 'hollow', each value is an (M, 3)
        array of Cartesian coordinates.
    """
    # ---- 1. Identify surface atoms ----
    surf_idx = _identify_surface_atoms(slab, z_tolerance=z_tolerance)
    surf_pos = slab.get_positions()[surf_idx]      # (N_surf, 3)
    z_surface = surf_pos[:, 2].mean()              # mean z of top layer

    # ---- 2. Build 2D point set (with PBC replication if requested) ----
    points_2d = surf_pos[:, :2]
    cell = slab.get_cell().array
    cell_2d = np.array([cell[0][:2], cell[1][:2]])

    if use_pbc:
        ext_points, ext_parent = _replicate_2d_pbc(points_2d, cell_2d)
    else:
        ext_points = points_2d
        ext_parent = np.arange(len(points_2d))

    # ---- 3. Top sites: simply the surface atoms in the central cell ----
    n_hat = np.array([0.0, 0.0, 1.0])  # surface normal (z-axis)
    top_sites = surf_pos.copy()
    top_sites[:, 2] = z_surface + d_top  # snap z to surface mean + offset

    # ---- 4. Delaunay triangulation on the extended 2D set ----
    tri = Delaunay(ext_points)

    # ---- 5. Bridge sites: midpoints of edges, deduped by midpoint location ----
    bridge_list = []
    seen_midpoints = []  # store accepted midpoints for geometric dedup
    midpoint_dedup_tol = 0.1  # Å

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p_i_2d = ext_points[simplex[i]]
                p_j_2d = ext_points[simplex[j]]
                midpoint_2d = 0.5 * (p_i_2d + p_j_2d)

                # Only keep midpoints that fall inside the central cell
                if not _is_inside_cell_2d(midpoint_2d, cell_2d):
                    continue

                # Geometric dedup: skip if a previously accepted midpoint
                # is within the tolerance
                duplicate = False
                for m in seen_midpoints:
                    if np.linalg.norm(midpoint_2d - m) < midpoint_dedup_tol:
                        duplicate = True
                        break
                if duplicate:
                    continue

                seen_midpoints.append(midpoint_2d)
                bridge_list.append(np.array([
                    midpoint_2d[0], midpoint_2d[1], z_surface + d_bridge
                ]))

    # ---- 6. Hollow sites: triangle centroids, with degenerate-triangle filter ----
    hollow_list = []
    seen_centroids = []
    centroid_dedup_tol = 0.1  # Å

    for simplex in tri.simplices:
        p1, p2, p3 = ext_points[simplex]
        if _triangle_min_angle_deg(p1, p2, p3) < min_triangle_angle:
            continue
        centroid_2d = (p1 + p2 + p3) / 3.0

        if not _is_inside_cell_2d(centroid_2d, cell_2d):
            continue

        duplicate = False
        for c in seen_centroids:
            if np.linalg.norm(centroid_2d - c) < centroid_dedup_tol:
                duplicate = True
                break
        if duplicate:
            continue

        seen_centroids.append(centroid_2d)
        hollow_list.append(np.array([
            centroid_2d[0], centroid_2d[1], z_surface + d_hollow
        ]))
    # ---- 7. De-duplicate hollow/bridge sites (different triangles can yield
    # very close centroids on near-symmetric lattices) ----
    def _dedup(arr_list, tol=0.1):
        if not arr_list:
            return np.zeros((0, 3))
        arr = np.array(arr_list)
        keep = []
        for p in arr:
            if not any(np.linalg.norm(p - q) < tol for q in keep):
                keep.append(p)
        return np.array(keep)

    return {
        "top":    top_sites,
        "bridge": np.array(bridge_list) if bridge_list else np.zeros((0, 3)),
        "hollow": np.array(hollow_list) if hollow_list else np.zeros((0, 3)),
    }


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ase.build import bcc110, fcc111, fcc100

    for name, builder in [
        ("Fe(110)",  lambda: bcc110("Fe", size=(4, 4, 3), vacuum=10.0)),
        ("Cu(111)",  lambda: fcc111("Cu", size=(4, 4, 3), vacuum=10.0)),
        ("Pd(100)",  lambda: fcc100("Pd", size=(4, 4, 3), vacuum=10.0)),
    ]:
        slab = builder()
        sites = find_adsorption_sites(slab)
        print(f"{name}: top={len(sites['top'])}, "
              f"bridge={len(sites['bridge'])}, hollow={len(sites['hollow'])}")