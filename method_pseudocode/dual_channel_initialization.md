# Pseudocode: geometry-aware dual-channel initialization

The production implementation of the dual-channel initialization procedure is retained in the in-house code base pending future software release. This document summarizes the reproducible algorithmic procedure.

## Channel 1: high-symmetry site initialization

```text
1. Read the clean catalyst slab.
2. Identify the top surface layer from atomic coordinates.
3. Project surface atoms onto the surface plane.
4. Replicate projected surface atoms across periodic boundaries to avoid edge artifacts.
5. Apply two-dimensional Delaunay triangulation.
6. Construct candidate adsorption sites:
   - top sites from surface atoms;
   - bridge sites from Delaunay edges;
   - hollow sites from Delaunay triangles.
7. Filter duplicate or symmetry-equivalent sites using geometric tolerances.
8. Place adsorbates at the candidate sites with chemically reasonable heights and orientations.
```

## Channel 2: near-surface grid initialization

```text
1. Define a three-dimensional near-surface sampling region above the catalyst.
2. Generate a quasi-uniform grid in the near-surface region.
3. Apply a height-dependent acceptance function so that points near the active surface are sampled more densely.
4. Remove grid points that clash with slab atoms or previously placed adsorbates using covalent-radius criteria.
5. Place adsorbates at accepted off-symmetry grid points with sampled orientations.
```

## Coadsorption assembly

```text
1. Pool candidate sites from the high-symmetry and grid channels.
2. Greedily assemble coadsorption configurations under pairwise clash constraints.
3. Canonicalize each configuration to remove duplicates caused by translation or permutation of identical adsorbates.
4. Export the resulting initial structures for DFT labeling or preliminary MLIP training.
```

Representative initialized structures are provided in `data/structures/initialization/`.
