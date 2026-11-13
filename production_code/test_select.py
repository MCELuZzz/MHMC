"""
Test selection pipeline end-to-end on a real candidate pool.

Usage:
    python test_selection.py path/to/candidate_pool.extxyz
"""

import sys
import json
from pathlib import Path

from ase.io import read, write

from mhmc.selection import (
    compute_soap_descriptors,
    KPCAReducer,
    stratified_fps_selection,
)


def main(pool_path: str, n_select: int = 50):
    print(f"\n{'='*70}")
    print(f"Loading candidate pool from {pool_path}")
    print(f"{'='*70}")

    atoms_list = read(pool_path, index=":")
    print(f"Loaded {len(atoms_list)} configurations")

    # Extract source tags from atoms.info
    source_tags = [a.info.get("source", "unknown") for a in atoms_list]
    from collections import Counter
    src_counter = Counter(source_tags)
    print(f"Source distribution:")
    for src, n in sorted(src_counter.items(), key=lambda x: -x[1]):
        print(f"  {src:20s}: {n}")

    # 1. SOAP descriptors
    print(f"\n{'-'*70}")
    print("Step 1: Compute SOAP descriptors")
    print(f"{'-'*70}")
    descriptors = compute_soap_descriptors(atoms_list, n_jobs=1)
    print(f"SOAP feature shape: {descriptors.shape}")

    # 2. KPCA reduction
    print(f"\n{'-'*70}")
    print("Step 2: KPCA dimensionality reduction")
    print(f"{'-'*70}")
    reducer = KPCAReducer(variance_threshold=0.95, max_components=50)
    reduced = reducer.fit_transform(descriptors)
    print(f"Reduced shape: {reduced.shape}")

    # 3. Stratified FPS
    print(f"\n{'-'*70}")
    print(f"Step 3: Stratified FPS (target {n_select})")
    print(f"{'-'*70}")
    result = stratified_fps_selection(
        features=reduced,
        source_tags=source_tags,
        n_select_total=n_select,
        seed=42,
    )
    print(f"\nFinal per-source breakdown:")
    for src, n in result["per_source_counts"].items():
        print(f"  {src:20s}: {n}")

    # Save selected configurations
    selected_atoms = [atoms_list[i] for i in result["selected_indices"]]
    out_path = "selected_test.extxyz"
    write(out_path, selected_atoms)
    print(f"\nSaved {len(selected_atoms)} selected configurations → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_selection.py path/to/candidate_pool.extxyz [n_select]")
        sys.exit(1)
    pool = sys.argv[1]
    n_sel = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(pool, n_sel)