#!/usr/bin/env python
"""
prepare_layerB.py — Layer B dataset preparation for Section 3.2.2

Given the output of `run_ablation.py`, this script:

  1. Combines per-seed candidate pools into per-mode pools (3 seeds → 1 pool per mode)
  2. Splits one mode's pool (default: coupled) into TRAIN + HOLDOUT partitions
     so the training and test sets cannot overlap
  3. FPS-selects N_train frames from each training pool (for retraining)
  4. FPS-selects N_holdout frames from the holdout partition (for the test set)
  5. Generates VASP input directories for every selected frame

After this script:
  USER runs VASP on each {output}/train_<mode>/vasp_inputs/ and
       {output}/holdout_<mode>/vasp_inputs/
  Then runs assemble_holdout.py to combine the labeled holdout with frames
  from the existing valid set into the final test set.

Usage
-----
    python prepare_layerB.py configs/al_loop.yaml ablation_results_Hookean/
    python prepare_layerB.py configs/al_loop.yaml ablation_results_Hookean/ \\
        --train-modes mh_only coupled \\
        --n-train 200 \\
        --n-holdout 50 \\
        --holdout-source coupled \\
        --output-dir layerB/

Design choices
--------------
* Holdout partition is set aside BEFORE training FPS so the training and
  test sets are mechanically disjoint, not just statistically.
* The split is a random hold-out fraction (default 10%) of the holdout-source
  pool's frames. Training FPS runs on the remaining 90%.
* The same SOAP+KPCA pipeline used in run_al.py is reused here for
  consistency with Section 2.3 — but stratified FPS quotas are renormalized
  to whatever sources actually exist in each pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from ase.io import read as ase_read, write as ase_write

from mhmc.selection.soap_features import compute_soap_descriptors
from mhmc.selection.kpca_reducer import KPCAReducer
from mhmc.selection.stratified_fps import stratified_fps_selection
from mhmc.al_loop.vasp_inputs import prepare_vasp_inputs


# ============================================================
#  Pool combination across seeds
# ============================================================

def combine_seeds_into_pool(
    ablation_root: Path,
    mode: str,
    output_path: Path,
) -> int:
    """Concatenate all <ablation_root>/<mode>_seed*/candidate_pool.extxyz
    into a single per-mode pool. Returns the number of frames written.
    """
    all_frames = []
    cells = sorted(ablation_root.glob(f"{mode}_seed*"))
    if not cells:
        raise FileNotFoundError(
            f"No cells matching {mode}_seed* in {ablation_root}"
        )
    for cell_dir in cells:
        pool = cell_dir / "candidate_pool.extxyz"
        if not pool.is_file() or pool.stat().st_size == 0:
            print(f"  WARN: {pool} missing/empty, skipping")
            continue
        frames = ase_read(pool, index=":")
        # Tag each frame with its source cell for traceability
        for f in frames:
            f.info["source_cell"] = cell_dir.name
        all_frames.extend(frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if all_frames:
        ase_write(output_path, all_frames)
    print(f"  {mode}: {len(all_frames)} frames combined → {output_path.name}")
    return len(all_frames)


# ============================================================
#  Random split for train/holdout partition
# ============================================================

def random_partition_pool(
    pool_path: Path,
    holdout_fraction: float,
    seed: int,
    out_train: Path,
    out_holdout: Path,
) -> tuple[int, int]:
    """Randomly split a pool into TRAIN (1-f) and HOLDOUT (f) partitions.
    Returns (n_train, n_holdout)."""
    frames = ase_read(pool_path, index=":")
    n = len(frames)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_holdout = max(1, int(round(n * holdout_fraction)))
    holdout_set = set(perm[:n_holdout].tolist())

    train_frames = [f for i, f in enumerate(frames) if i not in holdout_set]
    holdout_frames = [f for i, f in enumerate(frames) if i in holdout_set]

    ase_write(out_train, train_frames)
    ase_write(out_holdout, holdout_frames)
    print(f"  split (seed={seed}, frac={holdout_fraction:.0%}): "
          f"{n} → {len(train_frames)} train + {len(holdout_frames)} holdout")
    return len(train_frames), len(holdout_frames)


# ============================================================
#  FPS selection on one pool + VASP input generation
# ============================================================

def fps_select_and_dump_vasp(
    pool_path: Path,
    n_select: int,
    output_dir: Path,
    al_cfg: dict,
    seed: int = 0,
) -> int:
    """Load the pool, run SOAP+KPCA+stratified FPS, write selected.extxyz,
    and generate VASP input directories.

    Parameters
    ----------
    pool_path : Path
        extxyz file containing the candidate pool.
    n_select : int
        Number of frames to FPS-select.
    output_dir : Path
        Output directory; will contain selected.extxyz and vasp_inputs/.
    al_cfg : dict
        Parsed al_loop.yaml — used for SOAP / KPCA / FPS / VASP settings.
    seed : int
        Seed for FPS reproducibility.

    Returns
    -------
    n_actually_selected : int
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = ase_read(pool_path, index=":")
    print(f"    loaded {len(frames)} frames")

    if len(frames) == 0:
        print(f"    SKIP: empty pool")
        return 0

    if len(frames) <= n_select:
        print(f"    pool ({len(frames)}) ≤ n_select ({n_select}): taking all")
        selected = list(frames)
        selected_indices = list(range(len(frames)))
        per_source_counts = {}
    else:
        # SOAP
        soap_cfg = al_cfg.get("selection", {}).get("soap", {})
        # Separate soap hyperparameters from non-hyperparameter keys
        _SOAP_HYPER_KEYS = {"r_cut", "n_max", "l_max", "sigma", "average"}
        soap_params = {k: v for k, v in soap_cfg.items()
                       if k in _SOAP_HYPER_KEYS}
        print(f"    computing SOAP descriptors...")
        soap = compute_soap_descriptors(
            frames,
            species=soap_cfg.get("species"),
            soap_params=soap_params or None,
            n_jobs=soap_cfg.get("n_jobs", -1),
            verbose=False,
        )

        # KPCA
        kpca_cfg = al_cfg.get("selection", {}).get("kpca", {})
        print(f"    fitting KPCA on {soap.shape[0]}x{soap.shape[1]} matrix...")
        kpca = KPCAReducer(
            variance_threshold=kpca_cfg.get("variance_threshold", 0.95),
            max_components=kpca_cfg.get("max_components", 50),
            random_state=seed,
        )
        proj = kpca.fit_transform(soap, verbose=False)
        print(f"    KPCA → {proj.shape[1]} components")

        # Stratified FPS by source tag
        source_tags = [f.info.get("source", "unknown") for f in frames]
        weights = al_cfg.get("selection", {}).get("source_weights")
        print(f"    stratified FPS to {n_select}...")
        fps_result = stratified_fps_selection(
            features=proj,
            source_tags=source_tags,
            n_select_total=n_select,
            source_weights=weights,
            seed=seed,
            verbose=False,
        )
        selected_indices = list(fps_result["selected_indices"])
        selected = [frames[i] for i in selected_indices]
        per_source_counts = fps_result["per_source_counts"]
        print(f"    per-source breakdown: {per_source_counts}")

    # Save the selected frames + their pool indices
    selected_path = output_dir / "selected.extxyz"
    ase_write(selected_path, selected)
    with (output_dir / "selected_meta.json").open("w") as f:
        json.dump({
            "n_selected": len(selected),
            "source_pool": str(pool_path),
            "n_in_pool": len(frames),
            "selected_indices_in_pool": [int(i) for i in selected_indices],
            "per_source_counts": {k: int(v) for k, v in per_source_counts.items()},
            "fps_seed": seed,
        }, f, indent=2)
    print(f"    selected → {selected_path}")

    # VASP input directories
    vasp_dir = output_dir / "vasp_inputs"
    vasp_cfg = al_cfg.get("vasp", {})
    metadata_list = [
        {
            "source":         f.info.get("source"),
            "iteration":      f.info.get("iteration"),
            "basin_id":       f.info.get("basin_id"),
            "energy_mlp":     f.info.get("energy"),
            "source_cell":    f.info.get("source_cell"),
            "pool_index":     int(idx),
        }
        for f, idx in zip(selected, selected_indices)
    ]
    prepare_vasp_inputs(
        selected_atoms=selected,
        output_dir=vasp_dir,
        potcar_library=Path(vasp_cfg["potcar_library"]),
        incar_params=vasp_cfg.get("incar_params"),
        kpoints_grid=tuple(vasp_cfg.get("kpoints_grid", [3, 3, 1])),
        magmom_by_element=vasp_cfg.get("magmom_by_element"),
        metadata_list=metadata_list,
        selected_indices=[int(i) for i in selected_indices],
        verbose=False,
    )
    print(f"    VASP inputs → {vasp_dir}")

    return len(selected)


# ============================================================
#  Main orchestration
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Layer B training and holdout datasets "
                    "from an ablation_results/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "al_config", type=Path,
        help="Path to al_loop.yaml (provides SOAP/KPCA/FPS/VASP settings).",
    )
    parser.add_argument(
        "ablation_root", type=Path,
        help="Path to the ablation_results_*/ directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("layerB"),
        help="Where to write Layer B outputs (default: layerB/).",
    )
    parser.add_argument(
        "--train-modes", type=str, nargs="+",
        default=["mh_only", "coupled"],
        help="Modes to retrain MLPs from (default: mh_only coupled).",
    )
    parser.add_argument(
        "--n-train", type=int, default=200,
        help="Number of frames to FPS-select per training mode (default 200).",
    )
    parser.add_argument(
        "--holdout-source", type=str, default="coupled",
        help='Mode to draw holdout frames from. Use "none" to skip the '
             "holdout step entirely (default: coupled).",
    )
    parser.add_argument(
        "--n-holdout", type=int, default=50,
        help="Number of frames to FPS-select for the holdout from coupled "
             "(default 50). Ignored if --holdout-source none.",
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.1,
        help="Fraction of holdout-source's pool to set aside before training "
             "FPS, ensuring no train/test overlap (default 10%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for partitioning and FPS (default 42).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan only, don't actually run.",
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.al_config.is_file():
        raise SystemExit(f"al_loop config not found: {args.al_config}")
    if not args.ablation_root.is_dir():
        raise SystemExit(f"ablation root not found: {args.ablation_root}")

    with args.al_config.open() as f:
        al_cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    do_holdout = args.holdout_source.lower() != "none"

    print("=" * 70)
    print("Layer B preparation")
    print("=" * 70)
    print(f"  al_config       : {args.al_config}")
    print(f"  ablation_root   : {args.ablation_root}")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  train_modes     : {args.train_modes}  (FPS to {args.n_train})")
    if do_holdout:
        print(f"  holdout_source  : {args.holdout_source}  "
              f"(set aside {args.holdout_fraction:.0%}, FPS to {args.n_holdout})")
        if args.holdout_source not in args.train_modes:
            print(f"  NOTE: holdout source '{args.holdout_source}' is NOT in "
                  f"train_modes; it will only be combined but not FPS-trained.")
    else:
        print(f"  holdout_source  : none (skipping holdout)")
    print(f"  seed            : {args.seed}")
    print()

    if args.dry_run:
        print("(--dry-run: stopping here)")
        return

    # =========================================================
    # Stage 1: Combine seeds → per-mode pools
    # =========================================================
    print("Stage 1: combining seeds")
    modes_needed = set(args.train_modes)
    if do_holdout:
        modes_needed.add(args.holdout_source)

    pool_paths: dict[str, Path] = {}
    for mode in sorted(modes_needed):
        out_path = args.output_dir / f"pool_{mode}.extxyz"
        n_frames = combine_seeds_into_pool(
            args.ablation_root, mode, out_path,
        )
        if n_frames > 0:
            pool_paths[mode] = out_path
        else:
            print(f"  WARN: no frames for mode '{mode}', skipping it")
    print()

    # =========================================================
    # Stage 2: For the holdout-source mode, split into train+holdout
    # =========================================================
    if do_holdout and args.holdout_source in pool_paths:
        print(f"Stage 2: partitioning {args.holdout_source} for holdout")
        original_pool = pool_paths[args.holdout_source]
        train_partition = args.output_dir / f"partition_{args.holdout_source}_train.extxyz"
        holdout_partition = args.output_dir / f"partition_{args.holdout_source}_holdout.extxyz"
        random_partition_pool(
            pool_path=original_pool,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            out_train=train_partition,
            out_holdout=holdout_partition,
        )
        # If holdout-source is also being trained, use the TRAIN partition
        if args.holdout_source in args.train_modes:
            pool_paths[args.holdout_source] = train_partition
        print()
    else:
        holdout_partition = None

    # =========================================================
    # Stage 3: FPS + VASP-input gen for each training mode
    # =========================================================
    print(f"Stage 3: FPS to {args.n_train} for each training mode")
    for mode in args.train_modes:
        if mode not in pool_paths:
            print(f"  SKIP {mode}: no pool available")
            continue
        print(f"  -- {mode} --")
        out_subdir = args.output_dir / f"train_{mode}"
        fps_select_and_dump_vasp(
            pool_path=pool_paths[mode],
            n_select=args.n_train,
            output_dir=out_subdir,
            al_cfg=al_cfg,
            seed=args.seed,
        )
    print()

    # =========================================================
    # Stage 4: FPS the holdout partition
    # =========================================================
    if do_holdout and holdout_partition is not None:
        print(f"Stage 4: FPS to {args.n_holdout} for holdout candidates "
              f"(from {args.holdout_source})")
        out_subdir = args.output_dir / f"holdout_from_{args.holdout_source}"
        fps_select_and_dump_vasp(
            pool_path=holdout_partition,
            n_select=args.n_holdout,
            output_dir=out_subdir,
            al_cfg=al_cfg,
            seed=args.seed + 100,
        )
        print()

    # =========================================================
    # Final summary
    # =========================================================
    print("=" * 70)
    print("DONE. Next steps:")
    print("=" * 70)
    n_total_dft = args.n_train * len(args.train_modes)
    if do_holdout:
        n_total_dft += args.n_holdout
    print(f"  1. Submit DFT for {n_total_dft} total VASP single-points:")
    for mode in args.train_modes:
        if mode in pool_paths:
            print(f"     {args.output_dir}/train_{mode}/vasp_inputs/  ({args.n_train})")
    if do_holdout:
        print(f"     {args.output_dir}/holdout_from_{args.holdout_source}/vasp_inputs/  ({args.n_holdout})")
    print(f"  2. After DFT, run assemble_holdout.py to build the test set.")
    print(f"  3. Train NequIP MLPs on each train_<mode>/labeled.extxyz.")
    print(f"  4. Evaluate each MLP on the test set → Figure 4.")
    print()


if __name__ == "__main__":
    main()
