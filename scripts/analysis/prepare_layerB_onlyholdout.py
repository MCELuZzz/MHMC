#!/usr/bin/env python
"""
prepare_mh_holdout.py — build the mh holdout test partition for the symmetric
2x2 cross-evaluation, disjoint-BY-CONSTRUCTION from the mh training set.

Why this exists
---------------
prepare_layerB.py only set aside a holdout partition for the *holdout-source*
mode (coupled). The mh_only 200 training frames were FPS-selected from the FULL
mh_only pool, so no holdout partition was ever reserved for mh. To test the
coupled MLP on mh's distribution (and mh's MLP on its own), we need a held-out
set drawn from mh's pool that PROVABLY does not overlap mh's training frames.

Approach
--------
    holdout = FPS(N) over ( pool_mh_only  -  mh_train_selected_indices )

Disjointness is enforced by set difference, then VERIFIED by an assertion at
the end (overlap with the training set must be exactly 0). We rely on the
training indices that prepare_layerB already saved in
    train_<mode>/selected_meta.json  ["selected_indices_in_pool"]
so we do NOT need to re-run / reproduce the training FPS, and the result is
independent of whatever --seed was used for the original training selection.

All SOAP / KPCA / stratified-FPS / VASP-input work reuses
prepare_layerB.fps_select_and_dump_vasp UNCHANGED, so this holdout is built
byte-identically to your train sets and the coupled holdout (same descriptors,
same VASP INCAR/KPOINTS behaviour) — important for a fair DFT comparison.

IMPORTANT: pass configs/al_loop_mh_only.yaml (NOT al_loop.yaml). mh's pool only
contains minimum/escape_md/relaxation sources; the default al_loop.yaml weights
reserve quota for walk_* sources that have zero frames here, which would
under-fill the selection (the bug-#6 trap).

Usage
-----
    python prepare_mh_holdout.py configs/al_loop_mh_only.yaml layerB/ \
        --mode mh_only --n-holdout 50 --seed 42

(prepare_mh_holdout.py must sit next to prepare_layerB.py so it can import it.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ase.io import read as ase_read, write as ase_write

# Reuse the EXACT selection + VASP-input logic from prepare_layerB.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_layerB import fps_select_and_dump_vasp  # noqa: E402


def load_train_indices(train_meta_path: Path) -> tuple[set[int], int, int]:
    """Read the selected_meta.json that prepare_layerB wrote for the train set.

    Returns (train_indices_set, n_in_pool_recorded, n_selected_recorded).
    """
    with train_meta_path.open() as fh:
        meta = json.load(fh)
    idx = meta.get("selected_indices_in_pool")
    if idx is None:
        raise SystemExit(
            f"{train_meta_path} has no 'selected_indices_in_pool' field — "
            "cannot guarantee disjointness from training. Aborting."
        )
    return (
        set(int(i) for i in idx),
        int(meta["n_in_pool"]),
        int(meta.get("n_selected", len(idx))),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the mh holdout partition (disjoint-by-construction "
                    "from mh training) for the symmetric 2x2 cross-evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("al_config", type=Path,
                   help="Pass configs/al_loop_mh_only.yaml (NOT al_loop.yaml).")
    p.add_argument("layerb_dir", type=Path,
                   help="prepare_layerB.py output dir (contains pool_<mode>.extxyz "
                        "and train_<mode>/selected_meta.json).")
    p.add_argument("--mode", default="mh_only",
                   help="Mode whose pool we draw the holdout from (default mh_only).")
    p.add_argument("--n-holdout", type=int, default=50,
                   help="Number of holdout frames to FPS-select (default 50).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; holdout FPS uses seed+100, mirroring "
                        "prepare_layerB's holdout convention (default 42).")
    args = p.parse_args()

    pool_path   = args.layerb_dir / f"pool_{args.mode}.extxyz"
    train_meta  = args.layerb_dir / f"train_{args.mode}" / "selected_meta.json"
    out_subdir  = args.layerb_dir / f"holdout_from_{args.mode}"
    subset_path = args.layerb_dir / f"pool_{args.mode}_minus_train.extxyz"

    for pth in (args.al_config, pool_path, train_meta):
        if not pth.exists():
            raise SystemExit(f"Required input not found: {pth}")

    with args.al_config.open() as fh:
        al_cfg = yaml.safe_load(fh)

    print("=" * 70)
    print(f"mh holdout preparation   (mode = {args.mode})")
    print("=" * 70)

    # ---- 1. load the pool + the saved training indices --------------------
    pool = ase_read(pool_path, index=":")
    train_idx, n_in_pool_rec, n_train_rec = load_train_indices(train_meta)
    print(f"  pool_{args.mode}.extxyz   : {len(pool)} frames")
    print(f"  recorded train selection : {n_train_rec} frames "
          f"(from a pool of {n_in_pool_rec})")

    # ---- 2. safety: the pool file MUST match what train was selected from --
    #     The saved indices are positions into this exact file. If the pool
    #     file has changed since training, the indices point at the wrong
    #     frames -> silent leakage. Refuse to continue.
    if len(pool) != n_in_pool_rec:
        raise SystemExit(
            f"POOL MISMATCH: {pool_path} currently has {len(pool)} frames, but "
            f"the training selection was made from a pool of {n_in_pool_rec}. "
            "The saved indices would reference the wrong frames. Aborting — "
            "regenerate pool_<mode>.extxyz or re-run prepare_layerB."
        )
    out_of_range = [i for i in train_idx if i >= len(pool)]
    if out_of_range:
        raise SystemExit(f"Train indices out of range: {out_of_range[:5]} ...")

    # ---- 3. set difference: candidates = every frame NOT in training ------
    subset_to_orig = [i for i in range(len(pool)) if i not in train_idx]
    subset_frames  = [pool[i] for i in subset_to_orig]
    print(f"  candidate pool (pool - train): {len(subset_frames)} frames")
    if len(subset_frames) <= args.n_holdout:
        print(f"  WARNING: only {len(subset_frames)} candidates for "
              f"{args.n_holdout} holdout frames — FPS will take (almost) all "
              "of them, with little diversity.")

    ase_write(subset_path, subset_frames)
    print(f"  wrote candidate subset -> {subset_path.name}")

    # ---- 4. FPS + VASP-input generation (reusing prepare_layerB verbatim) -
    print(f"\n  FPS to {args.n_holdout} via prepare_layerB.fps_select_and_dump_vasp")
    fps_select_and_dump_vasp(
        pool_path=subset_path,
        n_select=args.n_holdout,
        output_dir=out_subdir,
        al_cfg=al_cfg,
        seed=args.seed + 100,
    )

    # ---- 5. VERIFY disjointness against the training set ------------------
    #     selected_indices_in_pool here are positions into the SUBSET file;
    #     map them back to original-pool indices via subset_to_orig.
    with (out_subdir / "selected_meta.json").open() as fh:
        hmeta = json.load(fh)
    subset_sel   = [int(j) for j in hmeta["selected_indices_in_pool"]]
    holdout_orig = [subset_to_orig[j] for j in subset_sel]
    overlap      = set(holdout_orig) & train_idx

    print("\n" + "-" * 70)
    print("  DISJOINTNESS CHECK  (holdout vs mh training)")
    print(f"    train frames            : {len(train_idx)}")
    print(f"    holdout frames selected : {len(holdout_orig)}")
    print(f"    overlap                 : {len(overlap)}")
    assert not overlap, (
        f"LEAKAGE DETECTED: {len(overlap)} holdout frames are also in mh "
        f"training: {sorted(overlap)[:10]}"
    )
    print("    PASS  ->  holdout intersection train = empty")
    print("-" * 70)

    with (out_subdir / "disjointness_check.json").open("w") as fh:
        json.dump({
            "mode": args.mode,
            "n_train": len(train_idx),
            "n_holdout": len(holdout_orig),
            "n_overlap": len(overlap),
            "holdout_indices_in_original_pool": sorted(int(i) for i in holdout_orig),
            "verified_disjoint": True,
        }, fh, indent=2)

    # ---- 6. next steps ----------------------------------------------------
    print(f"\nDONE.  VASP inputs -> {out_subdir}/vasp_inputs/  "
          f"({len(holdout_orig)} frames)")
    print("Next:")
    print(f"  1. Run VASP on {out_subdir}/vasp_inputs/   (+{len(holdout_orig)} single-points)")
    print(f"  2. collect_vasp_outputs.py  ->  labeled.extxyz")
    print(f"  3. Evaluate BOTH MLPs (mh, coupled) on this labeled set; this fills")
    print(f"     the 'mh-pool' column of the 2x2 cross-evaluation matrix.")


if __name__ == "__main__":
    main()