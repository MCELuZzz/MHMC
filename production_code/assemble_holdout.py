#!/usr/bin/env python
"""
assemble_holdout.py — compose the final Layer B holdout test set.

After prepare_layerB.py + DFT, this script combines:
    A) N_valid frames FPS-selected from the existing labeled valid set
       (already DFT-labeled — free, no new compute)
    B) N_coupled frames from layerB/holdout_from_coupled/, AFTER you've
       collected their DFT results and merged into a labeled extxyz

into a single holdout_test.extxyz that the retrained MLPs will be evaluated on.

Usage
-----
    python assemble_holdout.py \\
        --valid-set data/HNNH/valid_combined.extxyz \\
        --n-from-valid 50 \\
        --coupled-holdout-labeled layerB/holdout_from_coupled/labeled.extxyz \\
        --output layerB/holdout_test.extxyz

The coupled-holdout labeled file is the extxyz containing the DFT-labeled
versions of frames originally written under
    layerB/holdout_from_coupled/selected.extxyz
You assemble it yourself from the VASP OUTCAR/vasprun.xml outputs — typically
by parsing each subdirectory with ASE's read_vasp_xml.

Notes
-----
* The N_valid frames are selected via FPS in SOAP-KPCA space (same machinery
  as the AL pipeline) so the test set has structural diversity, not random.
* If --no-fps-on-valid is passed, valid-set frames are randomly sampled.
* Frames written carry an info["test_source"] tag = "valid" or "coupled".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read as ase_read, write as ase_write


def _select_from_valid(
    valid_frames: list,
    n_select: int,
    seed: int,
    fps: bool,
    al_cfg: dict = None,
) -> list:
    """Select n_select frames from the valid set. FPS-based if fps=True,
    otherwise random.
    """
    if len(valid_frames) <= n_select:
        print(f"  valid set has {len(valid_frames)} ≤ {n_select} requested; "
              "taking all")
        return list(valid_frames)

    rng = np.random.default_rng(seed)

    if not fps:
        idx = rng.choice(len(valid_frames), size=n_select, replace=False)
        return [valid_frames[i] for i in sorted(idx)]

    # FPS path
    from mhmc.selection.soap_features import compute_soap_descriptors
    from mhmc.selection.kpca_reducer import KPCAReducer

    soap_cfg = (al_cfg or {}).get("selection", {}).get("soap", {})
    kpca_cfg = (al_cfg or {}).get("selection", {}).get("kpca", {})
    _SOAP_HYPER_KEYS = {"r_cut", "n_max", "l_max", "sigma", "average"}
    soap_params = {k: v for k, v in soap_cfg.items()
                   if k in _SOAP_HYPER_KEYS}

    print(f"  SOAP on {len(valid_frames)} valid frames...")
    soap = compute_soap_descriptors(
        valid_frames,
        species=soap_cfg.get("species"),
        soap_params=soap_params or None,
        n_jobs=soap_cfg.get("n_jobs", -1),
        verbose=False,
    )
    print(f"  KPCA fit + plain FPS to {n_select}...")
    kpca = KPCAReducer(
        variance_threshold=kpca_cfg.get("variance_threshold", 0.95),
        max_components=kpca_cfg.get("max_components", 50),
        random_state=seed,
    )
    proj = kpca.fit_transform(soap, verbose=False)

    # Plain FPS in KPCA space (no source-stratification needed for valid set
    # since it usually doesn't have the same source tags)
    selected_indices = _plain_fps(proj, n_select, seed)
    return [valid_frames[int(i)] for i in selected_indices]


def _plain_fps(features: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Greedy farthest-point sampling in feature space."""
    n = len(features)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    for _ in range(n_select - 1):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = np.linalg.norm(features - features[idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return np.array(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble final Layer B holdout test set."
    )
    parser.add_argument(
        "--valid-set", type=Path, required=True,
        help="Existing DFT-labeled valid set extxyz (e.g. data/HNNH/valid_combined.extxyz).",
    )
    parser.add_argument(
        "--n-from-valid", type=int, default=50,
        help="Frames to take from the valid set (default 50).",
    )
    parser.add_argument(
        "--coupled-holdout-labeled", type=Path, required=True,
        help="DFT-labeled extxyz from layerB/holdout_from_coupled/ "
             "(you assemble this from VASP outputs).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output holdout test extxyz.",
    )
    parser.add_argument(
        "--al-config", type=Path, default=None,
        help="Optional al_loop.yaml to read SOAP/KPCA settings for FPS on valid set.",
    )
    parser.add_argument(
        "--no-fps-on-valid", action="store_true",
        help="Random-sample from valid set instead of FPS (faster).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for sampling (default 42).",
    )
    args = parser.parse_args()

    al_cfg = None
    if args.al_config and args.al_config.is_file():
        with args.al_config.open() as f:
            al_cfg = yaml.safe_load(f)
    elif not args.no_fps_on_valid:
        print("WARN: --al-config not provided; falling back to random sampling")
        args.no_fps_on_valid = True

    print("Loading labeled inputs...")
    valid_frames = ase_read(args.valid_set, index=":")
    print(f"  valid set: {len(valid_frames)} frames at {args.valid_set}")

    coupled_holdout = ase_read(args.coupled_holdout_labeled, index=":")
    print(f"  coupled holdout (labeled): {len(coupled_holdout)} frames")

    # ---- Select from valid set ----
    print(f"\nSelecting {args.n_from_valid} from valid set "
          f"({'FPS' if not args.no_fps_on_valid else 'random'})...")
    chosen_valid = _select_from_valid(
        valid_frames=valid_frames,
        n_select=args.n_from_valid,
        seed=args.seed,
        fps=not args.no_fps_on_valid,
        al_cfg=al_cfg,
    )
    for f in chosen_valid:
        f.info["test_source"] = "valid"
    print(f"  took {len(chosen_valid)} frames from valid")

    # ---- All labeled coupled-holdout frames go in ----
    for f in coupled_holdout:
        f.info["test_source"] = "coupled_holdout"

    # ---- Compose final test set ----
    test_frames = list(chosen_valid) + list(coupled_holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase_write(args.output, test_frames)

    # ---- Summary ----
    n_valid = sum(1 for f in test_frames if f.info["test_source"] == "valid")
    n_coupled = sum(1 for f in test_frames if f.info["test_source"] == "coupled_holdout")
    summary = {
        "output":                    str(args.output),
        "n_total":                   len(test_frames),
        "n_from_valid":              n_valid,
        "n_from_coupled_holdout":    n_coupled,
        "valid_source":              str(args.valid_set),
        "coupled_holdout_source":    str(args.coupled_holdout_labeled),
        "fps_on_valid":              not args.no_fps_on_valid,
        "seed":                      args.seed,
    }
    summary_path = args.output.with_suffix(".meta.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"Holdout test set written: {args.output}")
    print(f"  total       : {len(test_frames)}")
    print(f"  from valid  : {n_valid}")
    print(f"  from coupled: {n_coupled}")
    print(f"  metadata    : {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
