#!/usr/bin/env python
"""
plot_coverage_kpca.py — SOAP-KPCA coverage comparison of two (or more) sampling
pools, projected into ONE shared KPCA embedding.

Purpose  (Section 3.2 — the Layer B "cause" figure)
---------------------------------------------------
The cross-evaluation matrix shows the *effect*: the MHMC-trained MLP transfers
to MH's region but not vice-versa. This figure shows the *cause* in data space —
that the MHMC sampling pool envelopes the MH pool and extends into regions MH
never visits. It is completely MLP-independent.

Method (why it is done this way)
--------------------------------
* All pools are concatenated and a SINGLE SOAP featurisation + SINGLE KPCA fit
  is run on the union, so every cloud lives in the SAME embedding and is
  directly comparable. Fitting KPCA per-pool would give incomparable axes and
  the "superset" claim would be meaningless.
* SOAP/KPCA settings are read from al_loop.yaml exactly as prepare_layerB.py
  does (same r_cut/n_max/l_max/sigma/average and variance_threshold/
  max_components), so this embedding matches Section 2.3.
* Large pools are randomly sub-sampled (--max-per-pool) before featurising,
  because the KPCA kernel matrix is O(N^2) in memory. A random sub-sample is an
  unbiased view of a pool's *support* (= coverage), which is exactly what the
  envelope argument needs; it does not bias the extent.
* Convex hulls are drawn as a visual "envelope" guide. (A hull slightly
  over-states support but makes relative extent obvious; switch off with
  --no-hull if you prefer raw clouds.)

Usage
-----
    python plot_coverage_kpca.py configs/al_loop.yaml

    python plot_coverage_kpca.py configs/al_loop.yaml \
        --pool MH=layerB/pool_mh_only.extxyz \
        --pool MHMC=layerB/pool_coupled.extxyz \
        --max-per-pool 2000 \
        --out layerB/coverage_kpca.png

Optionally mark a set of frames on top of the clouds (e.g. the coupled-holdout
frames where the MH model's force error is largest) to nail the cause->effect
link visually:

    python plot_coverage_kpca.py configs/al_loop.yaml \
        --highlight "MH-model |F| err > 1 eV/A=layerB/mh_highF_frames.extxyz"

(LABEL=PATH; LABEL must not contain '='.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ase.io import read as ase_read  # noqa: E402

# Same selection backend as prepare_layerB.py / run_al.py
from mhmc.selection.soap_features import compute_soap_descriptors  # noqa: E402
from mhmc.selection.kpca_reducer import KPCAReducer  # noqa: E402

try:
    from scipy.spatial import ConvexHull
    _HAVE_HULL = True
except Exception:
    _HAVE_HULL = False

# Exactly the key set prepare_layerB uses to separate SOAP hyperparameters
# from non-hyperparameter config entries (the bug-#3 fix).
_SOAP_HYPER_KEYS = {"r_cut", "n_max", "l_max", "sigma", "average"}

# Colour-blind-friendly cloud colours, cycled across pools.
_PALETTE = ["#2C7FB8", "#E6550D", "#31A354", "#756BB1", "#C51B8A"]


def parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"expected LABEL=PATH, got: {spec!r}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def load_subsampled(path: Path, max_n: int, rng: np.random.Generator) -> list:
    frames = ase_read(path, index=":")
    n = len(frames)
    if max_n and n > max_n:
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
        frames = [frames[i] for i in idx]
        print(f"    {path.name}: {n} frames -> sub-sampled to {len(frames)}")
    else:
        print(f"    {path.name}: {n} frames (all kept)")
    return frames


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shared SOAP-KPCA coverage plot for two or more pools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("al_config", type=Path,
                   help="al_loop.yaml (provides SOAP/KPCA settings).")
    p.add_argument("--pool", action="append", default=None,
                   help="LABEL=PATH (repeatable). Default: MH and MHMC pools "
                        "under layerB/.")
    p.add_argument("--highlight", action="append", default=None,
                   help="LABEL=PATH (repeatable) — frames to mark on top.")
    p.add_argument("--max-per-pool", type=int, default=2000,
                   help="Random sub-sample cap per pool (0 = keep all). "
                        "Keeps the KPCA kernel tractable (default 2000).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for sub-sampling and KPCA (default 42).")
    p.add_argument("--out", type=Path, default=Path("layerB/coverage_kpca.png"))
    p.add_argument("--no-hull", action="store_true",
                   help="Do not draw convex-hull envelopes.")
    args = p.parse_args()

    if args.pool is None:
        args.pool = [
            "MH=layerB/pool_mh_only.extxyz",
            "MHMC=layerB/pool_coupled.extxyz",
        ]

    if not args.al_config.is_file():
        raise SystemExit(f"al_config not found: {args.al_config}")

    rng = np.random.default_rng(args.seed)
    al_cfg = yaml.safe_load(args.al_config.read_text())
    soap_cfg = al_cfg.get("selection", {}).get("soap", {})
    kpca_cfg = al_cfg.get("selection", {}).get("kpca", {})
    soap_params = {k: v for k, v in soap_cfg.items() if k in _SOAP_HYPER_KEYS}

    pools = [parse_labeled_path(s) for s in args.pool]
    highlights = [parse_labeled_path(s) for s in (args.highlight or [])]

    # ---- gather all frames into one list; remember each group's slice ------
    all_frames: list = []
    group_slices: dict[str, tuple[int, int]] = {}
    cursor = 0

    print("Loading pools:")
    for label, path in pools:
        if not path.exists():
            raise SystemExit(f"pool not found: {path}")
        fr = load_subsampled(path, args.max_per_pool, rng)
        all_frames.extend(fr)
        group_slices[label] = (cursor, cursor + len(fr))
        cursor += len(fr)

    hl_slices: dict[str, tuple[int, int]] = {}
    if highlights:
        print("Loading highlight sets (kept in full):")
        for label, path in highlights:
            if not path.exists():
                raise SystemExit(f"highlight not found: {path}")
            fr = ase_read(path, index=":")
            print(f"    {path.name}: {len(fr)} frames")
            all_frames.extend(fr)
            hl_slices[label] = (cursor, cursor + len(fr))
            cursor += len(fr)

    print(f"\nTotal frames for the shared embedding: {len(all_frames)}")
    if len(all_frames) > 8000:
        print("  WARNING: KPCA kernel memory grows as O(N^2); "
              f"{len(all_frames)} frames may be heavy — lower --max-per-pool "
              "if it OOMs.")

    # ---- ONE SOAP featurisation on the union -------------------------------
    print("Computing SOAP on the union (shared featurisation)...")
    soap = compute_soap_descriptors(
        all_frames,
        species=soap_cfg.get("species"),
        soap_params=soap_params or None,
        n_jobs=soap_cfg.get("n_jobs", -1),
        verbose=False,
    )
    print(f"  SOAP matrix: {soap.shape}")

    # ---- ONE KPCA fit on the union -> shared 2D embedding ------------------
    print("Fitting shared KPCA...")
    kpca = KPCAReducer(
        variance_threshold=kpca_cfg.get("variance_threshold", 0.95),
        max_components=kpca_cfg.get("max_components", 50),
        random_state=args.seed,
    )
    proj = kpca.fit_transform(soap, verbose=False)
    print(f"  KPCA -> {proj.shape[1]} components (plotting the first 2)")
    if proj.shape[1] < 2:
        raise SystemExit("KPCA returned <2 components; cannot make a 2D plot.")
    xy = proj[:, :2]

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=200)

    for i, (label, _) in enumerate(pools):
        s, e = group_slices[label]
        colour = _PALETTE[i % len(_PALETTE)]
        pts = xy[s:e]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.35, c=colour,
                   edgecolors="none", zorder=2, label=f"{label}  (n={e - s})")
        if (not args.no_hull) and _HAVE_HULL and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                loop = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[loop, 0], pts[loop, 1], "-", c=colour, lw=1.6,
                        alpha=0.9, zorder=3)
            except Exception as ex:  # degenerate hull etc.
                print(f"  (hull skipped for {label}: {ex})")

    for label, _ in highlights:
        s, e = hl_slices[label]
        pts = xy[s:e]
        ax.scatter(pts[:, 0], pts[:, 1], s=46, marker="X",
                   facecolors="#D7191C", edgecolors="k", linewidths=0.6,
                   zorder=5, label=f"{label}  (n={e - s})")

    ax.set_xlabel("KPCA component 1")
    ax.set_ylabel("KPCA component 2")
    ax.set_title("Configuration-space coverage\n(shared SOAP-KPCA embedding)")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nSaved:\n  {out}\n  {pdf}")


if __name__ == "__main__":
    main()
