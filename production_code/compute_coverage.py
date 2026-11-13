#!/usr/bin/env python
"""
compute_coverage.py — robust configuration-space coverage metric for panel (b).

Replaces "convex hull area in KPCA-2D" (outlier-sensitive, arbitrary units) with
an occupancy-based coverage a reviewer is far less likely to nitpick:

    coverage(mode, seed) =  (# KPCA-2D grid cells occupied by this pool)
                            ---------------------------------------------
                            (# cells occupied by the UNION of all pools)

i.e. the fraction of the *explored* 2-D configuration space each sampler reaches.
Cell occupancy is binary, so one stray frame cannot inflate the score (the
failure mode of convex-hull area), and the [0, 1] normalisation makes the
absolute bin size cancel out of the between-mode ordering.

Fairness: every (mode, seed) pool is randomly sub-sampled to a COMMON size
(matched sampling effort) before featurising, so "coupled covers more" cannot be
an artefact of coupled merely having more frames.

SOAP/KPCA reuse the same backend as prepare_layerB.py / run_al.py, so the
embedding matches Section 2.3.

Usage
-----
    python compute_coverage.py configs/al_loop.yaml ablation_results_Hookean/ \
        --modes mh_only walk_only serial coupled \
        --seeds 42 43 44 \
        --n-per-pool 1500 --grid 40 \
        --out layerB/coverage_metric.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ase.io import read as ase_read  # noqa: E402
from mhmc.selection.soap_features import compute_soap_descriptors  # noqa: E402
from mhmc.selection.kpca_reducer import KPCAReducer  # noqa: E402

try:
    from paper_style import apply_style
    apply_style()
except Exception:
    pass

_SOAP_HYPER_KEYS = {"r_cut", "n_max", "l_max", "sigma", "average"}

# Match the colours already used in your (a)/(b)/(c)/(d) figure.
MODE_COLORS = {
    "mh_only":   "#7F7F7F",  # gray
    "walk_only": "#8C564B",  # brown
    "serial":    "#FF7F0E",  # orange
    "coupled":   "#D62728",  # red
}
MODE_LABELS = {
    "mh_only":   "MH only",
    "walk_only": "Walk only",
    "serial":    "Serial",
    "coupled":   "Coupled (full)",
}


def occupied_cells(xy: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray) -> set:
    """Set of (ix, iy) grid cells occupied by 2-D points xy."""
    ix = np.clip(np.digitize(xy[:, 0], edges_x) - 1, 0, len(edges_x) - 2)
    iy = np.clip(np.digitize(xy[:, 1], edges_y) - 1, 0, len(edges_y) - 2)
    return set(map(tuple, np.column_stack([ix, iy]).tolist()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("al_config", type=Path, help="al_loop.yaml (SOAP/KPCA settings).")
    p.add_argument("ablation_root", type=Path, help="ablation_results_*/ directory.")
    p.add_argument("--modes", nargs="+",
                   default=["mh_only", "walk_only", "serial", "coupled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--n-per-pool", type=int, default=1500,
                   help="Cap; the actual matched N is min(this, smallest pool).")
    p.add_argument("--grid", type=int, default=40, help="Bins per KPCA axis.")
    p.add_argument("--seed", type=int, default=0, help="Sub-sampling/KPCA seed.")
    p.add_argument("--out", type=Path, default=Path("layerB/coverage_metric.png"))
    args = p.parse_args()

    if not args.al_config.is_file():
        raise SystemExit(f"al_config not found: {args.al_config}")
    rng = np.random.default_rng(args.seed)

    al_cfg = yaml.safe_load(args.al_config.read_text())
    soap_cfg = al_cfg.get("selection", {}).get("soap", {})
    kpca_cfg = al_cfg.get("selection", {}).get("kpca", {})
    soap_params = {k: v for k, v in soap_cfg.items() if k in _SOAP_HYPER_KEYS}

    # ---- locate every (mode, seed) pool, record its size ------------------
    pool_paths: dict[tuple[str, int], Path] = {}
    sizes: list[int] = []
    print("Pools:")
    for mode in args.modes:
        for seed in args.seeds:
            pth = args.ablation_root / f"{mode}_seed{seed}" / "candidate_pool.extxyz"
            if not pth.exists():
                raise SystemExit(f"pool not found: {pth}")
            n = len(ase_read(pth, index=":"))
            pool_paths[(mode, seed)] = pth
            sizes.append(n)
            print(f"    {mode}_seed{seed}: {n} frames")

    n_common = min(args.n_per_pool, min(sizes))
    print(f"\nMatched sample size per pool: {n_common}")

    # ---- load + sub-sample to matched N; remember slices ------------------
    all_frames: list = []
    slices: dict[tuple[str, int], tuple[int, int]] = {}
    cursor = 0
    for key, pth in pool_paths.items():
        frames = ase_read(pth, index=":")
        if len(frames) > n_common:
            idx = np.sort(rng.choice(len(frames), n_common, replace=False))
            frames = [frames[i] for i in idx]
        all_frames.extend(frames)
        slices[key] = (cursor, cursor + len(frames))
        cursor += len(frames)

    # ---- one shared SOAP + KPCA embedding ---------------------------------
    print(f"Computing SOAP on {len(all_frames)} frames (shared)...")
    soap = compute_soap_descriptors(
        all_frames, species=soap_cfg.get("species"),
        soap_params=soap_params or None, n_jobs=soap_cfg.get("n_jobs", -1),
        verbose=False,
    )
    kpca = KPCAReducer(
        variance_threshold=kpca_cfg.get("variance_threshold", 0.95),
        max_components=kpca_cfg.get("max_components", 50),
        random_state=args.seed,
    )
    proj = kpca.fit_transform(soap, verbose=False)
    if proj.shape[1] < 2:
        raise SystemExit("KPCA returned <2 components.")
    xy = proj[:, :2]

    # ---- grid over the union extent; global occupied cells ----------------
    pad_x = 0.02 * (xy[:, 0].max() - xy[:, 0].min() + 1e-9)
    pad_y = 0.02 * (xy[:, 1].max() - xy[:, 1].min() + 1e-9)
    edges_x = np.linspace(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x, args.grid + 1)
    edges_y = np.linspace(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y, args.grid + 1)
    global_cells = occupied_cells(xy, edges_x, edges_y)
    n_global = len(global_cells)
    print(f"Union occupies {n_global} / {args.grid**2} cells")

    # ---- per (mode, seed) coverage fraction -------------------------------
    rows = []
    per_mode: dict[str, list[float]] = {m: [] for m in args.modes}
    for (mode, seed), (s, e) in slices.items():
        cov = len(occupied_cells(xy[s:e], edges_x, edges_y)) / n_global
        per_mode[mode].append(cov)
        rows.append({"mode": mode, "seed": seed, "coverage": cov})

    # ---- plot: bars (mean) + error bars (std) + per-seed points -----------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    x = np.arange(len(args.modes))
    means = [float(np.mean(per_mode[m])) for m in args.modes]
    stds = [float(np.std(per_mode[m], ddof=1)) if len(per_mode[m]) > 1 else 0.0
            for m in args.modes]

    ax.bar(x, means, yerr=stds, capsize=4,
           color=[MODE_COLORS.get(m, "#999999") for m in args.modes],
           edgecolor="white", linewidth=0.9, zorder=3)
    for xi, m in enumerate(x):
        pts = per_mode[args.modes[m]]
        jit = (rng.random(len(pts)) - 0.5) * 0.12
        ax.scatter(np.full(len(pts), xi) + jit, pts, s=22, c="0.2",
                   zorder=5, edgecolors="white", linewidths=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in args.modes], rotation=12)
    ax.set_ylabel("Configuration-space coverage\n"
                  "(fraction of explored KPCA cells, matched N)")
    ax.set_title("Configurational coverage")
    ax.set_ylim(0, max(means[i] + stds[i] for i in range(len(means))) * 1.25)
    ax.grid(axis="y")
    ax.xaxis.grid(False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))

    csv_path = out.with_name(out.stem + ".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "seed", "coverage"])
        w.writeheader()
        w.writerows(rows)

    print("\nPer-mode coverage (mean ± std):")
    for m in args.modes:
        print(f"    {MODE_LABELS.get(m, m):>15}: "
              f"{np.mean(per_mode[m]):.3f} ± "
              f"{np.std(per_mode[m], ddof=1) if len(per_mode[m]) > 1 else 0:.3f}")
    print(f"\nSaved:\n  {out}\n  {out.with_suffix('.pdf')}\n  {csv_path}")


if __name__ == "__main__":
    main()
