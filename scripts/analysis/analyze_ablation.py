#!/usr/bin/env python
"""
analyze_ablation.py — produce Section 3.2 figures from ablation_results/

Reads the output of run_ablation.py and generates:

  1. figure_3.png / .pdf   — production figure with three panels:
       (a) basin discovery curve     (cumulative n_basins vs iter)
       (b) lowest energy found       (running min(E_min) vs iter)
       (c) configurational coverage  (KPCA convex-hull area, bar per mode)
     Each curve aggregates across seeds with mean ± std band.

  2. diagnostics.png / .pdf — 2x2 sanity panel showing per-mode mean
       trajectories of T_used, rho, raw E_min, and new-basins-per-iter.
       Use this first to verify that the ablation runs are well-behaved.

  3. per_cell_metrics.csv  — flat table of all per-cell scalar metrics
       (final n_basins, final lowest E, KPCA area, mean rho/T, etc.)
       for ad-hoc analysis in pandas / Excel.

Usage
-----
    python analyze_ablation.py ablation_results/
    python analyze_ablation.py ablation_results/ --quick           # skip SOAP+KPCA
    python analyze_ablation.py ablation_results/ --samples-per-cell 800
    python analyze_ablation.py ablation_results/ --output figs/    # custom output dir
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
#  Display conventions: fixed mode order, colors, labels
# ============================================================

MODE_ORDER = ["mh_only", "walk_only", "serial", "coupled_no_chain", "coupled"]

MODE_COLORS = {
    "mh_only":          "#888888",  # gray   — pure-MH baseline
    "walk_only":        "#8c564b",  # brown  — long-MD baseline
    "serial":           "#ff7f0e",  # orange — uncoupled MH+walk
    "coupled_no_chain": "#1f77b4",  # blue   — partial ablation
    "coupled":          "#d62728",  # red    — full algorithm (hero)
}

MODE_LABELS = {
    "mh_only":          "MH only",
    "walk_only":        "Walk only",
    "serial":           "Serial",
    "coupled_no_chain": "Coupled (no chain)",
    "coupled":          "Coupled (full)",
}


# ============================================================
#  Load run.jsonl into a dict of numpy arrays
# ============================================================

def load_cell_log(jsonl_path: Path) -> Optional[dict]:
    """Parse a run.jsonl into {field_name: np.ndarray}.

    Returns None if file is missing or empty.
    """
    if not jsonl_path.is_file():
        return None

    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping bad line in {jsonl_path}: {e}")

    if not records:
        return None

    # Union of all keys (in case some iters lack some fields)
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    out = {}
    for k in all_keys:
        vals = [r.get(k) for r in records]
        # Try to convert to float array; fall back to object array
        try:
            arr = np.array([
                np.nan if v is None or v is False else
                (1.0 if v is True else float(v))
                for v in vals
            ], dtype=float)
            out[k] = arr
        except (TypeError, ValueError):
            out[k] = np.array(vals, dtype=object)

    out["_n_iters"] = len(records)
    return out


# ============================================================
#  Per-cell derived metrics
# ============================================================

def compute_per_cell_metrics(log: dict) -> dict:
    """Compute basin_discovery, running min E, etc. from a parsed run.jsonl."""
    n_iters = log["_n_iters"]

    # Basin discovery: n_basins_total is logged as cumulative count
    n_basins = log["n_basins_total"].astype(float)

    # Running min of E_min (best minimum ever found at or before each iter)
    E_min = log["E_min"]
    running_min_E = np.minimum.accumulate(E_min)

    # New basins per iter (for diagnostics) — from is_new_basin flag
    is_new = (log["is_new_basin"] > 0.5)  # bool from float
    new_per_iter = is_new.astype(float)

    return {
        "n_iters":            n_iters,
        "n_basins_cum":       n_basins,
        "lowest_E":           running_min_E,
        "E_min_raw":          E_min,
        "new_basins_per_iter": new_per_iter,
        "T_used":             log["T_used"],
        "rho":                log["rho"],
        "R":                  log["R"],
    }


# ============================================================
#  Aggregate seeds within a mode
# ============================================================

def aggregate_seeds(per_seed_metrics: list[dict]) -> dict:
    """Stack per-seed arrays, compute mean ± std at each iter.

    Truncates to the shortest seed (so partial crashes don't poison aggregates).
    """
    n_common = min(m["n_iters"] for m in per_seed_metrics)

    keys = ["n_basins_cum", "lowest_E", "E_min_raw",
            "new_basins_per_iter", "T_used", "rho", "R"]

    out = {"n_iters_common": n_common, "n_seeds": len(per_seed_metrics)}
    for k in keys:
        stack = np.stack([m[k][:n_common] for m in per_seed_metrics])
        out[f"{k}_mean"] = np.nanmean(stack, axis=0)
        out[f"{k}_std"]  = np.nanstd(stack, axis=0)
        out[f"{k}_all"]  = stack

    return out


# ============================================================
#  KPCA coverage area (the expensive part)
# ============================================================

def compute_kpca_coverage(
    cells: list[dict],
    samples_per_cell: int = 500,
    random_state: int = 0,
) -> Optional[dict]:
    """Compute SOAP + shared KPCA + per-cell convex-hull area.

    Mutates each cell dict in `cells` with two new fields:
        cell["kpca_area"]:    float (2D convex-hull area, NaN if failed)
        cell["kpca_points"]:  (n, 2) ndarray or None

    Returns a dict with the pooled 2D projection for optional scatter plotting,
    or None if the SOAP/KPCA dependencies aren't available.
    """
    try:
        from ase.io import read as ase_read
        from mhmc.selection.soap_features import compute_soap_descriptors
        from mhmc.selection.kpca_reducer import KPCAReducer
        from scipy.spatial import ConvexHull, QhullError
    except ImportError as e:
        warnings.warn(f"KPCA coverage skipped: {e}")
        return None

    rng = np.random.default_rng(random_state)

    # ---- 1. Load + subsample frames from every cell's candidate_pool ----
    all_frames = []
    for cell in cells:
        pool_path = Path(cell["candidate_pool"])
        if not pool_path.is_file():
            warnings.warn(f"{pool_path} missing; cell {cell['run_name']} skipped")
            cell["frame_slice"] = None
            continue
        frames = ase_read(pool_path, index=":")
        if len(frames) == 0:
            cell["frame_slice"] = None
            continue
        if len(frames) > samples_per_cell:
            idx = rng.choice(len(frames), size=samples_per_cell, replace=False)
            frames = [frames[int(i)] for i in sorted(idx)]
        start = len(all_frames)
        all_frames.extend(frames)
        cell["frame_slice"] = (start, len(all_frames))

    if not all_frames:
        warnings.warn("No frames available; skipping KPCA")
        return None

    print(f"  SOAP on {len(all_frames)} pooled frames "
          f"({len(cells)} cells, up to {samples_per_cell} per cell)...")
    soap = compute_soap_descriptors(all_frames, verbose=False, n_jobs=-1)

    # ---- 2. Shared KPCA, forced to keep exactly 2 components for plotting ----
    print(f"  Fitting KPCA on {soap.shape[0]} x {soap.shape[1]} SOAP matrix...")
    # variance_threshold=1.0 means "never satisfied" -> keep up to max_components
    kpca = KPCAReducer(
        variance_threshold=1.0,
        max_components=2,
        random_state=random_state,
    )
    proj = kpca.fit_transform(soap, verbose=False)
    if proj.shape[1] < 2:
        warnings.warn("KPCA returned < 2 components; padding with zeros")
        pad = np.zeros((proj.shape[0], 2 - proj.shape[1]))
        proj = np.column_stack([proj, pad])
    proj_2d = proj[:, :2]

    # ---- 3. Per-cell convex hull area in shared 2D space ----
    for cell in cells:
        sl = cell.get("frame_slice")
        if sl is None:
            cell["kpca_area"] = np.nan
            cell["kpca_points"] = None
            continue
        a, b = sl
        pts = proj_2d[a:b]
        cell["kpca_points"] = pts
        if len(pts) < 3:
            cell["kpca_area"] = np.nan
            continue
        try:
            hull = ConvexHull(pts)
            # scipy: 2D ConvexHull.volume is the area
            cell["kpca_area"] = float(hull.volume)
        except QhullError:
            cell["kpca_area"] = np.nan

    return {"projection_2d": proj_2d}


# ============================================================
#  Figure 3 (production) — 3 panels
# ============================================================

def make_figure_3(
    per_mode: dict,
    cells: list[dict],
    have_kpca: bool,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # ----- (a) Basin discovery curve -----
    ax = axes[0]
    for mode in MODE_ORDER:
        if mode not in per_mode:
            continue
        agg = per_mode[mode]
        iters = np.arange(1, agg["n_iters_common"] + 1)
        m, s = agg["n_basins_cum_mean"], agg["n_basins_cum_std"]
        ax.plot(iters, m, color=MODE_COLORS[mode], lw=2.0,
                label=f"{MODE_LABELS[mode]}  (n={agg['n_seeds']})")
        ax.fill_between(iters, m - s, m + s,
                        color=MODE_COLORS[mode], alpha=0.18, linewidth=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative # of basins")
    ax.set_title("(a) Basin discovery")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ----- (b) Lowest energy found -----
    ax = axes[1]
    for mode in MODE_ORDER:
        if mode not in per_mode:
            continue
        agg = per_mode[mode]
        iters = np.arange(1, agg["n_iters_common"] + 1)
        m, s = agg["lowest_E_mean"], agg["lowest_E_std"]
        ax.plot(iters, m, color=MODE_COLORS[mode], lw=2.0,
                label=MODE_LABELS[mode])
        ax.fill_between(iters, m - s, m + s,
                        color=MODE_COLORS[mode], alpha=0.18, linewidth=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lowest energy found (eV)")
    ax.set_title("(b) Lowest energy")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ----- (c) KPCA coverage area -----
    ax = axes[2]
    if have_kpca:
        per_mode_area: dict[str, list[float]] = {}
        for cell in cells:
            area = cell.get("kpca_area")
            if area is None or not np.isfinite(area):
                continue
            per_mode_area.setdefault(cell["mode"], []).append(area)

        x_pos, means, stds, colors, labels = [], [], [], [], []
        for i, mode in enumerate(MODE_ORDER):
            if mode not in per_mode_area:
                continue
            areas = np.array(per_mode_area[mode])
            x_pos.append(i)
            means.append(areas.mean())
            stds.append(areas.std())
            colors.append(MODE_COLORS[mode])
            labels.append(MODE_LABELS[mode])

        if means:
            ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor="black",
                   capsize=4, alpha=0.85, error_kw=dict(lw=1.2))
            # Overlay individual seeds as dots
            for i, mode in enumerate(MODE_ORDER):
                if mode not in per_mode_area:
                    continue
                areas = per_mode_area[mode]
                xs = np.full(len(areas), MODE_ORDER.index(mode)) \
                     + np.random.uniform(-0.08, 0.08, size=len(areas))
                ax.scatter(xs, areas, color="black", s=14, zorder=3, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Convex hull area in KPCA-2D")
            ax.set_title("(c) Configurational coverage")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No valid KPCA areas",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "(KPCA panel skipped — use without --quick)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#888")
        ax.set_axis_off()

    plt.tight_layout()
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")
    print(f"  -> {pdf}")


# ============================================================
#  Diagnostics figure — 2x2 sanity panel
# ============================================================

def make_diagnostics(per_mode: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def _plot(ax, key, title, ylabel, hline=None, hline_label=None):
        for mode in MODE_ORDER:
            if mode not in per_mode:
                continue
            agg = per_mode[mode]
            iters = np.arange(1, agg["n_iters_common"] + 1)
            m = agg[f"{key}_mean"]
            s = agg[f"{key}_std"]
            ax.plot(iters, m, color=MODE_COLORS[mode], lw=1.8,
                    label=MODE_LABELS[mode])
            ax.fill_between(iters, m - s, m + s,
                            color=MODE_COLORS[mode], alpha=0.15, linewidth=0)
        if hline is not None:
            ax.axhline(hline, color="k", linestyle="--", lw=0.8,
                       alpha=0.6,
                       label=hline_label if hline_label else None)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "T_used",
          "T_used trajectory",
          "T_used (K)",
          hline=None)
    _plot(axes[0, 1], "rho",
          "Walk acceptance ρ",
          "ρ (walk accept rate)",
          hline=0.3, hline_label="ρ* = 0.3")
    _plot(axes[1, 0], "E_min_raw",
          "E_min per iteration (raw, not running min)",
          "E_min this iter (eV)")
    _plot(axes[1, 1], "new_basins_per_iter",
          "New basin found this iteration",
          "P(new basin)")

    # One legend at top, not per-panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(MODE_ORDER), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")
    print(f"  -> {pdf}")


# ============================================================
#  Per-cell scalar table (CSV)
# ============================================================

def write_per_cell_csv(cells: list[dict], metrics: list[dict],
                       output_path: Path) -> None:
    rows = []
    for cell, m in zip(cells, metrics):
        row = {
            "run_name":         cell["run_name"],
            "mode":             cell["mode"],
            "seed":             cell["seed"],
            "n_iters":          m["n_iters"],
            "final_n_basins":   float(m["n_basins_cum"][-1]),
            "final_lowest_E":   float(m["lowest_E"][-1]),
            "mean_T_used":      float(np.nanmean(m["T_used"])),
            "mean_rho":         float(np.nanmean(m["rho"])),
            "mean_R":           float(np.nanmean(m["R"])),
            "kpca_area":        cell.get("kpca_area", np.nan),
        }
        rows.append(row)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {output_path}")


# ============================================================
#  Console summary
# ============================================================

def print_console_summary(per_mode: dict, cells: list[dict]) -> None:
    """Print a quick text summary by mode for instant inspection."""
    print("\n" + "=" * 76)
    print(f"  {'mode':<20} {'n_seeds':>8} {'final_basins':>15} {'lowest_E (eV)':>18}")
    print("-" * 76)
    for mode in MODE_ORDER:
        if mode not in per_mode:
            continue
        agg = per_mode[mode]
        nb_m = agg["n_basins_cum_mean"][-1]
        nb_s = agg["n_basins_cum_std"][-1]
        le_m = agg["lowest_E_mean"][-1]
        le_s = agg["lowest_E_std"][-1]
        print(f"  {MODE_LABELS[mode]:<20} {agg['n_seeds']:>8} "
              f"{nb_m:>7.1f} ± {nb_s:<5.1f}  {le_m:>10.4f} ± {le_s:<5.4f}")
    print("=" * 76 + "\n")


# ============================================================
#  Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ablation_results/ and produce Section 3.2 figures."
    )
    parser.add_argument(
        "ablation_root", type=Path,
        help="Path to ablation_results/ (containing ablation_summary.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for figures (default: <ablation_root>/figures/)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip the expensive SOAP+KPCA panel.",
    )
    parser.add_argument(
        "--samples-per-cell", type=int, default=500,
        help="Max frames sampled per cell for SOAP+KPCA (default: 500).",
    )
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Seed for subsampling + KPCA (default: 0).",
    )
    args = parser.parse_args()

    root = args.ablation_root
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    summary_path = root / "ablation_summary.json"
    if not summary_path.is_file():
        sys.exit(f"Missing {summary_path}; did run_ablation.py finish?")

    output_dir = args.output or (root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load global summary ----
    with summary_path.open() as f:
        global_summary = json.load(f)
    cells = global_summary["cells"]
    print(f"\nFound {len(cells)} cells in {summary_path}")

    # ---- Load per-cell logs + compute metrics ----
    print("Parsing run.jsonl files...")
    metrics: list[dict] = []
    valid_cells: list[dict] = []
    for cell in cells:
        log = load_cell_log(Path(cell["log_jsonl"]))
        if log is None:
            warnings.warn(f"  {cell['run_name']}: log missing/empty, skipped")
            continue
        m = compute_per_cell_metrics(log)
        metrics.append(m)
        valid_cells.append(cell)

    if not metrics:
        sys.exit("No valid cells found; nothing to analyze.")

    # ---- Group by mode and aggregate over seeds ----
    per_mode: dict[str, dict] = {}
    for mode in MODE_ORDER:
        seeds_for_mode = [
            m for c, m in zip(valid_cells, metrics) if c["mode"] == mode
        ]
        if seeds_for_mode:
            per_mode[mode] = aggregate_seeds(seeds_for_mode)
            print(f"  {mode}: {len(seeds_for_mode)} seed(s), "
                  f"common length = {per_mode[mode]['n_iters_common']} iters")

    # ---- Optional KPCA coverage ----
    have_kpca = False
    if not args.quick:
        print("\nComputing KPCA coverage (this is the slow step)...")
        kp = compute_kpca_coverage(
            valid_cells,
            samples_per_cell=args.samples_per_cell,
            random_state=args.random_state,
        )
        have_kpca = kp is not None
        if not have_kpca:
            print("  KPCA failed/unavailable; figure 3 (c) will be blank.")
    else:
        print("\n--quick: skipping SOAP+KPCA.")

    # ---- Produce outputs ----
    print("\nGenerating figures...")
    make_figure_3(per_mode, valid_cells, have_kpca,
                  output_dir / "figure_3")
    make_diagnostics(per_mode, output_dir / "diagnostics")
    write_per_cell_csv(valid_cells, metrics,
                       output_dir / "per_cell_metrics.csv")

    print_console_summary(per_mode, valid_cells)
    print(f"All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()