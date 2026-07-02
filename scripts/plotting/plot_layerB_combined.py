#!/usr/bin/env python
"""
plot_layerB_combined_v2.py — 2x2 combined figure for Section 3.2.

Panels
------
(a) Basin discovery
    From ablation_results/ run.jsonl files, aggregated across seeds.
    Only plots: mh_only, walk_only, serial, coupled
    (coupled_no_chain is ignored).

(b) Configurational coverage
    Rebuilt using the new occupancy-based KPCA grid coverage metric:
        coverage = occupied cells by one pool / occupied cells by union of pools
    with matched sample size per pool.

(c) Force MAE
(d) Energy MAE
    Rebuilt using the new multi-seed / three-test plotting logic:
      * bars = mean over seeds
      * error bars = std over seeds
      * black points = individual seeds
      * hatched bars = in-distribution / home test
      * tests = MH, MHMC, Neutral
      * Energy MAE uses log y-axis

Recommended usage
-----------------
First compute the bottom-panel CSVs using your new evaluation script, e.g.
    seed_metrics_summary.csv
    group_metrics_summary.csv
Then run this script:

python plot_layerB_combined_v2.py \
    --ablation-root ablation_results_Hookean \
    --al-config configs/al_loop.yaml \
    --seed-summary-csv eval_cross_multiseed_three_tests_ep100/seed_metrics_summary.csv \
    --group-summary-csv eval_cross_multiseed_three_tests_ep100/group_metrics_summary.csv \
    --out layerB/layerB_combined_v2.png
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ase.io import read as ase_read


# =============================================================================
# 字体集中控制区 —— 改这里就能改整张图所有文字
# =============================================================================
# 字体族优先级：第一个在系统里装了的就用它（Agg 后端只能用已安装字体）。
# 想用中文请把中文字体名放进来，例如 "Noto Sans CJK SC" / "SimHei"。
FONT_FAMILY_PREFERENCE: tuple[str, ...] = (
    "Arial",
    "Liberation Sans",
    "Nimbus Sans",
    "DejaVu Sans",
)

# 每个元素的字号。改某一项只影响对应文字。
FONT = {
    "base":           9.5,   # 全局兜底（任何没单独指定字号的文字）
    "axis_label":     12,    # x/y 轴标题，如 "Iteration" / "Force MAE"（坐标轴标题）
    "axis_title":     14,    # 子图标题，如 "Basin discovery"（标题）
    "tick":           10,     # 坐标轴刻度数字 + 类别刻度标签（如 "MH only"）
    "legend":         14,   # 所有图例文字（c,d 子图图例 + 底部总图例 + panel a 图例）
    "panel_label":    16,    # (a)(b)(c)(d) 角标
    "bar_annotation": 14,   # c,d 柱子顶部的数值标注
}

# 字重（"normal" / "bold"）。想让标题或轴标题加粗就改这里。
FONT_WEIGHT = {
    "axis_label":  "normal",
    "axis_title":  "normal",
    "panel_label": "bold",
}
# =============================================================================


# -----------------------------------------------------------------------------
# Optional paper style + robust font fallback
# -----------------------------------------------------------------------------
try:
    from paper_style import apply_style
except Exception:
    def apply_style() -> None:
        plt.rcParams.update({
            "font.size": FONT["base"],
            "axes.labelsize": FONT["axis_label"],
            "axes.titlesize": FONT["axis_title"],
            "axes.labelweight": FONT_WEIGHT["axis_label"],
            "axes.titleweight": FONT_WEIGHT["axis_title"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 300,
        })


def set_available_sans_font(preferred: tuple[str, ...] = FONT_FAMILY_PREFERENCE) -> str:
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            return name
    plt.rcParams["font.family"] = "sans-serif"
    return "sans-serif"


# -----------------------------------------------------------------------------
# Shared display conventions
# -----------------------------------------------------------------------------
RAW_MODE_ORDER = ["mh_only", "walk_only", "serial", "coupled_no_chain", "coupled"]
PLOT_MODE_ORDER = ["mh_only", "walk_only", "serial", "coupled"]

MODE_COLORS = {
    "mh_only": "#7F7F7F",
    "walk_only": "#8C564B",
    "serial": "#FF7F0E",
    "coupled_no_chain": "#1F77B4",
    "coupled": "#D62728",
}
MODE_LABELS = {
    "mh_only": "MH only",
    "walk_only": "Walk only",
    "serial": "Serial",
    "coupled_no_chain": "Coupled (no chain)",
    "coupled": "Coupled (full)",
}

CROSS_MODELS = ["MH", "MHMC"]
CROSS_TESTS = ["MH", "MHMC", "Neutral"]
CROSS_MODEL_COLORS = {
    "MH": MODE_COLORS["mh_only"],
    "MHMC": MODE_COLORS["coupled"],
}
CROSS_PANELS = [
    ("Force MAE", "forces_mae_meV_A", "meV / Å", False),
    ("Energy MAE", "energy_mae_meV_atom", "meV / atom ", False),
]

SOAP_HYPER_KEYS = {"r_cut", "n_max", "l_max", "sigma", "average"}


# -----------------------------------------------------------------------------
# Panel (a): basin discovery from ablation results
# -----------------------------------------------------------------------------
def load_cell_log(jsonl_path: Path) -> Optional[dict]:
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

    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    out = {}
    for k in all_keys:
        vals = [r.get(k) for r in records]
        try:
            arr = np.array([
                np.nan if v is None or v is False else (1.0 if v is True else float(v))
                for v in vals
            ], dtype=float)
            out[k] = arr
        except (TypeError, ValueError):
            out[k] = np.array(vals, dtype=object)
    out["_n_iters"] = len(records)
    return out


def compute_basin_metrics(log: dict) -> dict:
    if "n_basins_total" not in log:
        raise KeyError("run.jsonl missing key: n_basins_total")
    return {
        "n_iters": int(log["_n_iters"]),
        "n_basins_cum": log["n_basins_total"].astype(float),
    }


def aggregate_basin_seeds(per_seed_metrics: list[dict]) -> dict:
    n_common = min(m["n_iters"] for m in per_seed_metrics)
    stack = np.stack([m["n_basins_cum"][:n_common] for m in per_seed_metrics])
    return {
        "n_iters_common": n_common,
        "n_seeds": len(per_seed_metrics),
        "n_basins_cum_mean": np.nanmean(stack, axis=0),
        "n_basins_cum_std": np.nanstd(stack, axis=0),
    }


def build_ablation_aggregates(ablation_root: Path) -> Tuple[dict, list[dict]]:
    summary_path = ablation_root / "ablation_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    with summary_path.open() as f:
        global_summary = json.load(f)
    cells = global_summary["cells"]

    valid_cells = []
    metrics = []
    for cell in cells:
        log = load_cell_log(Path(cell["log_jsonl"]))
        if log is None:
            warnings.warn(f"{cell['run_name']}: log missing/empty, skipped")
            continue
        try:
            m = compute_basin_metrics(log)
        except Exception as e:
            warnings.warn(f"{cell['run_name']}: failed metrics: {e}")
            continue
        valid_cells.append(cell)
        metrics.append(m)

    if not valid_cells:
        raise SystemExit("No valid ablation cells found.")

    per_mode = {}
    for mode in RAW_MODE_ORDER:
        seeds_for_mode = [m for c, m in zip(valid_cells, metrics) if c.get("mode") == mode]
        if seeds_for_mode:
            per_mode[mode] = aggregate_basin_seeds(seeds_for_mode)
    return per_mode, valid_cells


# -----------------------------------------------------------------------------
# Panel (b): new occupancy-based coverage metric
# -----------------------------------------------------------------------------
def occupied_cells(xy: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray) -> set:
    ix = np.clip(np.digitize(xy[:, 0], edges_x) - 1, 0, len(edges_x) - 2)
    iy = np.clip(np.digitize(xy[:, 1], edges_y) - 1, 0, len(edges_y) - 2)
    return set(map(tuple, np.column_stack([ix, iy]).tolist()))


def compute_coverage_rows(
    al_config: Path,
    ablation_root: Path,
    modes: list[str],
    seeds: list[int],
    n_per_pool: int,
    grid: int,
    random_state: int,
) -> list[dict]:
    try:
        from mhmc.selection.soap_features import compute_soap_descriptors
        from mhmc.selection.kpca_reducer import KPCAReducer
    except ImportError as e:
        raise SystemExit(f"Coverage calculation requires mhmc selection modules: {e}")

    if not al_config.is_file():
        raise SystemExit(f"al_config not found: {al_config}")

    rng = np.random.default_rng(random_state)
    al_cfg = yaml.safe_load(al_config.read_text())
    soap_cfg = al_cfg.get("selection", {}).get("soap", {})
    kpca_cfg = al_cfg.get("selection", {}).get("kpca", {})
    soap_params = {k: v for k, v in soap_cfg.items() if k in SOAP_HYPER_KEYS}

    pool_paths: dict[tuple[str, int], Path] = {}
    sizes: list[int] = []
    for mode in modes:
        for seed in seeds:
            pth = ablation_root / f"{mode}_seed{seed}" / "candidate_pool.extxyz"
            if not pth.exists():
                raise SystemExit(f"pool not found: {pth}")
            n = len(ase_read(pth, index=":"))
            pool_paths[(mode, seed)] = pth
            sizes.append(n)

    n_common = min(n_per_pool, min(sizes))

    all_frames = []
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

    soap = compute_soap_descriptors(
        all_frames,
        species=soap_cfg.get("species"),
        soap_params=soap_params or None,
        n_jobs=soap_cfg.get("n_jobs", -1),
        verbose=False,
    )
    kpca = KPCAReducer(
        variance_threshold=kpca_cfg.get("variance_threshold", 0.95),
        max_components=kpca_cfg.get("max_components", 50),
        random_state=random_state,
    )
    proj = kpca.fit_transform(soap, verbose=False)
    if proj.shape[1] < 2:
        raise SystemExit("KPCA returned <2 components.")
    xy = proj[:, :2]

    pad_x = 0.02 * (xy[:, 0].max() - xy[:, 0].min() + 1e-9)
    pad_y = 0.02 * (xy[:, 1].max() - xy[:, 1].min() + 1e-9)
    edges_x = np.linspace(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x, grid + 1)
    edges_y = np.linspace(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y, grid + 1)
    global_cells = occupied_cells(xy, edges_x, edges_y)
    n_global = len(global_cells)

    rows = []
    for (mode, seed), (s, e) in slices.items():
        cov = len(occupied_cells(xy[s:e], edges_x, edges_y)) / n_global
        rows.append({"mode": mode, "seed": seed, "coverage": float(cov)})
    return rows


def read_coverage_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "mode": str(r["mode"]).strip(),
                "seed": int(r["seed"]),
                "coverage": float(r["coverage"]),
            })
    return rows

def save_individual_panels(
    per_mode: dict,
    coverage_rows: list[dict],
    group_data: dict,
    seed_data: dict,
    out_dir: Path,
    figsize: tuple[float, float] = (5.7, 4.3),
    formats: tuple[str, ...] = ("png", "pdf"),
) -> None:
    """把 (a)(b)(c)(d) 四个子图分别画成独立 figure 存到 out_dir。"""
    apply_style()
    set_available_sans_font()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, stem: Path) -> None:
        fig.tight_layout()
        for ext in formats:
            fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {stem.name}.{{{','.join(formats)}}}")

    # (a) basin discovery
    fig, ax = plt.subplots(figsize=figsize)
    draw_basin_panel(ax, per_mode)
    _save(fig, out_dir / "panel_a_basin_discovery")

    # (b) coverage
    fig, ax = plt.subplots(figsize=figsize)
    draw_coverage_panel(ax, coverage_rows)
    _save(fig, out_dir / "panel_b_coverage")

    # (c,d) cross panels —— 组图里是底部共享图例，单独画时给每张补一个图例
    cd_names = ["panel_c_force_mae", "panel_d_energy_mae"]
    cd_legend = [
        Patch(facecolor=CROSS_MODEL_COLORS["MH"], edgecolor="white", label="MH model"),
        Patch(facecolor=CROSS_MODEL_COLORS["MHMC"], edgecolor="white", label="Coupled model"),
        Patch(facecolor="0.82", edgecolor="0.45", hatch="////", label="in-distribution"),
    ]
    for (title, key, ylabel, logy), name in zip(CROSS_PANELS, cd_names):
        fig, ax = plt.subplots(figsize=figsize)
        draw_cross_panel(ax, group_data, seed_data, key, title, ylabel, logy)
        ax.legend(handles=cd_legend, loc="best",
                  fontsize=FONT["legend"], framealpha=0.9)
        _save(fig, out_dir / name)

    print(f"Individual panels saved to: {out_dir}")

# -----------------------------------------------------------------------------
# Panels (c,d): new multi-seed / three-test bar plots
# -----------------------------------------------------------------------------
def read_group_summary_for_plot(path: Path) -> Dict[Tuple[str, str], dict]:
    data: Dict[Tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["model_group"].strip()
            t = r["test"].replace("_test", "").strip()
            data[(m, t)] = r
    missing = [(m, t) for m in CROSS_MODELS for t in CROSS_TESTS if (m, t) not in data]
    if missing:
        raise SystemExit(f"group summary CSV is missing cells: {missing}")
    return data


def read_seed_summary_for_plot(path: Path) -> Dict[Tuple[str, str], List[dict]]:
    data: Dict[Tuple[str, str], List[dict]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = r["model_group"].strip()
            t = r["test"].replace("_test", "").strip()
            data.setdefault((m, t), []).append(r)
    return data


def fmt(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def add_panel_labels(
    axes,
    labels=("a", "b", "c", "d"),
    x: float = -0.13,
    y: float = 1.08,
    fontsize: float = FONT["panel_label"],
    fontweight: str = FONT_WEIGHT["panel_label"],
) -> None:
    flat_axes = np.ravel(axes)
    for ax, lab in zip(flat_axes, labels):
        ax.text(
            x, y, f"({lab})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize,
            fontweight=fontweight,
            clip_on=False,
        )


def draw_basin_panel(ax, per_mode: dict) -> None:
    for mode in PLOT_MODE_ORDER:
        if mode not in per_mode:
            continue
        agg = per_mode[mode]
        iters = np.arange(1, agg["n_iters_common"] + 1)
        m = agg["n_basins_cum_mean"]
        s = agg["n_basins_cum_std"]
        ax.plot(iters, m, color=MODE_COLORS[mode], lw=2.0,
                label=f"{MODE_LABELS[mode]}  (n={agg['n_seeds']})")
        ax.fill_between(iters, m - s, m + s,
                        color=MODE_COLORS[mode], alpha=0.18, linewidth=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative # of basins")
    ax.set_title("Basin discovery")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=FONT["legend"], framealpha=0.9)


def draw_coverage_panel(ax, coverage_rows: list[dict]) -> None:
    per_mode: dict[str, list[float]] = {m: [] for m in PLOT_MODE_ORDER}
    for r in coverage_rows:
        mode = r["mode"]
        if mode in per_mode:
            per_mode[mode].append(float(r["coverage"]))

    x = np.arange(len(PLOT_MODE_ORDER))
    means = [float(np.mean(per_mode[m])) if per_mode[m] else np.nan for m in PLOT_MODE_ORDER]
    stds = [float(np.std(per_mode[m], ddof=1)) if len(per_mode[m]) > 1 else 0.0 for m in PLOT_MODE_ORDER]

    ax.bar(x, means, yerr=stds, capsize=4,
           color=[MODE_COLORS[m] for m in PLOT_MODE_ORDER],
           edgecolor="white", linewidth=0.9, zorder=3)

    rng = np.random.default_rng(0)
    for xi, mode in enumerate(PLOT_MODE_ORDER):
        pts = per_mode[mode]
        if not pts:
            continue
        jit = (rng.random(len(pts)) - 0.5) * 0.12
        ax.scatter(np.full(len(pts), xi) + jit, pts, s=22, c="0.2",
                   zorder=5, edgecolors="white", linewidths=0.4)

    ymax = np.nanmax(np.array(means) + np.array(stds))
    ax.set_ylim(0, ymax * 1.25 if np.isfinite(ymax) and ymax > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in PLOT_MODE_ORDER], rotation=12)
    ax.set_ylabel("Configuration-space coverage\n(fraction of explored KPCA cells, matched N)")
    ax.set_title("Configurational coverage")
    ax.grid(axis="y")
    ax.xaxis.grid(False)


def draw_cross_panel(ax, group_data, seed_data, key: str, title: str, ylabel: str, logy: bool):
    x = np.arange(len(CROSS_TESTS), dtype=float)
    width = 0.38
    vals_all = []

    seed_jitter = {42: -0.055, 43: 0.0, 44: 0.055}

    for mi, model_group in enumerate(CROSS_MODELS):
        offs = (mi - 0.5) * width
        means = []
        stds = []
        seed_vals_by_test = []

        for test_label in CROSS_TESTS:
            g = group_data[(model_group, test_label)]
            mean_v = float(g[f"{key}_mean"])
            std_v = float(g[f"{key}_std"])
            means.append(mean_v)
            stds.append(std_v)
            vals_all.append(mean_v)

            seed_rows = seed_data.get((model_group, test_label), [])
            vals = [(int(r["seed"]), float(r[key])) for r in seed_rows]
            seed_vals_by_test.append(vals)
            vals_all.extend([v for _, v in vals])

        bars = ax.bar(
            x + offs, means, width,
            yerr=stds, capsize=3.0,
            color=CROSS_MODEL_COLORS.get(model_group, "0.6"),
            edgecolor="white", linewidth=0.9,
            zorder=3, label=f"{model_group} model",
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "zorder": 4},
        )

        for b, test_label in zip(bars, CROSS_TESTS):
            if model_group == test_label:
                b.set_hatch("////")

        for test_i, vals in enumerate(seed_vals_by_test):
            for seed, val in vals:
                jitter = seed_jitter.get(seed, 0.0)
                ax.scatter(x[test_i] + offs + jitter, val,
                           s=17, c="0.15", edgecolors="white",
                           linewidths=0.35, zorder=5)

        for b, mean_v in zip(bars, means):
            xytext = (0, 5) if logy else (0, 3)
            ax.annotate(
                fmt(mean_v),
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=xytext, textcoords="offset points",
                ha="center", va="bottom", fontsize=FONT["bar_annotation"], zorder=6,
            )

    finite_vals = [v for v in vals_all if np.isfinite(v) and v > 0]
    if not finite_vals:
        finite_vals = [1.0]
    lo, hi = min(finite_vals), max(finite_vals)

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(lo * 0.45, hi * 3.2)
    else:
        ax.set_ylim(0, hi * 1.25)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} test" for t in CROSS_TESTS])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.xaxis.grid(False)


# -----------------------------------------------------------------------------
# Main figure assembly
# -----------------------------------------------------------------------------
def make_combined_figure(
    per_mode: dict,
    coverage_rows: list[dict],
    group_data: dict,
    seed_data: dict,
    out_path: Path,
    panel_label_x: float,
    panel_label_y: float,
    panel_label_fontsize: float,
) -> None:
    apply_style()
    set_available_sans_font()

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))

    draw_basin_panel(axes[0, 0], per_mode)
    draw_coverage_panel(axes[0, 1], coverage_rows)
    for ax, (title, key, ylabel, logy) in zip(axes[1, :], CROSS_PANELS):
        draw_cross_panel(ax, group_data, seed_data, key, title, ylabel, logy)

    add_panel_labels(
        axes,
        x=panel_label_x,
        y=panel_label_y,
        fontsize=panel_label_fontsize,
    )

    legend_handles = [
        Patch(facecolor=CROSS_MODEL_COLORS["MH"], edgecolor="white", label="MH model"),
        Patch(facecolor=CROSS_MODEL_COLORS["MHMC"], edgecolor="white", label="Coupled model"),
        Patch(facecolor="0.82", edgecolor="0.45", hatch="////", label="in-distribution"),
        # Patch(facecolor="white", edgecolor="0.15", label="points = seeds"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.005), frameon=False,
               fontsize=FONT["legend"])

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_path}\n  {out_path.with_suffix('.pdf')}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the updated 2x2 combined figure for Section 3.2."
    )
    p.add_argument("--ablation-root", type=Path, default=Path("data/figure_data/ablation_results_Hookean/"),
                   help="ablation_results_*/ directory containing ablation_summary.json")
    p.add_argument("--seed-summary-csv", type=Path, default=Path("data/figure_data/seed_metrics_summary.csv"),
                   help="seed_metrics_summary.csv from the new 3-test evaluation script")
    p.add_argument("--group-summary-csv", type=Path, default=Path("data/figure_data/group_metrics_summary.csv"),
                   help="group_metrics_summary.csv from the new 3-test evaluation script")

    # Coverage metric inputs: either provide precomputed coverage CSV,
    # or provide al_config so this script can compute panel (b) itself.
    p.add_argument("--coverage-csv", type=Path, default=Path("data/figure_data/coverage_metric.csv"),
                   help="Optional precomputed coverage_metric.csv")
    p.add_argument("--al-config", type=Path, default=Path("configs/al_loop.yaml"),
                   help="al_loop.yaml for computing panel (b) if --coverage-csv is not given")
    p.add_argument("--coverage-modes", nargs="+",
                   default=["mh_only", "walk_only", "serial", "coupled"])
    p.add_argument("--coverage-seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--n-per-pool", type=int, default=1500)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--random-state", type=int, default=0)

    p.add_argument("--out", type=Path, default=Path("layerB/layerB_combined_v2.png"))
    p.add_argument("--panel-label-x", type=float, default=-0.13)
    p.add_argument("--panel-label-y", type=float, default=1.08)
    p.add_argument("--panel-label-fontsize", type=float, default=FONT["panel_label"])
    p.add_argument("--panels-dir", type=Path, default=Path("layerB/layerB_figures"),
                   help="单独保存每个子图的目录；默认是 --out 同级的 'panels/' 子目录")
    args = p.parse_args()

    print("Loading ablation data for panel (a)...")
    per_mode, _ = build_ablation_aggregates(args.ablation_root)

    print("Preparing coverage data for panel (b)...")
    if args.coverage_csv is not None:
        coverage_rows = read_coverage_csv(args.coverage_csv)
    else:
        if args.al_config is None:
            raise SystemExit("Need either --coverage-csv or --al-config for panel (b).")
        coverage_rows = compute_coverage_rows(
            al_config=args.al_config,
            ablation_root=args.ablation_root,
            modes=args.coverage_modes,
            seeds=args.coverage_seeds,
            n_per_pool=args.n_per_pool,
            grid=args.grid,
            random_state=args.random_state,
        )

    print("Loading bottom-panel summaries for panels (c,d)...")
    group_data = read_group_summary_for_plot(args.group_summary_csv)
    seed_data = read_seed_summary_for_plot(args.seed_summary_csv)

    print("Drawing combined figure...")
    make_combined_figure(
        per_mode=per_mode,
        coverage_rows=coverage_rows,
        group_data=group_data,
        seed_data=seed_data,
        out_path=args.out,
        panel_label_x=args.panel_label_x,
        panel_label_y=args.panel_label_y,
        panel_label_fontsize=args.panel_label_fontsize,
    )
    panels_dir = args.panels_dir if args.panels_dir is not None else args.out.parent / "panels"
    print("Saving individual panels...")
    save_individual_panels(
        per_mode=per_mode,
        coverage_rows=coverage_rows,
        group_data=group_data,
        seed_data=seed_data,
        out_dir=panels_dir,
    )


if __name__ == "__main__":
    main()