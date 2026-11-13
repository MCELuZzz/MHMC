#!/usr/bin/env python
"""
Batch driver for Section 3.2 ablation experiments.

Runs the MHMC sampler under multiple (mode, seed) combinations and collects
results into  <output_root>/<mode>_seed<N>/  subdirectories for downstream
analysis (analyze_ablation.py).

Each combination runs in its own subprocess for isolation:
  - No GPU memory accumulation across runs
  - No calculator state leaking between runs
  - One run's crash does NOT kill the rest of the ablation

The MHMC code itself is NOT modified; this script only patches the base
MHMC yaml (mode, seed, output paths) for each cell, then drives the
subprocesses with a simple n_workers pool.

Usage
-----
    python run_ablation.py configs/ablation.yaml
    python run_ablation.py configs/ablation.yaml --dry-run
    python run_ablation.py configs/ablation.yaml --modes coupled,coupled_no_chain
    python run_ablation.py configs/ablation.yaml --seeds 42 --modes coupled

Resumption
----------
By default the driver SKIPS any cell whose output directory already contains
a 'run.jsonl' from a previous successful run. Use --force to redo everything.

This script also serves as its own worker. When invoked with --worker, it
runs a single MHMC pass for one (mode, seed) cell in the current process.
This dual-role design avoids the need for a separate worker script.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
#  Worker mode: execute one MHMC config in this process
# ============================================================

def _worker_main(patched_yaml: Path, output_dir: Path) -> None:
    """Run a single MHMC pass.

    Invoked by the driver via subprocess.Popen([..., '--worker', ...]).
    Reads the patched yaml, builds the initial Atoms + calculator,
    instantiates MHMCSampler, and runs it.
    """
    from ase.io import read as ase_read

    from mhmc.core.sampler import MHMCConfig, MHMCSampler

    with patched_yaml.open() as f:
        cfg = yaml.safe_load(f)

    # ---- 1. Initial atoms (slab + adsorbate, pre-MD) ----
    # The MHMC start point is the first frame of `initial_dataset`
    # (the validated step-2.1 data containing slab+adsorbate), NOT the
    # bare `slab_file` (which is the pure Fe slab kept for reference /
    # AIMD baselines). An optional integer can override the frame index.
    init_path = cfg["system"]["initial_dataset"]
    init_index = int(cfg.get("system", {}).get("initial_index", 0))
    init_atoms = ase_read(init_path, index=init_index)

    # ---- 2. Calculator (delegate to run.py for guaranteed consistency) ----
    # We deliberately reuse run.build_calculator instead of re-implementing
    # the NequIP + Hookean wiring here. This guarantees that the ablation
    # uses byte-identical calculator construction to the production AL
    # pipeline, so any 3.2 conclusion is apples-to-apples.
    from run import build_calculator
    calc = build_calculator(cfg["system"], cfg.get("hookean"))

    # ---- 3. Translate yaml -> MHMCConfig ----
    mhmc_cfg = _yaml_to_mhmcconfig(cfg)

    # ---- 4. Run sampler ----
    sampler = MHMCSampler(
        initial_atoms=init_atoms,
        calculator=calc,
        config=mhmc_cfg,
    )
    basin_db = sampler.run()

    # ---- 5. Per-run summary ----
    energies = [b.energy for b in basin_db.basins]
    summary = {
        "mode": mhmc_cfg.mode,
        "seed": mhmc_cfg.seed,
        "n_iters": mhmc_cfg.n_iters,
        "n_basins_total": len(basin_db),
        "lowest_E": min(energies) if energies else None,
        "highest_E": max(energies) if energies else None,
        "basin_energies": energies,
    }
    out_summary = output_dir / "run_summary.json"
    with out_summary.open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[worker] run complete, summary -> {out_summary}")


def _yaml_to_mhmcconfig(cfg: dict):
    """Translate an Fe110.yaml-style dict into MHMCConfig kwargs."""
    from mhmc.core.sampler import MHMCConfig
    return MHMCConfig(
        # Topology
        slab_n_atoms=cfg["topology"]["slab_n_atoms"],
        fixed_indices=cfg["topology"].get("fixed_indices"),
        # Basin
        eps_E=cfg["basin"]["eps_E"],
        eps_R=cfg["basin"]["eps_R"],
        # Escape
        T_init=cfg["escape"]["T_init"],
        md_steps=cfg["escape"]["md_steps"],
        md_timestep_fs=cfg["escape"]["md_timestep_fs"],
        relax_fmax=cfg["escape"]["relax_fmax"],
        relax_steps=cfg["escape"]["relax_steps"],
        # Walk
        T_walk=cfg["walk"]["T_walk"],
        n_walk_steps=cfg["walk"]["n_walk_steps"],
        walk_sigma=cfg["walk"]["walk_sigma"],
        dE_basin=cfg["walk"]["dE_basin"],
        # Feedback
        lambda_=cfg["feedback"]["lambda_"],
        rho_star=cfg["feedback"]["rho_star"],
        R_star=cfg["feedback"]["R_star"],
        gamma_rho=cfg["feedback"]["gamma_rho"],
        gamma_R=cfg["feedback"]["gamma_R"],
        T_min=cfg["feedback"]["T_min"],
        T_max=cfg["feedback"]["T_max"],
        # Chain
        E_diff_init=cfg["chain"]["E_diff_init"],
        E_diff_min=cfg["chain"]["E_diff_min"],
        E_diff_max=cfg["chain"]["E_diff_max"],
        chain_window=cfg["chain"]["chain_window"],
        chain_acc_target=cfg["chain"]["chain_acc_target"],
        chain_adapt_rate=cfg["chain"]["chain_adapt_rate"],
        reject_T_factor=cfg["chain"]["reject_T_factor"],
        # Dataset
        md_stride=cfg["dataset"]["md_stride"],
        relax_stride=cfg["dataset"]["relax_stride"],
        walk_rejected_keep_ratio=cfg["dataset"]["walk_rejected_keep_ratio"],
        dataset_flush_every=cfg["dataset"]["dataset_flush_every"],
        # Loop
        n_iters=cfg["loop"]["n_iters"],
        mode=cfg["loop"]["mode"],
        checkpoint_every=cfg["loop"]["checkpoint_every"],
        # Output
        log_path=cfg["output"]["log_path"],
        checkpoint_path=cfg["output"]["checkpoint_path"],
        dataset_extxyz_path=cfg["output"]["dataset_extxyz_path"],
        dataset_index_path=cfg["output"]["dataset_index_path"],
        # Reproducibility
        seed=cfg["reproducibility"]["seed"],

        # Surface confinement (optional; default None = disabled)
        z_ceiling_offset=cfg.get("topology", {}).get("z_ceiling_offset"),
        z_floor_offset=cfg.get("topology", {}).get("z_floor_offset"),
        z_confine_k=cfg.get("topology", {}).get("z_confine_k", 10.0),
    )


# ============================================================
#  Driver helpers
# ============================================================

def _deep_update(d: dict, u: dict) -> None:
    """Recursively update nested dict d with values from u (in place)."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def _patch_yaml(
    base_cfg: dict,
    mode: str,
    seed: int,
    output_dir: Path,
    overrides: Optional[dict] = None,
) -> dict:
    """Build the patched MHMC config for one ablation cell.

    Order of operations (later wins):
      1. deep-copy of base config
      2. apply user 'overrides' from ablation.yaml
      3. force loop.mode and reproducibility.seed
      4. force output.* paths into output_dir
    """
    patched = copy.deepcopy(base_cfg)

    if overrides:
        _deep_update(patched, overrides)

    patched.setdefault("loop", {})
    patched.setdefault("reproducibility", {})
    patched["loop"]["mode"] = mode
    patched["reproducibility"]["seed"] = int(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    patched["output"] = {
        "log_path":            str(output_dir / "run.jsonl"),
        "checkpoint_path":     str(output_dir / "state.pkl"),
        "dataset_extxyz_path": str(output_dir / "candidate_pool.extxyz"),
        "dataset_index_path": str(output_dir / "candidate_pool_index.jsonl"),
    }
    return patched


def _is_already_done(output_dir: Path) -> bool:
    """A cell is 'already done' iff run.jsonl exists and is non-empty
    AND run_summary.json exists (worker only writes it on success)."""
    log = output_dir / "run.jsonl"
    summary = output_dir / "run_summary.json"
    return log.is_file() and log.stat().st_size > 0 and summary.is_file()


def _spawn_worker(patched_yaml: Path, output_dir: Path, log_file: Path):
    """Spawn one MHMC subprocess. Returns a Popen with log_fh attached."""
    import subprocess
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--config", str(patched_yaml),
        "--output-dir", str(output_dir),
    ]
    log_fh = log_file.open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    return proc


# ============================================================
#  Driver: orchestrate parallel subprocesses
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Section 3.2 ablation runner."
    )
    parser.add_argument(
        "ablation_config", type=Path, nargs="?",
        help="ablation.yaml path (omit only when using --worker)",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help="Comma-separated mode subset, overrides ablation.yaml",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed subset, overrides ablation.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs without executing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run cells whose outputs already exist.",
    )

    # Internal worker flags (used when this script invokes itself)
    parser.add_argument("--worker", action="store_true",
                        help="(internal) run one MHMC config in this process")
    parser.add_argument("--config", type=Path, default=None,
                        help="(internal) patched MHMC yaml for the worker")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(internal) per-run output directory")

    args = parser.parse_args()

    # -------- Worker branch --------
    if args.worker:
        if args.config is None or args.output_dir is None:
            parser.error("--worker requires --config and --output-dir")
        _worker_main(args.config, args.output_dir)
        return

    # -------- Driver branch --------
    if args.ablation_config is None:
        parser.error("ablation_config is required")

    with args.ablation_config.open() as f:
        abl_cfg = yaml.safe_load(f)

    base_config_path = Path(abl_cfg["base_config"])
    output_root      = Path(abl_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    modes = (args.modes.split(",") if args.modes else abl_cfg["modes"])
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [int(s) for s in abl_cfg["seeds"]])
    overrides = abl_cfg.get("overrides", {}) or {}
    n_workers = max(1, int(abl_cfg.get("n_workers", 1)))

    # Validate mode names against MHMCSampler.VALID_MODES
    from mhmc.core.sampler import MHMCSampler
    bad = set(modes) - set(MHMCSampler.VALID_MODES)
    if bad:
        sys.exit(
            f"Unknown modes: {sorted(bad)}\n"
            f"Valid: {MHMCSampler.VALID_MODES}"
        )

    with base_config_path.open() as f:
        base_cfg = yaml.safe_load(f)

    # ---- Build the plan ----
    plan = []
    for mode in modes:
        for seed in seeds:
            run_name = f"{mode}_seed{seed}"
            output_dir = output_root / run_name
            plan.append((mode, seed, run_name, output_dir))

    print(f"\nAblation plan")
    print(f"  base config : {base_config_path}")
    print(f"  output root : {output_root}")
    print(f"  total cells : {len(plan)}")
    print(f"  parallelism : {n_workers} worker(s)")
    print(f"  overrides   : {overrides if overrides else '(none)'}\n")

    todo = []
    for mode, seed, run_name, output_dir in plan:
        if (not args.force) and _is_already_done(output_dir):
            print(f"  [SKIP]  {run_name}  (already done)")
            continue
        print(f"  [QUEUE] {run_name}  -> {output_dir}")
        todo.append((mode, seed, run_name, output_dir))

    if not todo:
        print("\nAll cells already done. Use --force to redo.")
        return

    if args.dry_run:
        print("\nDry run, exiting.")
        return

    # ---- Patch yamls ----
    cells = []
    for mode, seed, run_name, output_dir in todo:
        patched_cfg = _patch_yaml(base_cfg, mode, seed, output_dir, overrides)
        patched_yaml = output_dir / "patched_config.yaml"
        with patched_yaml.open("w") as f:
            yaml.dump(patched_cfg, f, sort_keys=False)
        cells.append((patched_yaml, output_dir, run_name))

    # ---- Dispatch with a simple worker pool ----
    pending = list(cells)
    running = []     # list of (Popen, run_name, t_start)

    print(f"\n{'='*60}")
    print(f"Starting {len(pending)} run(s) with {n_workers} worker(s)")
    print(f"{'='*60}")

    while pending or running:
        # Fill up to n_workers
        while pending and len(running) < n_workers:
            patched_yaml, output_dir, run_name = pending.pop(0)
            log_file = output_dir / "stdout.log"
            print(f"\n[start] {run_name}")
            proc = _spawn_worker(patched_yaml, output_dir, log_file)
            running.append((proc, run_name, time.time()))

        # Reap finished
        still = []
        for proc, run_name, t0 in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, run_name, t0))
                continue
            proc._log_fh.close()  # type: ignore[attr-defined]
            elapsed = time.time() - t0
            status = "OK" if ret == 0 else f"FAIL (exit {ret})"
            print(f"[done ] {run_name}  ({elapsed:.0f}s)  {status}")
            if ret != 0:
                print(f"        check stdout.log in the cell directory")
        running = still

        if running:
            time.sleep(2.0)

    # ---- Aggregate global summary ----
    global_summary = {
        "base_config": str(base_config_path),
        "output_root": str(output_root),
        "overrides":   overrides,
        "cells":       [],
    }
    for mode, seed, run_name, output_dir in plan:
        entry = {
            "run_name":       run_name,
            "mode":           mode,
            "seed":           seed,
            "output_dir":     str(output_dir),
            "candidate_pool": str(output_dir / "candidate_pool.extxyz"),
            "log_jsonl":      str(output_dir / "run.jsonl"),
        }
        per_summary = output_dir / "run_summary.json"
        if per_summary.is_file():
            with per_summary.open() as f:
                entry["summary"] = json.load(f)
        global_summary["cells"].append(entry)

    summary_path = output_root / "ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(global_summary, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f"Ablation complete.")
    print(f"  global summary -> {summary_path}")
    print(f"  per-cell dirs  -> {output_root}/<mode>_seed<N>/")
    print(f"\nNext step: python analyze_ablation.py {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
