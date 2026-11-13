"""
Run a single MHMC experiment described by a YAML config.

Usage
-----
    python run.py configs/Fe110.yaml
    python run.py configs/Fe110.yaml --mode mh_only --seed 1

Output behavior
---------------
Each run automatically creates a timestamped subdirectory under outputs/:
    outputs/2026-05-05_143022_Fe110_NH3_coupled_seed42/
        ├── config_used.yaml   # snapshot of resolved config
        ├── run.jsonl          # per-iteration log
        └── state.pkl          # final checkpoint

For ablation sweeps:
    for mode in mh_only walk_only serial coupled; do
      for seed in 1 2 3; do
        python run.py configs/Fe110.yaml --mode $mode --seed $seed
      done
    done
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from ase.io import read

from mhmc.core.sampler import MHMCSampler, MHMCConfig
from mhmc.physics.hookean import HookeanWrappedCalculator


# ------------------------------------------------------------
#  Calculator factory
# ------------------------------------------------------------
def build_calculator(system_cfg: dict, hookean_cfg: dict | None):
    """
    Build the calculator. Replace the body to plug in your MLP.
    """
    # ---- NequIP MLP ----
    from nequip.ase import NequIPCalculator
    model_path = Path(system_cfg["mlp_model"])
    if not model_path.is_file():
        raise FileNotFoundError(
            f"NequIP model not found: {model_path}. "
            "Set system.mlp_model in the YAML config."
        )
    base = NequIPCalculator.from_compiled_model(
        compile_path=str(model_path),
        device=system_cfg.get("device", "cuda"),
    )
    # --------------------

    if hookean_cfg and hookean_cfg.get("enabled", False):
        bond_pairs = [tuple(p) for p in hookean_cfg["bond_pairs"]]
        return HookeanWrappedCalculator(
            base_calc=base,
            bond_pairs=bond_pairs,
            k=hookean_cfg.get("k", 10.0),
            beta=hookean_cfg.get("beta", 1.5),
        )
    return base


# ------------------------------------------------------------
#  Config loader (with auto-timestamped output directory)
# ------------------------------------------------------------
def load_config(yaml_path: str, overrides: dict) -> tuple[MHMCConfig, dict, dict]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # ----- Apply CLI overrides to YAML before deciding output paths -----
    # so that --mode / --seed flow into the directory name.
    if overrides.get("mode") is not None:
        raw["loop"]["mode"] = overrides["mode"]
    if overrides.get("seed") is not None:
        raw["reproducibility"]["seed"] = overrides["seed"]
    if overrides.get("n_iters") is not None:
        raw["loop"]["n_iters"] = overrides["n_iters"]

    # ----- Auto-generate timestamped output directory -----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sys_name = raw["system"].get("name", "run")
    mode = raw["loop"].get("mode", "coupled")
    seed = raw["reproducibility"].get("seed", 0)

    run_dir = Path("outputs") / f"{timestamp}_{sys_name}_{mode}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write resolved output paths back into raw (CLI flags can still override below)
    raw["output"]["log_path"] = str(run_dir / "run.jsonl")
    raw["output"]["checkpoint_path"] = str(run_dir / "state.pkl")
    raw["output"]["dataset_extxyz_path"] = str(run_dir / "candidate_pool.extxyz")
    raw["output"]["dataset_index_path"] = str(run_dir / "candidate_pool_index.jsonl")

    # ----- Apply CLI overrides for explicit log/checkpoint paths -----
    if overrides.get("log_path") is not None:
        raw["output"]["log_path"] = overrides["log_path"]
    if overrides.get("checkpoint_path") is not None:
        raw["output"]["checkpoint_path"] = overrides["checkpoint_path"]

    # ----- Snapshot the resolved config to the run directory -----
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)

    # ----- Build MHMCConfig from resolved YAML -----
    cfg = MHMCConfig(
        slab_n_atoms     = raw["topology"]["slab_n_atoms"],
        fixed_indices    = raw["topology"].get("fixed_indices"),

        eps_E            = raw["basin"]["eps_E"],
        eps_R            = raw["basin"]["eps_R"],

        T_init           = raw["escape"]["T_init"],
        md_steps         = raw["escape"]["md_steps"],
        md_timestep_fs   = raw["escape"]["md_timestep_fs"],
        relax_fmax       = raw["escape"]["relax_fmax"],
        relax_steps      = raw["escape"]["relax_steps"],

        T_walk           = raw["walk"]["T_walk"],
        n_walk_steps     = raw["walk"]["n_walk_steps"],
        walk_sigma       = raw["walk"]["walk_sigma"],
        dE_basin         = raw["walk"]["dE_basin"],

        lambda_          = raw["feedback"]["lambda_"],
        rho_star         = raw["feedback"]["rho_star"],
        R_star           = raw["feedback"]["R_star"],
        gamma_rho        = raw["feedback"]["gamma_rho"],
        gamma_R          = raw["feedback"]["gamma_R"],
        T_min            = raw["feedback"]["T_min"],
        T_max            = raw["feedback"]["T_max"],

        n_iters          = raw["loop"]["n_iters"],
        mode             = raw["loop"]["mode"],
        checkpoint_every = raw["loop"]["checkpoint_every"],

        log_path         = raw["output"]["log_path"],
        checkpoint_path  = raw["output"]["checkpoint_path"],

        seed             = raw["reproducibility"]["seed"],

        # Surface confinement (optional; default None = disabled)
        z_ceiling_offset = raw.get("topology", {}).get("z_ceiling_offset"),
        z_floor_offset   = raw.get("topology", {}).get("z_floor_offset"),
        z_confine_k      = raw.get("topology", {}).get("z_confine_k", 10.0),

        # NEW: chain control fields
        E_diff_init      = raw["chain"]["E_diff_init"],
        E_diff_min       = raw["chain"]["E_diff_min"],
        E_diff_max       = raw["chain"]["E_diff_max"],
        chain_window     = raw["chain"]["chain_window"],
        chain_acc_target = raw["chain"]["chain_acc_target"],
        chain_adapt_rate = raw["chain"]["chain_adapt_rate"],
        reject_T_factor  = raw["chain"]["reject_T_factor"],   # NEW   
            
            # NEW: dataset collection fields
        md_stride                = raw["dataset"]["md_stride"],
        relax_stride             = raw["dataset"]["relax_stride"],
        walk_rejected_keep_ratio = raw["dataset"]["walk_rejected_keep_ratio"],
        dataset_flush_every      = raw["dataset"]["dataset_flush_every"],
        dataset_extxyz_path      = raw["output"]["dataset_extxyz_path"],
        dataset_index_path       = raw["output"]["dataset_index_path"],
    
    )

    return cfg, raw["system"], raw.get("hookean")


# ------------------------------------------------------------
#  Entry
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to YAML config")
    p.add_argument("--mode", default=None,
                   choices=["mh_only", "walk_only", "serial", "coupled_no_chain", "coupled"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_iters", type=int, default=None)
    p.add_argument("--log_path", default=None,
                   help="Override automatic log path")
    p.add_argument("--checkpoint_path", default=None,
                   help="Override automatic checkpoint path")
    args = p.parse_args()

    overrides = {
        "mode": args.mode,
        "seed": args.seed,
        "n_iters": args.n_iters,
        "log_path": args.log_path,
        "checkpoint_path": args.checkpoint_path,
    }
    cfg, sys_cfg, hk_cfg = load_config(args.config, overrides)

    # Initial structure: take first frame from initial dataset
    init_path = Path(sys_cfg["initial_dataset"])
    if not init_path.is_file():
        raise FileNotFoundError(
            f"Initial dataset not found: {init_path}. "
            "Set system.initial_dataset in the YAML config."
        )
    initial_index = int(sys_cfg.get("initial_index", 0))
    initial = read(str(init_path), index=initial_index)

    calc = build_calculator(sys_cfg, hk_cfg)

    print(f"[run_mhmc] system={sys_cfg['name']}  mode={cfg.mode}  "
          f"seed={cfg.seed}  n_iters={cfg.n_iters}")
    print(f"[run_mhmc] log -> {cfg.log_path}")
    print(f"[run_mhmc] ckpt -> {cfg.checkpoint_path}")

    sampler = MHMCSampler(initial_atoms=initial, calculator=calc, config=cfg)
    db = sampler.run()
    print("\n[run_mhmc] ===== Summary =====")
    print(f"  basins discovered: {len(db)}")
    s = db.summary()
    print(f"  energy range (eV): {s['energy_range']}")
    print(f"  total walk frames stored: {s['total_members']}")


if __name__ == "__main__":
    main()