#!/usr/bin/env python
"""Pre-flight check for the MHMC handoff repository."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def check_import(name: str, import_name: str | None = None) -> bool:
    target = import_name or name
    try:
        mod = importlib.import_module(target)
        version = getattr(mod, "__version__", "")
        print(f"[ OK ] {name:14s} {version}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name:14s} {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Check MHMC dependencies and input paths.")
    ap.add_argument("config", nargs="?", default="configs/Fe110.yaml")
    args = ap.parse_args()

    print("=== Python dependencies ===")
    core = [
        ("numpy", None),
        ("scipy", None),
        ("PyYAML", "yaml"),
        ("ASE", "ase"),
        ("NequIP", "nequip"),
    ]
    optional = [
        ("tqdm", None),
        ("scikit-learn", "sklearn"),
        ("DScribe", "dscribe"),
        ("pandas", None),
        ("matplotlib", None),
    ]
    core_ok = all(check_import(a, b) for a, b in core)
    print("\n=== Optional / AL dependencies ===")
    for a, b in optional:
        check_import(a, b)

    try:
        import yaml
    except Exception:
        print("\nCannot inspect YAML paths because PyYAML is unavailable.")
        return 2

    cfg_path = Path(args.config)
    print(f"\n=== Config: {cfg_path} ===")
    if not cfg_path.is_file():
        print(f"[FAIL] config not found: {cfg_path}")
        return 2

    raw = yaml.safe_load(cfg_path.read_text())
    sys_cfg = raw.get("system", {})
    required_paths = {
        "initial_dataset": sys_cfg.get("initial_dataset"),
        "mlp_model": sys_cfg.get("mlp_model"),
    }
    paths_ok = True
    for key, value in required_paths.items():
        if not value:
            print(f"[FAIL] system.{key} is not configured")
            paths_ok = False
            continue
        p = Path(value)
        status = " OK " if p.is_file() else "FAIL"
        print(f"[{status}] system.{key}: {p}")
        paths_ok &= p.is_file()

    slab_n = raw.get("topology", {}).get("slab_n_atoms")
    print(f"[INFO] slab_n_atoms = {slab_n}")
    device = sys_cfg.get("device", "cuda")
    print(f"[INFO] device       = {device}")

    t_init = raw.get("escape", {}).get("T_init")
    t_min = raw.get("feedback", {}).get("T_min")
    if t_init is not None and t_min is not None and t_init < t_min:
        print(
            f"[WARN] escape.T_init ({t_init}) < feedback.T_min ({t_min}). "
            "The first escape can run below the later clipping bound."
        )

    print("\n=== Result ===")
    if core_ok and paths_ok:
        print("Core prerequisites look present. Next: run a 1-iteration smoke test.")
        print(f"  python run.py {cfg_path} --mode coupled --seed 42 --n_iters 1")
        return 0
    print("Fix the FAIL items above before launching MHMC.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
