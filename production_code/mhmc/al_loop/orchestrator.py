"""
Single-round AL orchestration.

A "round" performs:
  1. Run MHMC sampler under the current MLP (via subprocess call to run.py)
  2. Compute SOAP descriptors from the candidate pool
  3. KPCA dimensionality reduction
  4. Stratified FPS selection
  5. Generate VASP input directories for the selected configurations

After step 5, control returns to the user, who:
  - Submits VASP jobs from the generated input directories
  - Collects DFT results
  - Merges into the cumulative training set
  - Trains a new MLP with NequIP
  - Starts the next round by passing the new MLP path

Reference: Section 2.3 of the manuscript.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from ase.io import read as ase_read, write as ase_write

from ..selection import (
    compute_soap_descriptors,
    KPCAReducer,
    stratified_fps_selection,
)
from .vasp_inputs import prepare_vasp_inputs

def _to_py(value):
    """Convert numpy scalar types to native Python types for JSON serialization."""
    if value is None:
        return None
    if hasattr(value, "item"):     # numpy scalar
        return value.item()
    return value

# ============================================================
#  Single-round entry point
# ============================================================

def run_al_round(
    al_config_path: Path,
    round_idx: int,
    mlp_model_path: Path,
    *,
    skip_mhmc: bool = False,
    candidate_pool_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Execute one AL round.

    Parameters
    ----------
    al_config_path : Path
        Path to the AL configuration yaml.
    round_idx : int
        Round index (0-based, but typically 1-based for user clarity).
    mlp_model_path : Path
        Path to the current MLP model used to drive this round's MHMC.
    skip_mhmc : bool
        If True, skip the MHMC step and use a pre-existing candidate pool.
        Useful for debugging selection on the same pool.
    candidate_pool_path : Path, optional
        If skip_mhmc=True, this points to the existing candidate pool.
    verbose : bool

    Returns
    -------
    dict with paths and summary stats for this round.
    """
    al_cfg = _load_al_config(al_config_path)

    # ---- Output directory for this round ----
    round_dir = Path(al_cfg["output_root"]) / f"al_round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"#  AL Round {round_idx}")
        print(f"#  MLP model: {mlp_model_path}")
        print(f"#  Output  : {round_dir}")
        print(f"{'#'*70}")

    # ============================================
    # Step 1: Run MHMC
    # ============================================
    if skip_mhmc:
        assert candidate_pool_path is not None, \
            "candidate_pool_path must be set if skip_mhmc=True"
        if verbose:
            print(f"\n[Step 1] Skipping MHMC, using existing pool: "
                  f"{candidate_pool_path}")
    else:
        candidate_pool_path = _run_mhmc(
            mhmc_config_path = Path(al_cfg["mhmc_config"]),
            mlp_model_path   = mlp_model_path,
            round_idx        = round_idx,
            round_dir        = round_dir,
            verbose          = verbose,
        )

    # ============================================
    # Step 2: Load candidate pool
    # ============================================
    if verbose:
        print(f"\n[Step 2] Loading candidate pool: {candidate_pool_path}")
    atoms_list = ase_read(str(candidate_pool_path), index=":")
    source_tags = [a.info.get("source", "unknown") for a in atoms_list]

    from collections import Counter
    src_counter = Counter(source_tags)
    if verbose:
        print(f"[Step 2] {len(atoms_list)} configurations loaded.")
        for src, n in sorted(src_counter.items(), key=lambda x: -x[1]):
            print(f"         {src:20s}: {n}")

    # ============================================
    # Step 3: Compute SOAP descriptors
    # ============================================
    soap_cfg = al_cfg["soap"]
    if verbose:
        print(f"\n[Step 3] Computing SOAP descriptors")
    descriptors = compute_soap_descriptors(
        atoms_list,
        soap_params=soap_cfg,
        n_jobs=al_cfg.get("n_jobs", 1),
        verbose=verbose,
    )

    # ============================================
    # Step 4: KPCA reduction
    # ============================================
    kpca_cfg = al_cfg["kpca"]
    if verbose:
        print(f"\n[Step 4] KPCA reduction")
    reducer = KPCAReducer(
        variance_threshold = kpca_cfg.get("variance_threshold", 0.95),
        max_components     = kpca_cfg.get("max_components", 50),
        gamma              = kpca_cfg.get("gamma"),  # None → median heuristic
        nystroem_components= kpca_cfg.get("nystroem_components", 0),
        random_state       = al_cfg.get("seed", 42),
    )
    reduced = reducer.fit_transform(descriptors, verbose=verbose)

    # ============================================
    # Step 5: Stratified FPS selection
    # ============================================
    fps_cfg = al_cfg["selection"]
    if verbose:
        print(f"\n[Step 5] Stratified FPS selection")
    result = stratified_fps_selection(
        features        = reduced,
        source_tags     = source_tags,
        n_select_total  = fps_cfg["n_select"],
        source_weights  = fps_cfg.get("source_weights"),
        seed            = al_cfg.get("seed", 42),
        verbose         = verbose,
    )
    selected_indices = result["selected_indices"]
    selected_atoms = [atoms_list[i] for i in selected_indices]

    # Save the selected pool as extxyz
    selected_extxyz = round_dir / "selected.extxyz"
    ase_write(str(selected_extxyz), selected_atoms)
    if verbose:
        print(f"[Step 5] Saved selected pool → {selected_extxyz}")

    # Save selection metadata
    sel_meta = round_dir / "selection_meta.jsonl"
    with sel_meta.open("w") as f:
        for idx, atoms in zip(selected_indices, selected_atoms):
            entry = {
                "pool_idx":       int(idx),
                "source":         atoms.info.get("source"),
                "iteration":      _to_py(atoms.info.get("iteration")),
                "basin_id":       _to_py(atoms.info.get("basin_id")),
                "energy":         _to_py(atoms.info.get("energy")),
                "frame_id":       _to_py(atoms.info.get("frame_id")),
            }
            f.write(json.dumps(entry) + "\n")

    # ============================================
    # Step 6: Generate VASP input directories
    # ============================================
    vasp_cfg = al_cfg["vasp"]
    vasp_dir = round_dir / "vasp_inputs"
    if verbose:
        print(f"\n[Step 6] Generating VASP input directories")

    metadata_list = []
    for atoms in selected_atoms:
        metadata_list.append({
            "source":     atoms.info.get("source"),
            "iteration":  _to_py(atoms.info.get("iteration")),
            "basin_id":   _to_py(atoms.info.get("basin_id")),
            "energy_mlp": _to_py(atoms.info.get("energy")),
        })

    prepare_vasp_inputs(
        selected_atoms     = selected_atoms,
        output_dir         = vasp_dir,
        potcar_library     = Path(vasp_cfg["potcar_library"]),
        incar_params       = vasp_cfg.get("incar", {}),
        kpoints_grid       = tuple(vasp_cfg.get("kpoints", [3, 3, 1])),
        magmom_by_element  = vasp_cfg.get("magmom_by_element"),
        metadata_list      = metadata_list,
        selected_indices   = selected_indices.tolist(),
        verbose            = verbose,
    )

    # ============================================
    # Summary
    # ============================================
    summary = {
        "round_idx":          round_idx,
        "mlp_model":          str(mlp_model_path),
        "candidate_pool":     str(candidate_pool_path),
        "n_candidates":       len(atoms_list),
        "n_soap_features":    int(descriptors.shape[1]),
        "n_kpca_components":  int(reduced.shape[1]),
        "kpca_gamma":         float(reducer.gamma_used_),
        "n_selected":         len(selected_atoms),
        "per_source_counts":  result["per_source_counts"],
        "round_dir":          str(round_dir),
        "vasp_inputs_dir":    str(vasp_dir),
    }
    summary_path = round_dir / "round_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*70}")
        print(f"AL round {round_idx} complete.")
        print(f"  Round directory : {round_dir}")
        print(f"  VASP inputs     : {vasp_dir}")
        print(f"  Summary         : {summary_path}")
        print(f"\nNext steps (manual):")
        print(f"  1. Submit VASP jobs in {vasp_dir}/*/")
        print(f"  2. Collect OUTCAR/vasprun.xml, merge into training set")
        print(f"  3. Train new NequIP MLP")
        print(f"  4. Run: python run_al.py {al_config_path} "
              f"--round {round_idx+1} --mlp_model <new_mlp_path>")
        print(f"{'='*70}")

    return summary


# ============================================================
#  Internal helpers
# ============================================================

def _load_al_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_mhmc(
    mhmc_config_path: Path,
    mlp_model_path: Path,
    round_idx: int,
    round_dir: Path,
    verbose: bool,
) -> Path:
    """
    Run the MHMC sampler as a subprocess and return the path to the
    generated candidate pool.

    We patch the MLP model path into the MHMC yaml before invoking it.
    Because run.py writes to its own outputs/<timestamp>_... directory
    regardless of the yaml's output section, we locate the freshly-written
    candidate pool by timestamp matching and copy it into the round_dir.
    """
    import time

    # Load original MHMC yaml
    with open(mhmc_config_path) as f:
        mhmc_cfg = yaml.safe_load(f)

    # Override the same model field consumed by run.py/build_calculator.
    mhmc_cfg.setdefault("system", {})
    mhmc_cfg["system"]["mlp_model"] = str(mlp_model_path)

    # Save the patched yaml
    patched_yaml = round_dir / f"mhmc_config_round_{round_idx:02d}.yaml"
    with patched_yaml.open("w") as f:
        yaml.dump(mhmc_cfg, f, sort_keys=False)

    if verbose:
        print(f"\n[Step 1] Running MHMC")
        print(f"         config: {patched_yaml}")

    # Record the timestamp just before invocation
    invocation_time = time.time()

    # Invoke run.py as subprocess
    cmd = [sys.executable, "run.py", str(patched_yaml)]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"MHMC subprocess failed with exit code {result.returncode}"
        )

    # MHMC writes to outputs/<timestamp>_<...>/ by default.
    # Find any candidate_pool.extxyz created after our invocation timestamp.
    outputs_root = Path("outputs")
    if not outputs_root.is_dir():
        raise FileNotFoundError(
            f"Expected MHMC output root 'outputs/' does not exist."
        )

    pool_candidates = []
    for pool_file in outputs_root.rglob("candidate_pool.extxyz"):
        # Match by modification time after the subprocess invocation
        if pool_file.stat().st_mtime >= invocation_time - 1.0:
            pool_candidates.append(pool_file)

    if not pool_candidates:
        raise FileNotFoundError(
            f"No candidate_pool.extxyz found in outputs/ "
            f"with mtime >= {invocation_time}.\n"
            f"MHMC may have failed silently or written elsewhere."
        )

    # Use the most recent one (the one we just produced)
    source_pool = sorted(pool_candidates, key=lambda p: p.stat().st_mtime)[-1]
    source_dir = source_pool.parent

    if verbose:
        print(f"[Step 1] MHMC wrote to: {source_dir}")

    # Copy the candidate pool and supporting artifacts into round_dir/mhmc/
    mhmc_subdir = round_dir / "mhmc"
    mhmc_subdir.mkdir(parents=True, exist_ok=True)

    # Files we want preserved per round: candidate pool + index + run log + checkpoint
    artifact_names = [
        "candidate_pool.extxyz",
        "candidate_pool_index.jsonl",
        "run.jsonl",
        "state.pkl",
    ]

    target_pool: Optional[Path] = None
    for name in artifact_names:
        src = source_dir / name
        if src.is_file():
            dst = mhmc_subdir / name
            shutil.copy2(src, dst)
            if name == "candidate_pool.extxyz":
                target_pool = dst

    if target_pool is None:
        raise FileNotFoundError(
            f"candidate_pool.extxyz expected but not in {source_dir}"
        )

    if verbose:
        print(f"[Step 1] Artifacts copied → {mhmc_subdir}")
        print(f"[Step 1] Candidate pool: {target_pool}")

    return target_pool