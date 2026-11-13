#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 candidate_pool.extxyz 中取最后一个 source == "minimum" 的 basin minimum，
写出 sanity_check.extxyz，并生成一个 VASP 5-step atoms-only relax 输入目录。

用法示例：
python sanity_check_last_min_to_vasp.py \
  --pool ablation_results_Hookean/coupled_seed42/candidate_pool.extxyz \
  --outdir sanity_check_vasp \
  --potcar-library /path/to/your/POTCAR_library \
  --kpoints 3 3 1

要求：
1. 当前目录下有 vasp_inputs.py
2. vasp_inputs.py 里包含 prepare_vasp_inputs()
3. POTCAR library 结构类似：
   /path/to/your/POTCAR_library/Fe/POTCAR
   /path/to/your/POTCAR_library/H/POTCAR
   /path/to/your/POTCAR_library/N/POTCAR
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write

# 这里假设你的文件名叫 vasp_inputs.py
# 如果你的文件名不是这个，比如 prepare_vasp_inputs.py，
# 就把下面这一行改成：
# from prepare_vasp_inputs import prepare_vasp_inputs
from mhmc.al_loop.vasp_inputs import prepare_vasp_inputs


def to_jsonable(x: Any) -> Any:
    """把 numpy 类型转成 json 能保存的普通 Python 类型。"""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def get_energy_safe(atoms) -> float | None:
    """
    尽量从 atoms 里取能量。
    extxyz 中可能在 atoms.info['energy']，
    也可能通过 atoms.get_potential_energy() 读取。
    """
    for key in ["energy", "Energy", "free_energy", "dft_energy", "mlp_energy"]:
        if key in atoms.info:
            try:
                return float(atoms.info[key])
            except Exception:
                pass

    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def find_last_minimum(pool):
    """
    找最后一个 a.info['source'] == 'minimum' 的结构。
    同时返回它在 candidate_pool.extxyz 中的原始 index。
    """
    minimum_items = []

    for i, atoms in enumerate(pool):
        source = atoms.info.get("source", None)
        if source == "minimum":
            minimum_items.append((i, atoms))

    if not minimum_items:
        raise RuntimeError(
            "没有在 candidate_pool.extxyz 中找到 source == 'minimum' 的结构。\n"
            "请先检查：\n"
            "  ase gui candidate_pool.extxyz\n"
            "或者打印：\n"
            "  python - << 'EOF'\n"
            "  from ase.io import read\n"
            "  pool = read('candidate_pool.extxyz', ':')\n"
            "  print(sorted(set(a.info.get('source') for a in pool)))\n"
            "  EOF"
        )

    return minimum_items[-1]


def patch_incar_for_5step_relax(
    incar_path: Path,
    *,
    system_name: str = "Sanity check last basin minimum",
    ibrion: int = 2,
    nsw: int = 5,
    isif: int = 2,
    potim: float = 0.2,
    ediffg: float = -0.01,
) -> None:
    """
    你的 prepare_vasp_inputs 默认生成 single-point INCAR：
        IBRION = -1
        NSW    = 0
        ISIF   = 2

    这里生成之后再 patch 成 5-step atoms relax。
    """

    text = incar_path.read_text()

    replacements = {
        "SYSTEM": f"SYSTEM = {system_name}",
        "IBRION": f"IBRION = {ibrion}",
        "NSW": f"NSW    = {nsw}",
        "ISIF": f"ISIF   = {isif}",
        "POTIM": f"POTIM  = {potim}",
        "EDIFFG": f"EDIFFG = {ediffg}",
    }

    def replace_or_append(text_in: str, key: str, newline: str) -> str:
        pattern = rf"^\s*{re.escape(key)}\s*=.*$"
        if re.search(pattern, text_in, flags=re.MULTILINE):
            return re.sub(pattern, newline, text_in, flags=re.MULTILINE)
        else:
            return text_in.rstrip() + "\n" + newline + "\n"

    for key, newline in replacements.items():
        text = replace_or_append(text, key, newline)

    # 加一段注释，避免之后忘记这是 sanity check relax
    marker = "# Sanity-check relaxation settings"
    if marker not in text:
        text = text.rstrip() + f"""

{marker}
# Purpose: relax last basin minimum for only {nsw} ionic steps.
# Cell is fixed by ISIF = {isif}; atoms are relaxed by IBRION = {ibrion}.
"""

    incar_path.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 5-step VASP sanity-check relaxation for the last basin minimum."
    )

    parser.add_argument(
        "--pool",
        type=str,
        default="ablation_results_Hookean/coupled_seed42/candidate_pool.extxyz",
        help="Path to candidate_pool.extxyz.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="sanity_check_vasp",
        help="Output root directory for VASP input.",
    )
    parser.add_argument(
        "--extxyz-out",
        type=str,
        default="sanity_check.extxyz",
        help="Output extxyz for the selected last minimum.",
    )
    parser.add_argument(
        "--potcar-library",
        type=str,
        required=True,
        help="Root of your own licensed POTCAR library. Expected: <potcar_library>/<element>/POTCAR",
    )
    parser.add_argument(
        "--kpoints",
        type=int,
        nargs=3,
        default=[3, 3, 1],
        metavar=("NKX", "NKY", "NKZ"),
        help="KPOINTS grid, default: 3 3 1",
    )

    # 常用 INCAR 参数
    parser.add_argument("--encut", type=float, default=400.0)
    parser.add_argument("--ediff", type=float, default=1.0e-5)
    parser.add_argument("--prec", type=str, default="High")
    parser.add_argument("--algo", type=str, default="Fast")
    parser.add_argument("--nelm", type=int, default=200)
    parser.add_argument("--nsim", type=int, default=4)
    parser.add_argument("--npar", type=int, default=2)
    parser.add_argument("--lreal", type=str, default="A")
    parser.add_argument("--gga", type=str, default="PE")
    parser.add_argument("--ismear", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--ivdw", type=int, default=12)

    # 5-step relax 参数
    parser.add_argument("--nsw", type=int, default=5)
    parser.add_argument("--ibrion", type=int, default=2)
    parser.add_argument("--isif", type=int, default=2)
    parser.add_argument("--potim", type=float, default=0.2)
    parser.add_argument("--ediffg", type=float, default=-0.01)

    args = parser.parse_args()

    pool_path = Path(args.pool)
    outdir = Path(args.outdir)
    extxyz_out = Path(args.extxyz_out)
    potcar_library = Path(args.potcar_library)

    if not pool_path.is_file():
        raise FileNotFoundError(f"candidate pool not found: {pool_path}")

    if not potcar_library.is_dir():
        raise NotADirectoryError(f"POTCAR library not found: {potcar_library}")

    print("=" * 80)
    print("Reading candidate pool")
    print("=" * 80)
    print(f"[INFO] pool: {pool_path}")

    pool = read(str(pool_path), index=":")
    print(f"[INFO] total frames: {len(pool)}")

    original_idx, last_min = find_last_minimum(pool)

    energy = get_energy_safe(last_min)
    formula = last_min.get_chemical_formula()
    source = last_min.info.get("source", None)
    iteration = last_min.info.get("iteration", None)
    basin_id = last_min.info.get("basin_id", None)

    print("\n" + "=" * 80)
    print("Selected last basin minimum")
    print("=" * 80)
    print(f"[INFO] original_pool_idx : {original_idx}")
    print(f"[INFO] formula           : {formula}")
    print(f"[INFO] n_atoms           : {len(last_min)}")
    print(f"[INFO] source            : {source}")
    print(f"[INFO] iteration         : {iteration}")
    print(f"[INFO] basin_id          : {basin_id}")
    print(f"[INFO] energy            : {energy}")

    # 写出 sanity_check.extxyz
    write(str(extxyz_out), last_min)
    print(f"\n[WRITE] extxyz: {extxyz_out}")

    # metadata
    metadata = {
        "purpose": "sanity_check_last_basin_minimum_5step_relax",
        "pool_path": str(pool_path),
        "original_pool_idx": int(original_idx),
        "source": to_jsonable(source),
        "iteration": to_jsonable(iteration),
        "basin_id": to_jsonable(basin_id),
        "energy": to_jsonable(energy),
        "formula": formula,
        "n_atoms": len(last_min),
        "all_info_keys": list(last_min.info.keys()),
    }

    incar_params = {
        "system_name": "Sanity check last basin minimum",
        "encut": args.encut,
        "ediff": args.ediff,
        "prec": args.prec,
        "algo": args.algo,
        "nelm": args.nelm,
        "nsim": args.nsim,
        "npar": args.npar,
        "lreal": args.lreal,
        "gga": args.gga,
        "ismear": args.ismear,
        "sigma": args.sigma,
        "ivdw": args.ivdw,
    }

    print("\n" + "=" * 80)
    print("Preparing VASP input")
    print("=" * 80)

    subdirs = prepare_vasp_inputs(
        selected_atoms=[last_min],
        output_dir=outdir,
        potcar_library=potcar_library,
        incar_params=incar_params,
        kpoints_grid=tuple(args.kpoints),
        metadata_list=[metadata],
        selected_indices=[original_idx],
        verbose=True,
    )

    run_dir = subdirs[0]

    # 关键：把默认 single-point INCAR 改成 5-step atoms relax
    patch_incar_for_5step_relax(
        run_dir / "INCAR",
        system_name="Sanity check last basin minimum",
        ibrion=args.ibrion,
        nsw=args.nsw,
        isif=args.isif,
        potim=args.potim,
        ediffg=args.ediffg,
    )

    # 额外写一个 summary json，方便之后查
    summary_path = outdir / "sanity_check_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "selected": metadata,
                "vasp_run_dir": str(run_dir),
                "extxyz_out": str(extxyz_out),
                "relax_settings": {
                    "IBRION": args.ibrion,
                    "NSW": args.nsw,
                    "ISIF": args.isif,
                    "POTIM": args.potim,
                    "EDIFFG": args.ediffg,
                },
                "incar_params": incar_params,
                "kpoints": args.kpoints,
                "potcar_library": str(potcar_library),
            },
            f,
            indent=2,
            default=to_jsonable,
        )

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"[OK] sanity extxyz : {extxyz_out}")
    print(f"[OK] VASP run dir  : {run_dir}")
    print(f"[OK] summary       : {summary_path}")
    print("\n接下来可以进入目录提交 VASP：")
    print(f"  cd {run_dir}")
    print("  ls")
    print("  grep -E 'IBRION|NSW|ISIF|POTIM|EDIFFG' INCAR")


if __name__ == "__main__":
    main()