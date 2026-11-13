"""
Quick post-processing of MHMC jsonl logs.

Usage
-----
    python analyze_mhmc.py outputs/Fe110_coupled.jsonl
    python analyze_mhmc.py outputs/Fe110_*.jsonl --compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_log(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def summary_one(df: pd.DataFrame) -> dict:
    return {
        "n_iters":             len(df),
        "n_basins_final":      int(df["n_basins_total"].max()) if len(df) else 0,
        "new_basin_rate":      float(df["is_new_basin"].mean()) if len(df) else 0.0,
        "mean_walk_acc_rho":   float(df["rho"].mean()),
        "mean_basin_spread_R": float(df["R"].mean()),
        "T_min_seen":          float(df["T_used"].min()),
        "T_max_seen":          float(df["T_used"].max()),
        "mean_T":              float(df["T_used"].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="One or more jsonl files")
    p.add_argument("--compare", action="store_true",
                   help="Print a comparison table across logs")
    args = p.parse_args()

    if args.compare:
        rows = []
        for path in args.logs:
            df = load_log(path)
            row = {"file": Path(path).name, **summary_one(df)}
            rows.append(row)
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
    else:
        for path in args.logs:
            df = load_log(path)
            print(f"\n===== {path} =====")
            for k, v in summary_one(df).items():
                print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()