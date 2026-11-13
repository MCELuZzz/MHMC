#!/usr/bin/env python
"""
AL loop driver. Run one AL round per invocation.

Usage:
    python run_al.py configs/al_loop.yaml --round 1 --mlp_model path/to/mlp.pt

After this command completes, manually:
  1. Submit VASP jobs in <round_dir>/vasp_inputs/*/
  2. Collect DFT results and merge into training set
  3. Train new MLP with NequIP
  4. Invoke again for the next round with the new MLP path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mhmc.al_loop import run_al_round


def main():
    parser = argparse.ArgumentParser(
        description="Active learning round driver."
    )
    parser.add_argument(
        "al_config",
        type=Path,
        help="Path to AL configuration yaml.",
    )
    parser.add_argument(
        "--round", "-r",
        type=int,
        required=True,
        help="Round index (1-based for user clarity).",
    )
    parser.add_argument(
        "--mlp_model", "-m",
        type=Path,
        required=True,
        help="Path to the MLP model (NequIP .pt or .pt2) for this round.",
    )
    parser.add_argument(
        "--skip_mhmc",
        action="store_true",
        help="Skip MHMC sampling, use --pool to provide an existing pool.",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="If --skip_mhmc, the existing candidate_pool.extxyz path.",
    )

    args = parser.parse_args()

    if args.skip_mhmc and args.pool is None:
        parser.error("--skip_mhmc requires --pool")

    run_al_round(
        al_config_path      = args.al_config,
        round_idx           = args.round,
        mlp_model_path      = args.mlp_model,
        skip_mhmc           = args.skip_mhmc,
        candidate_pool_path = args.pool,
    )


if __name__ == "__main__":
    main()