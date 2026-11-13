"""
Diagnostic runner for DualChannelInitializer.

Prints all channel- and configuration-level statistics needed for
reporting in Section 2.1 of the manuscript.
"""

import argparse

from ase.io import read, write
try:
    from .dual_channel_init import DualChannelInitializer
except ImportError:  # allow: python MLP_init/initial.py
    from dual_channel_init import DualChannelInitializer


def run_one_system(slab_file, adsorbates, n_configs, seed=42, label=""):
    print("=" * 70)
    print(f"System: {label}  |  slab: {slab_file}")
    print(f"Adsorbates: {adsorbates}  |  n_configs target: {n_configs}  |  seed: {seed}")
    print("-" * 70)

    slab = read(slab_file)

    init = DualChannelInitializer(
        slab=slab,
        d_top=1.5, d_bridge=1.3, d_hollow=1.2,
        alpha=0.5,
        epsilon=0.15,
        z_min_offset=1.2,
        z_max_offset=4.0,
        grid_density=0.4,
        seed=seed,
    )

    # ---- Channel-level statistics (set inside __init__) ----
    s = init.stats
    print(f"Channel 1 (Delaunay):")
    print(f"    top    sites:   {s['n_top']}")
    print(f"    bridge sites:   {s['n_bridge']}")
    print(f"    hollow sites:   {s['n_hollow']}")
    print(f"    channel-1 total:{s['n_channel1_total']}")
    print(f"Channel 2 (Grid):")
    print(f"    grid points:    {s['n_channel2_grid']}")
    print(f"Candidate pool total: {s['n_candidate_pool']}")
    print()

    # ---- Generate configurations ----
    configs, diag = init.generate_configurations(
        adsorbates=adsorbates,
        n_configs=n_configs,
    )

    print(f"\nConfiguration generation:")
    print(f"    requested:        {diag['n_target']}")
    print(f"    generated:        {diag['n_generated']}")
    print(f"    total trials:     {diag['n_trials']}")
    print(f"    clash failures:   {diag['n_clash_failures']}")
    print(f"    hash collisions:  {diag['n_hash_collisions']}")
    print(f"    overall success:  {diag['success_rate']*100:.1f}%")

    out = f"init_{label}_n{n_configs}.extxyz"
    write(out, configs)
    print(f"\nSaved -> {out}")
    print("=" * 70)
    print()

    return s, diag


def main():
    parser = argparse.ArgumentParser(
        description="Generate dual-channel initial configurations from a user-provided slab."
    )
    parser.add_argument("--slab", required=True, help="Path to your own slab structure file.")
    parser.add_argument("--adsorbates", default="H,H,H,H,N,N",
                        help="Comma-separated adsorbate symbols; default: H,H,H,H,N,N")
    parser.add_argument("--n-configs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="surface")
    args = parser.parse_args()

    run_one_system(
        slab_file=args.slab,
        adsorbates=[x.strip() for x in args.adsorbates.split(",") if x.strip()],
        n_configs=args.n_configs,
        seed=args.seed,
        label=args.label,
    )


if __name__ == "__main__":
    main()
