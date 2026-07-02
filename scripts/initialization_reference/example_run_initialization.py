#!/usr/bin/env python
"""Minimal example for the public dual-channel initialization reference code.

Run from repository root:
    python scripts/initialization_reference/example_run_initialization.py
"""
from pathlib import Path
from ase.io import read, write

from dual_channel_initializer_reference import DualChannelInitializer

root = Path(__file__).resolve().parents[2]
slab_path = root / "data/structures/initialization/Fe110_slab.extxyz"
out_path = root / "data/structures/initialization/example_generated_HHN.extxyz"

slab = read(slab_path)
initializer = DualChannelInitializer(
    slab=slab,
    d_top=1.5,
    d_bridge=1.3,
    d_hollow=1.2,
    alpha=0.5,
    epsilon=0.15,
    z_min_offset=1.2,
    z_max_offset=4.0,
    grid_density=0.6,
    seed=42,
)
configs, diagnostics = initializer.generate_configurations(
    adsorbates=["H", "H", "N"],
    n_configs=5,
)
write(out_path, configs)
print(diagnostics)
print(f"Wrote {len(configs)} structures to {out_path}")
