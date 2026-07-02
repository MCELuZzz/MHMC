# Release notes for the cleaned reproducibility package

This package was cleaned from an internal code-review archive for public GitHub deposition.

## Removed from the public package

- `mhmc/core/`: full production coupled MH--MC implementation.
- Full production dual-channel initialization workflow code and system-specific templates. A cleaned reference implementation of selected initialization building blocks is provided in `scripts/initialization_reference/`.
- `mhmc/al_loop/`: internal active-learning orchestration and VASP input generation implementation.
- `__pycache__/` and `*.pyc` files.
- Temporary/debug scripts such as `view.py`, `basin.py`, `basin_v2.py`, `test_select.py`, and VASP sanity-check utilities.
- The compiled NequIP model file, pending author approval for public release.
- Local absolute paths and direct references to local VASP pseudopotential libraries.

## Retained in the public package

- Machine-readable structural datasets in `extxyz` format.
- Sanitized configuration templates.
- Analysis and plotting scripts.
- Reference implementations of selected dual-channel initialization building blocks.
- Reference implementation of the SOAP/KPCA/FPS selection module.
- Pseudocode-level method documentation explaining the unpublished in-house workflow.
- Data and software availability statement draft.
