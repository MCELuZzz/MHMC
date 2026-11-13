# MHMC production source code

This folder contains the Python source code and configuration templates for the coupled minima-hopping–Monte Carlo (MH–MC) sampler and its active-learning workflow.

**This is a code-only release.** It does not include private datasets, DFT trajectories, structure files, compiled NequIP model weights, VASP executables, VASP license files, or POTCAR files.

## Contents

- `run.py`: run one MHMC sampling calculation.
- `run_ablation.py`: run sampling-mode and seed ablations.
- `run_al.py`: orchestrate one active-learning round.
- `mhmc/core/`: MH escape, basin classification, basin-confined Monte Carlo, feedback, and sampler control.
- `mhmc/io/`: candidate-pool collection, logging, and state/checkpoint writing.
- `mhmc/selection/`: SOAP descriptors, KPCA, and stratified farthest-point sampling.
- `mhmc/al_loop/`: active-learning orchestration and VASP input-file generation.
- `MLP_init/`: Delaunay/grid dual-channel initialization.
- `configs/`: example YAML templates with placeholder paths.
- `analyze*.py`, `compute_coverage.py`, `assemble_holdout.py`, and related scripts: analysis and research utilities.

## Requirements

Install the dependencies listed in `requirements.txt` in an environment compatible with your compiled NequIP model:

```bash
pip install -r requirements.txt
```

NequIP and PyTorch versions must be compatible with the compiled model supplied by the user. The compiled model is not included in this repository.

## Configure a sampling run

Edit `configs/Fe110.yaml` and replace the three placeholders:

```yaml
system:
  slab_file: "<PATH_TO_YOUR_SLAB.extxyz>"
  initial_dataset: "<PATH_TO_YOUR_INITIAL_STRUCTURES.extxyz>"
  mlp_model: "<PATH_TO_YOUR_COMPILED_NEQUIP_MODEL.pt2>"
  device: cuda
```

These files must be supplied locally by the user; no private structure dataset or model is included in this code release.

Run from the `production_code/` directory:

```bash
python check_setup.py configs/Fe110.yaml
python run.py configs/Fe110.yaml --mode coupled --seed 42 --n_iters 1
```

Available sampling modes are `mh_only`, `walk_only`, `serial`, `coupled_no_chain`, and `coupled`.

For a longer run, increase `--n_iters` after confirming that the one-iteration test works.

## Dual-channel initialization

The initialization utility also requires a slab structure supplied by the user:

```bash
python -m MLP_init.initial \
  --slab /path/to/your/slab.extxyz \
  --adsorbates H,H,H,H,N,N \
  --n-configs 100 \
  --seed 42 \
  --label Fe110_4H2N
```

It writes generated initial configurations to the current working directory.

## Active learning and VASP input preparation

The active-learning scripts perform candidate generation, SOAP calculation, KPCA reduction, stratified FPS selection, and VASP input preparation.

Before using `configs/al_loop.yaml` or `configs/al_loop_mh_only.yaml`, set:

```yaml
vasp:
  potcar_library: "<PATH_TO_YOUR_LICENSED_POTCAR_LIBRARY>"
```

The expected local directory layout is:

```text
<YOUR_LICENSED_POTCAR_LIBRARY>/
├── Fe/POTCAR
├── N/POTCAR
└── H/POTCAR
```

**VASP is a separately licensed software package.** This repository does not contain the VASP executable, a VASP license, or any POTCAR files. The scripts only prepare VASP input files and, when requested, assemble POTCAR from the user's own locally supplied licensed POTCAR library. Users must run VASP under their own valid license and must not upload POTCAR files to this repository.

The current active-learning driver prepares input directories; VASP job submission, DFT result collection, dataset merging, and NequIP retraining are external steps.

## Runtime outputs and checkpoints

Runtime outputs are written to local output directories and are excluded from the repository. The sampler writes state/checkpoint files, but this source release does not provide a complete command-line restart workflow that automatically resumes from a saved state.

## Data and model availability

This folder adds source code only. It does not add or redistribute datasets, structures, DFT labels, trajectories, or trained model weights. Any data already present in the parent repository remain unchanged.

## Citation and reuse

Please cite the associated MHMC manuscript when using this source code. The repository's existing license and reuse notice remains applicable; public visibility does not by itself grant additional redistribution rights.
