"""
Active learning loop orchestration.

Each round of AL consists of:
  1. Run MHMC sampler with the current MLP (delegated to run.py)
  2. Compute SOAP + KPCA reduction on the candidate pool
  3. Stratified FPS selection of N_sel configurations
  4. Generate per-configuration VASP input directories for DFT labeling

Steps 5+ (DFT submission, retraining, deployment of new MLP) are
performed manually by the user. The orchestrator stops after step 4.

Reference: Section 2.3 of the manuscript.
"""

from .vasp_inputs import prepare_vasp_inputs
from .orchestrator import run_al_round

__all__ = [
    "prepare_vasp_inputs",
    "run_al_round",
]