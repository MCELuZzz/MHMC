"""
Main MHMC loop with the final two-decision-diamond design.

Algorithm flow (per iteration):
  1. Escape MD at T_t  →  candidate minimum m_t^new
  2. Basin classify & register (always; basin upgrade if deeper)
  3. D2: MH accept/reject  using  E_new - E_accepted ≤ E_diff(t)
       ACCEPT: → basin-confined walk → D4 feedback → update seed/T
       REJECT: → skip walk → seed reverts → T amplified by kappa
  4. E_diff(t) adaptation via sliding chain acceptance rate

Five ablation modes:
  - "mh_only"          : pure MH (D2 only, no walk, no feedback)
  - "walk_only"        : never escapes; runs walks at the initial basin only
  - "serial"           : MH + walk run unconditionally; no feedback, no chain
  - "coupled_no_chain" : full feedback but D2 always accepts (no chain control)
  - "coupled"          : full feedback + chain control (default)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

from .basin import BasinDatabase
from .escape import escape_phase
from .walk import walk_phase
from .feedback import select_representative, update_temperature
from ..io.logger import JsonlLogger, save_checkpoint
from ..io.dataset import DatasetCollector

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================
#  Configuration container
# ============================================================

@dataclass
class MHMCConfig:
    # --- Topology ---
    slab_n_atoms: int
    fixed_indices: Optional[list[int]] = None

    # --- Basin classification ---
    eps_E: float = 0.05
    eps_R: float = 0.4

    # --- Escape phase ---
    T_init: float = 2000.0
    md_steps: int = 500
    md_timestep_fs: float = 1.0
    relax_fmax: float = 0.05
    relax_steps: int = 200

    # --- Walk phase ---
    T_walk: float = 600.0
    n_walk_steps: int = 80
    walk_sigma: float = 0.15
    dE_basin: float = 0.8

    # --- Feedback (accept path) ---
    lambda_: float = 2.0
    rho_star: float = 0.3
    R_star: float = 0.5
    gamma_rho: float = 1.0
    gamma_R: float = 0.5
    T_min: float = 800.0
    T_max: float = 5000.0

    # --- Chain control (D2) ---
    E_diff_init: float = 0.5
    E_diff_min: float = 0.1
    E_diff_max: float = 2.0
    chain_window: int = 10
    chain_acc_target: float = 0.5
    chain_adapt_rate: float = 0.1
    reject_T_factor: float = 1.2          # NEW: kappa for reject path

    # --- Dataset collection ---
    md_stride: int = 50
    relax_stride: int = 5
    walk_rejected_keep_ratio: float = 0.3
    dataset_flush_every: int = 200

    # --- Loop control ---
    n_iters: int = 200
    mode: str = "coupled"
    checkpoint_every: int = 50

    # --- Output paths ---
    log_path: str = "outputs/mhmc.jsonl"
    checkpoint_path: str = "outputs/mhmc_state.pkl"
    dataset_extxyz_path: str = "outputs/candidate_pool.extxyz"
    dataset_index_path: str = "outputs/candidate_pool_index.jsonl"


    # --- Reproducibility ---
    seed: int = 42

    # --- Surface confinement (optional ASE Hookean plane constraints) ---
    # Both offsets are referenced to the slab's max z, computed at __init__.
    # Set either or both to None (default) to disable that wall.
    z_ceiling_offset: Optional[float] = None   # Å above slab top — soft ceiling
    z_floor_offset:   Optional[float] = None   # Å above slab top — soft floor
    z_confine_k:      float = 10.0             # eV/Å^2, shared spring constant


# ============================================================
#  Sampler
# ============================================================

class MHMCSampler:

    VALID_MODES = ("mh_only", "walk_only", "serial",
                   "coupled_no_chain", "coupled")

    def __init__(
        self,
        initial_atoms: Atoms,
        calculator: Calculator,
        config: MHMCConfig,
    ):
        if config.mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got '{config.mode}'"
            )

        self.cfg = config
        self.calc = calculator
        self.rng = np.random.default_rng(config.seed)

        # Basin database
        self.basin_db = BasinDatabase(
            slab_n_atoms=config.slab_n_atoms,
            eps_E=config.eps_E,
            eps_R=config.eps_R,
        )

        # Relax initial structure into m_0
        atoms_init = initial_atoms.copy()
        atoms_init.calc = calculator

        # Build constraint list: FixAtoms (slab) + optional Hookean planes.
        # ASE constraints propagate through atoms.copy() automatically, so
        # escape MD / walk / relax inherit them with no further wiring.
        constraints = []
        if config.fixed_indices:
            from ase.constraints import FixAtoms
            constraints.append(FixAtoms(indices=config.fixed_indices))

        # --- Optional surface confinement ---
        if (config.z_ceiling_offset is not None) or (config.z_floor_offset is not None):
            # Sanity: if both set, ceiling must be above floor
            if (config.z_ceiling_offset is not None
                and config.z_floor_offset is not None
                and config.z_ceiling_offset < config.z_floor_offset):
                raise ValueError(
                    f"z_ceiling_offset ({config.z_ceiling_offset}) must be "
                    f">= z_floor_offset ({config.z_floor_offset})"
                )
            from ase.constraints import Hookean

            slab_top_z = float(atoms_init.positions[:config.slab_n_atoms, 2].max())
            n_total = len(atoms_init)
            k = config.z_confine_k

            msg = [f"[confine] slab_top_z = {slab_top_z:.2f} Å"]
            if config.z_ceiling_offset is not None:
                z_max = slab_top_z + float(config.z_ceiling_offset)
                # plane (0,0,1,-z_max): d = z - z_max, active when z > z_max
                plane_top = (0.0, 0.0,  1.0, -z_max)
                for ads_idx in range(config.slab_n_atoms, n_total):
                    constraints.append(Hookean(a1=ads_idx, a2=plane_top, k=k))
                msg.append(f"ceiling = {z_max:.2f} Å (offset {config.z_ceiling_offset:+.2f})")
            if config.z_floor_offset is not None:
                z_min = slab_top_z + float(config.z_floor_offset)
                # plane (0,0,-1, z_min): d = -z + z_min, active when z < z_min
                plane_bot = (0.0, 0.0, -1.0,  z_min)
                for ads_idx in range(config.slab_n_atoms, n_total):
                    constraints.append(Hookean(a1=ads_idx, a2=plane_bot, k=k))
                msg.append(f"floor = {z_min:.2f} Å (offset {config.z_floor_offset:+.2f})")
            msg.append(f"k = {k:.1f} eV/Å²")
            print(", ".join(msg))

        if constraints:
            atoms_init.set_constraint(constraints)

        BFGS(atoms_init, logfile=None).run(
            fmax=config.relax_fmax, steps=config.relax_steps,
        )

        E_init = float(atoms_init.get_potential_energy())

        basin0, _ = self.basin_db.classify(atoms_init, E_init, current_iter=-1)
        self._initial_basin = basin0

        # Loop state
        self.seed_atoms: Atoms = basin0.representative.copy()
        self.T_current: float = config.T_init

        # Chain state (only used in "coupled" mode)
        self.current_accepted_min: Atoms = basin0.representative.copy()
        self.E_current_accepted: float = E_init
        self.E_diff_current: float = config.E_diff_init
        self._chain_acc_history: deque[bool] = deque(maxlen=config.chain_window)

        # Logger
        self.logger = JsonlLogger(config.log_path)

        # Dataset collector
        self.dataset = DatasetCollector(
            extxyz_path=config.dataset_extxyz_path,
            index_path=config.dataset_index_path,
            flush_every=config.dataset_flush_every,
        )
        # Register the initial relaxed structure
        self.dataset.register_frame(
            atoms=basin0.representative,
            source="initial",
            iteration=-1,
            basin_id=basin0.id,
            energy=E_init,
        )

    # --------------------------------------------------------
    #  D2 helper: chain accept/reject + E_diff adaptation
    # --------------------------------------------------------
    def _d2_decision(self, E_new: float) -> bool:
        """Return True iff the new minimum is admitted as the next link
        of the propagation chain."""
        if self.cfg.mode != "coupled":
            # All other modes effectively always accept (no chain control)
            return True
        delta = E_new - self.E_current_accepted
        return delta <= self.E_diff_current

    def _update_E_diff(self) -> None:
        """Adapt E_diff based on the rolling chain acceptance rate."""
        if len(self._chain_acc_history) < max(3, self.cfg.chain_window // 2):
            return
        acc_rate = sum(self._chain_acc_history) / len(self._chain_acc_history)
        log_factor = self.cfg.chain_adapt_rate * (
            self.cfg.chain_acc_target - acc_rate
        )
        self.E_diff_current = float(np.clip(
            self.E_diff_current * np.exp(log_factor),
            self.cfg.E_diff_min,
            self.cfg.E_diff_max,
        ))

    # --------------------------------------------------------
    #  Main loop
    # --------------------------------------------------------
    def run(self) -> BasinDatabase:
        if HAS_TQDM:
            iterator = tqdm(
                range(self.cfg.n_iters),
                desc=f"MHMC[{self.cfg.mode}]",
                ncols=130,
            )
        else:
            iterator = range(self.cfg.n_iters)

        try:
            for t in iterator:
                record = {"iter": t, "mode": self.cfg.mode}

                # ============================================
                # Step 1: Escape (MD + relaxation)
                # ============================================
                md_traj: list = []
                relax_traj: list = []

                if self.cfg.mode == "walk_only":
                    new_min = self._initial_basin.representative.copy()
                    E_new = self._initial_basin.energy
                    basin = self._initial_basin
                    is_new = False
                else:
                    new_min, E_new, md_traj, relax_traj = escape_phase(
                        seed=self.seed_atoms,
                        calculator=self.calc,
                        T_escape=self.T_current,
                        md_steps=self.cfg.md_steps,
                        md_timestep_fs=self.cfg.md_timestep_fs,
                        relax_fmax=self.cfg.relax_fmax,
                        relax_steps=self.cfg.relax_steps,
                        fixed_indices=self.cfg.fixed_indices,
                        rng=self.rng,
                        md_stride=self.cfg.md_stride,
                        relax_stride=self.cfg.relax_stride,
                    )
                    # ============================================
                    # Step 2: Classify & register (always, even before D2)
                    # ============================================
                    basin, is_new = self.basin_db.classify(
                        new_min, E_new, current_iter=t,
                    )

                record.update({
                    "basin_id": basin.id,
                    "is_new_basin": bool(is_new),
                    "basin_n_upgrades": basin.n_upgrades,
                    "E_min": E_new,
                    "E_basin_min": basin.energy,
                    "n_basins_total": len(self.basin_db),
                    "T_used": self.T_current,
                })

                # ============================================
                # Step 2.5: Archive escape/relaxation/minimum to dataset
                # ============================================
                self.dataset.register_many(
                    md_traj,
                    source="escape_md",
                    iteration=t,
                    basin_id=basin.id,
                )
                self.dataset.register_many(
                    relax_traj,
                    source="relaxation",
                    iteration=t,
                    basin_id=basin.id,
                )
                self.dataset.register_frame(
                    atoms=new_min,
                    source="minimum",
                    iteration=t,
                    basin_id=basin.id,
                    energy=E_new,
                )

                # ============================================
                # Step 3 (D2): MH accept/reject
                # ============================================
                d2_accepted = self._d2_decision(E_new)

                # Track for E_diff adaptation (only meaningful in "coupled")
                if self.cfg.mode == "coupled":
                    self._chain_acc_history.append(d2_accepted)

                # ============================================
                # Step 3a/3a': branch on D2
                # ============================================
                frames: list = []
                rho: float = 0.0
                R: float = 0.0
                rej_frames: list = []

                # Determine whether to run walk:
                # - walk_only: yes (always run on initial basin)
                # - mh_only:   no
                # - serial:    yes (always)
                # - coupled_no_chain: yes (always)
                # - coupled:   yes only if D2 accepts
                run_walk = (
                    self.cfg.mode == "walk_only" or
                    self.cfg.mode == "serial" or
                    self.cfg.mode == "coupled_no_chain" or
                    (self.cfg.mode == "coupled" and d2_accepted)
                )

                if run_walk and self.cfg.mode != "mh_only":
                    walk_target = (
                        self._initial_basin.representative
                        if self.cfg.mode == "walk_only"
                        else basin.representative
                    )
                    frames, rho, R, rej_frames = walk_phase(
                        m_t=walk_target,
                        calculator=self.calc,
                        slab_n_atoms=self.cfg.slab_n_atoms,
                        T_walk=self.cfg.T_walk,
                        n_steps=self.cfg.n_walk_steps,
                        sigma=self.cfg.walk_sigma,
                        dE_basin=self.cfg.dE_basin,
                        rng=self.rng,
                        rejected_keep_ratio=self.cfg.walk_rejected_keep_ratio,
                    )
                    if self.cfg.mode != "walk_only":
                        basin.add_members(frames)

                    # Archive walk frames
                    self.dataset.register_many(
                        frames,
                        source="walk_accepted",
                        iteration=t,
                        basin_id=basin.id,
                    )
                    self.dataset.register_many(
                        rej_frames,
                        source="walk_rejected",
                        iteration=t,
                        basin_id=basin.id,
                    )

                record.update({
                    "rho": rho,
                    "R": R,
                    "n_walk_frames": len(frames),
                    "n_walk_rejected_kept": len(rej_frames),
                    "d2_accepted": d2_accepted if self.cfg.mode == "coupled" else None,
                    "E_chain_current": self.E_current_accepted,
                    "E_diff_current": self.E_diff_current,
                    "chain_acc_rate_window": (
                        sum(self._chain_acc_history) / len(self._chain_acc_history)
                        if self._chain_acc_history else None
                    ),
                })

                # ============================================
                # Step 3c/3c': feedback (seed + T update)
                # ============================================
                if self.cfg.mode == "coupled" and d2_accepted:
                    # ACCEPT path: full feedback + chain advance
                    self.current_accepted_min = new_min.copy()
                    self.E_current_accepted = E_new
                    self._coupled_feedback(basin, frames, rho, R)

                elif self.cfg.mode == "coupled" and not d2_accepted:
                    # REJECT path: skip walk feedback, simple revert + amplify T
                    self.seed_atoms = self.current_accepted_min.copy()
                    self.T_current = float(np.clip(
                        self.cfg.reject_T_factor * self.T_current,
                        self.cfg.T_min,
                        self.cfg.T_max,
                    ))

                elif self.cfg.mode == "coupled_no_chain":
                    # No chain control: always run feedback (same as ACCEPT path
                    # but without updating current_accepted_min)
                    self._coupled_feedback(basin, frames, rho, R)

                elif self.cfg.mode in ("mh_only", "serial"):
                    self.seed_atoms = basin.representative.copy()
                    self.T_current = self.cfg.T_init

                elif self.cfg.mode == "walk_only":
                    pass

                # ============================================
                # Step 4: update E_diff (only in "coupled")
                # ============================================
                if self.cfg.mode == "coupled":
                    self._update_E_diff()

                # ============================================
                # Step 5: log + progress + checkpoint
                # ============================================
                record.update({
                    "T_next": self.T_current,
                    "seed_basin_id": basin.id,
                    "n_dataset_frames": (
                        self.dataset.n_written + len(self.dataset._buffer)
                    ),
                })
                self.logger.log(record)

                if HAS_TQDM:
                    postfix = {
                        "n_basins": len(self.basin_db),
                        "new": "Y" if record["is_new_basin"] else "n",
                        "E": f"{record['E_min']:.3f}",
                        "rho": f"{record['rho']:.2f}",
                        "T": f"{int(self.T_current)}",
                        "frames": record["n_dataset_frames"],
                    }
                    if record["d2_accepted"] is not None:
                        postfix["D2"] = "A" if record["d2_accepted"] else "R"
                        postfix["E_diff"] = f"{self.E_diff_current:.2f}"
                    iterator.set_postfix(postfix)

                if (t + 1) % self.cfg.checkpoint_every == 0:
                    save_checkpoint(self._snapshot(), self.cfg.checkpoint_path)
                    self.dataset.flush()

            save_checkpoint(self._snapshot(), self.cfg.checkpoint_path)

        finally:
            self.dataset.close()

        return self.basin_db

    # --------------------------------------------------------
    #  Helper: coupled feedback (used by both "coupled" accept
    #  and "coupled_no_chain")
    # --------------------------------------------------------
    def _coupled_feedback(self, basin, frames, rho, R) -> None:
        """Pick seed_{t+1} via score function; update T via G function."""
        rep = select_representative(
            frames=frames,
            m_t=basin.representative,
            energy_m_t=basin.energy,
            slab_n_atoms=self.cfg.slab_n_atoms,
            lambda_=self.cfg.lambda_,
        )
        self.seed_atoms = (
            rep if rep is not None else basin.representative.copy()
        )
        self.T_current = update_temperature(
            T_old=self.T_current,
            rho_t=rho, R_t=R,
            rho_star=self.cfg.rho_star, R_star=self.cfg.R_star,
            gamma_rho=self.cfg.gamma_rho, gamma_R=self.cfg.gamma_R,
            T_min=self.cfg.T_min, T_max=self.cfg.T_max,
        )

    # --------------------------------------------------------
    #  Snapshot for checkpoint / resume
    # --------------------------------------------------------
    def _snapshot(self) -> dict:
        return {
            "config": self.cfg,
            "basin_db": self.basin_db,
            "seed_atoms": self.seed_atoms,
            "T_current": self.T_current,
            "current_accepted_min": self.current_accepted_min,
            "E_current_accepted": self.E_current_accepted,
            "E_diff_current": self.E_diff_current,
            "chain_acc_history": list(self._chain_acc_history),
            "rng_state": self.rng.bit_generator.state,
        }