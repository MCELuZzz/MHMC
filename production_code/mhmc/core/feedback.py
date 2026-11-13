"""
Bidirectional feedback mechanism: the methodological core of this work.

Two feedback channels couple the basin-confined walk back to the next
MH escape:

  1. Representative-state selection
     -----------------------------------
     From the walk's accepted frames F_t, choose a single seed for the
     next escape that is energetically still inside the basin yet
     structurally displaced from the basin minimum. This prevents the
     next escape from repeatedly inheriting the same local curvature
     information.

         S(f) = (E(f) - E(m_t))  -  lambda * d_ads(f, m_t)
         seed_{t+1} = argmin_{f in F_t} S(f)

  2. Escape temperature update (full G function with rho_t and R_t)
     -----------------------------------
     The walk acceptance ratio rho_t and basin spread R_t together
     decide whether the next escape should be more or less aggressive.

         T_{t+1} = T_t * exp[ gamma_rho * (rho_star - rho_t)
                              - gamma_R   * (R_t - R_star) ],
         T_{t+1} clipped to [T_min, T_max].

Reference: Section 2.2 of the manuscript, feedback mechanism.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from .basin import adsorbate_distance


# ============================================================
#  (1) Representative-state selection
# ============================================================

def select_representative(
    frames: list[Atoms],
    m_t: Atoms,
    energy_m_t: float,
    slab_n_atoms: int,
    lambda_: float = 2.0,
) -> Optional[Atoms]:
    """
    Select the seed for the next escape from the walk's accepted frames.

    Parameters
    ----------
    frames : list of ase.Atoms
        Walk-accepted frames in the current basin.
    m_t : ase.Atoms
        Current basin minimum (reference for distance and energy).
    energy_m_t : float
        Potential energy of m_t (eV).
    slab_n_atoms : int
        Number of slab atoms.
    lambda_ : float
        Trade-off coefficient between energy proximity and structural
        displacement (eV/Å). Larger -> prefer more displaced frames.

    Returns
    -------
    seed : ase.Atoms or None
        The chosen representative; None if `frames` is empty (caller
        should then fall back to m_t itself).
    """
    if not frames:
        return None

    best_score = np.inf
    best_frame = None
    for f in frames:
        # Energy was stored in atoms.info during the walk phase
        # (cf. walk.py); the calculator does not survive Atoms.copy().
        E_f = f.info.get("potential_energy")
        if E_f is None:
            # Fallback: only happens if the frame was constructed elsewhere.
            raise RuntimeError(
                "Frame is missing 'potential_energy' in atoms.info. "
                "Frames passed to select_representative must come from walk_phase."
            )
        dE = float(E_f - energy_m_t)
        d = adsorbate_distance(f, m_t, slab_n_atoms)
        score = dE - lambda_ * d
        if score < best_score:
            best_score = score
            best_frame = f
    return best_frame.copy() if best_frame is not None else None

# ============================================================
#  (2) Escape temperature update (full G function)
# ============================================================

def update_temperature(
    T_old: float,
    rho_t: float,
    R_t: float,
    rho_star: float = 0.3,
    R_star: float = 0.5,
    gamma_rho: float = 1.0,
    gamma_R: float = 0.5,
    T_min: float = 800.0,
    T_max: float = 5000.0,
) -> float:
    """
    Update the next-iteration escape temperature using both feedback signals.

    The exponential form ensures T stays positive and changes smoothly
    even with noisy (rho_t, R_t):

        T_{t+1} = T_old * exp[ gamma_rho (rho_star - rho_t)
                               - gamma_R (R_t - R_star) ]

    Intuition:
      - low rho_t  (walk struggling)  -> basin is tight    -> increase T
      - high rho_t (walk easy)        -> basin well-explored -> decrease T
      - large R_t  (broad spread)     -> basin well-mapped -> decrease T
      - small R_t  (narrow spread)    -> need to move on   -> increase T

    Parameters (all defaults match Table S2):
      rho_star  : target acceptance ratio
      R_star    : target spread (Å)
      gamma_rho : sensitivity to acceptance-ratio mismatch
      gamma_R   : sensitivity to spread mismatch (Å^-1)
      T_min, T_max : hard clipping bounds (K)
    """
    log_factor = gamma_rho * (rho_star - rho_t) - gamma_R * (R_t - R_star)
    T_new = T_old * float(np.exp(log_factor))
    return float(np.clip(T_new, T_min, T_max))