"""
SOAP descriptor computation for active-learning selection.

Each configuration is encoded by a SOAP (Smooth Overlap of Atomic
Positions) descriptor, averaged over atoms to yield a fixed-length,
translation-, rotation-, and permutation-invariant vector.

Reference: Section 2.3 of the manuscript, Eq. (1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms


# Default SOAP hyperparameters (used unless overridden)
DEFAULT_SOAP_PARAMS = {
    "r_cut":   5.0,     # Å, radial cutoff (covers second coordination shell)
    "n_max":   8,       # number of radial basis functions
    "l_max":   6,       # max spherical harmonic order
    "sigma":   0.5,     # Å, Gaussian broadening
    "average": "inner", # per-atom descriptors averaged via inner-product
}


def compute_soap_descriptors(
    atoms_list: list[Atoms],
    species: Optional[list[str]] = None,
    soap_params: Optional[dict] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute SOAP descriptors for a list of configurations.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        Configurations from the MHMC candidate pool.
    species : list of str, optional
        Chemical species in the dataset. If None, auto-detected by scanning
        all configurations.
    soap_params : dict, optional
        SOAP hyperparameters. Defaults to DEFAULT_SOAP_PARAMS if not given.
    n_jobs : int
        Parallel jobs for dscribe SOAP. Use -1 for all available cores.
    verbose : bool

    Returns
    -------
    descriptors : np.ndarray of shape (N, D)
        N is the number of configurations, D is the SOAP feature dimension
        (depends on n_max, l_max, and number of species).

    Notes
    -----
    Requires dscribe (`pip install dscribe`). All configurations must use
    the same chemical species set; otherwise the descriptor dimension
    differs per configuration and the array cannot be stacked.
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError as e:
        raise ImportError(
            "dscribe is required for SOAP descriptors. "
            "Install with: pip install dscribe"
        ) from e

    params = {**DEFAULT_SOAP_PARAMS}
    if soap_params:
        params.update(soap_params)

    # Auto-detect species if not provided
    if species is None:
        species_set: set[str] = set()
        for a in atoms_list:
            species_set.update(a.get_chemical_symbols())
        species = sorted(species_set)
        if verbose:
            print(f"[SOAP] auto-detected species: {species}")

    # Construct SOAP descriptor object
    soap = SOAP(
        species=species,
        periodic=True,                     # surface slabs have PBC
        r_cut=params["r_cut"],
        n_max=params["n_max"],
        l_max=params["l_max"],
        sigma=params["sigma"],
        average=params["average"],
        sparse=False,
    )

    if verbose:
        print(f"[SOAP] computing descriptors for {len(atoms_list)} configurations...")
        print(f"[SOAP] params: r_cut={params['r_cut']} A, "
              f"n_max={params['n_max']}, l_max={params['l_max']}, "
              f"sigma={params['sigma']} A, average='{params['average']}'")

    # Compute descriptors
    descriptors = soap.create(atoms_list, n_jobs=n_jobs)
    descriptors = np.asarray(descriptors, dtype=np.float32)

    if verbose:
        print(f"[SOAP] done. shape = {descriptors.shape}, "
              f"memory = {descriptors.nbytes / 1024**2:.1f} MB")

    return descriptors