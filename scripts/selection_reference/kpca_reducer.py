"""
Kernel-PCA dimensionality reduction for SOAP descriptors.

Projects high-dimensional SOAP features (~10^3-10^4) onto a low-dimensional
subspace via Kernel-PCA with a Gaussian (RBF) kernel, retaining components
that together explain at least a specified fraction of total variance.

Reference: Section 2.3 of the manuscript, Eq. (2).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class KPCAReducer:
    """
    Kernel-PCA-based dimensionality reduction with automatic component
    selection by cumulative variance.

    Parameters
    ----------
    variance_threshold : float
        Retain leading components that together explain at least this
        fraction of total variance (default 0.95).
    gamma : float, optional
        RBF kernel coefficient. If None, automatically set via the
        median-distance heuristic:
            gamma = 1 / (2 * median_pairwise_squared_distance)
        which centers the kernel around typical configurational distances.
    max_components : int
        Hard cap on the number of components retained (default 50);
        prevents memory blowup for very large datasets.
    random_state : int
        Seed for reproducibility.

    Notes
    -----
    For datasets larger than ~5000 points, full KPCA is memory-intensive
    (O(N^2) Gram matrix). For such cases set `nystroem_components` > 0
    to enable Nystroem kernel approximation; this trades a small amount
    of accuracy for substantial memory savings.
    """

    def __init__(
        self,
        variance_threshold: float = 0.95,
        gamma: Optional[float] = None,
        max_components: int = 50,
        random_state: int = 0,
        nystroem_components: int = 0,
    ):
        self.variance_threshold = variance_threshold
        self.gamma = gamma
        self.max_components = max_components
        self.random_state = random_state
        self.nystroem_components = nystroem_components

        # Filled after fit
        self.gamma_used_: Optional[float] = None
        self.n_components_: Optional[int] = None
        self._kpca = None
        self._nystroem = None

    # ----------------------------------------------------------
    #  Median heuristic for gamma
    # ----------------------------------------------------------
    @staticmethod
    def _median_heuristic_gamma(X: np.ndarray, n_subsample: int = 1000,
                                  rng: Optional[np.random.Generator] = None) -> float:
        """Set gamma so that the RBF kernel decays over the median pairwise
        distance scale of the data."""
        if rng is None:
            rng = np.random.default_rng(0)
        n = len(X)
        idx = rng.choice(n, size=min(n_subsample, n), replace=False)
        sample = X[idx]
        # Pairwise squared distances
        diff = sample[:, None, :] - sample[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1).ravel()
        sq_dist = sq_dist[sq_dist > 0]  # exclude self-distances
        median_sq = float(np.median(sq_dist))
        return 1.0 / (2.0 * median_sq) if median_sq > 0 else 1.0

    # ----------------------------------------------------------
    #  Fit + transform
    # ----------------------------------------------------------
    def fit_transform(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit KPCA on X and return the reduced features.

        Parameters
        ----------
        X : (N, D) array of SOAP descriptors.
        verbose : bool

        Returns
        -------
        X_reduced : (N, n_components_) array.
        """
        from sklearn.decomposition import KernelPCA
        from sklearn.kernel_approximation import Nystroem

        rng = np.random.default_rng(self.random_state)

        # Auto-set gamma
        if self.gamma is None:
            self.gamma_used_ = self._median_heuristic_gamma(X, rng=rng)
            if verbose:
                print(f"[KPCA] gamma (median heuristic) = {self.gamma_used_:.4g}")
        else:
            self.gamma_used_ = self.gamma

        # Decide between full KPCA and Nystroem approximation
        use_nystroem = (
            self.nystroem_components > 0
            and len(X) > self.nystroem_components
        )

        if use_nystroem:
            if verbose:
                print(f"[KPCA] using Nystroem approximation with "
                      f"{self.nystroem_components} components")
            self._nystroem = Nystroem(
                kernel="rbf",
                gamma=self.gamma_used_,
                n_components=self.nystroem_components,
                random_state=self.random_state,
            )
            X_feat = self._nystroem.fit_transform(X)
            # Then linear PCA in Nystroem feature space
            from sklearn.decomposition import PCA
            self._kpca = PCA(
                n_components=min(self.max_components, X_feat.shape[1]),
                random_state=self.random_state,
            )
            X_full = self._kpca.fit_transform(X_feat)
            explained = self._kpca.explained_variance_ratio_
        else:
            if verbose:
                print(f"[KPCA] using exact KPCA on {len(X)} samples...")
            self._kpca = KernelPCA(
                kernel="rbf",
                gamma=self.gamma_used_,
                n_components=min(self.max_components, len(X)),
                eigen_solver="auto",
                random_state=self.random_state,
            )
            X_full = self._kpca.fit_transform(X)
            # KernelPCA reports eigenvalues; convert to explained variance ratio
            eigs = self._kpca.eigenvalues_
            explained = eigs / eigs.sum()

        # Select leading components by cumulative variance
        cumulative = np.cumsum(explained)
        n_keep = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        n_keep = min(n_keep, X_full.shape[1])
        self.n_components_ = n_keep

        if verbose:
            print(f"[KPCA] retained {n_keep} components "
                  f"(cumulative variance = {cumulative[n_keep-1]:.3f})")

        return X_full[:, :n_keep]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data into the previously fit KPCA space."""
        if self._kpca is None:
            raise RuntimeError("Call fit_transform first.")

        if self._nystroem is not None:
            X_feat = self._nystroem.transform(X)
            return self._kpca.transform(X_feat)[:, :self.n_components_]
        else:
            return self._kpca.transform(X)[:, :self.n_components_]