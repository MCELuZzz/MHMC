"""
Optional Hookean penalty calculator wrapper.

Wraps a base ASE calculator and adds a one-sided harmonic restoring
energy term whenever a bonded pair (i, j) exceeds a soft cutoff
distance beta * r0. This prevents unphysical bond dissociation during
high-temperature MD, without affecting the energy landscape near
equilibrium geometries.

Reference: Section 2.2, optional structure-preserving penalty (mentioned
briefly; full description in SI).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class HookeanWrappedCalculator(Calculator):
    """
    A calculator wrapper that adds Hookean penalties to a base calculator.

    Parameters
    ----------
    base_calc : ase.Calculator
        The underlying calculator (typically the MLP).
    bond_pairs : list of (i, j, r0) tuples
        Atom index pairs (i, j) and their equilibrium bond length r0 (Å).
    k : float
        Harmonic force constant (eV/Å^2). Default 10.0.
    beta : float
        Soft cutoff multiplier; the penalty activates at r > beta * r0.
        Default 1.5.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calc: Calculator,
        bond_pairs: list[tuple[int, int, float]],
        k: float = 10.0,
        beta: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.bond_pairs = list(bond_pairs)
        self.k = k
        self.beta = beta

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Delegate to base calculator
        self.base_calc.calculate(atoms, properties, system_changes)
        E = float(self.base_calc.results.get("energy", 0.0))
        F = np.array(self.base_calc.results.get("forces", np.zeros((len(atoms), 3))))

        # Add Hookean penalties for bonds beyond beta * r0
        for (i, j, r0) in self.bond_pairs:
            d_vec = atoms.get_distance(i, j, mic=True, vector=True)
            r = float(np.linalg.norm(d_vec))
            r_cut = self.beta * r0
            if r <= r_cut:
                continue

            # One-sided Hookean penalty, continuous at r_cut:
            #   E_pen = 0.5 * k * (r - r_cut)^2,  r > r_cut
            # ASE get_distance(i, j, vector=True) points from i -> j,
            # so the restoring force on i points toward j.
            dr = r - r_cut
            E += 0.5 * self.k * dr ** 2
            unit = d_vec / r
            f_pen = self.k * dr * unit
            F[i] += f_pen
            F[j] -= f_pen

        self.results["energy"] = E
        self.results["forces"] = F