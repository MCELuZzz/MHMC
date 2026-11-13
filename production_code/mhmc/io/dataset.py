"""
Unified dataset collector for the MHMC sampling pipeline.

Every Atoms object produced by the sampler — whether an escape-MD frame,
a relaxation step, a relaxed minimum, a walk-accepted frame, or a
walk-rejected trial — is registered here with structured metadata. The
combined collection forms the candidate pool that downstream Section 2.3
draws from for FPS-based active-learning selection.

Design choices:
  * Metadata lives in `atoms.info`, which extxyz preserves natively.
  * Frames are buffered in memory and flushed to disk every `flush_every`
    iterations to avoid per-frame I/O overhead during NequIP-bottlenecked
    runs.
  * A parallel jsonl index file records per-frame metadata for fast
    post-processing queries (e.g. "give me all walk_accepted frames from
    iter > 100 in basin 5").

Source tags:
  - "initial"        : the initial relaxed structure m_0
  - "escape_md"      : MD trajectory frames during escape
  - "relaxation"     : BFGS trajectory frames during minimization
  - "minimum"        : the relaxed minimum at the end of escape
  - "walk_accepted"  : Metropolis-accepted frame in the walk phase
  - "walk_rejected"  : Metropolis-rejected trial in the walk phase
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import write as ase_write


VALID_SOURCES = (
    "initial",
    "escape_md",
    "relaxation",
    "minimum",
    "walk_accepted",
    "walk_rejected",
)


class DatasetCollector:
    """
    Buffer + writer for sampler-produced configurations.

    Parameters
    ----------
    extxyz_path : str
        Path to the candidate pool extxyz file. Frames are appended.
    index_path : str
        Path to the per-frame metadata jsonl file.
    flush_every : int
        Flush buffer to disk every this many `register_frame` calls.
        Default 200.
    """

    def __init__(
        self,
        extxyz_path: str,
        index_path: str,
        flush_every: int = 200,
    ):
        self.extxyz_path = Path(extxyz_path)
        self.index_path = Path(index_path)
        self.extxyz_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate at start of run
        self.extxyz_path.write_text("")
        self.index_path.write_text("")

        self.flush_every = flush_every
        self._buffer: list[Atoms] = []
        self._index_buffer: list[dict] = []
        self._counter: int = 0
        self._total_written: int = 0

    # --------------------------------------------------------
    #  Public API
    # --------------------------------------------------------
    def register_frame(
        self,
        atoms: Atoms,
        source: str,
        iteration: int,
        basin_id: Optional[int] = None,
        energy: Optional[float] = None,
        chain_accepted: Optional[bool] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Register one configuration with metadata.

        Parameters
        ----------
        atoms : ase.Atoms
            The configuration. A copy is taken so the original is not
            modified.
        source : str
            One of VALID_SOURCES.
        iteration : int
            MHMC iteration index (-1 for the pre-loop initial frame).
        basin_id : int, optional
            Basin this frame belongs to (None when not yet classified).
        energy : float, optional
            Potential energy (eV). If None and atoms has a calculator,
            a get_potential_energy call is attempted.
        chain_accepted : bool, optional
            Whether the iteration's chain decision was accept (only
            meaningful for source=="minimum").
        extra : dict, optional
            Free-form additional metadata.
        """
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source must be one of {VALID_SOURCES}, got '{source}'"
            )

        frame = atoms.copy()

        # Energy: prefer explicit arg, then atoms.info, then calculator
        if energy is None:
            energy = atoms.info.get("potential_energy")
        if energy is None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = None

        info = {
            "source": source,
            "iteration": int(iteration),
            "basin_id": int(basin_id) if basin_id is not None else None,
            "energy": float(energy) if energy is not None else None,
            "chain_accepted": (
                bool(chain_accepted) if chain_accepted is not None else None
            ),
            "frame_id": self._total_written + len(self._buffer),
        }
        if extra:
            info["extra"] = extra

        # Embed into atoms.info for extxyz survival
        # (avoid clobbering any existing user fields)
        for k, v in info.items():
            frame.info[k] = v

        self._buffer.append(frame)
        self._index_buffer.append(info)
        self._counter += 1

        if self._counter >= self.flush_every:
            self.flush()

    def register_many(self, atoms_list: list[Atoms], **kwargs) -> None:
        """Register a batch of frames sharing the same metadata."""
        for a in atoms_list:
            self.register_frame(a, **kwargs)

    def flush(self) -> None:
        """Write buffer to disk and clear it."""
        if not self._buffer:
            return
        try:
            # Strip constraints before writing — they aren't needed for
            # downstream SOAP/KPCA/FPS selection, and unfamiliar constraint
            # types (e.g. Hookean) can confuse ASE's extxyz writer.
            write_buffer = []
            for f in self._buffer:
                g = f.copy()
                g.set_constraint([])   # drop ALL constraints
                write_buffer.append(g)

            ase_write(str(self.extxyz_path), write_buffer, append=True)
            with self.index_path.open("a") as f:
                for record in self._index_buffer:
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            import traceback
            print(f"[DatasetCollector] flush failed: {e}")
            print(f"[DatasetCollector] traceback:\n{traceback.format_exc()}")
            # Do not silently discard buffered frames. Abort the run so the
            # failure is visible and the in-memory buffer remains inspectable.
            raise
        else:
            self._total_written += len(self._buffer)
            self._buffer.clear()
            self._index_buffer.clear()
            self._counter = 0

    def close(self) -> None:
        """Final flush at end of run."""
        self.flush()

    @property
    def n_written(self) -> int:
        return self._total_written