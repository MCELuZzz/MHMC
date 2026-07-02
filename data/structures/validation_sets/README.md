# Validation structures

This directory contains validation structures in machine-readable `extxyz` format.

Included files:

```text
HNNH_valid_combined.extxyz
valid_combined.extxyz
```

These files are intended for checking structure loading, validation-set analysis, and reproduction of representative MLIP/DFT comparison workflows. They can be read using ASE:

```python
from ase.io import read
atoms = read('valid_combined.extxyz', ':')
```
