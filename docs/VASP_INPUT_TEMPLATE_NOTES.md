# VASP input notes

VASP pseudopotential files are license-restricted and are not redistributed in this package.

For reproduction, users should provide their own licensed pseudopotential library and update the placeholder in the configuration templates:

```text
<PATH_TO_LICENSED_VASP_POTCAR_LIBRARY>
```

The expected local structure is typically:

```text
<PATH_TO_LICENSED_VASP_POTCAR_LIBRARY>/Fe/POTCAR
<PATH_TO_LICENSED_VASP_POTCAR_LIBRARY>/H/POTCAR
<PATH_TO_LICENSED_VASP_POTCAR_LIBRARY>/N/POTCAR
```

Element order must be kept consistent between POSCAR/CONTCAR, MAGMOM, and the concatenated POTCAR used for the corresponding calculation.
