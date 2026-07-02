# Sanity scan notes

This public repository was checked for common restricted VASP runtime files, compiled caches, local absolute paths, and repository-placement placeholders.

The repository intentionally contains explanatory mentions of restricted filenames such as `POTCAR`, `WAVECAR`, `CHGCAR`, `OUTCAR`, and `vasprun.xml`, but it does not include the corresponding raw VASP files from production calculations. Configuration templates use the placeholder:

```text
<PATH_TO_LICENSED_VASP_POTCAR_LIBRARY>
```

Users must provide their own licensed VASP pseudopotential library when reproducing VASP calculations.
