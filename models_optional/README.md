# Optional trained model files

The original internal check package contained a compiled NequIP model file. It is not included in this cleaned public package by default because distribution of trained models should be approved by all authors before public release.

If the authors decide to distribute the trained model, place it here with a neutral file name such as:

```text
FeNH_NequIP_compiled_model.pt2
```

and update the manuscript Data and Software Availability statement accordingly.

Minimum model metadata to report:

```yaml
model_type: NequIP compiled ASE model
chemical_system: Fe-N-H on Fe(110)
training_data: see ../data/structures/
software: NequIP
license_note: model file distributed only if approved by the authors
```
