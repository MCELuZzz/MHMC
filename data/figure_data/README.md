# Figure data

This directory contains example generated figure outputs and should also contain the machine-readable CSV/source tables underlying each manuscript figure, for example:

```text
figure1_energy_parity.csv
figure1_force_parity.csv
figure2_basin_counts.csv
figure2_coverage_metric.csv
figure2_cross_evaluation_metrics.csv
figure3_coverage_adsorption_energies.csv
figure4_reaction_profiles.csv
figure5_transferability_data.csv
```

The plotting scripts in `../../scripts/plotting/` assume that the corresponding CSV files are available in this directory or are passed through command-line options.
