# Pseudocode: basin classification and feedback quantities

## Basin classification

A relaxed minimum is assigned to an existing basin when both energy and structural criteria are satisfied.

```text
for each basin representative in basin_database:
    compute energy difference
    compute permutation-invariant adsorbate RMSD under the minimum-image convention
    if energy difference < threshold and RMSD < threshold:
        assign the minimum to this basin
        update the representative if the new minimum has lower energy
        stop
if no basin matches:
    register the minimum as a new basin
```

The permutation-invariant RMSD is required because exchanging indistinguishable adsorbates should not create a false new basin.

## Feedback variables

The basin-confined walk returns two main scalar statistics:

```text
walk_acceptance_ratio = number_of_accepted_local_trials / number_of_total_trials
basin_spread = average adsorbate-RMSD of accepted local frames relative to the basin minimum
```

These quantities are used to update:

```text
1. the next escape seed, chosen as a locally displaced but energetically basin-confined frame;
2. the next escape temperature, increased or decreased according to local walk statistics;
3. the propagation-chain acceptance tolerance, adjusted using a sliding-window acceptance rate.
```
