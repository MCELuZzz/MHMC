# Pseudocode: propagation-controlled coupled MH--MC sampling

This document provides a pseudocode-level description of the coupled MH--MC sampling workflow used in the manuscript. The full production implementation is retained in the authors' in-house code base pending future software release.

## Inputs

- Initial structure or current propagation seed `S_current`.
- Initial escape temperature `T_esc`.
- Basin database `B`.
- Maximum number of sampling iterations.
- Basin-identification thresholds, including energy and adsorbate-RMSD thresholds.
- Basin-confined walk parameters, including walk temperature, energy envelope, number of trial moves, and perturbation scale.

## One iteration

```text
1. Escape step
   a. Assign velocities to S_current according to T_esc.
   b. Run a short escape trajectory.
   c. Locally relax the escaped structure to obtain a proposed minimum S_min.

2. Basin classification and registration
   a. Compare S_min against the basin database using energy and permutation-invariant adsorbate RMSD.
   b. If S_min belongs to an existing basin, update the basin representative if it is lower in energy.
   c. Otherwise, register S_min as a new basin.

3. Propagation-chain acceptance
   a. Evaluate the minima-hopping accept/reject criterion using the current chain minimum and the proposed minimum.
   b. If rejected:
      - keep the previous accepted minimum as the next propagation seed;
      - increase T_esc for the next escape attempt;
      - still retain informative generated configurations in the candidate pool when appropriate.
   c. If accepted:
      - advance the propagation chain to S_min;
      - enter the basin-confined walk stage.

4. Basin-confined walk
   a. Starting from the accepted basin representative, perform local perturbation trials.
   b. Apply an energy-envelope filter to keep accepted frames within the basin neighborhood.
   c. Apply a Metropolis-style probability to locally accepted trials.
   d. Store accepted local frames and representative rejected/local frames in the candidate pool.

5. Bidirectional feedback
   a. Compute local-walk statistics such as walk acceptance ratio and basin spread.
   b. Select a structurally displaced but energetically basin-confined frame as the next escape seed.
   c. Update the next escape temperature using the walk statistics.
   d. Update the chain acceptance tolerance using a sliding-window acceptance estimate.

6. Candidate-pool update
   Add escape frames, relaxation snapshots, minima, walk-accepted frames, and selected rejected/local frames to the candidate pool with source metadata.
```

## Output

- Accumulated basin database.
- Candidate structure pool with source labels.
- Metadata describing the source stage, basin assignment, and sampling iteration for each candidate structure.
