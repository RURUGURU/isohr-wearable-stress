# Major-Revision Result Ledger

Reviewer-requested analyses were measured on 2026-07-25. The same exact Python
package versions were migrated on 2026-07-27 to the root `env.yaml` Conda
manifest. The canonical cross-corpus run was regenerated and synchronized with
the manuscript and figures on 2026-07-28 in the `easd` environment.

## Reproduction

- Canonical cohort reconstruction: WESAD 1,133 windows/15 subjects;
  Stress-Predict 2,846/35; Nurse 9,226/15.
- Nurse reconstruction remains canonical with the host timezone forced to
  `Asia/Seoul`: 9,226 windows, 7,347 stress, 1,879 non-stress, 15 subjects.
- The regenerated canonical log is
  `revision/MDPI_Healthcare_revision_20260727/results/run_crosscorpus.txt`.
  It includes a machine-readable `ANALYSIS_AUDIT_JSON` line from the same
  process that produced the numerical output.
- Single-SCR, logistic, cohort, movement, and subject-bootstrap estimates match
  the preceding ledger exactly. Dependency-sensitive histogram boosting and
  single-draw iso-HR values changed by at most 0.009; the manuscript and F1/F3
  were synchronized to the regenerated log.

### 2026-07-28 runtime audit

- A fresh environment created only from root `env.yaml` passed strict quality
  checks and all nine non-integration tests.
- The full cross-corpus command reconstructed the same 1,133/2,846/9,226
  windows and the same 15/35/15 subjects. Primary single-SCR/logistic estimates
  and subject-bootstrap intervals were identical to the preceding ledger; the
  dependency-sensitive GBM/single-draw iso-HR drift remained bounded by 0.009.
- `scripts/run_crosscorpus.sh` now writes stdout to a temporary file in the
  authoritative results directory and replaces `run_crosscorpus.txt` only after
  a successful run. A regression test verifies this atomic persistence boundary.
- Actual EDA execution was `cvxEDA(neurokit2):232`; backend failures were zero.
  One malformed IBI row was excluded and is now reported as an anonymous
  aggregate rather than silently dropped.
- Nurse reconstruction reported 42 survey intervals without a covering session
  and one signal-loading exclusion without participant IDs or paths. The
  canonical cohort remained exactly 9,226 windows, 7,347 stress, 1,879
  non-stress, and 15 subjects.
- Regression tests were observed failing before the fix for zero/non-finite
  protocol values, unreported EDA fallback, and silent Nurse input exclusion;
  the same four targeted tests passed after the fix.

## Reviewer-Requested Analyses

### Nurse operational utility

- Prevalence: 0.7963.
- Single SCR: AUROC 0.5533, AP 0.8256.
- Logistic: AUROC 0.5412, AP 0.8260, Brier 0.2428, ECE 0.2376,
  balanced accuracy 0.5240, sensitivity 0.5834, FPR 0.5354.
- Boosting: AUROC 0.5592, AP 0.8283, Brier 0.3442, ECE 0.3778,
  balanced accuracy 0.5408, sensitivity 0.3562, FPR 0.2746.
- Artifact: `revision_nurse_utility.json`; figure:
  `../figures/F4_nurse_utility.png`.

### Window overlap

- Non-overlapping 64 s windows preserve the primary LODO pattern.
- Single/logistic AUROC: WESAD 0.896/0.890; Stress-Predict 0.564/0.563;
  Nurse 0.554/0.545.
- Artifact: `revision_nonoverlap.txt`.

### Five-minute window

- 300 s/150 s reduces windows to 166/557/1,770.
- LODO single/logistic AUROC: WESAD 0.917/0.913; Stress-Predict
  0.562/0.556; Nurse 0.564/0.522.
- WESAD iso-HR matched sample falls to 26 windows and is not interpreted.
- Artifact: `revision_window300.txt`.

### Direct HRV ablation

- With-HRV minus without-HRV AUROC at 64 s:
  WESAD +0.0168; Stress-Predict +0.0034; Nurse +0.0046.
- At 300 s: WESAD -0.0058; Stress-Predict +0.0095; Nurse -0.0102.
- Artifact: `revision_hrv_window_sensitivity.json`. Table `tab:hrv_sens` in the
  manuscript is the sole presentation of this result; the earlier duplicate
  figure was removed at Reviewer 1's request.

### Stress-Predict label alignment

- Interval shifts: -64, -32, 0, +32, +64 s.
- Single SCR AUROC range: 0.5482--0.5822.
- Within-corpus logistic AUROC range: 0.6140--0.6367.
- Artifact: `revision_stress_predict_alignment.json`; figure:
  `../figures/F6_stress_predict_alignment.png`.

### Feature distribution shift

- Mean robust-standardized Wasserstein distance:
  WESAD--Stress-Predict 0.9567; WESAD--Nurse 1.5523;
  Stress-Predict--Nurse 1.6131.
- Largest individual shifts: EDA decay, SCL slope, and phasic area.
- Artifact: `revision_feature_shift.json`; figure:
  `../figures/F5_feature_shift.png`.

## Second Revision Pass (2026-07-28)

Added after the first response letter, to answer R1-4, R2-1 and R2-2 with
measurements rather than narrowed wording. The corpus builders gained
accelerometer extraction; the canonical run was repeated and every previously
reported value is byte-identical (only wall-clock timings differ).

### Unsupervised domain-adaptation sweep

Leave-one-dataset-out, identical protocol, four transductive methods. Subspace
alignment and transfer components report the target-maximizing component count
(an optimistic upper bound); the full 2--8 grid is stored in the artifact.

- WESAD (single 0.897): logistic 0.890, CORAL 0.879, subspace 0.895,
  transfer components 0.865, importance weighting 0.897.
- Stress-Predict (single 0.563): 0.564 / 0.564 / 0.567 / 0.565 / 0.565.
- Nurse (single 0.553): 0.541 / 0.542 / 0.574 / 0.571 / 0.543.
- Exactly one paired difference excludes zero: Nurse single vs subspace alignment
  (-0.020 [-0.039, -0.002]). It uses the oracle hyperparameters and its
  subject-bootstrap interval [0.431, 0.707] still includes chance.
- Transfer components sweep components crossed with three RBF bandwidths after an
  audit found the earlier single-bandwidth result was convention-dependent; the
  embedding is standardized so the value no longer depends on the eigenvector
  normalization convention either.
- Artifact: `revision_adaptation.json`; figure `../figures/F8_adaptation.png`.

### Supervised target-adaptation curve

Source corpora plus k labeled target subjects, evaluated on the remaining
subjects, 20 fixed-seed draws.

- Within each draw the source-only model is re-scored on the same held-out
  subjects, so the headline quantity is the paired gain, not raw AUROC across k.
- Mean paired gain at k=8: WESAD +0.002, Stress-Predict +0.001,
  Nurse +0.029 +/- 0.024.
- Artifact: `revision_target_supervised.json`.

### Accelerometer context-aware model

Four activity features (magnitude mean and SD, ENMO, mean absolute jerk) computed
identically in all three corpora from the 32 Hz wrist accelerometer, appended to
the eight transfer features.

- Within-corpus LOSO improves everywhere: WESAD 0.876 to 0.898 (LR) and 0.886 to
  0.930 (GBM); Stress-Predict 0.628 to 0.681 and 0.633 to 0.671; Nurse 0.450 to
  0.468 and 0.489 to 0.572.
- Leave-one-dataset-out degrades everywhere: WESAD 0.890 to 0.836, Stress-Predict
  0.564 to 0.520, Nurse 0.541 to 0.519.
- Signed movement-label association is direction-inconsistent: jerk alone gives
  0.828 (WESAD), 0.481 (Stress-Predict), 0.590 (Nurse). Activity-only transfer to
  WESAD is 0.792, i.e. largely protocol identity.
- Artifact: `revision_context_aware.json`; figure `../figures/F9_context_aware.png`.

### Leave-one-dataset-out error stratification

Fixed source-trained scores stratified without refitting; within-stratum AUROC
combined with n_stress x n_non-stress weights; small cells (<20 windows or
<3 subjects) reported with counts but no AUROC.

- Nurse: pooled 0.541 falls to 0.489 conditioning on Mahalanobis distance to
  source and 0.499 conditioning on movement tertile.
- WESAD: pooled 0.890 falls to 0.826 (distance) and 0.857 (movement) but stays
  clearly above chance; in the 102 windows at or above the 0.0796 g gate it
  collapses to 0.530 (single) and 0.467 (logistic).
- Stress-Predict: every stratified estimate within 0.036 of pooled.
- Per-subject logistic AUROC, anonymized and independently sorted: WESAD 15/15
  evaluable, median 0.985; Stress-Predict 33/35, median 0.589; Nurse 13/15,
  median 0.511, range 0.370--0.725.
- Artifact: `revision_error_stratification.json`; figure
  `../figures/F10_error_strata.png`.

### Equivalent commands

```bash
bash scripts/run_revision.sh adaptation
bash scripts/run_revision.sh target-supervised
bash scripts/run_revision.sh context-aware
bash scripts/run_revision.sh error-stratification
```

## Current Equivalent Commands

Run these commands from the repository root. Each shell entry point uses the
`easd` Conda environment and resolves the authoritative revision directory from
`scripts/analysis_config.ini`.

```bash
bash scripts/setup_env.sh
bash scripts/run_revision.sh nurse-utility
bash scripts/run_revision.sh sensitivity nonoverlap
bash scripts/run_revision.sh sensitivity window300
bash scripts/run_revision.sh stress-predict-alignment
bash scripts/run_revision.sh feature-shift
bash scripts/run_revision.sh hrv-window-sensitivity
```

Verification status is owned by `../QA_STATUS.md`; this file records
measurements only.
