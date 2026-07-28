# Glossary

The concepts you need before reading the code or the paper. Each entry ends with
where it is implemented.

## Binary stress task

Every analysis is a binary **stress vs non-stress** window classification. The
label definitions differ by corpus and are not interchangeable:

- **WESAD** — stress = TSST condition (label 2); non-stress = baseline (1) and
  meditation (4). Amusement (3) is excluded. A window is kept only if one
  condition covers more than 90 % of it.
- **Stress-Predict** — stress = Stroop test and interview; non-stress = the relax
  and baseline blocks. Hyperventilation and questionnaire blocks are excluded.
  Labels come from a protocol time log, so the alignment is a modelling choice
  and its sensitivity is measured.
- **Nurse** — stress = self-reported level 2; non-stress = level 0. The ambiguous
  level 1 is excluded. Labels are retrospective survey intervals covering hours
  of work, and roughly 80 % of retained windows are stress.

*Implemented in* `analysis_crosscorpus.build_wesad / build_sp / build_nurse`.

## Single-SCR baseline

The count of skin-conductance responses in a window, used **directly as a score
with no training at all**. Its decision direction is fixed a priori (more SCRs →
more stress) and is never flipped using test labels, so a below-chance AUROC is
reported as below chance.

Two consequences make it the bar every model has to clear. It is
**corpus-invariant by construction** — its within-corpus and cross-corpus AUROC
on a given target are identical, because nothing is fit. And because AUROC is
threshold-free and the score is continuous, no operating threshold is chosen for
the primary analysis.

*Implemented in* `analysis_crosscorpus.single_auroc`.

## LOSO vs LODO

- **LOSO** (leave-one-subject-out) — within one corpus, hold out one subject at a
  time. Answers "does this generalize to a new person in the same protocol?"
- **LODO** (leave-one-dataset-out) — train on the union of the other corpora and
  test on the held-out corpus. Answers "does this survive a change of stressor,
  population, device fit and labeling regime?"

The paper's claims live at the LODO level. Subject-level splits are asserted, not
assumed: `assert_no_subject_overlap` raises if a subject ever appears on both
sides.

*Implemented in* `utils_analysis.loso`, `analysis_crosscorpus.cross`.

## Transfer-robust feature set

Eight of the 22 window features, chosen to be device- and ambient-invariant:
phasic SCR count, mean and maximum SCR amplitude, phasic area, tonic (SCL) slope,
EDA decay, temperature slope, HR slope.

Absolute means — mean HR, mean skin conductance, mean temperature — are
deliberately excluded, because their level depends on the device, the ambient
conditions and the individual rather than on the state being measured. Every
cross-corpus model uses exactly this subset.

*Implemented in* `analysis_crosscorpus.TRANSFER` and `TCOLS`.

## Activity features

Four accelerometer-derived quantities computed identically in all three corpora
from the 32 Hz wrist magnitude signal in g: mean, standard deviation, ENMO
(Euclidean norm minus one), and mean absolute jerk.

They are kept **outside** `FEATS` on purpose. They are used for movement
stratification and, appended sideways, for the context-aware model. They are not
part of the primary feature set, because the movement–stress association points
in different directions in different protocols.

*Implemented in* `utils_analysis.activity_features` and `ACC_FEATS`.

## iso-HR matching

The core methodological device. Heart rate rises under stress but also under
physical activity, so a model can score well by detecting arousal rather than
stress. iso-HR matching removes that route:

1. assign every intact window to a 5 bpm HR bin, **within subject**;
2. in each subject–bin cell, set the retained count to the smaller class count;
3. sample that many whole windows from each class;
4. concatenate and evaluate subject-wise as usual.

No signal is spliced and no feature is recomputed — the unit of selection is the
intact window. Bins containing only one class contribute nothing. Because the
subsample is random, the whole procedure is repeated over seven seeds and the
spread is reported.

Two modes exist. `global` matches once over the whole corpus and then runs LOSO
on the matched set; `strict` matches training and held-out subjects independently
inside each fold. Leakage-sensitive claims are checked in `strict`.

*Implemented in* `utils_analysis.match_indices` and the `mode` argument of
`utils_analysis.loso`.

## HR-leakage index

The matched-set AUROC of the all-feature model minus that of the HR-level-free
model (everything except mean HR and mean IBI). Near zero means HR *level*
contributes little once the other signals are present.

**The caveat matters as much as the definition.** The index is informative only
where a discriminative signal exists at all. In a corpus where every feature set
sits near chance, the index is near zero trivially — a floor effect that carries
no evidence about confounding. This is why the paper treats iso-HR as diagnostic
mainly in WESAD.

*Implemented in* the iso-HR decomposition block of
`analysis_crosscorpus.run_analysis`.

## Subject-cluster bootstrap and within-subject permutation

Two procedures answering different questions, and the paper leans on the first.

- **Subject-cluster bootstrap** — resample *people* with replacement, 1,000 times.
  Quantifies generalization to new subjects. This is the primary, more
  conservative summary.
- **Within-subject permutation** — shuffle labels inside each subject, keeping the
  subject set and the fixed scores. Tests only whether a within-corpus
  association exceeds chance. With many windows it reaches significance even at
  AUROC ≈ 0.55, which is why it is a secondary check.

Overlapping windows are handled by construction: resampling is at the subject
cluster, so windows that share samples never straddle a resampling boundary.

*Implemented in* `utils_analysis.boot_ci`, `cluster_perm_p`,
`analysis_statistics.perm_p_within_subject`, `paired_boot_diff`.

## Negative control

Labels are permuted **within** each subject, preserving every subject's class
balance, and the whole within-corpus pipeline is re-run unchanged. If the
scrambled AUROC is not near chance, label information is reaching the model
through feature construction, the split, or the evaluation path.

Note that a *below*-chance value cannot indicate leakage — leakage inflates the
statistic. On Nurse the control returns ≈0.30, which reflects between-subject
score offsets against very unequal per-subject prevalences.

*Implemented in* `analysis_diagnostics.run_negative_control`.

## Between-stratum share

In the error stratification, pooled AUROC minus the within-stratum aggregate
(within-stratum AUROCs combined with `n_stress × n_non_stress` weights, which is
the pooled within-stratum Mann–Whitney statistic). A large positive share means
apparent discrimination comes from score offsets *between* strata rather than
ranking *inside* them.

*Implemented in* `analysis_error_strata.stratified_auroc`.

## Historical name

Internal notes and older commits call this project **IsoHR-Source**. The
public-facing name is the one in `CITATION.cff`.
