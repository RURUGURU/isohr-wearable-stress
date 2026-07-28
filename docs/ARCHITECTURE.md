# Architecture

How a raw Empatica recording becomes a number in the manuscript. Read this
first if you have just inherited the repository.

## 1. Pipeline

```
data/raw/{wesad,stress_predict,nurse}/          immutable third-party input
        │
        │  analysis_crosscorpus.build_wesad() / build_sp() / build_nurse()
        │    · segment into 64 s windows, 50 % overlap (analysis_config.ini)
        │    · decompose EDA into tonic/phasic with cvxEDA (utils_analysis.cvx_decompose)
        │    · one row per window: (subject_id, label, feature_vector, {activity})
        ▼
FEATS — 22 window features                       utils_analysis.FEATS
        │   HR level · cardiac dynamics/HRV · EDA · temperature · recovery
        │
        ├─► TCOLS — 8 transfer features          analysis_crosscorpus.TRANSFER
        │     device- and ambient-invariant subset used by every cross-corpus model
        │
        └─► ACC_FEATS — 4 activity features      utils_analysis.ACC_FEATS
              kept OUTSIDE FEATS; appended sideways only for the context-aware model
        │
        │  evaluation (utils_analysis.loso, boot_ci, cluster_perm_p)
        │    · LOSO  — within corpus, leave one subject out
        │    · LODO  — across corpora, leave one dataset out
        │    · uncertainty by subject-cluster bootstrap, never by window
        ▼
results/*.txt and results/*.json                 the only source of paper numbers
        │
        │  plot_figures.py parses the logs; analysis modules write their own figures
        ▼
figures/F*.png ──► manuscript_healthcare_revision_20260727.tex ──► the PDF
```

Two rules follow from this diagram and are enforced by tests:

- **Nothing writes a number into the manuscript by hand.** Figures are generated
  from the result logs, and `plot_figures._require_rows` fails loudly if a log's
  row count changes.
- **`FEATS` order is frozen.** Reordering it silently changes `IDX`, `NONHR`, the
  iso-HR decomposition and every stored result. `ACC_FEATS` exists as a separate
  block precisely so activity features could be added without touching it.

## 2. What each file does

### Configuration and paths

| File | Purpose |
|---|---|
| `scripts/analysis_config.ini` | The only place protocol constants live: window, step, seed, iso-HR bin width, movement gate, authoritative revision directory. |
| `scripts/utils_config.py` | Parses that ini into a frozen `AnalysisSettings`, rejecting invalid values before any data is read. |
| `scripts/utils_paths.py` | Resolves every path from the repository root. `EASD_PROJECT_ROOT` overrides it. |
| `scripts/utils_time.py` | Nurse survey timestamps to epochs, independent of host timezone. |

### Signal processing and corpus construction

| File | Purpose |
|---|---|
| `scripts/utils_analysis.py` | Signal loading, EDA decomposition, the 22 window features, the 4 activity features, iso-HR matching, LOSO, bootstrap, permutation, Holm. The numerical core. |
| `scripts/analysis_crosscorpus.py` | The three corpus builders and the canonical cross-corpus run. `build_all()` is the single entry point that constructs all three cohorts. |

### Analyses

| File | Purpose | Writes |
|---|---|---|
| `scripts/analysis_statistics.py` | Cluster-aware permutation + Holm, 7-seed iso-HR stability, paired subject-bootstrap. | `run_crosscorpus_stats.txt` |
| `scripts/analysis_diagnostics.py` | Pairwise feature shift, Stress-Predict label-alignment sensitivity, label-scramble negative control. | `revision_feature_shift.json`, `revision_stress_predict_alignment.json`, `revision_negative_control.json`, F5, F6 |
| `scripts/analysis_hrv.py` | HRV-feature ablation at 64 s and 300 s. | `revision_hrv_window_sensitivity.json` |
| `scripts/analysis_adaptation.py` | Four unsupervised adaptation methods; supervised target-adaptation curve. | `revision_adaptation.json`, `revision_target_supervised.json`, F8 |
| `scripts/analysis_context.py` | Accelerometer context-aware transfer model. | `revision_context_aware.json`, F9 |
| `scripts/analysis_error_strata.py` | LODO error stratification by movement, HR band, beat quality, distance to source. | `revision_error_stratification.json`, F10 |
| `scripts/metrics.py` | Nurse operating metrics at the fixed 0.5 threshold: AP, Brier, ECE, balanced accuracy, sensitivity, FPR. | — |
| `scripts/main_revision.py` | The single CLI surface for every reviewer-requested analysis. | `revision_nurse_utility.json`, F4 |
| `scripts/plot_figures.py` | Regenerates F1–F3 by parsing the canonical logs. | F1, F2, F3 |

### Entry points

Every `scripts/*.sh` resolves the project root from its own location and runs
inside the `easd` conda environment (`EASD_ENV_NAME` overrides the name).

| Script | Runs |
|---|---|
| `setup_env.sh` | Creates the pinned environment from `env.yaml`. |
| `run_quality.sh` | basedpyright + ruff over the maintained surface. |
| `run_tests.sh [fast\|full]` | `fast` skips tests needing raw data. |
| `run_crosscorpus.sh` | The canonical analysis; writes `run_crosscorpus.txt` atomically. |
| `run_statistics.sh` | The statistics companion. |
| `run_revision.sh <command>` | All reviewer analyses; `--help` lists them. |
| `run_figures.sh` | Regenerates F1–F3 from the logs. |
| `build_manuscript.sh` | Builds the PDF and cleans LaTeX aux files. |

## 3. Result artifacts

Everything under `revision/MDPI_Healthcare_revision_20260727/results/`.

| Artifact | Produced by | Contains |
|---|---|---|
| `run_crosscorpus.txt` | `run_crosscorpus.sh` | Cohort counts, within-corpus LOSO, LODO with CORAL, Nurse movement tertiles, iso-HR decomposition, subject-bootstrap CIs, and the `ANALYSIS_AUDIT_JSON` line recording EDA backend and input exclusions. |
| `run_crosscorpus_stats.txt` | `run_statistics.sh` | Within-subject permutation p-values with Holm correction, iso-HR stability over 7 seeds, paired subject-bootstrap differences against the single feature. |
| `revision_nurse_utility.json` | `run_revision.sh nurse-utility` | Nurse AUROC, average precision, Brier, ECE, balanced accuracy, sensitivity, FPR at a fixed 0.5 threshold. |
| `revision_nonoverlap.txt` | `run_revision.sh sensitivity nonoverlap` | The full protocol with non-overlapping 64 s windows. |
| `revision_window300.txt` | `run_revision.sh sensitivity window300` | The full protocol with 300 s windows. |
| `revision_hrv_window_sensitivity.json` | `run_revision.sh hrv-window-sensitivity` | LOSO AUROC with and without six HRV/quality variables at both window lengths. |
| `revision_stress_predict_alignment.json` | `run_revision.sh stress-predict-alignment` | AUROC under protocol-label shifts of −64 to +64 s. |
| `revision_feature_shift.json` | `run_revision.sh feature-shift` | Pairwise Wasserstein distance per transfer feature, divided by pooled IQR. |
| `revision_negative_control.json` | `run_revision.sh negative-control` | Within-subject label-scramble AUROC per corpus. The leakage control. |
| `revision_adaptation.json` | `run_revision.sh adaptation` | Four adaptation methods per target: AUROC, bootstrap CI, paired difference against the single feature, and the full hyperparameter grid. |
| `revision_target_supervised.json` | `run_revision.sh target-supervised` | AUROC and paired gain when k target subjects are labeled. |
| `revision_context_aware.json` | `run_revision.sh context-aware` | Within-corpus and transfer AUROC with and without activity features, plus the signed movement–label association. |
| `revision_error_stratification.json` | `run_revision.sh error-stratification` | Per-stratum AUROC with counts and status, within-stratum aggregate, and anonymized per-subject distributions. |
| `revision_environment.json` | recorded manually | OS, hardware, exact package versions, seed. |
| `REVISION_RESULTS.md` | maintained by hand | Narrative ledger tying each artifact to the reviewer request that motivated it. |

## 4. Figures

Figure **file names do not match rendered figure numbers** — F7 was removed as a
duplicate of the HRV table, and the new sections were inserted mid-document.

| File | Rendered as | Produced by |
|---|---|---|
| — | Figure 1 (TikZ workflow) | drawn in the `.tex` |
| `F1_crosscorpus.png` | Figure 2 | `plot_figures.py` |
| `F8_adaptation.png` | Figure 3 | `analysis_adaptation.py` |
| `F9_context_aware.png` | Figure 4 | `analysis_context.py` |
| `F6_stress_predict_alignment.png` | Figure 5 | `analysis_diagnostics.py` |
| `F5_feature_shift.png` | Figure 6 | `analysis_diagnostics.py` |
| `F3_isohr_decomp.png` | Figure 7 | `plot_figures.py` |
| `F2_nurse_movement.png` | Figure 8 | `plot_figures.py` |
| `F4_nurse_utility.png` | Figure 9 | `main_revision.py` |
| `F10_error_strata.png` | Figure 10 | `analysis_error_strata.py` |

MDPI production may ask for `Figure1.png`-style names at typesetting; rename at
that point, not before, or the `\includegraphics` paths break.

## 5. Environment

`env.yaml` is authoritative: it pins Python 3.11.13 and every package. Create it
with `bash scripts/setup_env.sh`.

`requirements.txt` mirrors only the pip section, for users who cannot use conda;
it cannot pin the interpreter. Regenerate it after editing `env.yaml`:

```bash
python3 - <<'EOF'
import pathlib, re
env = pathlib.Path("env.yaml").read_text()
pins = re.findall(r'^      - ([a-zA-Z0-9_.\-]+==[0-9][^\s]*)$', env, re.M)
path = pathlib.Path("requirements.txt")
head = path.read_text().split("annotated-doc")[0]
path.write_text(head + "\n".join(pins) + "\n")
EOF
```

`results/revision_environment.json` records the OS, hardware and exact versions
under which the deposited numbers were produced.

`.github/workflows/ci.yml` runs the data-free surface (type check, lint, fast
tests, CLI reachability) on every push. Anything calling `build_all()` cannot run
in CI because the corpora are not redistributed; verify those locally.

## 6. Cost of a full reproduction

`build_all()` reconstructs all three cohorts in roughly 250 s and holds them in
memory (Nurse is the large one at 9,226 windows). Each `run_revision.sh` command
rebuilds independently, so running all of them takes about 40 minutes. Peak
memory stays well under 8 GB; no GPU is used.
