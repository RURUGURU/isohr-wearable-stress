# Iso-heart-rate, cross-corpus evaluation of wrist-wearable stress detection

Reproducible analysis code for the manuscript *When Wellness Stress Detection
Enters Healthcare Workflows: An Iso-Heart-Rate, Cross-Corpus Evaluation of Wrist
Wearables* (submitted to MDPI *Healthcare*).

The study asks whether binary stress-versus-non-stress performance from a wrist
wearable reflects stress-specific physiology rather than heart-rate arousal, and
whether it survives a move from controlled protocols (WESAD, Stress-Predict) into
a real nursing workflow (Nurse). It is an evaluation study, not a proposal for a
new classifier.

한국어 문서는 [docs/README.ko.md](docs/README.ko.md)를 참고하세요.

## Start here

| If you want to… | Read |
|---|---|
| understand how the code fits together | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| understand iso-HR, LOSO/LODO, the transfer feature set | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| avoid the traps a newcomer hits | [docs/GOTCHAS.md](docs/GOTCHAS.md) |
| obtain the datasets | [data/README.md](data/README.md) |
| see what was measured and why | `revision/MDPI_Healthcare_revision_20260727/results/REVISION_RESULTS.md` |

## Quick start

```bash
bash scripts/setup_env.sh          # create the pinned `easd` conda environment
bash scripts/run_quality.sh        # strict type + lint gate      (no data needed)
bash scripts/run_tests.sh fast     # tests that need no raw data  (no data needed)
bash scripts/run_revision.sh --help
```

Those four work on a fresh clone. Everything below needs the corpora in place —
see [data/README.md](data/README.md) for the expected layout and each dataset's
licence.

```bash
bash scripts/run_crosscorpus.sh    # the canonical analysis  (~4 min)
bash scripts/run_statistics.sh     # permutation, 7-seed iso-HR, paired bootstrap
bash scripts/run_tests.sh full     # adds the real-data Nurse reconstruction
bash scripts/run_figures.sh        # regenerate F1-F3 from the logs
bash scripts/build_manuscript.sh   # build the PDF
```

Paths resolve from the repository root; set `EASD_PROJECT_ROOT` to override and
`EASD_ENV_NAME` to use a differently named conda environment.

## Analyses

Every reviewer-requested analysis runs through one CLI. Each command rebuilds the
corpora independently (~4 min each); running all of them takes about 40 minutes.

| Command (`bash scripts/run_revision.sh …`) | Produces |
|---|---|
| `nurse-utility` | Nurse PR / calibration / FPR metrics, figure F4 |
| `sensitivity nonoverlap` | full protocol with non-overlapping windows |
| `sensitivity window300` | full protocol with 300 s windows |
| `hrv-window-sensitivity` | direct HRV-feature ablation at both window lengths |
| `stress-predict-alignment` | protocol-label shift sensitivity, figure F6 |
| `feature-shift` | pairwise Wasserstein feature shift, figure F5 |
| `negative-control` | within-subject label-scramble leakage control |
| `adaptation` | CORAL / subspace alignment / TCA / importance weighting, figure F8 |
| `target-supervised` | supervised target-adaptation curve |
| `context-aware` | accelerometer context-aware transfer model, figure F9 |
| `error-stratification` | LODO error stratification, figure F10 |

## Interpretation boundaries

These are the sentences that stop a reader — or a new lab member — from
overclaiming. They are maintained here and nowhere else.

- WESAD cross-corpus transfer is clearly above chance; Stress-Predict is
  marginal; the Nurse subject-bootstrap interval includes chance.
- Nurse floor effects and sparse, imbalanced, retrospective labels mean
  physiological non-transfer cannot be separated from ground-truth limitations.
- iso-HR evidence is diagnostic mainly in WESAD. A near-zero HR-leakage index in
  a corpus that is near chance everywhere is a floor effect, not evidence about
  confounding.
- Paired-bootstrap equivalence statements apply only to the comparisons actually
  tested, never to models in general.
- Feature-shift Wasserstein distances and error strata are domain discrepancies
  and associations. They do not identify a causal driver.
- The four evaluated adaptation methods are shallow and transductive. Adversarial
  representation learning was not evaluated and the result must not be
  generalized to it.
- Activity context raises within-corpus performance and lowers transfer, because
  the movement–stress association reverses across protocols. Do not read it as
  activity degrading the electrodermal signal.
- Do not claim public reproducibility is complete until the repository URL and
  archival DOI are filled into the manuscript.

## Repository layout

`scripts/` holds all code, `data/raw/` is an immutable third-party input that is
never redistributed, `revision/` holds the manuscript sources with their result
ledgers and figures, and `docs/` holds the documents linked above. The per-file
purpose table lives in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#2-what-each-file-does) so that it has
one owner rather than being duplicated in prose.

Protocol constants — window, step, seed, iso-HR bin width, movement gate — live
only in `scripts/analysis_config.ini`. Exact package versions live only in
`env.yaml` and `results/revision_environment.json`.

`revision/MDPI_Healthcare_revision_20260727` holds the current manuscript. The
frozen 2026-06-30 submission snapshot and the peer-review correspondence are kept
in the authors' working copy but are not published here.

## Data and licences

No corpus is redistributed. Obtain each from its original source under its own
terms; WESAD's EULA prohibits redistribution, and the Nurse corpus contains
participant-level survey rows. Full table in [data/README.md](data/README.md).
Result artifacts report input exclusions only as anonymous aggregate counts.

## Contributing

`AGENTS.md` holds the working rules for anyone editing the code: what may be
changed, which surfaces are strictly typed, and which numerical invariants must
be re-verified after a change.

## Licence and citation

Analysis code, documentation and the environment manifests are MIT
([LICENSE](LICENSE)). Manuscript text and figures are CC BY 4.0, and MDPI's
LaTeX template files remain MDPI's property — see [NOTICE](NOTICE) for the exact
scope and for the dataset terms.

Please cite both the software and the article — see [CITATION.cff](CITATION.cff).
