# Expected layout of `data/raw`

No dataset is redistributed in this repository. Obtain each corpus from its
original source under its own licence, then place it so that the paths below
resolve. `scripts/utils_paths.py` derives every dataset path from the
repository root (override with the `EASD_PROJECT_ROOT` environment variable).

```text
data/raw/
├── wesad/
│   └── WESAD/
│       ├── S2/S2.pkl
│       ├── S3/S3.pkl
│       └── ...                       # one directory per subject
├── stress_predict/
│   ├── Raw_data/
│   │   ├── S01/{ACC,EDA,HR,IBI,TEMP}.csv
│   │   └── ...                       # S01 .. S35
│   └── Processed_data/
│       └── Time_logs.xlsx
└── nurse/
    ├── SurveyResults.xlsx
    └── <participant>/<session>.zip   # extracted alongside as <session>/
```

All three corpora are required; every analysis in the manuscript uses all of
them.

## Sources and licences

| Corpus | Source | Licence / terms | Cite |
|---|---|---|---|
| WESAD | UCI Machine Learning Repository / University of Siegen | EULA: non-commercial research use, **redistribution prohibited** | Schmidt et al., ICMI 2018 |
| Stress-Predict | <https://github.com/italha-d/Stress-Predict-Dataset> | MIT (Copyright © 2022 Talha Iqbal) | Iqbal et al. 2022 |
| Nurse | Dryad data package | Verify the current terms on the Dryad landing page before redistribution | Hosseini et al., *Sci. Data* 2022, plus the Dryad DOI |

The Nurse corpus contains participant-level survey rows. None of it is copied
into any result artifact: input exclusions are reported only as anonymous aggregate counts
(see the `ANALYSIS_AUDIT_JSON` line in `results/run_crosscorpus.txt`).
