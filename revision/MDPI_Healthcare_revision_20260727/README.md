# MDPI Healthcare Major Revision, 2026-07-27

이 폴더는 2026-06-30 제출본을 기준으로 세 리뷰를 반영한 독립 메이저 리비전
패키지입니다.

## 기준과 수정 원칙

- 기준선: `../MDPI_Healthcare_submission_20260630`
- 리뷰 원문: `reviews/review1.txt`, `reviews/review2.txt`,
  `reviews/review3.txt`
- 수정 원칙: 기준 원고의 제목, 실제 저자·소속·교신저자·funding,
  Healthcare workflow 맥락, 표와 인용 구조를 보존하고 리뷰 근거를 기존
  Methods–Results–Discussion 흐름 안에 통합했습니다.
- 다른 임시 원고를 기준선으로 삼거나 PDF를 손으로 편집하지 않았습니다.

## 주요 파일

```text
.
├── manuscript_healthcare_revision_20260727.tex
├── manuscript_healthcare_revision_20260727.pdf
├── references.bib
├── Definitions/
├── figures/
│   ├── F1_crosscorpus.png
│   ├── F2_nurse_movement.png
│   ├── F3_isohr_decomp.png
│   ├── F4_nurse_utility.png
│   ├── F5_feature_shift.png
│   ├── F6_stress_predict_alignment.png
│   ├── F8_adaptation.png
│   ├── F9_context_aware.png
│   └── F10_error_strata.png
├── results/
├── reviews/
├── response_to_reviewers.md
├── major_revision_matrix.md
└── QA_STATUS.md
```

## 리뷰 반영 내용

- abstract에서 binary stress-vs-non-stress task를 명시
- Nurse 결론을 sparse·retrospective·imbalanced label 한계에 맞게 축소
- iso-HR 인과 해석을 WESAD 중심의 진단 근거로 제한
- SCR count가 within-fold importance에서 선택된 연속 score임을 명시
- subject-cluster bootstrap와 within-subject permutation 범위를 설명
- 64초 non-overlap과 300초 window sensitivity 추가
- 64초·300초 HRV 직접 ablation 추가
- Nurse AP, Brier, ECE, balanced accuracy, sensitivity, FPR 추가
- Stress-Predict label alignment sensitivity 추가
- corpus 간 robust Wasserstein feature shift 추가
- CORAL/GBM 결론을 시험한 모델에만 한정
- 단계별 iso-HR 알고리즘 추가

## 결과 provenance

- 본문 LODO/LOSO: `results/run_crosscorpus.txt`
- permutation/Holm/7-seed/paired bootstrap:
  `results/run_crosscorpus_stats.txt`
- reviewer 분석: `results/REVISION_RESULTS.md`와 `results/revision_*`

`run_crosscorpus.txt`의 `ANALYSIS_AUDIT_JSON`은 같은 실행에서 관찰한 EDA
backend, fallback, malformed 행, corpus 입력 제외의 익명 집계입니다.

F1–F3은 root `scripts/plot_figures.py`가 위 로그를 파싱해 생성합니다. 수치
배열을 그림 코드에 하드코딩하지 않습니다.

## 재현

저장소 root에서 실행합니다.

```bash
bash scripts/setup_env.sh
bash scripts/run_quality.sh
bash scripts/run_tests.sh full
bash scripts/run_figures.sh
bash scripts/build_manuscript.sh
```

리뷰 분석 명령:

```bash
bash scripts/run_revision.sh --help
```

## 해석 경계

- WESAD는 명확한 cross-corpus signal을 보입니다.
- Stress-Predict 효과는 작고 marginal합니다.
- Nurse interval은 chance를 포함합니다.
- 약한 corpus의 iso-HR floor effect로 stressor-type causality를 판단하지
  않습니다.
- PhysioNet stress와 exercise는 별도 세션입니다.
- 시험하지 않은 adaptation·multimodal 방법의 가능성을 배제하지 않습니다.

## 제출 전 필수 외부 작업

Data Availability의 `[repository URL; archival DOI]`를 실제 public versioned
repository와 archival DOI로 교체해야 합니다. 교체 후 PDF를 다시 빌드하고
모든 페이지를 재검사해야 합니다.
