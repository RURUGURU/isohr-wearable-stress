# Working rules for this repository

**Updated:** 2026-07-28

이 문서는 코드를 수정하는 사람이 지켜야 할 규칙만 담습니다. 구조·개념·함정
설명은 중복을 피하기 위해 다른 문서가 단독으로 소유합니다.

- 코드 구조와 데이터 흐름: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 개념 정의(iso-HR, LOSO/LODO, 전이 특징 집합 등): [docs/GLOSSARY.md](docs/GLOSSARY.md)
- 알려진 함정: [docs/GOTCHAS.md](docs/GOTCHAS.md)
- 해석 경계(논문 주장 한계선): [README.md](README.md#interpretation-boundaries)

`git log`와 `git status`가 무엇이 바뀌었는지에 대한 유일한 권위입니다. 문서에
적힌 트리나 목록이 아니라 저장소 이력을 먼저 보세요.

## 프로젝트 성격

새 SOTA classifier 제안이 아니라 측정·진단 연구입니다. 성능을 올리는 변경보다
측정의 정직성을 지키는 변경이 항상 우선합니다.

## 권위 경로

- 원 제출 기준선(읽기 전용): `revision/MDPI_Healthcare_submission_20260630`
- 현재 리비전: `revision/MDPI_Healthcare_revision_20260727`
- 권위 원고와 PDF: 위 폴더의 `manuscript_healthcare_revision_20260727.{tex,pdf}`
- 결과 원장: `results/run_crosscorpus.txt`, `results/run_crosscorpus_stats.txt`,
  `results/REVISION_RESULTS.md`

기준선 폴더는 수정하지 않습니다. 후속 리뷰 수정은 날짜가 명시된 새 revision
폴더에서만 수행합니다.

## 수치 불변조건

바꾸려면 전체 재생성과 재검토가 필요한 값입니다.

- 기본 창 64초, step 32초, seed 42, iso-HR 구간 5 bpm, 절대 움직임 기준
  0.0796 g. 모두 `scripts/analysis_config.ini`에만 정의합니다.
- primary split은 피험자 단위 LOSO, 교차코퍼스는 leave-one-dataset-out.
- SCR count는 연속 점수이며 AUROC에 threshold가 필요하지 않습니다.
- Nurse 확률 모델의 운용 threshold는 target 라벨 tuning 없이 0.5 고정입니다.
- test 라벨로 below-chance AUROC 방향을 뒤집지 않습니다.
- `FEATS`, `IDX`, `HRLEVEL`, `NONHR` 순서를 바꾸면 모든 consumer와 결과를
  재생성하고 검토해야 합니다. 활동 특징을 `ACC_FEATS`로 분리해 둔 이유입니다.
- 코호트 수는 WESAD 1,133 / Stress-Predict 2,846 / Nurse 9,226 창,
  피험자 15 / 35 / 15명입니다. `test_nurse.py`가 Nurse를 고정합니다.

## 데이터 규칙

- `data/raw`는 immutable input입니다. derived feature·result·figure를 절대
  이 트리에 쓰지 않습니다.
- 피험자 단위로 split·bootstrap·deduplicate합니다.
- Nurse 609 session ZIP과 WESAD 피험자별 E4 ZIP은 builder 입력이므로 삭제하지
  않습니다. archive와 extracted tree를 서로 다른 관측으로 세지 않습니다.
- 설문·질문지·time log·signal의 행 단위 내용을 로그나 외부 artifact에
  복사하지 않습니다. 제외 사유는 익명 집계 건수로만 보고합니다.
- 공개 데이터라고 재배포 허가를 가정하지 않고 각 license를 따릅니다.

## 데이터셋 라벨

- WESAD: stress 2, amusement 3(제외), meditation 4.
- Stress-Predict: `Time_logs.xlsx`의 protocol 구간을 창 중심 시각으로
  라벨링합니다. 정렬을 바꾸면 shift sensitivity를 다시 실행합니다.
- Nurse: stress level 2 대 0만 사용하고 level 1은 제외합니다.

## EDA backend

순서는 `cvxEDA(neurokit2) -> SciPy Butterworth -> moving average`입니다.
backend가 바뀌면 특징 수치가 바뀌므로, 측정 실행마다 실제 선택 횟수, fallback
실패 횟수, malformed signal-row 제외 건수를 식별자 없이 기록합니다. 설치
가능한 backend를 실제 사용 backend처럼 보고하거나 malformed file을 조용히
제외하지 않습니다.

## 명명

실제 신경망 학습 loop가 없으므로 `model/train/val` 같은 이름을 인위적으로
만들지 않습니다. 새 파일은 실제 역할에 따라 `analysis_*`, `utils_*`, `metrics`,
`main_*`, `plot_*`, `test_*`로 이름 짓습니다.

## QA Gate

코드 변경 후:

- 엄격 표면 basedpyright 0 error, Ruff 0 finding (`scripts/run_quality.sh`)
- `scripts/run_tests.sh fast` 통과, 원자료가 있으면 `full`도 통과
- 수치 정의·split·threshold가 바뀌지 않았는지 확인
- builder를 건드렸다면 `scripts/run_crosscorpus.sh`를 다시 돌려 정본 로그와
  diff하고, 코호트 수와 주요 표가 동일한지 확인
- 개인 절대경로·secret·PII가 artifact에 없는지 확인

원고 변경 후:

- `latexmk -pdf -interaction=nonstopmode -halt-on-error`
- 최종 pass 기준 undefined reference/citation 0건, fatal error 0건
- page count와 모든 페이지 렌더링, figure clipping 확인
- 표·caption과 결과 로그 수치 일치 확인

테스트를 실행하지 못한 경우 Pass로 쓰지 않고 `Not run`과 이유를 남깁니다.

## 유지보수 표면 경계

`pyrightconfig.json`과 `run_quality.sh`가 다루는 파일만 엄격 검사 대상입니다.
`utils_analysis.py`, `analysis_crosscorpus.py`, `analysis_statistics.py`는
발표 수치에 고정되어 있어 의도적으로 제외했고, 대규모 타입·포맷 개조 대신 실제
결과·음성 대조군·integration 회귀로 관리합니다. 자세한 배경은
[docs/GOTCHAS.md](docs/GOTCHAS.md)에 있습니다.

## 제출 전 외부 blocker

- 실제 public versioned repository URL
- archival DOI
- portal 요구 시 최신 MDPI template 확인

URL/DOI를 입력한 뒤 권위 PDF를 다시 빌드하고 전체 시각 QA를 반복합니다.
