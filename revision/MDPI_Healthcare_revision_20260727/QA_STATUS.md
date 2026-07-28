# QA/QC Status

검증일: 2026-07-28 (2차 리비전 패스)

## 기준선과 내용

- 2026-06-30 제출본에서 시작: Pass
- 실제 저자·소속·교신저자·funding 보존: Pass
- review1–review3 자체 포함: Pass
- response letter와 action matrix 포함: Pass
- public repository URL: Pass — 원고 Data Availability에 삽입 완료
- archival DOI: Not run — Zenodo 저장소 토글이 OFF라 아직 발급 불가

## 코드와 환경

- `env.yaml` 기반 `easd` Conda 환경: Pass
- Python 3.11.13 핵심 import smoke: Pass
- shell syntax: Pass
- strict basedpyright surface: Pass, 0 error (신규 3개 모듈 포함 16개 파일)
- strict Ruff surface: Pass, 0 finding
- fast tests: Pass, 26 passed
- full tests: Pass, 28 passed
- Nurse 실자료 재구성: Pass
  - windows 9,226 / stress 7,347 / non-stress 1,879 / subjects 15
- Nurse 입력 제외 익명 집계: Pass (no covering session 42, signal loading 1)
- 실제 EDA backend 추적: Pass — cvxEDA(neurokit2) 232, fallback failure 0
- 권위 결과 로그 원자 저장: Pass

## 2차 패스 회귀 검증

- 빌더에 가속도 추출을 추가한 뒤 정본 분석 재실행: Pass
  - `run_crosscorpus.txt`의 모든 코호트 수, LOSO, LODO, movement tertile,
    iso-HR, subject-bootstrap 값이 이전 로그와 **바이트 단위 동일**
    (차이는 실행 시간 표기뿐)
- 신규 활동 특징의 `acc_mag_std`가 기존 movement 정의(`np.std`)와 동일함을
  단위 테스트로 고정: Pass
- 적응 기법 무이동(no-shift) 항등성 불변식 테스트: Pass
- TCA 고정 seed 재현성 테스트: Pass
- 신규 결과 artifact PII 검사(`W_`/`P_`/`N_` 접두사 부재): Pass

## 수치와 그림

- 신규 4개 분석 실행 및 artifact 저장: Pass
  - `revision_adaptation.json`, `revision_context_aware.json`,
    `revision_error_stratification.json`, `revision_target_supervised.json`
- 원고 수치 대 artifact 독립 교차검증: Pass
  - 표 tab:adapt 93개 값, tab:context 21개 값, tab:errstrata 45개 값,
    기존 표 전체를 스크립트로 대조 — 반올림 오류 0건
- 그림 F1–F6, F8–F10 생성 및 참조: Pass
- F7(HRV window sensitivity)는 tab:hrv_sens와 완전 중복이라 삭제: Pass

## 원고

- LaTeX build: Pass
- undefined reference/citation: Pass, 최종 pass 기준 0건
- overfull hbox: Pass, 0건
- fatal error: Pass
- page count: 27 pages
- 표 병합(R1-11): Pass — `tab:clinicalutility` + `tab:futurevalidation` →
  `tab:validationmap`, 정보 손실 없음, dangling ref 0건
- 로컬 절대경로 포함 가능 LaTeX 보조 파일 정리: Pass
- TeX LSP: Not run, `texlab` 미설치

## 공개 릴리스 준비

- `.gitignore`: Pass — data/(22 GB), .omo/, __pycache__, peer-review 문서 제외
- `git add -A` 후 실제 스테이징: 102 파일 / 5.8 MB, raw data 0건
- `LICENSE`(MIT + CC BY 4.0 + MDPI 템플릿 carve-out): Pass
- `CITATION.cff`, `.zenodo.json`: Pass
- `data/README.md`(레이아웃 + 4개 코퍼스 라이선스 의무): Pass
- README 영문 릴리스 섹션: Pass
- 참여자 식별자 노출 1건(구 `results_isohr_full.md`의 S02): 익명화 후
  해당 legacy 원장 자체를 삭제: Pass
- 실제 remote 생성/push: Pass — https://github.com/RURUGURU/isohr-wearable-stress
- GitHub 라이선스 자동 인식: Pass — MIT (LICENSE를 순수 MIT로 분리, 범위는 NOTICE)
- GitHub Actions CI(데이터 불요 검사): Pass
- 개인 이메일 제거: Pass — 제1저자 개인 주소 삭제, 교신저자 기관 주소 유지
- Data Availability에 저장소 URL 삽입: Pass — archival DOI만 미정
- Zenodo webhook: Not run — 저장소 토글이 아직 OFF라 release해도 아카이빙되지 않음

## 잔여 위험

- 초록 236단어로 MDPI 권장 200단어를 초과합니다. 추가 분석 서술을 줄이지
  않으면 더 줄이기 어려우므로 편집 단계에서 조정이 필요할 수 있습니다.
- Nurse 대상 subspace alignment가 단일 특징을 유의하게 앞섭니다(+0.020).
  성분 수를 target 성능으로 고른 상한이며 구간이 chance를 포함한다는 점을
  본문·응답서에 명시했지만, 리뷰어가 추가 질문할 수 있습니다.
- 활동 맥락 특징은 corpus별로 라벨과의 연관 방향이 달라 전이를 낮춥니다.
  이는 결과로 보고했으나 인과 해석으로 읽히지 않도록 문구를 유지해야 합니다.
- 그림 파일명(F1–F10)과 렌더링된 Figure 번호가 더 이상 일치하지 않습니다.
  MDPI 제작 단계에서 Figure1.png 형식 요구 시 재명명이 필요합니다.
- 300초 WESAD iso-HR 표본은 26개 창으로 감소해 해석하지 않습니다.
- Nurse label 품질과 15명 표본은 현장 일반화 결론을 제한합니다.
- archival DOI는 Zenodo 저장소 토글을 켠 뒤 release를 발행해야 발급됩니다.
- MDPI가 전 저자 이메일을 요구하면 제1저자 기관 주소를 추가해야 합니다.
- portal template 최신 여부는 제출 시점에 확인이 필요합니다.
