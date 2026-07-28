# 손목 웨어러블 스트레스 탐지의 iso-HR·교차코퍼스 평가

MDPI *Healthcare* 투고 논문 *When Wellness Stress Detection Enters Healthcare
Workflows: An Iso-Heart-Rate, Cross-Corpus Evaluation of Wrist Wearables*의
재현 가능한 분석 코드입니다.

손목 웨어러블의 이진 stress/non-stress 판별이 심박수 각성이 아니라 스트레스
특이적 생리를 반영하는지, 그리고 통제된 프로토콜(WESAD, Stress-Predict)에서
실제 간호 현장(Nurse)으로 옮겼을 때 유지되는지를 평가합니다. 새 분류기를
제안하는 연구가 아니라 측정·진단 연구입니다.

영문 문서가 기본이며 이 문서는 요약본입니다: [README.md](../README.md)

## 먼저 읽을 것

| 목적 | 문서 |
|---|---|
| 코드 구조와 데이터 흐름 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| iso-HR, LOSO/LODO, 전이 특징 집합 개념 | [GLOSSARY.md](GLOSSARY.md) |
| 처음 들어온 사람이 빠지는 함정 | [GOTCHAS.md](GOTCHAS.md) |
| 데이터셋 배치와 라이선스 | [../data/README.md](../data/README.md) |

## 실행

원자료 없이 동작하는 명령:

```bash
bash scripts/setup_env.sh          # env.yaml로 easd conda 환경 생성
bash scripts/run_quality.sh        # 엄격 타입·lint 게이트
bash scripts/run_tests.sh fast     # 원자료가 필요 없는 테스트
bash scripts/run_revision.sh --help
```

원자료가 필요한 명령:

```bash
bash scripts/run_crosscorpus.sh    # 정본 분석 (약 4분)
bash scripts/run_statistics.sh     # 순열검정·7-seed iso-HR·paired bootstrap
bash scripts/run_tests.sh full     # 실제 Nurse 재구성 포함
bash scripts/run_figures.sh        # 로그에서 F1-F3 재생성
bash scripts/build_manuscript.sh   # PDF 빌드
```

경로는 저장소 루트에서 해석하며 `EASD_PROJECT_ROOT`로, conda 환경 이름은
`EASD_ENV_NAME`으로 바꿀 수 있습니다.

리뷰어 요청 분석 목록은 영문 README의 명령 표를 참고하세요. 목록을 두 곳에
두면 반드시 어긋나므로 여기에는 복제하지 않습니다.

## 해석 경계

논문 주장의 한계선입니다. 전체 목록은 영문 README가 단일 출처이며, 핵심만
옮기면 다음과 같습니다.

- WESAD 교차코퍼스 전이는 명확히 우연 이상, Stress-Predict는 marginal,
  Nurse의 피험자 bootstrap 구간은 우연을 포함합니다.
- Nurse의 floor effect와 sparse·불균형·회고적 라벨 때문에 생리학적 비전이와
  ground-truth 한계를 분리할 수 없습니다.
- iso-HR는 주로 WESAD에서 진단적입니다. 모든 특징이 우연 근처인 corpus에서
  HR-leakage 지수가 0에 가까운 것은 floor effect이지 혼입의 증거가 아닙니다.
- feature-shift Wasserstein 값과 오류 층화는 domain discrepancy·연관이며
  인과적 driver를 규명하지 않습니다.
- 평가한 적응 기법 네 가지는 모두 얕고 transductive합니다. adversarial
  표현학습은 평가하지 않았으므로 결과를 일반화하면 안 됩니다.
- 활동 맥락은 corpus 내부 성능을 올리고 전이를 낮춥니다. 이는 움직임과
  스트레스의 연관 방향이 프로토콜마다 뒤집히기 때문이며, 활동이 EDA 신호를
  훼손한다는 뜻이 아닙니다.
- 공개 저장소 URL과 archival DOI가 원고에 채워지기 전에는 공개 재현성이
  완료됐다고 쓰지 않습니다.

## 규약

코드를 수정하는 사람이 지켜야 할 규칙은 [../AGENTS.md](../AGENTS.md)에 있습니다.
프로토콜 상수는 `scripts/analysis_config.ini`에만, 패키지 버전은 `env.yaml`과
`results/revision_environment.json`에만 둡니다.

현재 원고는 `revision/MDPI_Healthcare_revision_20260727`입니다. 2026-06-30
제출본 스냅샷과 리뷰 correspondence는 저자 작업 사본에만 두고 공개하지
않습니다.

## 라이선스

`scripts/`·`docs/`·환경 매니페스트는 MIT([../LICENSE](../LICENSE)), 원고
본문과 그림은 CC BY 4.0, MDPI 템플릿은 MDPI 소유입니다. 정확한 범위와 데이터셋
이용 조건은 [../NOTICE](../NOTICE)를 보세요.

Zenodo 아카이브: https://doi.org/10.5281/zenodo.21638616 (concept DOI, 항상
최신 release를 가리킵니다).
