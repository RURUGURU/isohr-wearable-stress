#!/usr/bin/env python3
# pyright: standard
"""WESAD·Stress-Predict·Nurse 공통의 신호 처리와 수치 연산 모듈.

# noqa: SIZE_OK
크기 근거: 논문 수치와 결합된 프로토콜이므로 공통 특징·추론 경로의 회귀
검증이 마련되기 전에는 편의상 여러 파일로 쪼개지 않는다.

[AUDIT][DATA] 데이터 누수 불변조건:
  - 학습/평가 분할은 피험자 단위 LOSO만 사용하며 한 피험자는 양쪽에 동시에
    나타나지 않는다(``assert_no_subject_overlap``).
  - iso-HR 매칭은 HR과 라벨을 이용해 피험자별 HR 구간의 두 클래스 수를
    맞추는 연구 설계용 부분표본 추출이다. 모델 적합에는 관여하지 않는다.
      * ``global``: 전체 자료에서 한 번 매칭한 뒤 매칭 집합으로 LOSO한다.
      * ``strict``: 각 fold 내부에서 학습 피험자와 평가 피험자를 독립적으로
        매칭한다. 모델은 매칭된 학습 자료에만 적합되고 평가 라벨이나 창은
        적합 과정에 들어가지 않는다.
  - HistGradientBoostingClassifier는 NaN을 직접 처리하므로 대치·스케일링에서
    학습 통계가 평가 자료로 전파되지 않는다.
  - EDA 분해는 각 세션 자체 신호만 사용하고, 창 특징은 해당 창 내부 표본만
    사용하므로 세션·피험자 간 정보나 미래 표본을 참조하지 않는다.
  - 50% 중첩 창은 같은 피험자 안에서만 표본을 공유한다. LOSO가 피험자 전체를
    한쪽에만 두므로 train/test 간 중첩 누수는 없다. 신뢰구간도 창이 아니라
    피험자 cluster를 재표집한다.
"""
import csv, math, os, re
from collections import Counter
from datetime import datetime
import numpy as np

from scripts.utils_config import SETTINGS

try:
    import neurokit2 as nk
    _HAS_NK = True
except ImportError:
    _HAS_NK = False
try:
    from scipy.signal import butter, filtfilt, find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

HistGradientBoostingClassifier = None
LogisticRegression = None
StandardScaler = None
SimpleImputer = None
Pipeline = None
roc_auc_score = None
_SKLEARN_IMPORT_ERROR = None
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    _SKLEARN_IMPORT_ERROR = exc

WIN = SETTINGS.primary_window_seconds
STEP = SETTINGS.primary_step_seconds
SEED = SETTINGS.random_seed

# [AUDIT][REPRO] 분석 프로세스에서 실제 선택된 EDA backend와 제외된 원시 행을
# 집계한다. 참여자·세션·경로는 기록하지 않아 재현성과 개인정보 보호를 함께 지킨다.
_EDA_BACKEND_COUNTS: Counter[str] = Counter()
_EDA_BACKEND_FAILURE_COUNTS: Counter[str] = Counter()
_INPUT_PARSE_EXCLUSIONS: Counter[str] = Counter()

# ---- 특징 스키마: 순서를 바꾸면 모든 저장 결과와 모델 입력이 달라진다. ----
FEATS = [
    "mean_hr", "mean_ibi",                       # HR 수준(비심박 특징에서는 제외)
    "hr_std", "hr_slope", "hr_recovery",         # HR 동역학
    "rmssd", "sdnn", "pnn50", "n_nn", "ibi_cov", "artifact_pct",  # HRV와 신뢰도
    "scl_mean", "scl_slope", "phasic_auc", "scr_count", "scr_amp_mean", "scr_amp_max",  # EDA
    "temp_mean", "temp_slope",                   # 피부 온도
    "hr_range", "hr_accel", "eda_decay",         # 회복 동역학
]
HRLEVEL = {"mean_hr", "mean_ibi"}
NONHR = [f for f in FEATS if f not in HRLEVEL]
IDX = {f: i for i, f in enumerate(FEATS)}

# ---- 활동(가속도) 맥락 특징 ----
# [AUDIT][DATA] FEATS와 분리해 유지한다. FEATS 순서를 바꾸면 iso-HR·HR-leakage를
# 포함한 모든 기존 수치를 재생성해야 하므로, 리뷰어가 요청한 context-aware 모델은
# 전이 특징 행렬에 이 블록을 옆으로 이어 붙이는 방식으로만 사용한다.
# 세 corpus 모두 Empatica 규격 32 Hz 3축 손목 ACC이며 1/64 g LSB이므로
# ``load_signal(..., ncol=3)``가 반환하는 g 단위 크기 신호에 동일하게 적용된다.
ACC_FS = 32.0
ACC_FEATS = ["acc_mag_mean", "acc_mag_std", "acc_enmo", "acc_jerk"]
ACC_IDX = {f: i for i, f in enumerate(ACC_FEATS)}


def activity_features(mag_w, fs=ACC_FS):
    """창 하나의 g 단위 가속도 크기에서 활동 맥락 특징 4개를 계산한다.

    ``acc_mag_std``는 기존 Nurse movement tertile이 쓰던 ``np.std`` 값과 정확히
    같은 정의를 유지해 이미 발표한 층화 결과가 바뀌지 않게 한다.
    """
    mag_w = np.asarray(mag_w, float)
    if len(mag_w) < 2:
        return np.full(len(ACC_FEATS), np.nan)
    enmo = float(np.mean(np.clip(mag_w - 1.0, 0.0, None)))   # 중력 1 g를 뺀 활동량
    jerk = float(np.mean(np.abs(np.diff(mag_w))) * fs)       # 초당 크기 변화량
    return np.array([float(np.mean(mag_w)), float(np.std(mag_w)), enmo, jerk], float)


def require_dependency(package_name, import_error, install_command):
    if import_error is None:
        return
    raise ModuleNotFoundError(
        f"Missing optional dependency '{package_name}' in the active Python environment. "
        f"Install it before running this experiment, for example: {install_command}"
    ) from import_error


def require_sklearn():
    require_dependency("scikit-learn", _SKLEARN_IMPORT_ERROR, "python3 -m pip install scikit-learn")


# ================= 신호 로딩과 창 특징 계산 =================
def _epoch(s):
    s = str(s).strip()
    try:
        return float(s)                      # Empatica 원자료: Unix epoch
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):  # PhysioNet: 날짜·시각 문자열
            try:
                return datetime.strptime(s, fmt).timestamp()
            except ValueError:
                continue
    raise ValueError("bad timestamp: %r" % s)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def load_signal(path, ncol=1):
    rs = _rows(path)
    start = _epoch(rs[0][0])
    fs = float(rs[1][0])
    out = []
    for r in rs[2:]:
        try:
            if ncol == 3:
                out.append(math.sqrt(float(r[0])**2 + float(r[1])**2 + float(r[2])**2) / 64.0)
            else:
                out.append(float(r[0]))
        except (ValueError, IndexError):
            _INPUT_PARSE_EXCLUSIONS["signal_rows"] += 1
    return start, fs, np.asarray(out, float)


def load_ibi(path):
    rs = _rows(path)
    start = _epoch(rs[0][0])
    out = []
    for r in rs[1:]:
        try:
            out.append((float(r[0]), float(r[1])))
        except (ValueError, IndexError):
            _INPUT_PARSE_EXCLUSIONS["ibi_rows"] += 1
    return start, (np.asarray(out, float) if out else np.empty((0, 2)))


def cvx_decompose(eda, fs):
    """세션별 tonic/phasic EDA를 분해하고 사용한 backend 순서를 보존한다.

    NeuroKit2 cvxEDA가 실패하면 Butterworth, 그마저 실패하면 이동평균을
    사용한다. backend 변경은 특징 수치를 바꾸므로 결과 로그에 함께 기록한다.
    """
    eda = np.asarray(eda, float)
    if len(eda) < int(8 * fs):
        _EDA_BACKEND_COUNTS["identity-short-signal"] += 1
        return eda.copy(), np.zeros_like(eda)
    if _HAS_NK:
        try:
            df = nk.eda_phasic(nk.signal_sanitize(eda), sampling_rate=int(fs), method="cvxeda")
            _EDA_BACKEND_COUNTS["cvxEDA(neurokit2)"] += 1
            return df["EDA_Tonic"].to_numpy(), df["EDA_Phasic"].to_numpy()
        except (ArithmeticError, RuntimeError, ValueError):
            _EDA_BACKEND_FAILURE_COUNTS["cvxEDA(neurokit2)"] += 1
    if _HAS_SCIPY:
        try:
            b, a = butter(2, 0.05 / (fs / 2), btype="high")
            phasic = filtfilt(b, a, eda)
            _EDA_BACKEND_COUNTS["butter-highpass"] += 1
            return eda - phasic, phasic
        except (ArithmeticError, RuntimeError, ValueError):
            _EDA_BACKEND_FAILURE_COUNTS["butter-highpass"] += 1
    k = max(1, int(4 * fs))
    tonic = np.convolve(eda, np.ones(k) / k, mode="same")
    _EDA_BACKEND_COUNTS["moving-average"] += 1
    return tonic, eda - tonic


def _slope(y):
    """1차 추세 기울기를 **표본당** 변화량으로 반환한다.

    회귀 x축이 표본 index이므로 단위는 signal-unit/sample이며 초당 값이 아니다.
    corpus마다 같은 신호의 표집률이 같으므로 corpus 간 비교에는 문제가 없지만,
    절대 단위로 해석하거나 표집률이 다른 신호끼리 비교해서는 안 된다.
    """
    if len(y) < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def hrv_from_ibi(ibi_d):
    """Malik 규칙으로 이상 박동을 제외하고 HRV와 측정 신뢰도 지표를 계산한다."""
    f = dict(rmssd=np.nan, sdnn=np.nan, pnn50=np.nan, n_nn=0.0,
             ibi_cov=0.0, artifact_pct=np.nan, mean_ibi=np.nan)
    if len(ibi_d) == 0:
        return f
    ms = np.asarray(ibi_d, float) * 1000.0
    f["ibi_cov"] = float(np.sum(ibi_d) / WIN)
    if len(ms) < 3:
        f["n_nn"] = float(len(ms))
        f["mean_ibi"] = float(np.mean(ms))
        return f
    # [QC][METRIC] 연속 NN 차이가 이전 NN의 20%를 넘으면 Malik 이상 박동으로 제외한다.
    flag = np.abs(np.diff(ms)) > 0.2 * ms[:-1]
    f["artifact_pct"] = float(np.mean(flag))
    keep = np.concatenate(([True], ~flag))
    nn = ms[keep]
    f["n_nn"] = float(len(nn))
    f["mean_ibi"] = float(np.mean(nn))
    if len(nn) >= 3:
        d = np.diff(nn)
        f["rmssd"] = float(np.sqrt(np.mean(d**2)))
        f["sdnn"] = float(np.std(nn))
        f["pnn50"] = float(np.mean(np.abs(d) > 50))
    return f


def window_features(hr_w, tonic_w, phasic_w, temp_w, ibi_d):
    f = {}
    f["mean_hr"] = float(np.mean(hr_w)) if len(hr_w) else np.nan
    f["hr_std"] = float(np.std(hr_w)) if len(hr_w) > 1 else np.nan
    sl = _slope(hr_w)
    f["hr_slope"] = sl
    f["hr_recovery"] = float(max(0.0, -sl)) if sl == sl else np.nan   # 감소하는 HR만 회복량
    h = hrv_from_ibi(ibi_d)
    f.update({k: h[k] for k in ("rmssd", "sdnn", "pnn50", "n_nn", "ibi_cov",
                                "artifact_pct", "mean_ibi")})
    if len(tonic_w) >= 8:
        f["scl_mean"] = float(np.mean(tonic_w))
        f["scl_slope"] = _slope(tonic_w)
        pos = np.clip(phasic_w, 0, None)
        f["phasic_auc"] = float(np.mean(pos))
        if len(phasic_w) > 2:
            pk = (phasic_w[1:-1] > phasic_w[:-2]) & (phasic_w[1:-1] > phasic_w[2:]) & (phasic_w[1:-1] > 0.01)
            amps = phasic_w[1:-1][pk]
            f["scr_count"] = int(np.sum(pk))
            f["scr_amp_mean"] = float(np.mean(amps)) if amps.size else 0.0
            f["scr_amp_max"] = float(np.max(amps)) if amps.size else 0.0
        else:
            f["scr_count"] = 0; f["scr_amp_mean"] = 0.0; f["scr_amp_max"] = 0.0
    else:
        for k in ("scl_mean", "scl_slope", "phasic_auc", "scr_amp_mean", "scr_amp_max"):
            f[k] = np.nan
        f["scr_count"] = 0
    f["temp_mean"] = float(np.mean(temp_w)) if len(temp_w) >= 8 else np.nan
    f["temp_slope"] = _slope(temp_w) if len(temp_w) >= 8 else np.nan
    # 리뷰어가 요구한 회복 동역학은 같은 창 내부 값으로만 계산한다.
    f["hr_range"] = float(np.max(hr_w) - np.min(hr_w)) if len(hr_w) > 1 else np.nan
    if len(hr_w) >= 5:
        f["hr_accel"] = float(np.polyfit(np.arange(len(hr_w)), hr_w, 2)[0])  # 2차 곡률
    else:
        f["hr_accel"] = np.nan
    f["eda_decay"] = _slope(phasic_w) if len(phasic_w) >= 8 else np.nan
    return np.array([f[k] for k in FEATS], float)


# ================= 피험자 내부 iso-HR 매칭 =================
def match_indices(
    pool,
    hr,
    y,
    subj,
    seed=SEED,
    bin_w=SETTINGS.iso_hr_bin_width_bpm,
):
    """각 피험자의 HR 구간 안에서 두 라벨 수가 같은 부분표본 index를 반환한다."""
    rng = np.random.default_rng(seed)
    keep = []
    pool = np.asarray(pool)
    for s in np.unique(subj[pool]):
        sp = pool[subj[pool] == s]
        b = (hr[sp] // bin_w).astype(int)
        for bid in np.unique(b):
            pos = sp[(b == bid) & (y[sp] == 1)]
            neg = sp[(b == bid) & (y[sp] == 0)]
            k = min(len(pos), len(neg))
            if k:
                keep += list(rng.choice(pos, k, replace=False))
                keep += list(rng.choice(neg, k, replace=False))
    return np.array(sorted(keep), int)


# ================= 피험자 단위 leave-one-subject-out 평가 =================
def assert_no_subject_overlap(subj, tr, te):
    assert not (set(subj[tr]) & set(subj[te])), "SUBJECT LEAKAGE!"


def _make_est(est):
    """GBM 또는 fold별 학습 자료에만 적합한 대치·표준화 포함 LR을 만든다."""
    require_sklearn()
    if est == "lr":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    return HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                          learning_rate=0.1, max_iter=200)


def loso(X, y, subj, cols, eval_idx, mode="none", est="gbm", seed=SEED):
    """LOSO의 평가 라벨·피험자·확률·원본 index를 매칭 모드에 따라 반환한다.

    ``none``은 비매칭, ``global``은 전체에서 한 번 매칭, ``strict``는 fold
    내부에서 학습과 평가를 독립 매칭한다. 누수에 민감한 주장은 ``strict``로
    확인해야 하며 어느 모드에서도 평가 피험자는 학습 집합에 포함되지 않는다.
    """
    hr = X[:, IDX["mean_hr"]]
    eval_idx = np.asarray(eval_idx)
    if mode == "global":
        M = match_indices(np.arange(len(y)), hr, y, subj, seed)
        eval_idx = eval_idx[np.isin(eval_idx, M)]
    probs = {}
    for s in np.unique(subj[eval_idx]):
        te = eval_idx[subj[eval_idx] == s]
        if mode == "strict":
            te = match_indices(te, hr, y, subj, seed)
            tr = match_indices(np.where(subj != s)[0], hr, y, subj, seed)
        elif mode == "global":
            tr = M[subj[M] != s]
        else:
            tr = np.where(subj != s)[0]
        assert_no_subject_overlap(subj, tr, te)
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        clf = _make_est(est)
        clf.fit(X[np.ix_(tr, cols)], y[tr])
        p = clf.predict_proba(X[np.ix_(te, cols)])[:, 1]
        for j, ix in enumerate(te):
            probs[ix] = p[j]
    idx = np.array(sorted(probs), int)
    return y[idx], subj[idx], np.array([probs[i] for i in idx]), idx


# ================= 피험자 구조를 보존하는 평가와 추론 =================
def auroc(y, p):
    require_sklearn()
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def per_subject_auroc(y, subj, p):
    """피험자별 AUROC의 macro 평균·중앙값·평가 가능 피험자 수를 반환한다.

    pooled AUROC가 피험자 사이 calibration offset 때문에 과대평가되는 위험을
    분리하기 위해 두 클래스가 모두 있는 피험자만 각각 평가한다.
    """
    vals = []
    for s in np.unique(subj):
        m = subj == s
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], p[m]))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.median(vals)), len(vals)


def boot_ci(y, subj, p, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    us = np.unique(subj)
    vals = []
    for _ in range(B):
        smp = rng.choice(us, len(us), replace=True)
        yy = np.concatenate([y[subj == s] for s in smp])
        pp = np.concatenate([p[subj == s] for s in smp])
        if len(np.unique(yy)) == 2:
            vals.append(roc_auc_score(yy, pp))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(y, p, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, p)
    c = sum(roc_auc_score(rng.permutation(y), p) >= obs for _ in range(B))
    return (c + 1) / (B + 1)


def per_bin_auroc(y, p, hr, bin_w=10, min_n=12):
    out = []
    b = (hr // bin_w * bin_w).astype(int)
    for v in sorted(np.unique(b)):
        m = b == v
        if np.sum(m) >= min_n and len(np.unique(y[m])) == 2:
            out.append((int(v), int(np.sum(m)), float(roc_auc_score(y[m], p[m]))))
    return out


def loso_importance(X, y, subj, cols, eval_idx, feat_names, match=True, seed=SEED):
    """평가 fold의 한 특징만 섞었을 때 AUROC 감소량으로 중요도를 계산한다."""
    rng = np.random.default_rng(seed)
    hr = X[:, IDX["mean_hr"]]
    if match:
        M = match_indices(np.arange(len(y)), hr, y, subj, seed)
        eval_idx = np.asarray(eval_idx)[np.isin(eval_idx, M)]
    else:
        M = np.arange(len(y))
    base = {}
    perm = {c: {} for c in cols}
    for s in np.unique(subj[eval_idx]):
        te = eval_idx[subj[eval_idx] == s]
        tr = M[subj[M] != s]
        if len(np.unique(y[tr])) < 2:
            continue
        clf = _make_est("gbm")
        clf.fit(X[np.ix_(tr, cols)], y[tr])
        Xte = X[np.ix_(te, cols)]
        bp = clf.predict_proba(Xte)[:, 1]
        for j, ix in enumerate(te):
            base[ix] = bp[j]
        for ci, c in enumerate(cols):
            Xp = Xte.copy()
            Xp[:, ci] = rng.permutation(Xp[:, ci])
            pp = clf.predict_proba(Xp)[:, 1]
            for j, ix in enumerate(te):
                perm[c][ix] = pp[j]
    idx = np.array(sorted(base), int)
    yv = y[idx]
    a0 = auroc(yv, np.array([base[i] for i in idx]))
    imp = []
    for c in cols:
        ap = auroc(yv, np.array([perm[c][i] for i in idx]))
        imp.append((feat_names[cols.index(c)], a0 - ap))
    imp.sort(key=lambda t: -t[1])
    return a0, imp


def matched_eval(X, y, subj, eval_idx, seed=SEED):
    """기술 통계가 공유하는 global iso-HR 평가 부분집합을 반환한다."""
    hr = X[:, IDX["mean_hr"]]
    M = match_indices(np.arange(len(y)), hr, y, subj, seed)
    return np.asarray(eval_idx)[np.isin(eval_idx, M)]


def univariate_auroc(X, y, subj, cols, eval_idx, feat_names, mode="global", seed=SEED):
    """매칭 집합에서 원 특징값 하나의 방향 비의존 분리력을 기술한다."""
    ei = matched_eval(X, y, subj, eval_idx, seed) if mode == "global" else np.asarray(eval_idx)
    yv = y[ei]
    res = []
    for j, c in enumerate(cols):
        v = X[ei, c]
        ok = ~np.isnan(v)
        if len(np.unique(yv[ok])) == 2:
            a = roc_auc_score(yv[ok], v[ok])
            res.append((feat_names[j], max(a, 1 - a)))
    res.sort(key=lambda t: -t[1])
    return res


def holm(pvals):
    """입력 순서를 보존한 Holm--Bonferroni 보정 p-value를 반환한다."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, a)
        adj[i] = prev
    return adj


def cluster_perm_p(X, y, subj, cols, eval_idx, mode="global", est="gbm", B=99, seed=SEED):
    """피험자 안에서만 라벨을 섞고 전체 LOSO·매칭을 다시 수행한다.

    피험자 구조와 클래스 수를 보존하므로 전체 라벨을 한꺼번에 섞는 검정보다
    엄격하며, 관측 AUROC 이상인 순열 비율로 단측 p-value를 구한다.
    """
    rng = np.random.default_rng(seed)
    y0, _, p0, _ = loso(X, y, subj, cols, eval_idx, mode, est, seed)
    obs = auroc(y0, p0)
    cnt = 0
    for _ in range(B):
        yp = y.copy()
        for s in np.unique(subj):
            m = np.where(subj == s)[0]
            yp[m] = rng.permutation(y[m])
        yb, _, pb, _ = loso(X, yp, subj, cols, eval_idx, mode, est, seed)
        if auroc(yb, pb) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


def compare_models(X, y, subj, cols, eval_idx, feat_names, seed=SEED):
    """같은 iso-HR 집합에서 단일 특징·LR·GBM을 동일 LOSO 조건으로 비교한다."""
    out = {}
    for est in ("lr", "gbm"):
        ye, se, pe, _ = loso(X, y, subj, cols, eval_idx, mode="global", est=est, seed=seed)
        m, md, n = per_subject_auroc(ye, se, pe)
        out[est] = dict(pooled=auroc(ye, pe), ps_mean=m, ps_med=md, n=n)
    uni = univariate_auroc(X, y, subj, cols, eval_idx, feat_names, mode="global", seed=seed)
    out["best_single"] = dict(feat=uni[0][0], auroc=uni[0][1]) if uni else None
    out["uni_top5"] = uni[:5]
    return out


def reset_analysis_audit():
    """한 실행의 backend·입력 제외 집계를 다른 실행과 섞지 않도록 초기화한다."""
    _EDA_BACKEND_COUNTS.clear()
    _EDA_BACKEND_FAILURE_COUNTS.clear()
    _INPUT_PARSE_EXCLUSIONS.clear()


def analysis_audit_snapshot() -> dict[str, object]:
    """식별자를 제외한 실행 시점 backend·파싱 감사값의 불변 복사본을 반환한다."""
    return {
        "schema_version": 1,
        "eda_backend_counts": dict(sorted(_EDA_BACKEND_COUNTS.items())),
        "eda_backend_failure_counts": dict(
            sorted(_EDA_BACKEND_FAILURE_COUNTS.items())
        ),
        "input_parse_exclusion_counts": dict(
            sorted(_INPUT_PARSE_EXCLUSIONS.items())
        ),
    }


def eda_backend():
    """설치 가능성이 아니라 이 프로세스에서 실제 사용한 backend별 횟수를 반환한다."""
    if not _EDA_BACKEND_COUNTS:
        return "not-run"
    return ", ".join(
        f"{backend}:{count}"
        for backend, count in sorted(_EDA_BACKEND_COUNTS.items())
    )


def eda_backend_failures():
    """fallback을 일으킨 backend 실패 횟수를 참여자 식별자 없이 반환한다."""
    if not _EDA_BACKEND_FAILURE_COUNTS:
        return "none"
    return ", ".join(
        f"{backend}:{count}"
        for backend, count in sorted(_EDA_BACKEND_FAILURE_COUNTS.items())
    )


def input_parse_exclusions():
    """원시 신호 파싱에서 제외한 행 수를 파일·참여자 식별자 없이 반환한다."""
    if not _INPUT_PARSE_EXCLUSIONS:
        return "none"
    return ", ".join(
        f"{reason}:{count}"
        for reason, count in sorted(_INPUT_PARSE_EXCLUSIONS.items())
    )
