#!/usr/bin/env python3
# pyright: standard
"""데이터셋 간 스트레스 신호 전이성과 단일 SCR 기준선을 비교한다.

# noqa: SIZE_OK
크기 근거: 세 corpus builder와 논문 수치가 결합된 프로토콜이므로 모든 builder의
회귀 검증이 마련되기 전에는 기계적으로 분할하지 않는다.

이진 stress(+) 대 non-stress(-)를 손목 E4로 평가한다. 데이터셋 간 전이에는
장치·주변 환경 차이를 줄이기 위해 절대 SCL/TEMP/HR 평균을 제외한 EDA phasic과
동역학 특징을 사용한다. corpus 내부 LOSO와 leave-one-dataset-out에서 고정
``scr_count``·LogReg·GBM·CORAL을 동일 자료로 비교한다.
"""
import csv, json, os, sys, glob, pickle, warnings, time
from collections import Counter
import numpy as np, pandas as pd  # noqa: E401  # noqa: PANDAS_OK
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts import utils_analysis as L
from scripts.utils_time import nurse_survey_epoch
from scripts.utils_paths import (
    DATA_RAW_DIR,
    NURSE_DIR,
    STRESS_PREDICT_PROCESSED_DIR,
    STRESS_PREDICT_RAW_DIR,
    WESAD_DIR,
    require_path,
)

nk = None
_NK_IMPORT_ERROR = None
roc_auc_score = None
_SKLEARN_METRICS_IMPORT_ERROR = None
try:
    import neurokit2 as nk
except ImportError as exc:
    _NK_IMPORT_ERROR = exc
try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    _SKLEARN_METRICS_IMPORT_ERROR = exc

ROOT = str(DATA_RAW_DIR)
TRANSFER = ["scr_count", "phasic_auc", "scr_amp_mean", "scr_amp_max",
            "scl_slope", "eda_decay", "temp_slope", "hr_slope"]
TCOLS = [L.IDX[f] for f in TRANSFER]
SCR = L.IDX["scr_count"]
_CORPUS_INPUT_EXCLUSIONS: Counter[str] = Counter()


def require_runtime_paths():
    require_path(DATA_RAW_DIR, "raw data root")
    require_path(WESAD_DIR, "WESAD dataset root")
    require_path(STRESS_PREDICT_RAW_DIR, "Stress-Predict raw-data root")
    require_path(STRESS_PREDICT_PROCESSED_DIR, "Stress-Predict processed-data root")
    require_path(NURSE_DIR, "Nurse dataset root")


def require_runtime_dependencies():
    L.require_dependency("neurokit2", _NK_IMPORT_ERROR, "python3 -m pip install neurokit2")
    if _SKLEARN_METRICS_IMPORT_ERROR is not None:
        L.require_dependency(
            "scikit-learn",
            _SKLEARN_METRICS_IMPORT_ERROR,
            "python3 -m pip install scikit-learn",
        )
    L.require_sklearn()


def _report_input_exclusions(corpus: str, exclusions: Counter[str]) -> None:
    for reason, count in exclusions.items():
        _CORPUS_INPUT_EXCLUSIONS[f"{corpus}.{reason}"] += count
    if exclusions:
        summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(exclusions.items())
        )
        print(f"{corpus} input exclusions: {summary}", file=sys.stderr)


# ---------- [DATA] WESAD: stress(2)=1, baseline(1)+meditation(4)=0 ----------
def build_wesad():
    rows, FB, FE, FL, FA = [], 64, 4, 700, 32
    exclusions: Counter[str] = Counter()
    for sd in sorted(glob.glob(ROOT + "/wesad/WESAD/S*")):
        pk = os.path.join(sd, os.path.basename(sd) + ".pkl")
        if not os.path.exists(pk):
            continue
        d = pickle.load(open(pk, "rb"), encoding="latin1"); w = d["signal"]["wrist"]
        bvp = np.asarray(w["BVP"], float).ravel(); eda = np.asarray(w["EDA"], float).ravel()
        temp = np.asarray(w["TEMP"], float).ravel(); lab = np.asarray(d["label"]).ravel()
        # [DATA] 손목 E4 ACC는 32 Hz 정수 1/64 g LSB이므로 CSV corpus와 동일하게 환산한다.
        # 흉부 RespiBAN ACC(700 Hz)는 부착 부위가 달라 corpus 간 비교에 쓰지 않는다.
        amag = np.sqrt((np.asarray(w["ACC"], float) ** 2).sum(axis=1)) / 64.0
        sid = "W_" + str(d.get("subject", os.path.basename(sd)))
        try:
            sig, _ = nk.ppg_process(bvp, sampling_rate=FB)
            rate = sig["PPG_Rate"].to_numpy(); peaks = np.where(sig["PPG_Peaks"].to_numpy() == 1)[0]
        except (ArithmeticError, RuntimeError, ValueError):
            exclusions["ppg_processing"] += 1
            continue
        tonic, phasic = L.cvx_decompose(eda, FE); nsec = len(lab) / FL; t = 0.0
        while t + L.WIN <= nsec:
            lw = lab[int(t * FL): int((t + L.WIN) * FL)]
            v, c = np.unique(lw, return_counts=True); cond = int(v[np.argmax(c)])
            if cond in (1, 2, 4) and c.max() / len(lw) > 0.9:
                hr_w = rate[int(t * FB): int((t + L.WIN) * FB)]
                pt = peaks / FB; pin = peaks[(pt >= t) & (pt < t + L.WIN)]
                ibi = np.diff(pin) / FB if len(pin) > 1 else np.empty(0)
                tw = slice(int(t * FE), int((t + L.WIN) * FE))
                fv = L.window_features(hr_w, tonic[tw], phasic[tw], temp[tw], ibi)
                if fv[L.IDX["mean_hr"]] == fv[L.IDX["mean_hr"]]:
                    aw = amag[int(t * FA): int((t + L.WIN) * FA)]
                    rows.append((sid, 1 if cond == 2 else 0, fv,
                                 {"activity": L.activity_features(aw)}))
            t += L.STEP
    _report_input_exclusions("WESAD", exclusions)
    return rows


# ---------- [DATA] Stress-Predict: Time_logs의 protocol 구간 라벨 ----------
# stress=Stroop+Interview, non-stress=Relax/Baseline이며 나머지 구간은 제외한다.
SP_PHASES = [(0, "Baseline/Questionniare", "Unnamed: 7"), (1, "Stroop Test", "Unnamed: 9"),
             (0, "Relax", "Unnamed: 11"), (1, "Interview", "Unnamed: 13"),
             (0, "Relax.1", "Unnamed: 15"), (-1, "Hyperventilation", "Unnamed: 17"),
             (0, "Relax.2", "Unnamed: 19"), (-1, "Questionniare", "Unnamed: 21"),
             (0, "Relax/Baseline", "Unnamed: 23")]


def _tosec(s):
    h, m, sec = [int(x) for x in str(s).strip().split(":")]
    return h * 3600 + m * 60 + sec


def build_sp(label_shift_seconds: float = 0.0):
    rows = []
    exclusions = Counter()
    tl = pd.read_excel(STRESS_PREDICT_PROCESSED_DIR / "Time_logs.xlsx").iloc[1:]
    tlmap = {str(r["S. ID."]).strip(): r for _, r in tl.iterrows()}
    for d in sorted(glob.glob(str(STRESS_PREDICT_RAW_DIR / "S*"))):
        sx = os.path.basename(d)
        if sx not in tlmap:
            continue
        r = tlmap[sx]
        try:
            raw = []
            for _, sc, ec in SP_PHASES:
                raw += [_tosec(r[sc]), _tosec(r[ec])]
        except (ValueError, AttributeError):
            exclusions["protocol_time"] += 1
            continue
        # [QC] 12시간제 AM/PM 전환 때문에 시각이 역행하면 12시간을 더해 단조 증가시킨다.
        adj, prev, add = [], -1, 0
        for v in raw:
            while v + add < prev:
                add += 12 * 3600
            adj.append(v + add); prev = v + add
        try:
            hr_s, _, hr = L.load_signal(d + "/HR.csv"); ed_s, ef, eda = L.load_signal(d + "/EDA.csv")
            tp_s, _, temp = L.load_signal(d + "/TEMP.csv"); ib_s, ibi = L.load_ibi(d + "/IBI.csv")
        except (OSError, UnicodeError, csv.Error, IndexError, ValueError):
            exclusions["signal_loading"] += 1
            continue
        # [QC] ACC는 맥락 특징 전용이다. 실패해도 창을 버리지 않고 NaN으로 남겨
        # 기존 코호트 수(2,846창/35명)를 그대로 유지한다.
        try:
            ac_s, _, acc = L.load_signal(d + "/ACC.csv", ncol=3)
        except (OSError, UnicodeError, csv.Error, IndexError, ValueError):
            exclusions["acc_loading"] += 1
            ac_s, acc = 0.0, np.empty(0)
        ref = max(hr_s, ed_s, tp_s)            # 세 신호가 모두 존재하는 실제 기록 시작점
        ivs = []                               # (라벨, epoch 시작, epoch 종료)
        for i, (lab, _, _) in enumerate(SP_PHASES):
            if lab == -1:
                continue
            ivs.append(
                (
                    lab,
                    ref + (adj[2 * i] - adj[0]) + label_shift_seconds,
                    ref + (adj[2 * i + 1] - adj[0]) + label_shift_seconds,
                )
            )
        tonic, phasic = L.cvx_decompose(eda, ef)
        end = min(hr_s + len(hr), ed_s + len(eda) / 4, tp_s + len(temp) / 4)
        t = ref
        while t + L.WIN <= end:
            tc = t + L.WIN / 2
            lab = next((lb for lb, a, b in ivs if a <= tc < b), None)
            if lab is not None:
                hw = hr[int(round(t - hr_s)): int(round(t - hr_s + L.WIN))]
                ew = slice(int(round((t - ed_s) * 4)), int(round((t - ed_s + L.WIN) * 4)))
                tw = temp[int(round((t - tp_s) * 4)): int(round((t - tp_s + L.WIN) * 4))]
                idd = ibi[(ibi[:, 0] >= t - ib_s) & (ibi[:, 0] < t - ib_s + L.WIN), 1] if len(ibi) else np.empty(0)
                if len(hw) >= 50:
                    fv = L.window_features(hw, tonic[ew], phasic[ew], tw, idd)
                    if fv[L.IDX["mean_hr"]] == fv[L.IDX["mean_hr"]]:
                        aw = acc[int(round((t - ac_s) * 32)): int(round((t - ac_s + L.WIN) * 32))]
                        rows.append(("P_" + sx, lab, fv,
                                     {"activity": L.activity_features(aw)}))
            t += L.STEP
    _report_input_exclusions("Stress-Predict", exclusions)
    return rows


# ---------- [DATA] Nurse: 병원 현장의 설문 기반 stress 라벨 ----------
# high-stress(level 2)=1, no-stress(level 0)=0이며 모호한 level 1은 제외한다.
def build_nurse():
    NUR = str(NURSE_DIR)
    exclusions = Counter()
    df = pd.read_excel(NUR + "/SurveyResults.xlsx"); df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).str.strip()
    df = df[pd.to_numeric(df["Stress level"], errors="coerce").notna()].copy()
    df["SL"] = df["Stress level"].astype(int)

    def toep(date, tt):
        return nurse_survey_epoch(pd.to_datetime(date).date(), tt)

    sess = {}
    for z in glob.glob(NUR + "/*/*.zip"):
        nid = os.path.basename(os.path.dirname(z)); ep = int(os.path.basename(z).split("_")[1][:-4])
        p = z[:-4]; eda = p + "/EDA.csv"; dur = 0
        if os.path.exists(eda):
            try:
                with open(eda) as f:
                    dur = (sum(1 for _ in f) - 2) / 4.0
            except (OSError, UnicodeError):
                exclusions["eda_duration"] += 1
        sess.setdefault(nid, []).append((ep, ep + dur, p))
    cache, rows = {}, []
    for _, r in df.iterrows():
        nid = r["ID"]; sl = int(r["SL"])
        if sl not in (0, 2):
            continue
        yv = 1 if sl == 2 else 0
        try:
            a, b = toep(r["date"], r["Start time"]), toep(r["date"], r["End time"])
        except (TypeError, ValueError, OverflowError):
            exclusions["survey_time"] += 1
            continue
        if b <= a:
            exclusions["nonpositive_interval"] += 1
            continue
        cov = [p for s, e, p in sess.get(nid, []) if s <= a < e]
        if not cov:
            exclusions["no_covering_session"] += 1
            continue
        p = cov[0]
        try:
            if p not in cache:
                ed_s, ef, eda = L.load_signal(p + "/EDA.csv"); hr_s, _, hr = L.load_signal(p + "/HR.csv")
                tp_s, _, temp = L.load_signal(p + "/TEMP.csv"); ib_s, ibi = L.load_ibi(p + "/IBI.csv")
                ac_s, _, acc = L.load_signal(p + "/ACC.csv", ncol=3)
                cache[p] = (ed_s, ef, eda, hr_s, hr, tp_s, temp, ib_s, ibi, ac_s, acc)
            ed_s, ef, eda, hr_s, hr, tp_s, temp, ib_s, ibi, ac_s, acc = cache[p]
        except (OSError, UnicodeError, csv.Error, IndexError, ValueError):
            exclusions["signal_loading"] += 1
            continue
        i0 = max(0, int((a - ed_s) * ef)); i1 = min(len(eda), int((b - ed_s) * ef))
        if i1 - i0 < 8 * ef:
            exclusions["short_eda_interval"] += 1
            continue
        tonic, phasic = L.cvx_decompose(eda[i0:i1], ef); base = ed_s + i0 / ef
        t = max(a, hr_s, base)
        endw = min(b, hr_s + len(hr), tp_s + len(temp) / 4, base + (i1 - i0) / ef)
        while t + L.WIN <= endw:
            hw = hr[int(round(t - hr_s)): int(round(t - hr_s + L.WIN))]
            e0 = int(round((t - base) * 4)); ew = slice(e0, e0 + int(L.WIN * 4))
            tw = temp[int(round((t - tp_s) * 4)): int(round((t - tp_s + L.WIN) * 4))]
            idd = ibi[(ibi[:, 0] >= t - ib_s) & (ibi[:, 0] < t - ib_s + L.WIN), 1] if len(ibi) else np.empty(0)
            if len(hw) >= 50 and ew.stop <= len(tonic):
                fv = L.window_features(hw, tonic[ew], phasic[ew], tw, idd)
                aw = acc[int(round((t - ac_s) * 32)): int(round((t - ac_s + L.WIN) * 32))]
                if fv[L.IDX["mean_hr"]] == fv[L.IDX["mean_hr"]]:
                    rows.append(("N_" + nid, yv, fv,
                                 {"activity": L.activity_features(aw)}))
            t += L.STEP
    _report_input_exclusions("Nurse", exclusions)
    return rows


def arr(rows):
    return (np.vstack([r[2] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[0] for r in rows]))


def acc_arr(rows):
    """창별 활동 맥락 특징을 ``L.ACC_FEATS`` 순서의 행렬로 모은다."""
    return np.vstack([r[3]["activity"] for r in rows])


def movement_g(rows):
    """기존 movement tertile과 동일한 정의(ACC 크기 표준편차, g)를 반환한다."""
    return acc_arr(rows)[:, L.ACC_IDX["acc_mag_std"]]


def single_auroc(y, X):
    if _SKLEARN_METRICS_IMPORT_ERROR is not None:
        L.require_dependency(
            "scikit-learn",
            _SKLEARN_METRICS_IMPORT_ERROR,
            "python3 -m pip install scikit-learn",
        )
    v = X[:, SCR]; ok = ~np.isnan(v)
    a = roc_auc_score(y[ok], v[ok]) if len(np.unique(y[ok])) == 2 else np.nan
    return a  # 원 방향 >0.5이면 stress에서 SCR count가 더 많다는 사전 가설과 일치한다.


def cross(XA, yA, XB, yB, est):
    clf = L._make_est(est); clf.fit(XA[:, TCOLS], yA)
    return L.auroc(yB, clf.predict_proba(XB[:, TCOLS])[:, 1])


def coral(XA, yA, XB, yB, eps=1e-3):
    """라벨 없는 target의 2차 통계에 source를 CORAL 정렬한 뒤 LR을 적합한다.

    이 구현은 시험한 CORAL이 일반 LR보다 나은지만 평가하며, 모든 domain
    adaptation 방법의 성능으로 일반화하지 않는다.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xa, Xb = XA[:, TCOLS], XB[:, TCOLS]
    imp = SimpleImputer(strategy="median").fit(Xa); Xa, Xb = imp.transform(Xa), imp.transform(Xb)
    sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)

    def msqrt(M, inv=False):
        w, V = np.linalg.eigh(M); w = np.clip(w, 1e-8, None)
        w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
        return V @ np.diag(w) @ V.T
    d = Xa.shape[1]
    Cs = np.cov(Xa, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xb, rowvar=False) + eps * np.eye(d)
    Xa_aligned = Xa @ msqrt(Cs, inv=True) @ msqrt(Ct)   # source 백색화 후 target 공분산으로 복원
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xa_aligned, yA)
    return L.auroc(yB, clf.predict_proba(Xb)[:, 1])


def build_all():
    require_runtime_paths()
    t0 = time.time()
    L.reset_analysis_audit()
    _CORPUS_INPUT_EXCLUSIONS.clear()
    corp, raw = {}, {}
    for nm, fn in (("WESAD", build_wesad), ("StrPred", build_sp), ("Nurse", build_nurse)):
        r = fn(); raw[nm] = r; X, y, s = arr(r)
        corp[nm] = (X, y, s)
        print(f"{nm:8} windows={len(y)}  +{int(y.sum())}/-{int((1-y).sum())}  subjects={len(np.unique(s))}  ({time.time()-t0:.0f}s)")
    print(f"EDA backend (observed decompositions): {L.eda_backend()}")
    print(f"EDA backend failures: {L.eda_backend_failures()}")
    print(f"Signal-row exclusions: {L.input_parse_exclusions()}")
    audit = L.analysis_audit_snapshot()
    audit["corpus_input_exclusion_counts"] = dict(
        sorted(_CORPUS_INPUT_EXCLUSIONS.items())
    )
    print(
        "ANALYSIS_AUDIT_JSON="
        + json.dumps(audit, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )
    return corp, raw


def run_analysis():
    require_runtime_dependencies()
    t0 = time.time()
    corp, raw = build_all()

    print("\n=== WITHIN-corpus LOSO (reference) ===")
    for nm, (X, y, s) in corp.items():
        full = np.arange(len(y))
        yl, sl, pl, _ = L.loso(X, y, s, TCOLS, full, mode="none", est="lr")
        yg, sg, pg, _ = L.loso(X, y, s, TCOLS, full, mode="none", est="gbm")
        print(f"  {nm:8}: single(scr_count)={single_auroc(y,X):.3f}  LR={L.auroc(yl,pl):.3f}  GBM={L.auroc(yg,pg):.3f}")

    print("\n=== (A) LEAVE-ONE-DATASET-OUT + domain-generalization (CORAL) ===")
    names = list(corp)
    print(f"  {'test':10} {'single':>7} {'LR':>7} {'GBM':>7} {'CORAL':>7}")
    for tgt in names:
        Xte, yte, _ = corp[tgt]
        src = [corp[n] for n in names if n != tgt]
        Xtr = np.vstack([c[0] for c in src]); ytr = np.concatenate([c[1] for c in src])
        a = single_auroc(yte, Xte); lr = cross(Xtr, ytr, Xte, yte, "lr")
        gbm = cross(Xtr, ytr, Xte, yte, "gbm"); cor = coral(Xtr, ytr, Xte, yte)
        print(f"  {tgt:10} {a:7.3f} {lr:7.3f} {gbm:7.3f} {cor:7.3f}")
    print("  single=fixed SCR rule (no train); LR/GBM/CORAL trained on other corpora.")
    print("  -> if CORAL & GBM do NOT beat LR/single, even a DG method cannot rescue transfer.")

    print("\n=== (C) NURSE stratified by movement (real-world activity confound) ===")
    Nr = raw["Nurse"]
    mov = movement_g(Nr); XN = np.vstack([r[2] for r in Nr])
    yN = np.array([r[1] for r in Nr]); sN = np.array([r[0] for r in Nr])
    ok = ~np.isnan(mov); mov, XN, yN, sN = mov[ok], XN[ok], yN[ok], sN[ok]
    q1, q2 = np.quantile(mov, [1 / 3, 2 / 3])
    print(f"  movement tertiles (ACC std, g): low<{q1:.3f}  mid<{q2:.3f}  high>=")
    for nm, m in (("low-move", mov < q1), ("mid-move", (mov >= q1) & (mov < q2)), ("high-move", mov >= q2)):
        Xs, ys, ss = XN[m], yN[m], sN[m]
        a = single_auroc(ys, Xs)
        full = np.arange(len(ys))
        yl, sl, pl, _ = L.loso(Xs, ys, ss, TCOLS, full, mode="none", est="lr")
        print(f"  {nm:9}: n={len(ys):5d} (+{int(ys.sum())}/-{int((1-ys).sum())})  "
              f"med-move={np.median(mov[m]):.3f}  single={a:.3f}  LR={L.auroc(yl,pl):.3f}")
    print("  -> tests whether real-world physical activity (high movement) degrades the EDA stress signal.")

    print("\n=== (2) iso-HR decomposition per corpus: transfer-weakness = HR-leakage or weak EDA? ===")
    allc = list(range(len(L.FEATS))); nonhr = [L.IDX[f] for f in L.NONHR]
    edatemp = [L.IDX[f] for f in ("scl_mean", "scl_slope", "phasic_auc", "scr_count",
                                  "scr_amp_mean", "scr_amp_max", "temp_mean", "temp_slope")]
    print(f"  {'corpus':10} {'unmatch_HR':>10} {'match_HR':>9} {'match_nonHR':>11} {'match_EDA+T':>11} {'HR-leak':>8} {'mN':>5}")
    for nm, (X, y, s) in corp.items():
        full = np.arange(len(y))
        a_un = L.auroc(*(lambda r: (r[0], r[2]))(L.loso(X, y, s, allc, full, mode="none")))
        yw, _, pw, _ = L.loso(X, y, s, allc, full, mode="global"); a_wm = L.auroc(yw, pw)
        yn, sn, pn, _ = L.loso(X, y, s, nonhr, full, mode="global"); a_nm = L.auroc(yn, pn)
        ye, _, pe, _ = L.loso(X, y, s, edatemp, full, mode="global"); a_em = L.auroc(ye, pe)
        print(f"  {nm:10} {a_un:10.3f} {a_wm:9.3f} {a_nm:11.3f} {a_em:11.3f} {a_wm-a_nm:+8.3f} {len(yn):5d}")
    print("  -> EDA+T(matched) ~0.5 for StrPred/Nurse => the EDA signal itself is weak (NOT just HR-leakage);")
    print("     WESAD keeps strong matched EDA => transfer-weakness is genuine stressor-type, not HR-confound alone.")

    print("\n=== (3) cross-corpus AUROC with subject-bootstrap 95% CI (single & LR) ===")
    for tgt in names:
        Xte, yte, ste = corp[tgt]; src = [corp[n] for n in names if n != tgt]
        Xtr = np.vstack([c[0] for c in src]); ytr = np.concatenate([c[1] for c in src])
        ps = Xte[:, SCR]; ok = ~np.isnan(ps)
        a_s = single_auroc(yte, Xte); lo_s, hi_s = L.boot_ci(yte[ok], ste[ok], ps[ok])
        clf = L._make_est("lr").fit(Xtr[:, TCOLS], ytr); pl = clf.predict_proba(Xte[:, TCOLS])[:, 1]
        a_l = L.auroc(yte, pl); lo_l, hi_l = L.boot_ci(yte, ste, pl)
        print(f"  test={tgt:9}: single={a_s:.3f} [{lo_s:.3f}-{hi_s:.3f}]   LR={a_l:.3f} [{lo_l:.3f}-{hi_l:.3f}]")
    print(f"\nTOTAL {time.time()-t0:.0f}s")


def main():
    require_runtime_dependencies()
    require_runtime_paths()
    run_analysis()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise SystemExit(str(exc))
