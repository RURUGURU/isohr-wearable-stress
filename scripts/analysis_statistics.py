#!/usr/bin/env python3
# pyright: standard
"""리뷰어 답변에 필요한 cross-corpus 추가 통계를 계산한다.

첫 분석에 없던 두 근거를 보완한다.
  1. LODO의 고정 SCR와 LR 점수에 대해 피험자 내부에서만 라벨을 섞는
     cluster-aware 단측 순열검정을 하고, 같은 검정군에 Holm 보정을 적용한다.
     이 귀무분포는 피험자 구조와 클래스 수를 보존한다.
  2. iso-HR 무작위 부분표본을 7개 seed로 반복해 매칭 EDA+TEMP AUROC와
     HR-leakage 지수의 평균·표준편차를 기록한다.

결과는 권위 리비전의 ``results/run_crosscorpus_stats.txt``에 저장한다. corpus
구축 로직은 ``analysis_crosscorpus.py``를 import해 중복 구현하지 않는다.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts import analysis_crosscorpus as C
from scripts import utils_analysis as L
from scripts.utils_paths import RESULTS_DIR

roc_auc_score = None
_SKLEARN_METRICS_IMPORT_ERROR = None
try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    _SKLEARN_METRICS_IMPORT_ERROR = exc

OUT = str(RESULTS_DIR / "run_crosscorpus_stats.txt")
SCR = L.IDX["scr_count"]
TCOLS = C.TCOLS
LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def perm_p_within_subject(y, subj, score, B=2000, seed=42):
    """피험자 내부 라벨 순열로 고정 점수 AUROC가 우연보다 큰지 단측 검정한다."""
    ok = ~np.isnan(score)
    y, subj, score = y[ok].astype(int), subj[ok], score[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    obs = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    usub = np.unique(subj)
    idx_by_s = {s: np.where(subj == s)[0] for s in usub}
    cnt = 0
    for _ in range(B):
        yp = y.copy()
        for s in usub:
            m = idx_by_s[s]
            yp[m] = rng.permutation(y[m])
        if len(np.unique(yp)) == 2 and roc_auc_score(yp, score) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


def coral_proba(XA, yA, XB, eps=1e-3):
    """CORAL target 확률을 반환해 같은 피험자 bootstrap에서 모델을 비교한다."""
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
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        Xa @ msqrt(Cs, inv=True) @ msqrt(Ct), yA)
    return clf.predict_proba(Xb)[:, 1]


def paired_boot_diff(scores, y, subj, ref="single", B=1000, seed=42):
    """피험자 cluster paired bootstrap으로 ``기준-모델`` AUROC 차이를 구한다.

    95% 신뢰구간이 0을 포함하면 시험한 표본과 비교에서 통계적으로 구분되지
    않는다는 뜻이며, 일반적인 모델 동등성을 의미하지 않는다.
    """
    us = np.unique(subj)
    rng = np.random.default_rng(seed)
    diffs = {m: [] for m in scores if m != ref}
    idx_by_s = {s: np.where(subj == s)[0] for s in us}
    for _ in range(B):
        smp = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([idx_by_s[s] for s in smp])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = {m: roc_auc_score(yy, sc[idx]) for m, sc in scores.items()}
        for m in diffs:
            diffs[m].append(a[ref] - a[m])
    return {m: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for m, v in diffs.items()}


def isohr_decomp(corp, seed):
    allc = list(range(len(L.FEATS)))
    nonhr = [L.IDX[f] for f in L.NONHR]
    edatemp = [L.IDX[f] for f in ("scl_mean", "scl_slope", "phasic_auc", "scr_count",
                                  "scr_amp_mean", "scr_amp_max", "temp_mean", "temp_slope")]
    res = {}
    for nm, (X, y, s) in corp.items():
        full = np.arange(len(y))
        yw, _, pw, _ = L.loso(X, y, s, allc, full, mode="global", seed=seed); a_wm = L.auroc(yw, pw)
        yn, _, pn, _ = L.loso(X, y, s, nonhr, full, mode="global", seed=seed); a_nm = L.auroc(yn, pn)
        ye, _, pe, _ = L.loso(X, y, s, edatemp, full, mode="global", seed=seed); a_em = L.auroc(ye, pe)
        res[nm] = dict(with_hr=a_wm, non_hr=a_nm, eda_t=a_em, hr_leak=a_wm - a_nm, mN=len(yn))
    return res


def main():
    if _SKLEARN_METRICS_IMPORT_ERROR is not None:
        L.require_dependency(
            "scikit-learn",
            _SKLEARN_METRICS_IMPORT_ERROR,
            "python3 -m pip install scikit-learn",
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    corp, raw = C.build_all()
    names = list(corp)

    # ---------- (1) 피험자 구조 보존 순열검정과 Holm 보정 ----------
    emit("\n=== (1) Cross-corpus LODO: cluster-aware permutation p (vs chance) + Holm ===")
    emit("  within-subject label shuffle, B=2000; one-sided P(AUROC>chance).")
    tests, pvals = [], []
    for tgt in names:
        Xte, yte, ste = corp[tgt]
        src = [corp[n] for n in names if n != tgt]
        Xtr = np.vstack([c[0] for c in src]); ytr = np.concatenate([c[1] for c in src])
        # 단일 SCR은 학습하지 않는 고정 연속 점수이다.
        a_s, p_s = perm_p_within_subject(yte, ste, Xte[:, SCR])
        tests.append((tgt, "single", a_s)); pvals.append(p_s)
        # LR은 source에서만 적합하고 target에서는 고정 확률만 산출한다.
        clf = L._make_est("lr").fit(Xtr[:, TCOLS], ytr)
        pl = clf.predict_proba(Xte[:, TCOLS])[:, 1]
        a_l, p_l = perm_p_within_subject(yte, ste, pl)
        tests.append((tgt, "LR", a_l)); pvals.append(p_l)
    adj = L.holm([p if p == p else 1.0 for p in pvals])
    emit(f"  {'test':9} {'model':7} {'AUROC':>7} {'p_raw':>9} {'p_Holm':>9} {'sig.05':>7}")
    for (tgt, mdl, a), p, pa in zip(tests, pvals, adj):
        sig = "*" if (pa == pa and pa < 0.05) else "ns"
        emit(f"  {tgt:9} {mdl:7} {a:7.3f} {p:9.4f} {pa:9.4f} {sig:>7}")

    # ---------- (2) 여러 seed에서 iso-HR 분해 안정성 ----------
    SEEDS = [42, 1, 7, 13, 99, 123, 2024]
    emit(f"\n=== (2) iso-HR decomposition stability over {len(SEEDS)} matching seeds ===")
    emit(f"  seeds={SEEDS}; mean +/- SD across seeds (matching is a random subsample).")
    agg = {nm: {k: [] for k in ("with_hr", "non_hr", "eda_t", "hr_leak", "mN")} for nm in names}
    for sd in SEEDS:
        r = isohr_decomp(corp, sd)
        for nm in names:
            for k in agg[nm]:
                agg[nm][k].append(r[nm][k])
    emit(f"  {'corpus':9} {'match_with_HR':>14} {'match_EDA+T':>14} {'HR-leak':>16} {'mN(mean)':>9}")
    for nm in names:
        def ms(k):
            v = np.array(agg[nm][k], float)
            return np.mean(v), np.std(v)
        wm, wms = ms("with_hr"); em, ems = ms("eda_t"); hm, hms = ms("hr_leak")
        mn = np.mean(agg[nm]["mN"])
        emit(f"  {nm:9} {wm:6.3f}+/-{wms:5.3f} {em:6.3f}+/-{ems:5.3f} {hm:+7.3f}+/-{hms:5.3f}   {mn:7.0f}")

    # ---------- (3) 단일 SCR 대비 paired subject-bootstrap AUROC 차이 ----------
    emit("\n=== (3) Paired subject-bootstrap AUROC difference (single - model), B=1000 ===")
    emit("  positive => single better; 95% CI containing 0 => statistically indistinguishable.")
    for tgt in names:
        Xte, yte, ste = corp[tgt]
        src = [corp[n] for n in names if n != tgt]
        Xtr = np.vstack([c[0] for c in src]); ytr = np.concatenate([c[1] for c in src])
        scores = {"single": Xte[:, SCR].astype(float)}
        scores["LR"] = L._make_est("lr").fit(Xtr[:, TCOLS], ytr).predict_proba(Xte[:, TCOLS])[:, 1]
        scores["GBM"] = L._make_est("gbm").fit(Xtr[:, TCOLS], ytr).predict_proba(Xte[:, TCOLS])[:, 1]
        scores["CORAL"] = coral_proba(Xtr, ytr, Xte)
        res = paired_boot_diff(scores, yte, ste, ref="single", B=1000)
        emit(f"  {tgt}:")
        for m in ("LR", "GBM", "CORAL"):
            mn, lo, hi = res[m]
            flag = "indistinguishable" if lo <= 0 <= hi else "DIFFERS"
            emit(f"     single - {m:5}: {mn:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {flag}")

    emit(f"\nTOTAL {time.time()-t0:.0f}s")
    with open(OUT, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise SystemExit(str(exc))
