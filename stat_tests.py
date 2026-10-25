"""Friedman test with Nemenyi critical difference, Holm procedure with a control algorithm,
pairwise Wilcoxon rank-sum tests with Holm correction and the Vargha-Delaney A12 effect size."""

import numpy as np
from scipy import stats

# studentised range statistic / sqrt(2), alpha = 0.05 (Demsar, 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def a12(x, y):
    """P(x < y) + 0.5 P(x = y); for minimisation, values above 0.5 favour x."""
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def holm_adjust(p):
    order = np.argsort(p)
    m = len(p)
    out = np.ones(m)
    running = 0.0
    for r, j in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[j]))
        out[j] = running
    return out


def analyse(res, algs, cases, ctrl, title=None):
    """res[alg][case] -> array of final values from independent runs."""
    k, n = len(algs), len(cases)
    M = np.array([[np.mean(res[a][c]) for a in algs] for c in cases])
    ranks = np.vstack([stats.rankdata(row) for row in M]).mean(0)
    chi2, p = stats.friedmanchisquare(*M.T)
    se = np.sqrt(k * (k + 1) / (6 * n))
    cd = Q05[k] * se

    ci = algs.index(ctrl)
    z = sorted([(a, (ranks[j] - ranks[ci]) / se) for j, a in enumerate(algs) if a != ctrl],
               key=lambda r: -abs(r[1]))
    holm_ctrl, stop = [], False
    for i, (a, zi) in enumerate(z):
        pv = 2 * stats.norm.sf(abs(zi))
        thr = 0.05 / (len(z) - i)
        rej = pv < thr and not stop
        stop = stop or not rej
        holm_ctrl.append((a, zi, pv, thr, rej))

    rivals = [a for a in algs if a != ctrl]
    P = np.ones((n, len(rivals)))
    A = np.zeros((n, len(rivals)))
    for i, c in enumerate(cases):
        x = np.asarray(res[ctrl][c])
        for j, a in enumerate(rivals):
            y = np.asarray(res[a][c])
            A[i, j] = a12(x, y)
            P[i, j] = 1.0 if np.ptp(np.r_[x, y]) == 0 else stats.ranksums(x, y).pvalue
    PH = np.vstack([holm_adjust(row) for row in P])
    wtl = {}
    for j, a in enumerate(rivals):
        sig = PH[:, j] < 0.05
        w = int((sig & (A[:, j] > 0.5)).sum())
        l = int((sig & (A[:, j] < 0.5)).sum())
        wtl[a] = (w, n - w - l, l)

    out = dict(algs=algs, cases=cases, mean=M, ranks=ranks, chi2=chi2, p=p, CD=cd,
               holm_ctrl=holm_ctrl, rivals=rivals, p_holm=PH, A12=A, wtl=wtl)
    if title:
        report(out, title)
    return out


def report(r, title):
    ctrl = [a for a in r["algs"] if a not in r["rivals"]][0]
    print(f"\n{title}: {len(r['algs'])} algorithms, {len(r['cases'])} cases")
    print(f"Friedman chi2 = {r['chi2']:.2f}, p = {r['p']:.3g}, Nemenyi CD = {r['CD']:.3f}")
    for i in np.argsort(r["ranks"]):
        print(f"  {r['algs'][i]:12s} {r['ranks'][i]:.3f}")
    print(f"Holm procedure, control = {ctrl}")
    for a, z, pv, thr, rej in r["holm_ctrl"]:
        print(f"  {a:12s} z = {z:+.3f}  p = {pv:.4f}  threshold = {thr:.4f}  {'significant' if rej else ''}")
    print(f"{ctrl} vs rival: W/T/L (Wilcoxon rank-sum, Holm)  mean A12")
    for j, a in enumerate(r["rivals"]):
        print(f"  {a:12s} {'%d/%d/%d' % r['wtl'][a]:>10s}  {r['A12'][:, j].mean():.3f}")
