import os

import numpy as np
from scipy import stats

from common import RESULTS_DIR, load
from stat_tests import analyse, a12, holm_adjust
from cd_diagram import draw_cd

SEVEN = ["D-ARO", "ARO", "LARO", "IARO", "GWO", "HHO", "WOA"]
NINE = SEVEN + ["L-SHADE", "CMA-ES"]
PHI = ["log", "none", "linear", "const", "trunc_linear", "restart"]
TAUS = [10, 5, 15, 20, 30]


def path(name):
    return os.path.join(RESULTS_DIR, name)


def main_sets():
    out = {}
    cls = load(path("classical.pkl"), None)
    if cls:
        res = cls["results"]
        out["classical"] = analyse(res, SEVEN, list(res["D-ARO"]), "D-ARO", "Classical set")
    c17 = load(path("cec2017.pkl"), None)
    if c17:
        res = c17["results"]
        cases = list(res["D-ARO"])
        out["cec2017"] = analyse(res, SEVEN, cases, "D-ARO", "CEC2017")
        for d in (30, 50, 100):
            sub = [c for c in cases if c.endswith(f"_D{d}")]
            if sub:
                analyse(res, SEVEN, sub, "D-ARO", f"CEC2017, D = {d}")
    c22 = load(path("cec2022.pkl"), None)
    if c22:
        res = c22["results"]
        cases = list(res["D-ARO"])
        out["cec2022"] = analyse(res, NINE, cases, "D-ARO", "CEC2022")
        analyse(res, SEVEN, cases, "D-ARO", "CEC2022, ARO family and swarm algorithms")
        for d in (10, 20):
            analyse(res, NINE, [c for c in cases if c.endswith(f"_D{d}")], "D-ARO", f"CEC2022, D = {d}")
    return out


def ablation():
    main = {"classical": "classical.pkl", "cec2017": "cec2017.pkl", "cec2022": "cec2022.pkl"}
    for suite, ref_file in main.items():
        abl = load(path(f"ablation_{suite}.pkl"), None)
        ref = load(path(ref_file), None)
        if not abl or not ref:
            continue
        ref = ref["results"]["D-ARO"]
        cases = list(abl["phi:none"])
        phi = {m: (ref if m == "log" else abl[f"phi:{m}"]) for m in PHI}
        analyse(phi, PHI, cases, "log", f"Ablation of phi, {suite}")
        tau = {f"tau={t}": (ref if t == 10 else abl[f"tau:{t}"]) for t in TAUS}
        analyse(tau, list(tau), cases, "tau=10", f"Sensitivity to tau, {suite}")
    d100 = load(path("ablation_cec2017-d100.pkl"), None)
    c17 = load(path("cec2017.pkl"), None)
    if d100 and c17:
        w = t = l = 0
        for case, y in d100["phi:none"].items():
            x = c17["results"]["D-ARO"][case]
            p = 1.0 if np.ptp(np.r_[x, y]) == 0 else stats.ranksums(x, y).pvalue
            e = a12(x, y)
            w, t, l = w + (p < 0.05 and e > 0.5), t + (p >= 0.05), l + (p < 0.05 and e < 0.5)
        print(f"\nCEC2017, D = 100, log vs none: W/T/L {w}/{t}/{l}")


def engineering():
    eng = load(path("engineering.pkl"), None)
    if not eng:
        return
    probs = list(dict.fromkeys(name for _, name in eng))
    rivals = [a for a in NINE if a != "D-ARO"]
    print("\nEngineering problems: best feasible value per run, Wilcoxon rank-sum with Holm correction")
    for name in probs:
        x = np.array([r["feas"] for r in eng[("D-ARO", name)]])
        p, A = [], []
        for a in rivals:
            y = np.array([r["feas"] for r in eng[(a, name)]])
            A.append(a12(x, y))
            p.append(1.0 if np.ptp(np.r_[x, y]) == 0 else stats.ranksums(x, y).pvalue)
        ph = holm_adjust(np.array(p))
        print(f"  {name} (D-ARO mean {x.mean():.6f}, feasible {np.isfinite(x).sum()}/{len(x)})")
        for a, pa, aa in zip(rivals, ph, A):
            y = np.array([r["feas"] for r in eng[(a, name)]])
            sign = "+" if pa < 0.05 and aa > 0.5 else "-" if pa < 0.05 else "="
            print(f"    {a:8s} mean {np.nanmean(np.where(np.isfinite(y), y, np.nan)):.6f}  "
                  f"feasible {np.isfinite(y).sum():2d}  p_Holm {pa:.3g}  A12 {aa:.3f}  {sign}")


if __name__ == "__main__":
    sets = main_sets()
    for name, r in sets.items():
        draw_cd(r["algs"], list(r["ranks"]), r["CD"], path(f"cd_{name}.png"))
    ablation()
    engineering()
