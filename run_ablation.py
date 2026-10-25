"""Ablation of the energy factor phi and sensitivity to tau.

Variants (tau = 10): none, linear, const, trunc_linear, restart.
Thresholds (phi = log): tau = 5, 15, 20, 30.
The reference configuration (phi = log, tau = 10) is D-ARO itself, taken from the main runs.

Usage:
    python run_ablation.py classical      57 classical cases
    python run_ablation.py cec2017        CEC2017, D = 30
    python run_ablation.py cec2022        CEC2022, D = 10 and 20
    python run_ablation.py cec2017-d100   CEC2017, D = 100, variant 'none' only
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import N_RUNS, RESULTS_DIR, n_proc, load, save, run_algorithm, cec_function
from benchmark_functions import get_function_info
from run_classical import CASES as CLASSICAL_CASES

CONFIGS = [("phi", m) for m in ("none", "linear", "const", "trunc_linear", "restart")] + \
          [("tau", t) for t in (5, 15, 20, 30)]
SUITES = {
    "classical": ([f"{n}_D{d}" for n, d in CLASSICAL_CASES], CONFIGS),
    "cec2017": ([f"F{f}_D30" for f in [1] + list(range(3, 31))], CONFIGS),
    "cec2022": ([f"F{f}_D{d}" for d in (10, 20) for f in range(1, 13)], CONFIGS),
    "cec2017-d100": ([f"F{f}_D100" for f in [1] + list(range(3, 31))], [("phi", "none")]),
}


def problem(suite, case):
    name, dim = case.split("_D")
    dim = int(dim)
    if suite == "classical":
        f, lb, ub, dim, _ = get_function_info(name, dim)
        return f, lb, ub, dim
    f = cec_function(2022 if suite == "cec2022" else 2017, int(name[1:]), dim)
    return f, np.full(dim, -100.0), np.full(dim, 100.0), dim


def job(args):
    suite, kind, val, case, seed = args
    f, lb, ub, dim = problem(suite, case)
    phi_mode, tau = (val, 10) if kind == "phi" else ("log", val)
    return run_algorithm("D-ARO", f, lb, ub, dim, seed, tau=tau, phi_mode=phi_mode)[0]


if __name__ == "__main__":
    suite = sys.argv[1]
    cases, configs = SUITES[suite]
    out = os.path.join(RESULTS_DIR, f"ablation_{suite}.pkl")
    store = load(out, {})
    t0 = time.time()
    with Pool(n_proc()) as pool:
        for case in cases:
            for kind, val in configs:
                key = f"{kind}:{val}"
                st = store.setdefault(key, {})
                if case in st:
                    continue
                st[case] = np.array(pool.map(job, [(suite, kind, val, case, s) for s in range(N_RUNS)]))
                save(store, out)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {case:9s} {key:16s} mean={st[case].mean():.4e}", flush=True)
