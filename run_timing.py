"""Run-time measurements of D-ARO (single process, one run timed at a time).

scalability: CEC2017 F1, F5, F10, F15, F20, F25, F30 at D = 30, 50, 100.
overhead:    D-ARO with and without the age counter (use_stagnation=True/False) on the same code.

Usage: python run_timing.py
"""

import os
import time

import numpy as np
from scipy import stats

from common import N_RUNS, POP, ITER, TAU, RESULTS_DIR, save, cec_function
from benchmark_functions import get_function_info
from daro import DARO

SCALABILITY = [(fid, d) for fid in (1, 5, 10, 15, 20, 25, 30) for d in (30, 50, 100)]
OVERHEAD = [(f"F{i}", d) for i in (1, 9, 13) for d in (30, 100, 500)] + \
           [(f"CEC17_F{i}", d) for i in (1, 15, 30) for d in (30, 50, 100)]


def problem(name, dim):
    if name.startswith("CEC17_F"):
        return cec_function(2017, int(name[7:]), dim), np.full(dim, -100.0), np.full(dim, 100.0), dim
    f, lb, ub, d, _ = get_function_info(name, dim)
    return f, lb, ub, d


def timed(f, lb, ub, dim, seed, use_stagnation=True):
    t0 = time.perf_counter()
    _, fit = DARO(f, lb, ub, dim, n_rabbits=POP, max_iter=ITER, stag_limit=TAU,
                  use_stagnation=use_stagnation, seed=seed).optimize()
    return time.perf_counter() - t0, fit


if __name__ == "__main__":
    scal = {}
    for fid, d in SCALABILITY:
        f, lb, ub, d = problem(f"CEC17_F{fid}", d)
        ts = np.array([timed(f, lb, ub, d, s)[0] for s in range(N_RUNS)])
        scal[(fid, d)] = ts
        print(f"CEC2017 F{fid:<2d} D={d:<3d} time={ts.mean():.3f} s (std {ts.std():.3f})", flush=True)
    save(scal, os.path.join(RESULTS_DIR, "timing_scalability.pkl"))

    over = {}
    for name, d in OVERHEAD:
        f, lb, ub, d = problem(name, d)
        runs = min(10, N_RUNS) if d >= 500 else N_RUNS
        on, off = np.zeros(runs), np.zeros(runs)
        for s in range(runs):
            on[s] = timed(f, lb, ub, d, s, True)[0]
            off[s] = timed(f, lb, ub, d, s, False)[0]
        p = stats.wilcoxon(on, off).pvalue
        over[(name, d)] = (on, off)
        print(f"{name:10s} D={d:<3d} with={on.mean():.3f} s without={off.mean():.3f} s "
              f"diff={100 * (on.mean() - off.mean()) / off.mean():+.2f}% p={p:.3g}", flush=True)
    save(over, os.path.join(RESULTS_DIR, "timing_overhead.pkl"))
