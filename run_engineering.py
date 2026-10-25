"""Constrained engineering problems.

The search is guided by a static penalty f(x) + lambda * sum(max(0, g_i(x))^2); the reported
result of each run is the best fully feasible solution (all g_i(x) <= 0, no tolerance).
N = 30, T = 500 (30 + 30 * 500 = 15,030 evaluations); L-SHADE and IPOP-CMA-ES use the same budget.

Usage: python run_engineering.py
"""

import os
import time
from multiprocessing import Pool

import numpy as np

from common import N_RUNS, RESULTS_DIR, TAU, n_proc, load, save, run_algorithm
from baselines import BudgetExhausted, run_lshade, run_cmaes
from engineering_problems import PROBLEMS

ALGS = ["D-ARO", "ARO", "LARO", "IARO", "GWO", "HHO", "WOA", "L-SHADE", "CMA-ES"]
POP, ITER = 30, 500
BUDGET = POP + POP * ITER
OUT = os.path.join(RESULTS_DIR, "engineering.pkl")


class Tracker:
    """Penalised objective that keeps the best feasible solution found."""

    def __init__(self, name, budget=None):
        self.obj, self.cons, self.fix, self.lam, lb, ub = PROBLEMS[name]
        self.lb, self.ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
        self.budget, self.n = budget, 0
        self.best_feas, self.best_x, self.first_feas = np.inf, None, None

    def __call__(self, x):
        if self.budget is not None and self.n >= self.budget:
            raise BudgetExhausted
        self.n += 1
        x = self.fix(np.asarray(x, dtype=float))
        f = float(self.obj(x))
        g = np.asarray(self.cons(x), dtype=float)
        feasible = np.all(g <= 0.0)
        if feasible and self.first_feas is None:
            self.first_feas = self.n
        if feasible and f < self.best_feas:
            self.best_feas, self.best_x = f, x.copy()
        return f + self.lam * float(np.sum(np.maximum(0.0, g) ** 2))


def job(args):
    alg, name, seed = args
    tr = Tracker(name, BUDGET if alg in ("L-SHADE", "CMA-ES") else None)
    lb, ub, dim = tr.lb, tr.ub, len(tr.lb)
    if alg == "L-SHADE":
        run_lshade(tr, lb, ub, dim, BUDGET, seed)
    elif alg == "CMA-ES":
        span = ub - lb
        run_cmaes(tr, lb, ub, BUDGET, seed, 0.3 * float(np.min(span)),
                  {"CMA_stds": (span / np.min(span)).tolist()})
    else:
        run_algorithm(alg, tr, lb, ub, dim, seed, pop=POP, iters=ITER, tau=TAU)
    return dict(feas=tr.best_feas, x=tr.best_x, fe=tr.n, first_feas=tr.first_feas)


if __name__ == "__main__":
    store = load(OUT, {})
    with Pool(n_proc()) as pool:
        for name in PROBLEMS:
            for alg in ALGS:
                if (alg, name) in store:
                    continue
                t = time.time()
                store[(alg, name)] = pool.map(job, [(alg, name, s) for s in range(N_RUNS)])
                save(store, OUT)
                feas = np.array([r["feas"] for r in store[(alg, name)]])
                ok = np.isfinite(feas)
                print(f"{name:16s} {alg:8s} feasible={ok.sum():2d}/{N_RUNS} "
                      f"best={feas[ok].min() if ok.any() else np.nan:.6f} "
                      f"mean={feas[ok].mean() if ok.any() else np.nan:.6f} ({time.time() - t:.0f}s)", flush=True)
