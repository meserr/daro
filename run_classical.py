"""Classical benchmark set: F1-F13 (D = 30, 100, 500) and F14-F31, 57 cases.

Usage: python run_classical.py
"""

import os
import time
from multiprocessing import Pool

import numpy as np

from common import N_RUNS, RESULTS_DIR, n_proc, load, save, run_algorithm
from benchmark_functions import FUNCTIONS, get_function_info

ALGS = ["D-ARO", "ARO", "LARO", "IARO", "GWO", "HHO", "WOA"]
CASES = [(f"F{i}", d) for i in range(1, 14) for d in (30, 100, 500)] + \
        [(f"F{i}", FUNCTIONS[f"F{i}"]["dim"]) for i in range(14, 32)]
OUT = os.path.join(RESULTS_DIR, "classical.pkl")


def job(args):
    alg, name, dim, seed = args
    f, lb, ub, d, _ = get_function_info(name, dim)
    return run_algorithm(alg, f, lb, ub, d, seed)


if __name__ == "__main__":
    store = load(OUT, {"results": {}, "convergences": {}})
    t0 = time.time()
    with Pool(n_proc()) as pool:
        for name, dim in CASES:
            key = f"{name}_D{dim}"
            for alg in ALGS:
                if key in store["results"].get(alg, {}):
                    continue
                res = pool.map(job, [(alg, name, dim, s) for s in range(N_RUNS)])
                store["results"].setdefault(alg, {})[key] = np.array([r[0] for r in res])
                store["convergences"].setdefault(alg, {})[key] = np.vstack([r[1] for r in res])
                save(store, OUT)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {key:9s} {alg:6s} "
                      f"mean={store['results'][alg][key].mean():.4e}", flush=True)
