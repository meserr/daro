"""CEC2017 (official definitions, minionpy): F1 and F3-F30 at D = 30, 50, 100.

Usage: python run_cec2017.py [D ...]
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import N_RUNS, RESULTS_DIR, n_proc, load, save, run_algorithm, cec_function

ALGS = ["D-ARO", "ARO", "LARO", "IARO", "GWO", "HHO", "WOA"]
FIDS = [1] + list(range(3, 31))
OUT = os.path.join(RESULTS_DIR, "cec2017.pkl")


def job(args):
    alg, fid, dim, seed = args
    f = cec_function(2017, fid, dim)
    return run_algorithm(alg, f, np.full(dim, -100.0), np.full(dim, 100.0), dim, seed)


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [30, 50, 100]
    store = load(OUT, {"results": {}, "convergences": {}})
    t0 = time.time()
    with Pool(n_proc()) as pool:
        for dim in dims:
            for fid in FIDS:
                key = f"F{fid}_D{dim}"
                for alg in ALGS:
                    if key in store["results"].get(alg, {}):
                        continue
                    res = pool.map(job, [(alg, fid, dim, s) for s in range(N_RUNS)])
                    store["results"].setdefault(alg, {})[key] = np.array([r[0] for r in res])
                    store["convergences"].setdefault(alg, {})[key] = np.vstack([r[1] for r in res])
                    save(store, OUT)
                    print(f"[{(time.time() - t0) / 60:6.1f} min] {key:8s} {alg:6s} "
                          f"mean={store['results'][alg][key].mean():.4e}", flush=True)
