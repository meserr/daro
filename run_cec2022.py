import os
import time
from multiprocessing import Pool

import numpy as np

from common import N_RUNS, RESULTS_DIR, n_proc, load, save, run_algorithm, cec_function
from baselines import Budget, run_lshade, run_cmaes

ALGS = ["D-ARO", "ARO", "LARO", "IARO", "GWO", "HHO", "WOA", "L-SHADE", "CMA-ES"]
CASES = [(fid, d) for d in (10, 20) for fid in range(1, 13)]
MAX_FE = 50_000
OUT = os.path.join(RESULTS_DIR, "cec2022.pkl")


def job(args):
    alg, fid, dim, seed = args
    f = cec_function(2022, fid, dim)
    lb, ub = np.full(dim, -100.0), np.full(dim, 100.0)
    if alg not in ("L-SHADE", "CMA-ES"):
        cnt = [0]

        def fc(x):
            cnt[0] += 1
            return f(x)
        fit, conv = run_algorithm(alg, fc, lb, ub, dim, seed)
        return fit, conv, cnt[0]
    b = Budget(f, MAX_FE)
    if alg == "L-SHADE":
        run_lshade(b, lb, ub, dim, MAX_FE, seed)
    else:
        run_cmaes(b, lb, ub, MAX_FE, seed, 0.3 * (ub[0] - lb[0]))
    return float(b.best), b.final_curve().astype(np.float32), b.n


if __name__ == "__main__":
    store = load(OUT, {"results": {}, "convergences": {}, "fe": {}})
    t0 = time.time()
    with Pool(n_proc()) as pool:
        for fid, dim in CASES:
            key = f"F{fid}_D{dim}"
            for alg in ALGS:
                if key in store["results"].get(alg, {}):
                    continue
                res = pool.map(job, [(alg, fid, dim, s) for s in range(N_RUNS)])
                store["results"].setdefault(alg, {})[key] = np.array([r[0] for r in res])
                store["convergences"].setdefault(alg, {})[key] = np.vstack([r[1] for r in res])
                store["fe"].setdefault(alg, {})[key] = np.array([r[2] for r in res])
                save(store, OUT)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {key:7s} {alg:8s} "
                      f"mean={store['results'][alg][key].mean():.4e}", flush=True)
