import os
import pickle

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

N_RUNS = 30
POP = 50
ITER = 1000
TAU = 10
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MEALPY_ALGS = ("ARO", "LARO", "IARO", "GWO", "HHO", "WOA")


def n_proc():
    return int(os.environ.get("NPROC", max(1, (os.cpu_count() or 2) // 2)))


def load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return default


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def cec_function(year, fid, dim):
    import minionpy as mp
    g = mp.CEC2017Functions(fid, dim) if year == 2017 else mp.CEC2022Functions(fid, dim)
    return lambda x: float(g(np.asarray(x, dtype=float)[None, :])[0])


def run_algorithm(alg, f, lb, ub, dim, seed, pop=POP, iters=ITER, tau=TAU, phi_mode="log"):
    if alg == "D-ARO":
        from daro import DARO
        opt = DARO(f, lb, ub, dim, n_rabbits=pop, max_iter=iters, stag_limit=tau,
                   phi_mode=phi_mode, seed=seed)
        _, fit = opt.optimize()
        return float(fit), opt.convergence.astype(np.float32)
    from mealpy_adapter import run_mealpy_algorithm
    _, fit, conv = run_mealpy_algorithm(alg, f, lb, ub, dim, pop_size=pop, max_iter=iters, seed=seed)
    return float(fit), np.asarray(conv, dtype=np.float32)
