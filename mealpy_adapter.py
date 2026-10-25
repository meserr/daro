import logging
import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based.ARO import OriginalARO, LARO, IARO
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.WOA import OriginalWOA

logging.getLogger("mealpy").setLevel(logging.WARNING)

ALGORITHMS = {
    "ARO": OriginalARO,
    "LARO": LARO,
    "IARO": IARO,
    "GWO": OriginalGWO,
    "HHO": OriginalHHO,
    "WOA": OriginalWOA,
}


def run_mealpy_algorithm(name, func, lb, ub, dim, pop_size=50, max_iter=1000, seed=None, **params):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    problem = {
        "obj_func": func,
        "bounds": FloatVar(lb=tuple(lb.tolist()), ub=tuple(ub.tolist())),
        "minmax": "min",
        "log_to": None,
    }
    model = ALGORITHMS[name](epoch=max_iter, pop_size=pop_size, **params)
    g_best = model.solve(problem, seed=seed)

    conv = np.array(model.history.list_global_best_fit)
    if len(conv) < max_iter:
        conv = np.concatenate([conv, np.full(max_iter - len(conv), conv[-1])])
    return np.array(g_best.solution), g_best.target.fitness, conv[:max_iter]
