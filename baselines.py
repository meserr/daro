"""L-SHADE (NiaPy) and IPOP-CMA-ES (pycma) run under a fixed budget of function evaluations."""

import numpy as np


class BudgetExhausted(Exception):
    pass


class Budget:
    """Wraps an objective, stops after max_fe evaluations and records the best value every `step` evaluations."""

    def __init__(self, f, max_fe, step=50):
        self.f, self.max_fe, self.step = f, max_fe, step
        self.n, self.best = 0, np.inf
        self.curve = np.empty(max_fe // step)

    def __call__(self, x):
        if self.n >= self.max_fe:
            raise BudgetExhausted
        v = self.f(x)
        self.n += 1
        if v < self.best:
            self.best = v
        if self.n % self.step == 0:
            self.curve[self.n // self.step - 1] = self.best
        return v

    def final_curve(self):
        k = self.n // self.step
        if k < len(self.curve):
            self.curve[k:] = self.best
        return self.curve


def run_lshade(f, lb, ub, dim, max_fe, seed):
    # Tanabe and Fukunaga (2014): N_init = 18D, r_arc = 2.6, p = 0.11, H = 6
    from niapy.task import Task
    from niapy.problems import Problem
    from niapy.algorithms.modified import LpsrSuccessHistoryAdaptiveDifferentialEvolution as LSHADE

    class P(Problem):
        def __init__(self):
            super().__init__(dimension=dim, lower=lb, upper=ub)

        def _evaluate(self, x):
            return f(x)

    try:
        LSHADE(population_size=18 * dim, extern_arc_rate=2.6, pbest_factor=0.11, hist_mem_size=6,
               seed=seed).run(Task(problem=P(), max_evals=max_fe))
    except BudgetExhausted:
        pass


def run_cmaes(f, lb, ub, max_fe, seed, sigma0, extra_opts=None):
    # IPOP restarts with doubling population size
    import cma
    x0 = np.random.default_rng(seed).uniform(lb, ub)
    opts = {"bounds": [np.asarray(lb).tolist(), np.asarray(ub).tolist()], "maxfevals": max_fe, "seed": seed + 1, "verbose": -9}
    opts.update(extra_opts or {})
    try:
        cma.fmin2(f, x0, sigma0, opts, restarts=20, incpopsize=2)
    except BudgetExhausted:
        pass
