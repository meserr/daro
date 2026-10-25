import numpy as np


class DARO:

    def __init__(self, func, lb, ub, dim, n_rabbits=50, max_iter=1000, stag_limit=10,
                 use_stagnation=True, phi_mode='log', seed=None):
        self.func = func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = dim
        self.n = n_rabbits
        self.T = max_iter
        self.stag_limit = stag_limit
        self.use_stagnation = use_stagnation
        self.phi_mode = phi_mode
        self.eps = 1e-16

        if seed is not None:
            np.random.seed(seed)

        self.X = self.lb + (self.ub - self.lb) * np.random.rand(self.n, self.dim)
        self.fitness = np.array([self.func(self.X[i]) for i in range(self.n)])
        best_idx = np.argmin(self.fitness)
        self.best_pos = self.X[best_idx].copy()
        self.best_fit = self.fitness[best_idx]
        if self.use_stagnation:
            self.age = np.zeros(self.n, dtype=int)
        self.convergence = np.zeros(self.T)

    def _phi(self, age):
        if self.phi_mode == 'log':
            return 1.0 + np.log(age + 1)
        if self.phi_mode == 'linear':
            return 1.0 + age
        if self.phi_mode == 'const':
            return 3.0
        if self.phi_mode == 'trunc_linear':
            return min(1.0 + age, 1.0 + np.log(self.T + 1))
        return 1.0

    def _R(self, t):
        d, T = self.dim, self.T
        r2 = np.random.rand()
        L = (np.e - np.e ** (((t - 1) / T) ** 2)) * np.sin(2 * np.pi * r2)
        r3 = np.random.rand()
        c = np.zeros(d)
        c[np.random.choice(d, int(np.ceil(r3 * d)), replace=False)] = 1.0
        return L * c

    def optimize(self):
        n, d, T = self.n, self.dim, self.T

        for t in range(1, T + 1):
            r = np.random.rand()
            A_global = 4.0 * (1.0 - t / T) * np.log(1.0 / (r + self.eps))

            for i in range(n):
                xi = self.X[i]

                if (self.use_stagnation and self.phi_mode == 'restart'
                        and self.age[i] >= self.stag_limit):
                    vi = self.lb + (self.ub - self.lb) * np.random.rand(d)
                    fi = self.func(vi)
                    self.X[i] = vi
                    self.fitness[i] = fi
                    self.age[i] = 0
                    if fi < self.best_fit:
                        self.best_fit = fi
                        self.best_pos = vi.copy()
                    continue

                A_i = A_global
                if self.use_stagnation and self.age[i] >= self.stag_limit:
                    A_i = A_global * self._phi(self.age[i])

                if A_i > 1:
                    j = np.random.randint(0, n - 1)
                    if j >= i:
                        j += 1
                    xj = self.X[j]
                    R = self._R(t)
                    r1 = np.random.rand()
                    n1 = np.random.randn(d)
                    vi = xj + R * (xi - xj) + np.round(0.5 * (0.05 + r1)) * n1
                else:
                    r4 = np.random.rand()
                    H = ((T - t + 1) / T) * r4
                    g = np.zeros(d)
                    g[np.random.randint(0, d)] = 1.0
                    burrow = xi + H * g * xi
                    R = self._R(t)
                    r5 = np.random.rand()
                    vi = xi + R * (r5 * burrow - xi)

                vi = np.clip(vi, self.lb, self.ub)
                fi = self.func(vi)

                if fi < self.fitness[i]:
                    self.X[i] = vi
                    self.fitness[i] = fi
                    if self.use_stagnation:
                        self.age[i] = 0
                    if fi < self.best_fit:
                        self.best_fit = fi
                        self.best_pos = vi.copy()
                elif self.use_stagnation:
                    self.age[i] += 1

            self.convergence[t - 1] = self.best_fit

        return self.best_pos, self.best_fit
