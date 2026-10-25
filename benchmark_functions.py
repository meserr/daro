"""Classical benchmark set used in the paper.

F1-F7 unimodal, F8-F13 multimodal (Yao et al., 1999), F14-F23 fixed-dimension multimodal,
F24-F31 composition functions built on the CEC2014 framework (Liang et al., 2013)
with identity rotation and fixed-seed shift vectors.
"""

import numpy as np


def _u(x, a, k, m):
    y = np.zeros_like(x)
    mask1 = x > a
    mask2 = x < -a
    y[mask1] = k * (x[mask1] - a) ** m
    y[mask2] = k * (-x[mask2] - a) ** m
    return y


def F1(x):
    """Sphere  |  [-100, 100]^D  |  f* = 0"""
    return np.sum(x ** 2)


def F2(x):
    """Schwefel 2.22  |  [-10, 10]^D  |  f* = 0"""
    ax = np.abs(x)
    return np.sum(ax) + np.prod(ax)


def F3(x):
    """Schwefel 1.2  |  [-100, 100]^D  |  f* = 0"""
    return np.sum(np.cumsum(x) ** 2)


def F4(x):
    """Schwefel 2.21 (max |x_i|)  |  [-100, 100]^D  |  f* = 0"""
    return np.max(np.abs(x))


def F5(x):
    """Rosenbrock  |  [-30, 30]^D  |  f* = 0"""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def F6(x):
    """Step  |  [-100, 100]^D  |  f* = 0"""
    return np.sum(np.floor(x + 0.5) ** 2)


def F7(x):
    """Quartic + noise  |  [-1.28, 1.28]^D  |  f* ≈ 0"""
    d = len(x)
    i = np.arange(1, d + 1)
    return np.sum(i * x ** 4) + np.random.rand()


def F8(x):
    """Schwefel 2.26  |  [-500, 500]^D  |  f* ≈ -418.9829*D"""
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    """Rastrigin  |  [-5.12, 5.12]^D  |  f* = 0"""
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def F10(x):
    """Ackley  |  [-32, 32]^D  |  f* = 0"""
    d = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
            + 20 + np.e)


def F11(x):
    """Griewank  |  [-600, 600]^D  |  f* = 0"""
    i = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1


def F12(x):
    """Penalized 1  |  [-50, 50]^D  |  f* = 0"""
    d = len(x)
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0]) ** 2
    term2 = np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
    term3 = (y[-1] - 1) ** 2
    penalty = np.sum(_u(x, 10, 100, 4))
    return (np.pi / d) * (term1 + term2 + term3) + penalty


def F13(x):
    """Penalized 2  |  [-50, 50]^D  |  f* = 0"""
    term1 = np.sin(3 * np.pi * x[0]) ** 2
    term2 = np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
    term3 = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)
    penalty = np.sum(_u(x, 5, 100, 4))
    return 0.1 * (term1 + term2 + term3) + penalty


_A14 = np.array([[-32, -16, 0, 16, 32] * 5,
                  np.repeat([-32, -16, 0, 16, 32], 5)], dtype=float)


def F14(x):
    """Shekel Foxholes  |  [-65.536, 65.536]^2  |  f* ≈ 1"""
    j = np.arange(25)
    val = j + 1 + np.sum(np.array([(x[k] - _A14[k]) ** 6 for k in range(2)]), axis=0)
    return (1.0 / 500 + np.sum(1.0 / val)) ** -1


_a15 = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                  0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
_b15 = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])


def F15(x):
    """Kowalik  |  [-5, 5]^4  |  f* ≈ 0.0003075"""
    num = x[0] * (_b15 ** 2 + _b15 * x[1])
    den = _b15 ** 2 + _b15 * x[2] + x[3]
    return np.sum((_a15 - num / den) ** 2)


def F16(x):
    """Six-Hump Camel  |  [-5, 5]^2  |  f* ≈ -1.0316"""
    x1, x2 = x[0], x[1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


def F17(x):
    """Branin  |  [-5, 10] x [0, 15]  |  f* ≈ 0.397887"""
    x1, x2 = x[0], x[1]
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def F18(x):
    """Goldstein-Price  |  [-2, 2]^2  |  f* = 3"""
    x1, x2 = x[0], x[1]
    t1 = (1 + (x1 + x2 + 1) ** 2 *
          (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
    t2 = (30 + (2 * x1 - 3 * x2) ** 2 *
          (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    return t1 * t2


_aH3 = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_cH3 = np.array([1, 1.2, 3, 3.2])
_pH3 = np.array([[0.3689, 0.1170, 0.2673],
                   [0.4699, 0.4387, 0.7470],
                   [0.1091, 0.8732, 0.5547],
                   [0.0381, 0.5743, 0.8828]])


def F19(x):
    """Hartman 3  |  [0, 1]^3  |  f* ≈ -3.8628"""
    outer = 0.0
    for i in range(4):
        inner = np.sum(_aH3[i] * (x - _pH3[i]) ** 2)
        outer -= _cH3[i] * np.exp(-inner)
    return outer


_aH6 = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
_cH6 = np.array([1, 1.2, 3, 3.2])
_pH6 = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])


def F20(x):
    """Hartman 6  |  [0, 1]^6  |  f* ≈ -3.3224"""
    outer = 0.0
    for i in range(4):
        inner = np.sum(_aH6[i] * (x - _pH6[i]) ** 2)
        outer -= _cH6[i] * np.exp(-inner)
    return outer


_aShekel = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3.6, 7, 3.6]], dtype=float)
_cShekel = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _shekel(x, m):
    outer = 0.0
    for i in range(m):
        inner = np.sum((x - _aShekel[i]) ** 2) + _cShekel[i]
        outer -= 1.0 / inner
    return outer


def F21(x):
    """Shekel 5  |  [0, 10]^4  |  f* ≈ -10.1532"""
    return _shekel(x, 5)


def F22(x):
    """Shekel 7  |  [0, 10]^4  |  f* ≈ -10.4029"""
    return _shekel(x, 7)


def F23(x):
    """Shekel 10  |  [0, 10]^4  |  f* ≈ -10.5364"""
    return _shekel(x, 10)


def _cf_elliptic(x):
    D = len(x)
    if D <= 1:
        return x[0] ** 2
    i = np.arange(D)
    return np.sum(1e6 ** (i / (D - 1)) * x ** 2)


def _cf_bent_cigar(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def _cf_discus(x):
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


def _cf_rosenbrock(x):
    z = x + 1
    return np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)


def _cf_rastrigin(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def _cf_ackley(x):
    D = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / D))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / D) + 20 + np.e)


def _cf_griewank(x):
    i = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1


def _cf_schwefel(x):
    D = len(x)
    z = x + 420.9687462275036
    return 418.9829 * D - np.sum(z * np.sin(np.sqrt(np.abs(z))))


def _cf_happycat(x):
    z = x - 1
    D = len(z)
    s2 = np.sum(z ** 2)
    s1 = np.sum(z)
    return np.abs(s2 - D) ** 0.25 + (0.5 * s2 + s1) / D + 0.5


def _cf_hgbat(x):
    z = x - 1
    D = len(z)
    s2 = np.sum(z ** 2)
    s1 = np.sum(z)
    return np.abs(s2 ** 2 - s1 ** 2) ** 0.5 + (0.5 * s2 + s1) / D + 0.5


def _cf_weierstrass(x):
    D = len(x)
    a, b, kmax = 0.5, 3, 20
    k = np.arange(kmax + 1)
    ak, bk = a ** k, b ** k
    x_ext = x[:, np.newaxis] + 0.5
    val = np.sum(ak * np.cos(2 * np.pi * bk * x_ext))
    val -= D * np.sum(ak * np.cos(np.pi * bk))
    return val


_CF_DIM = 30


def _make_shifts(n, seed):
    rng = np.random.RandomState(seed)
    return [rng.uniform(-80, 80, _CF_DIM) for _ in range(n)]


_sh24 = _make_shifts(5, 2024)
_sh25 = _make_shifts(3, 2025)
_sh26 = _make_shifts(3, 2026)
_sh27 = _make_shifts(5, 2027)
_sh28 = _make_shifts(5, 2028)
_sh29 = _make_shifts(3, 2029)
_sh30 = _make_shifts(3, 2030)
_sh31 = _make_shifts(3, 2031)


def _composition(x, sub_funcs, sigmas, lambdas, biases, f_opt, shifts):
    D = len(x)
    N = len(sub_funcs)
    w = np.zeros(N)
    for i in range(N):
        diff = x - shifts[i][:D]
        w[i] = np.exp(-np.sum(diff ** 2) / (2 * D * sigmas[i] ** 2))
    w_sum = np.sum(w)
    w = w / w_sum if w_sum > 0 else np.ones(N) / N
    result = 0.0
    for i in range(N):
        z = x - shifts[i][:D]
        result += w[i] * (lambdas[i] * sub_funcs[i](z) + biases[i])
    return result + f_opt


def F24(x):
    """CF1 (N=5): Rosenbrock+Elliptic+BentCigar+Discus+Elliptic  |  [-100,100]^30  |  f*=2300"""
    return _composition(x,
        [_cf_rosenbrock, _cf_elliptic, _cf_bent_cigar, _cf_discus, _cf_elliptic],
        [10, 20, 30, 40, 50], [1, 1e-6, 1e-26, 1e-6, 1e-6],
        [0, 100, 200, 300, 400], 2300, _sh24)


def F25(x):
    """CF2 (N=3): Schwefel+Rastrigin+HGBat  |  [-100,100]^30  |  f*=2400"""
    return _composition(x,
        [_cf_schwefel, _cf_rastrigin, _cf_hgbat],
        [20, 20, 20], [10, 1, 1],
        [0, 100, 200], 2400, _sh25)


def F26(x):
    """CF3 (N=3): Schwefel+Rastrigin+HGBat  |  [-100,100]^30  |  f*=2500"""
    return _composition(x,
        [_cf_schwefel, _cf_rastrigin, _cf_hgbat],
        [10, 30, 50], [10, 1, 1],
        [0, 100, 200], 2500, _sh26)


def F27(x):
    """CF4 (N=5): Schwefel+HappyCat+HGBat+Rastrigin+Ackley  |  [-100,100]^30  |  f*=2600"""
    return _composition(x,
        [_cf_schwefel, _cf_happycat, _cf_hgbat, _cf_rastrigin, _cf_ackley],
        [10, 20, 30, 40, 50], [10, 1, 1, 1, 1],
        [0, 100, 200, 300, 400], 2600, _sh27)


def F28(x):
    """CF5 (N=5): HGBat+Rastrigin+Schwefel+Weierstrass+Elliptic  |  [-100,100]^30  |  f*=2700"""
    return _composition(x,
        [_cf_hgbat, _cf_rastrigin, _cf_schwefel, _cf_weierstrass, _cf_elliptic],
        [10, 20, 30, 40, 50], [10, 10, 2.5, 1e-26, 1e-6],
        [0, 100, 200, 300, 400], 2700, _sh28)


def F29(x):
    """CF6 (N=3): HGBat+Rastrigin+Schwefel  |  [-100,100]^30  |  f*=2800"""
    return _composition(x,
        [_cf_hgbat, _cf_rastrigin, _cf_schwefel],
        [20, 20, 20], [10, 10, 2.5],
        [0, 100, 200], 2800, _sh29)


def F30(x):
    """CF7 (N=3): Rastrigin+HappyCat+Ackley  |  [-100,100]^30  |  f*=2900"""
    return _composition(x,
        [_cf_rastrigin, _cf_happycat, _cf_ackley],
        [10, 30, 50], [10, 1, 1],
        [0, 100, 200], 2900, _sh30)


def F31(x):
    """CF8 (N=3): Ackley+Discus+Griewank  |  [-100,100]^30  |  f*=3000"""
    return _composition(x,
        [_cf_ackley, _cf_discus, _cf_griewank],
        [10, 30, 50], [10, 1e-6, 1],
        [0, 100, 200], 3000, _sh31)


FUNCTIONS = {
    "F1":  {"func": F1,  "lb": -100,      "ub": 100,      "dim": None,  "fmin": 0},
    "F2":  {"func": F2,  "lb": -10,       "ub": 10,       "dim": None,  "fmin": 0},
    "F3":  {"func": F3,  "lb": -100,      "ub": 100,      "dim": None,  "fmin": 0},
    "F4":  {"func": F4,  "lb": -100,      "ub": 100,      "dim": None,  "fmin": 0},
    "F5":  {"func": F5,  "lb": -30,       "ub": 30,       "dim": None,  "fmin": 0},
    "F6":  {"func": F6,  "lb": -100,      "ub": 100,      "dim": None,  "fmin": 0},
    "F7":  {"func": F7,  "lb": -1.28,     "ub": 1.28,     "dim": None,  "fmin": 0},
    "F8":  {"func": F8,  "lb": -500,      "ub": 500,      "dim": None,  "fmin": None},
    "F9":  {"func": F9,  "lb": -5.12,     "ub": 5.12,     "dim": None,  "fmin": 0},
    "F10": {"func": F10, "lb": -32,       "ub": 32,       "dim": None,  "fmin": 0},
    "F11": {"func": F11, "lb": -600,      "ub": 600,      "dim": None,  "fmin": 0},
    "F12": {"func": F12, "lb": -50,       "ub": 50,       "dim": None,  "fmin": 0},
    "F13": {"func": F13, "lb": -50,       "ub": 50,       "dim": None,  "fmin": 0},
    "F14": {"func": F14, "lb": -65.536,   "ub": 65.536,   "dim": 2,     "fmin": 0.998},
    "F15": {"func": F15, "lb": -5,        "ub": 5,        "dim": 4,     "fmin": 0.0003075},
    "F16": {"func": F16, "lb": -5,        "ub": 5,        "dim": 2,     "fmin": -1.0316},
    "F17": {"func": F17, "lb": [-5, 0],   "ub": [10, 15], "dim": 2,     "fmin": 0.397887},
    "F18": {"func": F18, "lb": -2,        "ub": 2,        "dim": 2,     "fmin": 3},
    "F19": {"func": F19, "lb": 0,         "ub": 1,        "dim": 3,     "fmin": -3.8628},
    "F20": {"func": F20, "lb": 0,         "ub": 1,        "dim": 6,     "fmin": -3.3224},
    "F21": {"func": F21, "lb": 0,         "ub": 10,       "dim": 4,     "fmin": -10.1532},
    "F22": {"func": F22, "lb": 0,         "ub": 10,       "dim": 4,     "fmin": -10.4029},
    "F23": {"func": F23, "lb": 0,         "ub": 10,       "dim": 4,     "fmin": -10.5364},
    "F24": {"func": F24, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2300},
    "F25": {"func": F25, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2400},
    "F26": {"func": F26, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2500},
    "F27": {"func": F27, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2600},
    "F28": {"func": F28, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2700},
    "F29": {"func": F29, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2800},
    "F30": {"func": F30, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 2900},
    "F31": {"func": F31, "lb": -100,      "ub": 100,      "dim": 30,    "fmin": 3000},
}


def get_function_info(name: str, dim: int = 30):
    """Returns (func, lb, ub, dim, fmin); dim is used only for F1-F13."""
    info = FUNCTIONS[name]
    func = info["func"]
    d = info["dim"] if info["dim"] is not None else dim

    lb_raw, ub_raw = info["lb"], info["ub"]
    if isinstance(lb_raw, list):
        lb = np.array(lb_raw, dtype=float)
        ub = np.array(ub_raw, dtype=float)
    else:
        lb = np.full(d, lb_raw, dtype=float)
        ub = np.full(d, ub_raw, dtype=float)

    fmin = info["fmin"]
    if fmin is None:
        fmin = -418.9829 * d

    return func, lb, ub, d, fmin
