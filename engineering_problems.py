import numpy as np


def welded_beam_obj(x):
    h, l, t, b = x[0], x[1], x[2], x[3]
    return 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)


def welded_beam_cons(x):
    h, l, t, b = x[0], x[1], x[2], x[3]
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    delta_max, tau_max, sigma_max = 0.25, 13600.0, 30000.0

    M = P * (L + l / 2)
    R = np.sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
    J = 2 * (np.sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2))
    tau1 = P / (np.sqrt(2) * h * l + 1e-30)
    tau2 = M * R / (J + 1e-30)
    tau = np.sqrt(tau1 ** 2 + 2 * tau1 * tau2 * l / (2 * R + 1e-30) + tau2 ** 2)
    sigma = 6 * P * L / (b * t ** 2 + 1e-30)
    delta = 4 * P * L ** 3 / (E * b * t ** 3 + 1e-30)
    Pc = (4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36) / L ** 2
          * (1 - t / (2 * L) * np.sqrt(E / (4 * G))))

    return [tau - tau_max,
            sigma - sigma_max,
            h - b,
            0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0,
            0.125 - h,
            delta - delta_max,
            P - Pc]


def pressure_vessel_obj(x):
    Ts, Th, R, L = x[0], x[1], x[2], x[3]
    return (0.6224 * Ts * R * L + 1.7781 * Th * R ** 2
            + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R)


def pressure_vessel_cons(x):
    Ts, Th, R, L = x[0], x[1], x[2], x[3]
    return [-Ts + 0.0193 * R,
            -Th + 0.00954 * R,
            -np.pi * R ** 2 * L - (4 / 3) * np.pi * R ** 3 + 1296000,
            L - 240]


def pressure_vessel_fix(x):
    x = np.array(x, dtype=float)
    x[0] = round(x[0] / 0.0625) * 0.0625
    x[1] = round(x[1] / 0.0625) * 0.0625
    return x


def spring_obj(x):
    d, D, N = x[0], x[1], x[2]
    return (N + 2) * D * d ** 2


def spring_cons(x):
    d, D, N = x[0], x[1], x[2]
    return [1 - D ** 3 * N / (71785 * d ** 4 + 1e-30),
            (4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4) + 1e-30) + 1 / (5108 * d ** 2 + 1e-30) - 1,
            1 - 140.45 * d / (D ** 2 * N + 1e-30),
            (d + D) / 1.5 - 1]


def speed_reducer_obj(x):
    b, m, z, l1, l2, d1, d2 = x
    return (0.7854 * b * m ** 2 * (3.3333 * z ** 2 + 14.9334 * z - 43.0934)
            - 1.508 * b * (d1 ** 2 + d2 ** 2)
            + 7.4777 * (d1 ** 3 + d2 ** 3)
            + 0.7854 * (l1 * d1 ** 2 + l2 * d2 ** 2))


def speed_reducer_cons(x):
    b, m, z, l1, l2, d1, d2 = x
    A1 = (745 * l1 / (m * z + 1e-30)) ** 2 + 16.9e6
    A2 = (745 * l2 / (m * z + 1e-30)) ** 2 + 157.5e6
    return [27 / (b * m ** 2 * z + 1e-30) - 1,
            397.5 / (b * m ** 2 * z ** 2 + 1e-30) - 1,
            1.93 * l1 ** 3 / (m * z * d1 ** 4 + 1e-30) - 1,
            1.93 * l2 ** 3 / (m * z * d2 ** 4 + 1e-30) - 1,
            np.sqrt(A1) / (110 * d1 ** 3 + 1e-30) - 1,
            np.sqrt(A2) / (85 * d2 ** 3 + 1e-30) - 1,
            b * m / 40 - 1,
            5 * m / (b + 1e-30) - 1,
            b / (12 * m + 1e-30) - 1,
            (1.5 * d1 + 1.9) / (l1 + 1e-30) - 1,
            (1.1 * d2 + 1.9) / (l2 + 1e-30) - 1]


def speed_reducer_fix(x):
    x = np.array(x, dtype=float)
    x[2] = float(np.round(x[2]))
    return x


def _identity(x):
    return x


PROBLEMS = {
    "Welded Beam": (welded_beam_obj, welded_beam_cons, _identity, 1e6,
                    [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0]),
    "Pressure Vessel": (pressure_vessel_obj, pressure_vessel_cons, pressure_vessel_fix, 1e9,
                        [0.0625, 0.0625, 10.0, 10.0], [6.1875, 6.1875, 200.0, 200.0]),
    "Tension Spring": (spring_obj, spring_cons, _identity, 1e6,
                       [0.05, 0.25, 2.0], [2.0, 1.3, 15.0]),
    "Speed Reducer": (speed_reducer_obj, speed_reducer_cons, speed_reducer_fix, 1e6,
                      [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
}
