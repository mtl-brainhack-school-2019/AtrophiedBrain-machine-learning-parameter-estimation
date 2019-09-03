import numpy as np


def random_beta(count=1):
    set = np.arange(0.01, 0.05, 0.005, dtype=float)
    set2 = np.arange(-0.05, -0.01, 0.005, dtype=float)
    set = np.concatenate((set, set2))

    return np.random.choice(set, size=count) / 1000


def random_alpha(count=1):
    set = np.arange(-0.05, -0.005, 0.005, dtype=float)
    set2 = np.arange(0.005, 0.05, 0.005, dtype=float)
    set = np.concatenate((set, set2))
    return np.random.choice(set, size=count)


def random_alpha_negative(count=1):
    set = np.arange(-0.03, -0.005, 0.005, dtype=float)
    return np.random.choice(set, size=count)


def random_alpha_positive(count=1):
    set = np.arange(-0.03, -0.005, 0.005, dtype=float)
    return np.random.choice(set, size=count)


def estimate(a, b, times, ivs, step):
    output = np.zeros((times.shape[0], ivs.shape[0]))
    output[0, :] = ivs
    i = 1
    for time in times[1:]:
        last = output[i - 1, :]

        at = a * last
        m = np.tile(last, (ivs.shape[0], 1))
        np.fill_diagonal(m, 0.0)
        bt = b * m.sum(-1)

        deriv = at + bt
        output[i, :] = last + deriv * step
        i += 1

    return output