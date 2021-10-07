import numpy as np
from scipy.stats import binom, poisson

def pa(params, model):
    amin, amax = params['amin'], params['amax']
    # bmin, bmax = params['bmin'], params['bmax']
    # p1, p2, p3 = params['p1'], params['p2'], params['p3']

    n = amax - amin + 1
    prob = np.ones((n,)) * (1 / n)
    val = np.arange(amin, amax + 1)

    return prob, val

def pb(params, model):
    # amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    # p1, p2, p3 = params['p1'], params['p2'], params['p3']

    m = bmax - bmin + 1
    prob = np.ones((m,)) * (1 / m)
    val = np.arange(bmin, bmax + 1)

    return prob, val

def pc_ab(a, b, params, model):
    # a,b scalars
    p1, p2 = params['p1'], params['p2']

    if model == 3:

        p_ksi = binom.pmf(np.arange(a + b), a, p1)
        p_eta = binom.pmf(np.arange(a + b), b, p2)

        prob = np.convolve(p_ksi, p_eta)[:a + b + 1]
        val = np.arange(a + b + 1)

        return prob, val

    if model == 4:

        prob = poisson.pmf(np.arange(a + b + 1), a * p1 + b * p2)
        val = np.arange(a + b + 1)

        # divide to prob.sum for avoiding generating mistakes when probs >> 0
        return prob / prob.sum(), val

def pc(params, model):
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    p1, p2 = params['p1'], params['p2']

    n = amax - amin + 1
    m = bmax - bmin + 1

    if model == 3:

        a_col = np.arange(amin, amax + 1)[..., np.newaxis]
        b_col = np.arange(bmin, bmax + 1)[..., np.newaxis]

        ksi_row = np.arange(amax + 1)
        eta_row = np.arange(bmax + 1)

        p_ksi = binom.pmf(ksi_row, a_col, p1)
        p_eta = binom.pmf(eta_row, b_col, p2)

        prob = np.zeros(amax + bmax + 1)

        for row in p_ksi:
            prob += np.sum(
                np.apply_along_axis(lambda eta_row: np.convolve(eta_row, row), 1, p_eta), axis=0
            ) * (1 / n) * (1 / m)

        return prob, np.arange(amax + bmax + 1)

    if model == 4:

        col = np.arange(amin, amax + 1)[..., np.newaxis] * p1
        row = np.arange(bmin, bmax + 1)[np.newaxis, ...] * p2

        ab = col + row
        lambdas_row = ab.ravel()[np.newaxis, ...]
        values_col = np.arange(amax + bmax + 1)[..., np.newaxis]

        table = poisson.pmf(values_col, lambdas_row)
        prob = np.sum(table, axis=1) * (1 / n) * (1 / m)

        return prob, np.arange(amax + bmax + 1)

def pd(params, model):
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    p3 = params['p3']

    pc_prob, _ = pc(params, model)

    k_col = np.arange(2 * (amax + bmax) + 1)[..., np.newaxis]
    n_row = np.arange(amax + bmax + 1)[np.newaxis, ...]

    table = binom.pmf(k_col, n_row, p3)

    prob = np.zeros(2 * (amax + bmax) + 1)

    for c in range(amax + bmax + 1):
        prob += np.roll(table[..., c], c) * pc_prob[c]

    return prob, np.arange(2 * (amax + bmax) + 1)

def pd_ab(a, b, params, model):
    # maybe refactor
    # code dupl with pd
    p3 = params['p3']

    pc_ab_prob, _ = pc_ab(a, b, params, model)

    k_col = np.arange(2 * (a + b) + 1)[..., np.newaxis]
    n_row = np.arange(a + b + 1)[np.newaxis, ...]

    table = binom.pmf(k_col, n_row, p3)

    prob = np.zeros(2 * (a + b) + 1)

    for c in range(a + b + 1):
        prob += np.roll(table[..., c], c) * pc_ab_prob[c]

    return prob, np.arange(2 * (a + b) + 1)

def generate(N, a, b, params, model):
    res = np.zeros((N, len(a), len(b)))
    for a_idx, a_val in enumerate(a):
        for b_idx, b_val in enumerate(b):
            p, v = pd_ab(a_val, b_val, params, model)
            res[..., a_idx, b_idx] = np.random.choice(v, size=N, p=p)
    return res

def pd_b(k, b, params, model):
    amin, amax = params['amin'], params['amax']
    p3 = params['p3']

    n = amax - amin + 1
    a_row = np.arange(amin, amax + 1)

    prob = np.zeros(amax + b + 1)

    for a in a_row:
        pc_ab_iter, _ = pc_ab(a, b, params, model)
        prob += np.pad(pc_ab_iter, (0, amax + b + 1 - len(pc_ab_iter)), 'constant')

    prob *= (1 / n)

    n_bin = np.arange(amax + b + 1)
    k_bin = k - n_bin
    weights = binom.pmf(k_bin, n_bin, p3)

    return np.dot(weights, prob)

def pc_ab_vectorized(params, model):
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    p1, p2 = params['p1'], params['p2']

    if model == 3:
        a_n_col = np.arange(amin, amax + 1)[..., np.newaxis]
        b_n_col = np.arange(bmin, bmax + 1)[..., np.newaxis]

        a_k_row = np.arange(amax + 1)[np.newaxis, ...]
        b_k_row = np.arange(bmax + 1)[np.newaxis, ...]

        a_table = binom.pmf(a_k_row, a_n_col, p1)
        b_table = binom.pmf(b_k_row, b_n_col, p2)

        p_table = np.zeros((amax - amin + 1, bmax - bmin + 1, amax + bmax + 1))

        for idx_a in np.arange(amax - amin + 1):
            p_table[idx_a] = np.apply_along_axis(lambda row: np.convolve(row, a_table[idx_a]), 1, b_table)

        return p_table

    if model == 4:
        p_table = np.zeros((amax - amin + 1, bmax - bmin + 1, amax + bmax + 1))

        col = np.arange(amin, amax + 1)[..., np.newaxis] * p1
        row = np.arange(bmin, bmax + 1)[np.newaxis, ...] * p2

        ab = (col + row)

        vals = np.arange(amax + bmax + 1)[np.newaxis, ...][np.newaxis, ...]

        p_table = poisson.pmf(vals, np.repeat(ab.reshape(ab.shape[0], ab.shape[1], 1), amax + bmax + 1, 2))

        return p_table

def pd_ab_vectorized(k, params, model):
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    p3 = params['p3']

    tensor = pc_ab_vectorized(params, model)

    n_bin = np.arange(amax + bmax + 1)
    k_bin = k - n_bin
    pd_c = binom.pmf(k_bin, n_bin, p3)[np.newaxis, ...][np.newaxis, ...]

    cube = tensor * pd_c

    return np.sum(cube, axis=2)

def pd_b_vectorized(k, params, model):
    amin, amax = params['amin'], params['amax']
    n = amax - amin + 1

    table = pd_ab_vectorized(k, params, model)

    return np.sum(table, axis=0) * (1 / n)

def pb_d_old(d, params, model):
    # Result: prob of size (bmax-bmin+1,k_d), val of size (bmax-bmin+1,)
    # d of size (k_d,N)
    bmin, bmax = params['bmin'], params['bmax']
    m = bmax - bmin + 1

    pb_prob = (1 / m)
    pd_prob, _ = pd(params, model)

    res_prob = np.zeros((bmax - bmin + 1, d.shape[0]))

    for idx_seq, d_seq in enumerate(d):
        iter_prob = np.ones(bmax - bmin + 1)
        denom = 1
        for d_n in d_seq:
            pd_b_iter = pd_b_vectorized(d_n, params, model)
            iter_prob *= pd_b_iter
            denom *= pd_prob[d_n]

        res_prob[..., idx_seq] = iter_prob * pb_prob / denom

    return res_prob, np.arange(bmin, bmax + 1)

def X(d, params, model):
    # d of size (k_d,N)
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']
    p3 = params['p3']

    pc_ab_prob = pc_ab_vectorized(params, model)
    c_n = np.arange(amax + bmax + 1).reshape(1, 1, amax + bmax + 1)
    d_n = np.repeat(d.reshape(d.shape[0], d.shape[1], 1), amax + bmax + 1, 2)

    n_bin = c_n
    k_bin = d_n - c_n

    pd_c_prob = binom.pmf(k_bin, n_bin, p3)

    res_x = np.zeros((d.shape[0], amax - amin + 1, bmax - bmin + 1))

    for res_x_idx, d_c_N in enumerate(pd_c_prob):
        res_x_iter = np.ones((amax - amin + 1, bmax - bmin + 1))
        for d_n_iter in d_c_N:
            iter_tensor = pc_ab_prob * d_n_iter
            res_x_iter *= iter_tensor.sum(axis = 2)
        res_x[res_x_idx] = res_x_iter

    return res_x

def pb_d(d, params, model):
    # d of size (k_d,N)
    # Result: prob of size (bmax-bmin+1,k_d), val of size (bmax-bmin+1,)
    bmin, bmax = params['bmin'], params['bmax']

    res = np.zeros((bmax - bmin + 1, len(d)))
    X_ab = X(d, params, model)
    for d_N_idx, d_N in enumerate(d):
        X_iter = X_ab[d_N_idx]
        iter_table = np.sum(X_iter, axis=0)
        denom = np.sum(iter_table)
        res[..., d_N_idx] = iter_table / denom

    return res, np.arange(bmin, bmax + 1)

def pb_ad(a, d, params, model):
    # Input: a of size (k_a,) and d of size (k_d,N)
    # Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
    amin, amax = params['amin'], params['amax']
    bmin, bmax = params['bmin'], params['bmax']

    res = np.zeros((bmax - bmin + 1, len(a), len(d)))
    X_ab = X(d, params, model)

    for d_N_idx, d_N in enumerate(d):
        X_iter = X_ab[d_N_idx][a - amin]
        denom = np.sum(X_iter, axis=1)[..., np.newaxis]

        res[..., d_N_idx] = np.swapaxes(X_iter / denom, 0, 1)

    return res, np.arange(bmin, bmax + 1)
