"""Optional numba-compiled kernels for the sequential decoders (Viterbi,
turbo max-log-MAP). Without numba the pure NumPy versions are used.

The decoders work on tiny blocks (tens to a few thousand bits) with a
strictly sequential trellis recursion, so they are bound by interpreter
overhead, not arithmetic: compiling the loops gives 30-100x, while a GPU
would spend more time launching kernels than computing."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:            # pragma: no cover - exercised without numba
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco


@njit(cache=True)
def viterbi_core(ext: np.ndarray, pred: np.ndarray, out: np.ndarray) -> np.ndarray:
    """ext: (3, N) LLRs, pred: (64, 2) predecessor states, out: (64, 2, 3)
    +-1 branch symbols. Returns N decided bits (traceback from best state)."""
    n = ext.shape[1]
    metric = np.zeros(64)
    new = np.empty(64)
    dec = np.empty((n, 64), dtype=np.uint8)
    for t in range(n):
        l0, l1, l2 = ext[0, t], ext[1, t], ext[2, t]
        best = -1e300
        for sp in range(64):
            s0 = pred[sp, 0]
            s1 = pred[sp, 1]
            m0 = metric[s0] + out[sp, 0, 0] * l0 + out[sp, 0, 1] * l1 + out[sp, 0, 2] * l2
            m1 = metric[s1] + out[sp, 1, 0] * l0 + out[sp, 1, 1] * l1 + out[sp, 1, 2] * l2
            if m1 > m0:
                new[sp] = m1
                dec[t, sp] = 1
            else:
                new[sp] = m0
                dec[t, sp] = 0
            if new[sp] > best:
                best = new[sp]
        for sp in range(64):
            metric[sp] = new[sp] - best
    s = 0
    for sp in range(64):
        if metric[sp] > metric[s]:
            s = sp
    bits = np.empty(n, dtype=np.uint8)
    for t in range(n - 1, -1, -1):
        bits[t] = s >> 5
        s = pred[s, dec[t, s]]
    return bits


@njit(cache=True)
def map_core(ls, lp, la, tx, tz, nxt, zb, prv_s, prv_u, tail_u, tail_z):
    """Max-log-MAP of one 8-state RSC (same semantics as turbo._map_decode)."""
    k = ls.shape[0]
    neg = -1e30
    alpha = np.empty((k + 1, 8))
    for s in range(8):
        alpha[0, s] = neg
    alpha[0, 0] = 0.0
    for t in range(k):
        lsa = ls[t] + la[t]
        best = neg
        for ns in range(8):
            m = neg
            for j in range(2):
                s = prv_s[ns, j]
                u = prv_u[ns, j]
                g = 0.5 * ((1.0 - 2.0 * u) * lsa + (1.0 - 2.0 * zb[s, u]) * lp[t])
                v = alpha[t, s] + g
                if v > m:
                    m = v
            alpha[t + 1, ns] = m
            if m > best:
                best = m
        for ns in range(8):
            alpha[t + 1, ns] -= best
    beta = np.empty(8)
    for s in range(8):
        beta[s] = neg
    beta[0] = 0.0
    nb = np.empty(8)
    for t in range(2, -1, -1):
        best = neg
        for s in range(8):
            g = 0.5 * (tx[t] * (1.0 - 2.0 * tail_u[s]) + tz[t] * (1.0 - 2.0 * tail_z[s]))
            nb[s] = beta[s >> 1] + g
            if nb[s] > best:
                best = nb[s]
        for s in range(8):
            beta[s] = nb[s] - best
    out = np.empty(k)
    for t in range(k - 1, -1, -1):
        lsa = ls[t] + la[t]
        m0 = neg
        m1 = neg
        best = neg
        for s in range(8):
            g0 = 0.5 * (lsa + (1.0 - 2.0 * zb[s, 0]) * lp[t])
            g1 = 0.5 * (-lsa + (1.0 - 2.0 * zb[s, 1]) * lp[t])
            b0 = g0 + beta[nxt[s, 0]]
            b1 = g1 + beta[nxt[s, 1]]
            if alpha[t, s] + b0 > m0:
                m0 = alpha[t, s] + b0
            if alpha[t, s] + b1 > m1:
                m1 = alpha[t, s] + b1
            nb[s] = b0 if b0 > b1 else b1
            if nb[s] > best:
                best = nb[s]
        out[t] = m0 - m1
        for s in range(8):
            beta[s] = nb[s] - best
    return out
