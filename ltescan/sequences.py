"""Reference sequences of TS 36.211: Gold sequence, PSS, SSS, CRS."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .params import N_RB_MAX

_NC = 1600


@lru_cache(maxsize=4096)
def _gold_cached(c_init: int, length: int) -> np.ndarray:
    n = length + _NC
    x1 = np.zeros(n + 31, dtype=np.uint8)
    x2 = np.zeros(n + 31, dtype=np.uint8)
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    # x(n+31) depends on x(n..n+3) -> we can compute 28 new values at a time
    i = 0
    while i < n:
        m = min(28, n - i)
        x1[i + 31:i + 31 + m] = x1[i + 3:i + 3 + m] ^ x1[i:i + m]
        x2[i + 31:i + 31 + m] = (x2[i + 3:i + 3 + m] ^ x2[i + 2:i + 2 + m]
                                 ^ x2[i + 1:i + 1 + m] ^ x2[i:i + m])
        i += m
    c = x1[_NC:_NC + length] ^ x2[_NC:_NC + length]
    c.setflags(write=False)
    return c


def gold(c_init: int, length: int) -> np.ndarray:
    """Pseudo-random sequence c(n) of 36.211 7.2 (uint8 0/1)."""
    return _gold_cached(int(c_init), int(length))


# ----------------------------------------------------------------------- PSS
PSS_ROOTS = (25, 29, 34)


@lru_cache(maxsize=3)
def pss(n_id_2: int) -> np.ndarray:
    u = PSS_ROOTS[n_id_2]
    n = np.arange(62)
    d = np.where(n < 31,
                 np.exp(-1j * np.pi * u * n * (n + 1) / 63),
                 np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))
    d.setflags(write=False)
    return d


# ----------------------------------------------------------------------- SSS
def _mseq(taps: tuple[int, ...]) -> np.ndarray:
    x = [0, 0, 0, 0, 1] + [0] * 26
    for i in range(26):
        x[i + 5] = sum(x[i + t] for t in taps) % 2
    return 1 - 2 * np.array(x)


_S_T = _mseq((2, 0))
_C_T = _mseq((3, 0))
_Z_T = _mseq((4, 2, 1, 0))


@lru_cache(maxsize=1024)
def sss(n_id_1: int, n_id_2: int, subframe5: bool) -> np.ndarray:
    qp = n_id_1 // 30
    q = (n_id_1 + qp * (qp + 1) // 2) // 30
    mp = n_id_1 + q * (q + 1) // 2
    m0 = mp % 31
    m1 = (m0 + mp // 31 + 1) % 31
    n = np.arange(31)
    s0 = _S_T[(n + m0) % 31]
    s1 = _S_T[(n + m1) % 31]
    c0 = _C_T[(n + n_id_2) % 31]
    c1 = _C_T[(n + n_id_2 + 3) % 31]
    z0 = _Z_T[(n + (m0 % 8)) % 31]
    z1 = _Z_T[(n + (m1 % 8)) % 31]
    d = np.empty(62)
    if not subframe5:
        d[0::2] = s0 * c0
        d[1::2] = s1 * c1 * z0
    else:
        d[0::2] = s1 * c0
        d[1::2] = s0 * c1 * z1
    d.setflags(write=False)
    return d


@lru_cache(maxsize=4)
def sss_table(n_id_2: int) -> np.ndarray:
    """All 336 SSS candidates (168 ids x {sf0, sf5}) as a (336, 62) matrix.
    Row 2*n1 = subframe 0, row 2*n1+1 = subframe 5."""
    t = np.empty((336, 62))
    for n1 in range(168):
        t[2 * n1] = sss(n1, n_id_2, False)
        t[2 * n1 + 1] = sss(n1, n_id_2, True)
    t.setflags(write=False)
    return t


# ----------------------------------------------------------------------- CRS
def crs_symbols(cp_ext: bool) -> list[tuple[int, list[int]]]:
    """Per slot: (symbol index in slot, list of ports with CRS in it)."""
    n_symb = 6 if cp_ext else 7
    return [(0, [0, 1]), (1, [2, 3]), (n_symb - 3, [0, 1])]


def crs_shift(port: int, l: int, ns: int, n_id: int, cp_ext: bool) -> int:
    """Subcarrier offset (0..5) of the CRS of `port` in symbol l of slot ns.
    For ports 0/1, l is either 0 or N_symb-3."""
    if port == 0:
        v = 0 if l == 0 else 3
    elif port == 1:
        v = 3 if l == 0 else 0
    elif port == 2:
        v = 3 * (ns % 2)
    else:
        v = 3 + 3 * (ns % 2)
    return (v + n_id % 6) % 6


@lru_cache(maxsize=8192)
def crs_seq(ns: int, l: int, n_id: int, cp_ext: bool) -> np.ndarray:
    """Full-length (2*N_RB_MAX) CRS QPSK sequence for slot ns, symbol l."""
    n_cp = 0 if cp_ext else 1
    c_init = (1 << 10) * (7 * (ns + 1) + l + 1) * (2 * n_id + 1) + 2 * n_id + n_cp
    c = gold(c_init, 4 * N_RB_MAX).astype(np.float64)
    r = ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2)
    r.setflags(write=False)
    return r


def crs(ns: int, l: int, port: int, n_id: int, n_rb: int, cp_ext: bool):
    """Return (subcarrier indices, CRS values) for one port/symbol."""
    k0 = crs_shift(port, l, ns, n_id, cp_ext)
    m = np.arange(2 * n_rb)
    k = 6 * m + k0
    r = crs_seq(ns, l, n_id, cp_ext)[m + N_RB_MAX - n_rb]
    return k, r
