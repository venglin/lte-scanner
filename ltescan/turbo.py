"""LTE turbo code (TS 36.212 5.1.3.2) with max-log-MAP decoder, plus
turbo rate matching (5.1.4.1) and DL-SCH transport block processing for the
single code-block case (TBS + 24 <= 6144 bits, i.e. all SI/paging/RAR)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .coding import CRC24A, PERM_TURBO, crc_attach, crc_check, subblock_perm

# Table 5.1.3-3: code block sizes and QPP parameters
TURBO_K = (list(range(40, 513, 8)) + list(range(528, 1025, 16))
           + list(range(1056, 2049, 32)) + list(range(2112, 6145, 64)))
_F1 = [3, 7, 19, 7, 7, 11, 5, 11, 7, 41, 103, 15, 9, 17, 9, 21, 101, 21, 57, 23, 13, 27, 11, 27, 85, 29, 33, 15, 17, 33, 103, 19, 19, 37, 19, 21, 21, 115, 193, 21, 133, 81, 45, 23, 243, 151, 155, 25, 51, 47, 91, 29, 29, 247, 29, 89, 91, 157, 55, 31, 17, 35, 227, 65, 19, 37, 41, 39, 185, 43, 21, 155, 79, 139, 23, 217, 25, 17, 127, 25, 239, 17, 137, 215, 29, 15, 147, 29, 59, 65, 55, 31, 17, 171, 67, 35, 19, 39, 19, 199, 21, 211, 21, 43, 149, 45, 49, 71, 13, 17, 25, 183, 55, 127, 27, 29, 29, 57, 45, 31, 59, 185, 113, 31, 17, 171, 209, 253, 367, 265, 181, 39, 27, 127, 143, 43, 29, 45, 157, 47, 13, 111, 443, 51, 51, 451, 257, 57, 313, 271, 179, 331, 363, 375, 127, 31, 33, 43, 33, 477, 35, 233, 357, 337, 37, 71, 71, 37, 39, 127, 39, 39, 31, 113, 41, 251, 43, 21, 43, 45, 45, 161, 89, 323, 47, 23, 47, 263]  # noqa: E501
_F2 = [10, 12, 42, 16, 18, 20, 22, 24, 26, 84, 90, 32, 34, 108, 38, 120, 84, 44, 46, 48, 50, 52, 36, 56, 58, 60, 62, 32, 198, 68, 210, 36, 74, 76, 78, 120, 82, 84, 86, 44, 90, 46, 94, 48, 98, 40, 102, 52, 106, 72, 110, 168, 114, 58, 118, 180, 122, 62, 84, 64, 66, 68, 420, 96, 74, 76, 234, 80, 82, 252, 86, 44, 120, 92, 94, 48, 98, 80, 102, 52, 106, 48, 110, 112, 114, 58, 118, 60, 122, 124, 84, 64, 66, 204, 140, 72, 74, 76, 78, 240, 82, 252, 86, 88, 60, 92, 846, 48, 28, 80, 102, 104, 954, 96, 110, 112, 114, 116, 354, 120, 610, 124, 420, 64, 66, 136, 420, 216, 444, 456, 468, 80, 164, 504, 172, 88, 300, 92, 188, 96, 28, 240, 204, 104, 212, 192, 220, 336, 228, 232, 236, 120, 244, 248, 168, 64, 130, 264, 134, 408, 138, 280, 142, 480, 146, 444, 120, 152, 462, 234, 158, 80, 96, 902, 166, 336, 170, 86, 174, 176, 178, 120, 182, 184, 186, 94, 190, 480]  # noqa: E501
QPP = {k: (f1, f2) for k, f1, f2 in zip(TURBO_K, _F1, _F2)}
assert len(TURBO_K) == len(_F1) == len(_F2) == 188


@lru_cache(maxsize=256)
def qpp(k: int) -> np.ndarray:
    f1, f2 = QPP[k]
    i = np.arange(k, dtype=np.int64)
    p = (f1 * i + f2 * i * i) % k
    p.setflags(write=False)
    return p


def code_block_size(b: int) -> int:
    for k in TURBO_K:
        if k >= b:
            return k
    raise ValueError(f"a {b}-bit block needs code block segmentation (not supported)")


# ------------------------------------------------------------- trellis
def _build_trellis():
    nxt = np.zeros((8, 2), dtype=np.int64)
    zb = np.zeros((8, 2), dtype=np.int64)
    for s in range(8):
        s1, s2, s3 = (s >> 2) & 1, (s >> 1) & 1, s & 1
        for u in range(2):
            a = u ^ s2 ^ s3
            zb[s, u] = a ^ s1 ^ s3
            nxt[s, u] = (a << 2) | (s1 << 1) | s2
    # predecessors of every state: (state, input) pairs
    prv = [[] for _ in range(8)]
    for s in range(8):
        for u in range(2):
            prv[nxt[s, u]].append((s, u))
    prv_s = np.array([[p[0] for p in q] for q in prv])
    prv_u = np.array([[p[1] for p in q] for q in prv])
    return nxt, zb, prv_s, prv_u


_NXT, _ZB, _PRV_S, _PRV_U = _build_trellis()
_XZ = 1.0 - 2.0 * _ZB                         # parity symbol per (s,u)
_TAIL_U = np.array([((s >> 1) & 1) ^ (s & 1) for s in range(8)])
_TAIL_Z = np.array([((s >> 2) & 1) ^ (s & 1) for s in range(8)])  # a=0


def rsc_encode(c: np.ndarray):
    """One constituent encoder. Returns (z[K], tail_x[3], tail_z[3])."""
    s = 0
    z = np.zeros(len(c), dtype=np.uint8)
    for k, u in enumerate(c):
        z[k] = _ZB[s, u]
        s = _NXT[s, u]
    tx, tz = [], []
    for _ in range(3):
        u = _TAIL_U[s]
        tx.append(u)
        tz.append(_ZB[s, u])
        s = _NXT[s, u]
    assert s == 0
    return z, np.array(tx, np.uint8), np.array(tz, np.uint8)


def turbo_encode(c: np.ndarray) -> np.ndarray:
    """Returns d of shape (3, K+4)."""
    c = np.asarray(c, dtype=np.uint8)
    k = len(c)
    z1, x1t, z1t = rsc_encode(c)
    z2, x2t, z2t = rsc_encode(c[qpp(k)])
    d = np.zeros((3, k + 4), dtype=np.uint8)
    d[0, :k], d[1, :k], d[2, :k] = c, z1, z2
    d[0, k:] = [x1t[0], z1t[1], x2t[0], z2t[1]]
    d[1, k:] = [z1t[0], x1t[2], z2t[0], x2t[2]]
    d[2, k:] = [x1t[1], z1t[2], x2t[1], z2t[2]]
    return d


NEG = -1e30


def _map_decode(ls, lp, la, tx, tz):
    """Max-log-MAP for one RSC. ls, lp, la: length K; tx, tz: tail LLRs.
    Returns a-posteriori LLR (length K)."""
    k = len(ls)
    lsa = ls + la
    # gamma[t, s, u]
    g = 0.5 * (lsa[:, None, None] * np.array([1.0, -1.0])[None, None, :]
               + lp[:, None, None] * _XZ[None, :, :])
    gt = 0.5 * (tx[:, None] * (1 - 2 * _TAIL_U)[None, :]
                + tz[:, None] * (1 - 2 * _TAIL_Z)[None, :])     # (3, 8)
    alpha = np.empty((k + 1, 8))
    alpha[0] = NEG
    alpha[0, 0] = 0.0
    for t in range(k):
        a = alpha[t][_PRV_S] + g[t][_PRV_S, _PRV_U]               # (8, 2)
        m = a.max(axis=1)
        alpha[t + 1] = m - m.max()
    beta = np.full(8, NEG)
    beta[0] = 0.0
    for t in range(2, -1, -1):                                    # tail
        nb = np.full(8, NEG)
        nb = beta[np.arange(8) >> 1] + gt[t]
        beta = nb - nb.max()
    out = np.empty(k)
    for t in range(k - 1, -1, -1):
        m0 = alpha[t] + g[t, :, 0] + beta[_NXT[:, 0]]
        m1 = alpha[t] + g[t, :, 1] + beta[_NXT[:, 1]]
        out[t] = m0.max() - m1.max()
        nb = np.maximum(m0, m1) - alpha[t]
        beta = nb - nb.max()
    return out


def turbo_decode(d_llr: np.ndarray, n_filler: int = 0, iters: int = 6,
                 check=None):
    """d_llr: (3, K+4) LLRs. Returns hard bits (K). If `check` callable is
    given, it is called with the bits after each iteration and decoding
    stops early once it returns True."""
    k = d_llr.shape[1] - 4
    p = qpp(k)
    ls = d_llr[0, :k].astype(np.float64).copy()
    if n_filler:
        ls[:n_filler] = 50.0
    lp1, lp2 = d_llr[1, :k], d_llr[2, :k]
    t = d_llr[:, k:]
    tx1 = np.array([t[0, 0], t[2, 0], t[1, 1]])
    tz1 = np.array([t[1, 0], t[0, 1], t[2, 1]])
    tx2 = np.array([t[0, 2], t[2, 2], t[1, 3]])
    tz2 = np.array([t[1, 2], t[0, 3], t[2, 3]])
    la1 = np.zeros(k)
    bits = (ls < 0).astype(np.uint8)
    for _ in range(iters):
        post1 = _map_decode(ls, lp1, la1, tx1, tz1)
        le1 = post1 - ls - la1
        la2 = le1[p]
        post2 = _map_decode(ls[p], lp2, la2, tx2, tz2)
        le2 = post2 - ls[p] - la2
        la1 = np.empty(k)
        la1[p] = le2
        post = np.empty(k)
        post[p] = post2
        bits = (post < 0).astype(np.uint8)
        if check is not None and check(bits):
            break
    return bits


# ---------------------------------------------------- rate matching
@lru_cache(maxsize=512)
def _turbo_w(k: int, n_filler: int) -> np.ndarray:
    """Circular buffer as indices into flattened (3, K+4) block (-1=NULL)."""
    d = k + 4
    r = -(-d // 32)
    kp = 32 * r
    nd = kp - d
    v0 = np.array(subblock_perm(d, PERM_TURBO))
    v1 = v0.copy()
    v0 = np.where((v0 >= 0) & (v0 >= n_filler), v0, -1)
    v1 = np.where((v1 >= 0) & (v1 >= n_filler), v1 + d, -1)
    kk = np.arange(kp)
    pi = (np.array(PERM_TURBO)[kk // r] + 32 * (kk % r) + 1) % kp
    v2 = np.where(pi - nd >= 0, pi - nd + 2 * d, -1)
    w = np.empty(3 * kp, dtype=np.int64)
    w[:kp] = v0
    w[kp::2] = v1
    w[kp + 1::2] = v2
    w.setflags(write=False)
    return w


def turbo_rm_index(k: int, n_filler: int, e: int, rv: int) -> np.ndarray:
    w = _turbo_w(k, n_filler)
    ncb = len(w)
    r = ncb // 96
    k0 = r * (2 * (-(-ncb // (8 * r))) * rv + 2)
    order = np.roll(w, -k0)
    valid = order[order >= 0]
    return valid[np.arange(e) % len(valid)]


def tb_encode(tb_bits: np.ndarray, e: int, rv: int) -> np.ndarray:
    b = crc_attach(tb_bits, CRC24A)
    k = code_block_size(len(b))
    f = k - len(b)
    c = np.concatenate([np.zeros(f, np.uint8), b])
    d = turbo_encode(c)
    return d.reshape(-1)[turbo_rm_index(k, f, e, rv)]


class TbSoftBuffer:
    """Accumulates soft bits of (re)transmissions of one transport block
    (HARQ chase/IR combining in the circular buffer)."""

    def __init__(self, tbs: int):
        self.tbs = tbs
        self.k = code_block_size(tbs + 24)
        self.f = self.k - tbs - 24
        self.acc = np.zeros(3 * (self.k + 4))
        self.n = 0

    def add(self, llr: np.ndarray, rv: int):
        idx = turbo_rm_index(self.k, self.f, len(llr), rv)
        np.add.at(self.acc, idx, llr)
        self.n += 1

    def decode(self, iters: int = 8):
        """Returns TB bits or None if CRC fails."""
        d = self.acc.reshape(3, self.k + 4)
        f = self.f

        def ok(bits):
            return crc_check(bits[f:], CRC24A)[0]
        bits = turbo_decode(d, f, iters, check=ok)
        if ok(bits):
            return bits[f:f + self.tbs]
        return None


def tb_decode(llr: np.ndarray, tbs: int, rv: int, iters: int = 8):
    sb = TbSoftBuffer(tbs)
    sb.add(llr, rv)
    return sb.decode(iters)
