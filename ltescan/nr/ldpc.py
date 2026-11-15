"""LDPC coding of the NR shared channels (TS 38.212 5.2.2, 5.3.2, 5.4.2):
transport block CRC, base graph selection, code block segmentation,
quasi-cyclic encoder, rate matching with redundancy versions and a
layered normalised min-sum decoder.

LLR convention as elsewhere: positive means bit 0."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..coding import CRC16, CRC24A, CRC24B, crc
from .ldpc_tables import BG1, BG2

BIG = 1e4                      # LLR of known (filler) bits
# 38.212 Table 5.3.2-1: lifting sizes Z = a * 2^j, set index i_LS by a
_LS_A = (2, 3, 5, 7, 9, 11, 13, 15)
LIFTING = sorted((a << j, i) for i, a in enumerate(_LS_A) for j in range(8) if a << j <= 384)


def select_bg(tbs: int, rate: float) -> int:
    """38.212 7.2.2 (LDPC base graph selection)."""
    if tbs <= 292 or (tbs <= 3824 and rate <= 0.67) or rate <= 0.25:
        return 2
    return 1


@dataclass(frozen=True)
class Segmentation:
    bg: int
    tbs: int
    tb_crc: tuple             # (poly, length)
    c: int                    # code blocks
    k_prime: int              # bits per block incl. CB CRC (without filler)
    k: int                    # bits per block incl. filler
    z: int
    cb_crc: int               # 0 or 24

    @property
    def n(self) -> int:
        """Encoded bits per block after removing the first 2Z (= N_cb here)."""
        return (66 if self.bg == 1 else 50) * self.z

    @property
    def i_ls(self) -> int:
        return next(i for z, i in LIFTING if z == self.z)


def segment(tbs: int, bg: int) -> Segmentation:
    """38.212 5.2.2 code block segmentation for a transport block."""
    tb_crc = CRC24A if tbs > 3824 else CRC16
    b = tbs + tb_crc[1]
    kcb = 8448 if bg == 1 else 3840
    if b <= kcb:
        lcb, c = 0, 1
    else:
        lcb = 24
        c = -(-b // (kcb - lcb))
    k_prime = (b + c * lcb) // c
    if bg == 1:
        kb = 22
    else:
        kb = 10 if b > 640 else 9 if b > 560 else 8 if b > 192 else 6
    z = next(z for z, _ in LIFTING if kb * z >= k_prime)
    k = (22 if bg == 1 else 10) * z
    return Segmentation(bg, tbs, tb_crc, c, k_prime, k, z, lcb)


# ------------------------------------------------------------ base graph
@lru_cache(maxsize=64)
def graph(bg: int, z: int):
    """Rows of the lifted parity-check matrix: list of (columns, shifts)
    with shift = V mod Z (38.212 5.3.2), plus (n_rows, n_cols, k_cols)."""
    i_ls = next(i for zz, i in LIFTING if zz == z)
    table = BG1 if bg == 1 else BG2
    n_rows, n_cols, k_cols = (46, 68, 22) if bg == 1 else (42, 52, 10)
    rows: list[list] = [[] for _ in range(n_rows)]
    for r, c, v in table:
        rows[r].append((c, v[i_ls] % z))
    out = []
    for r in rows:
        cols = np.array([c for c, _ in r], dtype=np.int64)
        sh = np.array([s for _, s in r], dtype=np.int64)
        out.append((cols, sh))
    return out, n_rows, n_cols, k_cols


def _gf2_inv(m: np.ndarray) -> np.ndarray:
    """Inverse of a binary matrix (Gauss-Jordan on bit-packed rows)."""
    n = m.shape[0]
    aug = np.packbits(np.concatenate([m.astype(np.uint8), np.eye(n, dtype=np.uint8)], axis=1),
                      axis=1)
    for col in range(n):
        byte, bit = col >> 3, 7 - (col & 7)
        colv = (aug[:, byte] >> bit) & 1
        cand = np.nonzero(colv[col:])[0]
        if len(cand) == 0:
            raise ValueError("singular LDPC core matrix")
        p = col + int(cand[0])
        if p != col:
            aug[[col, p]] = aug[[p, col]]
            colv[[col, p]] = colv[[p, col]]
        rows = np.nonzero(colv)[0]
        rows = rows[rows != col]
        aug[rows] ^= aug[col]
    return np.unpackbits(aug, axis=1)[:, n:2 * n]


@lru_cache(maxsize=16)
def _core_inverse(bg: int, z: int) -> np.ndarray:
    rows, _, _, kc = graph(bg, z)
    m = np.zeros((4 * z, 4 * z), dtype=np.uint8)
    r = np.arange(z)
    for i in range(4):
        for c, s in zip(*rows[i]):
            if c >= kc:
                m[i * z + r, (c - kc) * z + (r + s) % z] ^= 1
    return _gf2_inv(m)


def encode(c: np.ndarray, bg: int, z: int) -> np.ndarray:
    """K = k_cols*Z information bits (filler bits as 0) -> the full
    codeword of n_cols*Z bits (38.212 5.3.2 before removing the first 2Z)."""
    rows, n_rows, n_cols, kc = graph(bg, z)
    v = np.zeros((n_cols, z), dtype=np.uint8)
    v[:kc] = np.asarray(c, dtype=np.uint8).reshape(kc, z)
    idx = np.arange(z)
    lam = np.zeros((4, z), dtype=np.uint8)
    for i in range(4):
        for col, s in zip(*rows[i]):
            if col < kc:
                lam[i] ^= v[col][(idx + s) % z]
    p = (_core_inverse(bg, z).astype(np.int64) @ lam.reshape(-1)) & 1
    v[kc:kc + 4] = p.reshape(4, z)
    for i in range(4, n_rows):
        acc = np.zeros(z, dtype=np.uint8)
        for col, s in zip(*rows[i]):
            if col != kc + i:
                acc ^= v[col][(idx + s) % z]
        v[kc + i] = acc
    return v.reshape(-1)


def syndrome_ok(cw: np.ndarray, bg: int, z: int) -> bool:
    rows, _, n_cols, _ = graph(bg, z)
    v = np.asarray(cw, dtype=np.uint8).reshape(n_cols, z)
    idx = np.arange(z)
    for cols, sh in rows:
        acc = np.zeros(z, dtype=np.uint8)
        for col, s in zip(cols, sh):
            acc ^= v[col][(idx + s) % z]
        if acc.any():
            return False
    return True


def decode(llr: np.ndarray, bg: int, z: int, max_iter: int = 30, alpha: float = 0.75,
           n_rows: int | None = None):
    """Layered normalised min-sum decoding. llr: n_cols*Z soft bits of the
    full codeword (0 for punctured/untransmitted, +BIG for filler). Only
    the first n_rows check rows are used (rows whose extension parity bit
    was never transmitted carry no information). Returns (hard bits of the
    full codeword, parity checks satisfied)."""
    rows, nr, n_cols, kc = graph(bg, z)
    n_rows = nr if n_rows is None else max(4, min(n_rows, nr))
    rows = rows[:n_rows]
    L = np.asarray(llr, dtype=np.float64).reshape(n_cols, z).copy()
    idx = np.arange(z)
    gidx = [(cols[:, None], (idx[None, :] + sh[:, None]) % z) for cols, sh in rows]
    R = [np.zeros((len(cols), z)) for cols, _ in rows]
    ok = False
    for _ in range(max_iter):
        for i, (ci, ki) in enumerate(gidx):
            q = L[ci, ki] - R[i]
            sgn = np.where(q < 0, -1.0, 1.0)
            a = np.abs(q)
            j = np.argmin(a, axis=0)
            m1 = a[j, idx]
            a[j, idx] = np.inf
            m2 = a.min(axis=0)
            mag = np.where(np.arange(len(q))[:, None] == j[None, :], m2[None, :], m1[None, :])
            rn = alpha * np.prod(sgn, axis=0)[None, :] * sgn * mag
            L[ci, ki] = q + rn
            R[i] = rn
        hard = (L < 0).astype(np.uint8)
        ok = all(not np.bitwise_xor.reduce(hard[ci, ki], axis=0).any() for ci, ki in gidx)
        if ok:
            break
    return (L < 0).astype(np.uint8).reshape(-1), ok


# ---------------------------------------------------------- rate matching
def k0(rv: int, bg: int, ncb: int, z: int) -> int:
    """38.212 Table 5.4.2.1-2."""
    num = {1: (0, 17, 33, 56), 2: (0, 13, 25, 43)}[bg][rv]
    den = 66 if bg == 1 else 50
    return (num * ncb // (den * z)) * z


def _filler_mask(seg: Segmentation) -> np.ndarray:
    """True for filler (NULL) positions of d (the codeword without 2Z)."""
    m = np.zeros(seg.n, dtype=bool)
    m[seg.k_prime - 2 * seg.z:seg.k - 2 * seg.z] = True
    return m


def _rm_positions(seg: Segmentation, rv: int, e: int) -> np.ndarray:
    """Indices into d of the E rate-matched bits (5.4.2.1 bit selection)."""
    ncb = seg.n
    filler = _filler_mask(seg)
    valid = np.nonzero(~filler)[0]
    start = k0(rv, seg.bg, ncb, seg.z)
    order = np.concatenate([valid[valid >= start], valid[valid < start]])
    reps = -(-e // len(order))
    return np.tile(order, reps)[:e]


def cb_sizes(seg: Segmentation, g: int, qm: int, n_layers: int = 1) -> list[int]:
    """E_r of every code block (5.4.2.1)."""
    unit = n_layers * qm
    q = g // unit
    out = []
    for r in range(seg.c):
        if r <= seg.c - (q % seg.c) - 1:
            out.append(unit * (q // seg.c))
        else:
            out.append(unit * -(-q // seg.c))
    return out


def _interleave(e: np.ndarray, qm: int) -> np.ndarray:
    """5.4.2.2: f_{i + j Qm} = e_{i E/Qm + j}."""
    return e.reshape(qm, -1).T.reshape(-1)


def _deinterleave(f: np.ndarray, qm: int) -> np.ndarray:
    return f.reshape(-1, qm).T.reshape(-1)


def _int_bits(v: int, n: int) -> np.ndarray:
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def dlsch_encode(tb: np.ndarray, rate: float, qm: int, g: int, rv: int = 0,
                 n_layers: int = 1) -> np.ndarray:
    """Transport block bits -> G coded and rate-matched bits."""
    tb = np.asarray(tb, dtype=np.uint8)
    seg = segment(len(tb), select_bg(len(tb), rate))
    b = np.concatenate([tb, _int_bits(crc(tb, seg.tb_crc), seg.tb_crc[1])])
    kp = seg.k_prime - seg.cb_crc
    out = []
    for r, e in enumerate(cb_sizes(seg, g, qm, n_layers)):
        blk = b[r * kp:(r + 1) * kp]
        if seg.cb_crc:
            blk = np.concatenate([blk, _int_bits(crc(blk, CRC24B), 24)])
        c = np.zeros(seg.k, dtype=np.uint8)
        c[:seg.k_prime] = blk
        d = encode(c, seg.bg, seg.z)[2 * seg.z:]
        out.append(_interleave(d[_rm_positions(seg, rv, e)], qm))
    return np.concatenate(out)


def dlsch_decode(rx: list[tuple[np.ndarray, int]], tbs: int, rate: float, qm: int,
                 n_layers: int = 1, max_iter: int = 30) -> np.ndarray | None:
    """Soft-combine and decode one transport block. rx: list of (G soft
    bits after descrambling, redundancy version) of transmissions of the
    same TB. Returns the TBS bits if the TB CRC passes, else None."""
    seg = segment(tbs, select_bg(tbs, rate))
    filler = _filler_mask(seg)
    rows, nr, n_cols, kc = graph(seg.bg, seg.z)
    bufs = [np.zeros(seg.n) for _ in range(seg.c)]
    for llr, rv in rx:
        sizes = cb_sizes(seg, len(llr), qm, n_layers)
        pos = 0
        for r, e in enumerate(sizes):
            np.add.at(bufs[r], _rm_positions(seg, rv, e), _deinterleave(llr[pos:pos + e], qm))
            pos += e
    kp = seg.k_prime - seg.cb_crc
    bits = []
    for r in range(seg.c):
        d = bufs[r]
        d[filler] = BIG
        full = np.concatenate([np.zeros(2 * seg.z), d])
        used = np.nonzero(d != 0)[0]
        last_col = (int(used.max()) + 2 * seg.z) // seg.z if len(used) else kc + 3
        n_rows = max(4, last_col - kc + 1)
        hard, ok = decode(full, seg.bg, seg.z, max_iter, n_rows=n_rows)
        if not ok:
            return None
        blk = hard[:seg.k_prime]
        if seg.cb_crc:
            if crc(blk, CRC24B) != 0:
                return None
        bits.append(blk[:kp])
    b = np.concatenate(bits)[:tbs + seg.tb_crc[1]]
    # all zeros passes any zero-initialised CRC: what a decoder outputs
    # without information, never accepted
    if crc(b, seg.tb_crc) != 0 or not b.any():
        return None
    return b[:tbs]
