"""Channel coding of TS 36.212: CRC, tail-biting convolutional code with
Viterbi decoder, sub-block interleaver and convolutional rate matching.

LLR convention used throughout: llr = log(P(b=0)/P(b=1)), i.e. positive
means bit 0 (BPSK symbol +1)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._jit import HAVE_NUMBA, viterbi_core

# ----------------------------------------------------------------------- bits


def int_to_bits(v: int, n: int) -> np.ndarray:
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bits_to_int(b) -> int:
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def bits_to_bytes(b) -> bytes:
    b = np.asarray(b, dtype=np.uint8)
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, np.uint8)])
    return np.packbits(b).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# ----------------------------------------------------------------------- CRC
CRC24A = (0x864CFB, 24)
CRC24B = (0x800063, 24)
CRC16 = (0x1021, 16)
CRC8 = (0x9B, 8)


@lru_cache(maxsize=8)
def _crc_table(poly: int, n: int) -> tuple[int, ...]:
    tab = []
    top = 1 << (n - 1)
    mask = (1 << n) - 1
    for byte in range(256):
        r = byte << (n - 8)
        for _ in range(8):
            r = ((r << 1) ^ poly) & mask if r & top else (r << 1) & mask
        tab.append(r)
    return tuple(tab)


def crc(bits, spec=CRC24A) -> int:
    """CRC remainder of a bit sequence (MSB first, zero init)."""
    poly, n = spec
    bits = np.asarray(bits, dtype=np.uint8)
    mask = (1 << n) - 1
    top = 1 << (n - 1)
    r = 0
    nbytes = len(bits) // 8
    if nbytes:
        tab = _crc_table(poly, n)
        for byte in np.packbits(bits[:nbytes * 8]):
            r = ((r << 8) & mask) ^ tab[((r >> (n - 8)) ^ int(byte)) & 0xFF]
    for b in bits[nbytes * 8:]:
        fb = ((r & top) != 0) ^ int(b)
        r = (r << 1) & mask
        if fb:
            r ^= poly
    return r


def crc_attach(bits, spec=CRC24A, mask: int = 0) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.concatenate([bits, int_to_bits(crc(bits, spec) ^ mask, spec[1])])


def crc_check(bits, spec=CRC24A) -> tuple[bool, int]:
    """Return (crc ok?, residual = received_crc XOR computed_crc)."""
    n = spec[1]
    bits = np.asarray(bits, dtype=np.uint8)
    res = crc(bits[:-n], spec) ^ bits_to_int(bits[-n:])
    return res == 0, res


# ---------------------------------------------------- convolutional code
CONV_POLY = (0o133, 0o171, 0o165)
_K = 7


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


@lru_cache(maxsize=1)
def _trellis():
    # state = c(k-1)<<5 | ... | c(k-6); register R = u<<6 | state
    n_st = 64
    pred = np.zeros((n_st, 2), dtype=np.int64)
    out = np.zeros((n_st, 2, 3), dtype=np.float64)   # +-1 symbols
    for sp in range(n_st):
        u = sp >> 5
        for b in range(2):
            s = ((sp & 31) << 1) | b
            pred[sp, b] = s
            reg = (u << 6) | s
            for i, g in enumerate(CONV_POLY):
                out[sp, b, i] = 1 - 2 * _parity(reg & g)
    return pred, out


def conv_encode(bits) -> np.ndarray:
    """Tail-biting rate-1/3 convolutional encoder. Returns (3, K) bits."""
    c = np.asarray(bits, dtype=np.uint8)
    k = len(c)
    # tail-biting: register starts with the last 6 input bits;
    # state holds c(k-1) as MSB ... c(k-6) as LSB
    s = 0
    for j in range(6):
        s |= int(c[k - 1 - j]) << (5 - j)
    d = np.zeros((3, k), dtype=np.uint8)
    for t in range(k):
        reg = (int(c[t]) << 6) | s
        for i, g in enumerate(CONV_POLY):
            d[i, t] = _parity(reg & g)
        s = (int(c[t]) << 5) | (s >> 1)
    return d


def viterbi_tb(llr: np.ndarray, wrap: int = 2) -> np.ndarray:
    """Tail-biting Viterbi decoder (wrap-around variant).
    llr: shape (3, K) soft values, positive = bit 0."""
    llr = np.asarray(llr, dtype=np.float64)
    k = llr.shape[1]
    ext = np.concatenate([llr] * (2 * wrap + 1), axis=1)
    pred, out = _trellis()
    if HAVE_NUMBA:
        return viterbi_core(np.ascontiguousarray(ext), pred, out)[wrap * k:(wrap + 1) * k]
    n = ext.shape[1]
    metric = np.zeros(64)
    dec = np.zeros((n, 64), dtype=np.uint8)
    # branch metric for all (sp,b): sum_i out[sp,b,i]*llr[i,t]
    bm_all = np.einsum("sbi,it->tsb", out, ext)
    for t in range(n):
        cand = metric[pred] + bm_all[t]          # (64, 2)
        choice = np.argmax(cand, axis=1)
        dec[t] = choice
        metric = cand[np.arange(64), choice]
        metric -= metric.max()
    s = int(np.argmax(metric))
    bits = np.zeros(n, dtype=np.uint8)
    for t in range(n - 1, -1, -1):
        bits[t] = s >> 5
        s = int(pred[s, dec[t, s]])
    return bits[wrap * k:(wrap + 1) * k]


# ------------------------------------------------- sub-block interleaver
PERM_CONV = (1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
             0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30)
PERM_TURBO = (0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
              1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31)


@lru_cache(maxsize=512)
def subblock_perm(d: int, perm: tuple = PERM_CONV) -> np.ndarray:
    """Output sequence of the sub-block interleaver as input indices
    (-1 = <NULL> dummy). Length 32*R."""
    r = -(-d // 32)
    nd = 32 * r - d
    y = np.concatenate([-np.ones(nd, dtype=np.int64), np.arange(d)])
    m = y.reshape(r, 32)[:, list(perm)]
    v = m.T.reshape(-1)
    v.setflags(write=False)
    return v


@lru_cache(maxsize=512)
def conv_rm_index(k: int, e: int) -> np.ndarray:
    """For conv rate matching: index into the flattened (3,K) coded block
    for each of the E output bits."""
    v = subblock_perm(k, PERM_CONV)
    w = np.concatenate([np.where(v >= 0, v + i * k, -1) for i in range(3)])
    w = w[w >= 0]
    idx = w[np.arange(e) % len(w)]
    idx.setflags(write=False)
    return idx


def conv_rate_match(d: np.ndarray, e: int) -> np.ndarray:
    return d.reshape(-1)[conv_rm_index(d.shape[1], e)]


def conv_rate_dematch(llr: np.ndarray, k: int) -> np.ndarray:
    idx = conv_rm_index(k, len(llr))
    acc = np.zeros(3 * k)
    np.add.at(acc, idx, llr)
    return acc.reshape(3, k)


# ----------------------------------------------------------------- QPSK


def qpsk_mod(bits) -> np.ndarray:
    b = np.asarray(bits, dtype=np.float64)
    return ((1 - 2 * b[0::2]) + 1j * (1 - 2 * b[1::2])) / np.sqrt(2)


def qpsk_llr(sym: np.ndarray, gain=None) -> np.ndarray:
    """Soft bits from equalised QPSK symbols. `gain` = per-symbol
    reliability (|h|^2/noise), optional."""
    s = np.asarray(sym)
    g = 1.0 if gain is None else gain
    out = np.empty(2 * len(s))
    out[0::2] = s.real * g
    out[1::2] = s.imag * g
    return out


def scramble_llr(llr: np.ndarray, c: np.ndarray) -> np.ndarray:
    return llr * (1 - 2 * c[:len(llr)].astype(np.float64))
