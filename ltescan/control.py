"""Downlink control region: REG layout, PCFICH, PHICH REG allocation and
PDCCH (CCE de-interleaving, blind decoding). TS 36.211 6.2.4, 6.7-6.8,
TS 36.212 5.3.3-5.3.4."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import coding
from .mimo import td_decode
from .params import PHICH_NG, CellConfig
from .sequences import gold

# --------------------------------------------------------------- REGs


def n_ctrl_symbols(cfg: CellConfig, cfi: int) -> int:
    return cfi + 1 if cfg.n_rb <= 10 else cfi


def _reg_has_crs(cfg: CellConfig, l: int) -> bool:
    if l == 0:
        return True
    if l == 1:
        return cfg.n_ports == 4
    if l == 3:
        return cfg.cp_ext
    return False


def reg_res(cfg: CellConfig, l: int, k0: int) -> np.ndarray:
    """Subcarriers carrying data of the REG starting at k0 in symbol l."""
    if _reg_has_crs(cfg, l):
        k = np.arange(k0, k0 + 6)
        return k[k % 3 != cfg.n_id % 3]
    return np.arange(k0, k0 + 4)


def reg_starts(cfg: CellConfig, l: int) -> np.ndarray:
    step = 6 if _reg_has_crs(cfg, l) else 4
    return np.arange(0, cfg.n_sc, step)


# ------------------------------------------------------------- PCFICH
CFI_CODE = {
    1: np.array(([0, 1, 1] * 11)[:32], dtype=np.uint8),
    2: np.array(([1, 0, 1] * 11)[:32], dtype=np.uint8),
    3: np.array(([1, 1, 0] * 11)[:32], dtype=np.uint8),
}


@lru_cache(maxsize=256)
def pcfich_regs(cfg: CellConfig) -> tuple[int, ...]:
    kbar = 6 * (cfg.n_id % (2 * cfg.n_rb))
    return tuple((kbar + (i * cfg.n_rb // 2) * 6) % cfg.n_sc for i in range(4))


def pcfich_scrambling(cfg: CellConfig, sf: int) -> np.ndarray:
    c_init = (sf + 1) * (2 * cfg.n_id + 1) * (1 << 9) + cfg.n_id
    return gold(c_init, 32)


@lru_cache(maxsize=1024)
def reg_index(cfg: CellConfig, regs: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(symbol, subcarrier) arrays of the data REs of a tuple of REGs, in
    order (4 REs per REG). Cached: the layout only depends on the config."""
    ls, ks = [], []
    for l, k0 in regs:
        k = reg_res(cfg, l, k0)
        ls.append(np.full(len(k), l))
        ks.append(k)
    ls = np.concatenate(ls)
    ks = np.concatenate(ks)
    ls.setflags(write=False)
    ks.setflags(write=False)
    return ls, ks


def _extract(grid, h, cfg, regs):
    """Gather REs (and channel) of a list of (l, k0) REGs."""
    ls, ks = reg_index(cfg, tuple(regs))
    return grid[ls, ks], h[:cfg.n_ports, ls, ks]


def decode_pcfich(grid, h, noise, cfg: CellConfig, sf: int):
    """Returns (cfi, confidence)."""
    regs = [(0, k0) for k0 in pcfich_regs(cfg)]
    r, hh = _extract(grid, h, cfg, regs)
    x, g = td_decode(r, hh, cfg.n_ports)
    llr = coding.qpsk_llr(x, g / max(noise, 1e-12))
    llr = coding.scramble_llr(llr, pcfich_scrambling(cfg, sf))
    scores = {c: float(np.dot(llr, 1 - 2 * code.astype(float))) for c, code in CFI_CODE.items()}
    best = max(scores, key=scores.get)
    tot = sum(abs(v) for v in scores.values()) + 1e-12
    return best, scores[best] / tot


# --------------------------------------------------------------- PHICH


def phich_groups(cfg: CellConfig, m_i: int = 1) -> int:
    ng = PHICH_NG[cfg.phich_ng]
    n = int(np.ceil(ng * cfg.n_rb / 8))
    n = 2 * n if cfg.cp_ext else n
    return n * m_i


@lru_cache(maxsize=256)
def phich_regs(cfg: CellConfig, m_i: int = 1) -> tuple[tuple[int, int], ...]:
    """(l, k0) of all REGs used by PHICH."""
    n_groups = phich_groups(cfg, m_i)
    if n_groups == 0:
        return ()
    pcf = set(pcfich_regs(cfg))
    n_sym = 3 if cfg.phich_ext else 1
    free = []
    for l in range(max(n_sym, 1)):
        st = reg_starts(cfg, l)
        if l == 0:
            st = [k for k in st if k not in pcf]
        free.append(list(st))
    n_l = [len(f) for f in free]
    n_mp = n_groups // 2 if cfg.cp_ext else n_groups
    out = []
    for mp in range(n_mp):
        for i in range(3):
            li = i if cfg.phich_ext else 0
            den = n_l[1] if cfg.phich_ext else n_l[0]
            ni = (cfg.n_id * n_l[li] // den + mp + i * n_l[li] // 3) % n_l[li]
            out.append((li, free[li][ni]))
    return tuple(out)


# --------------------------------------------------------------- PDCCH


@lru_cache(maxsize=512)
def pdcch_regs(cfg: CellConfig, cfi: int, m_i: int = 1) -> tuple[tuple[int, int], ...]:
    """PDCCH REGs (l, k0) in the order of quadruplet mapping."""
    n_ctrl = n_ctrl_symbols(cfg, cfi)
    used = {(0, k) for k in pcfich_regs(cfg)} | set(phich_regs(cfg, m_i))
    starts = [set(reg_starts(cfg, l).tolist()) for l in range(n_ctrl)]
    out = []
    for k in range(cfg.n_sc):
        for l in range(n_ctrl):
            if k in starts[l] and (l, k) not in used:
                out.append((l, k))
    return tuple(out)


@lru_cache(maxsize=512)
def quad_permutation(n_reg: int, n_id: int) -> np.ndarray:
    """perm[i] = index of quadruplet w(.) carried by REG i."""
    v = coding.subblock_perm(n_reg, coding.PERM_CONV)
    v = v[v >= 0]
    i = np.arange(n_reg)
    p = v[(i + n_id) % n_reg]
    p.setflags(write=False)
    return p


def pdcch_scrambling(cfg: CellConfig, sf: int, n_bits: int) -> np.ndarray:
    return gold(sf * (1 << 9) + cfg.n_id, n_bits)


def pdcch_llr(grid, h, noise, cfg: CellConfig, sf: int, cfi: int, m_i: int = 1):
    """Soft bits of all CCEs of a subframe, shape (n_cce, 72), and the
    REG list (for statistics)."""
    regs = pdcch_regs(cfg, cfi, m_i)
    n_reg = len(regs)
    r, hh = _extract(grid, h, cfg, regs)
    x, g = td_decode(r, hh, cfg.n_ports)
    quads = x.reshape(n_reg, 4)
    gains = g.reshape(n_reg, 4)
    perm = quad_permutation(n_reg, cfg.n_id)
    w = np.empty_like(quads)
    wg = np.empty_like(gains)
    w[perm] = quads
    wg[perm] = gains
    llr = coding.qpsk_llr(w.reshape(-1), wg.reshape(-1) / max(noise, 1e-12))
    llr = coding.scramble_llr(llr, pdcch_scrambling(cfg, sf, len(llr)))
    n_cce = n_reg // 9
    return llr[:n_cce * 72].reshape(n_cce, 72), regs


def cce_energy(cce_llr: np.ndarray) -> np.ndarray:
    """Mean |LLR| per CCE (crude 'is this CCE in use' indicator)."""
    return np.mean(np.abs(cce_llr), axis=1)


@dataclass
class DciCandidate:
    cce: int
    agg: int
    size: int
    rnti: int
    bits: np.ndarray
    reliability: float      # fraction of re-encoded bits agreeing with rx


def decode_candidate(cce_llr: np.ndarray, cce: int, agg: int, size: int) -> DciCandidate:
    llr = cce_llr[cce:cce + agg].reshape(-1)
    k = size + 16
    d = coding.conv_rate_dematch(llr, k)
    bits = coding.viterbi_tb(d)
    _, res = coding.crc_check(bits, coding.CRC16)
    # reliability: re-encode and measure the LLR-weighted disagreement with
    # the received soft bits. A genuine candidate only disagrees on weak
    # bits; a misaligned/noise candidate disagrees on ~10 % of strong ones.
    enc = coding.conv_rate_match(coding.conv_encode(bits), len(llr))
    mag = np.abs(llr)
    tot = mag.sum()
    if tot <= 0:
        rel = 0.0
    else:
        wrong = (llr < 0) != (enc == 1)
        rel = float(1 - mag[wrong].sum() / tot)
    return DciCandidate(cce, agg, size, res, bits[:size], rel)


def common_search_space(n_cce: int):
    """(cce, aggregation) candidates of the common search space."""
    out = []
    n = min(n_cce, 16)
    for agg, m_count in ((4, 4), (8, 2)):
        n_l = n // agg
        for m in range(min(m_count, n_l)):
            out.append((agg * m, agg))
    return out


def ue_search_space(n_cce: int, rnti: int, sf: int):
    """UE-specific search space candidates (TS 36.213 9.1.1)."""
    y = rnti
    for _ in range(sf + 1):
        y = (39827 * y) % 65537
    out = []
    for agg, m_count in ((1, 6), (2, 6), (4, 2), (8, 2)):
        n_l = n_cce // agg
        if n_l == 0:
            continue
        for m in range(m_count):
            out.append((agg * ((y + m) % n_l), agg))
    return out
