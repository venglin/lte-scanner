"""PDSCH for broadcast (SI/P/RA-RNTI) transmissions: TBS, VRB->PRB mapping,
RE mapping, demodulation and DL-SCH decoding. TS 36.211 6.3-6.4, 6.2.3,
TS 36.213 7.1.6-7.1.7."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import coding
from .control import n_ctrl_symbols
from .dci import Dci, gap_values, n_vrb_dist, rbg_size
from .mimo import td_decode
from .params import CellConfig
from .sequences import crs_shift, crs_symbols, gold
from .turbo import TbSoftBuffer

# TS 36.213 Table 7.1.7.2.1-1, columns N_PRB = 2 and 3 (all that format 1A
# with SI/P/RA-RNTI can use) -- values from srsRAN tbs_tables.h
TBS_PRB2 = [32, 56, 72, 104, 120, 144, 176, 224, 256, 296, 328, 376, 440, 488, 552, 600,
            632, 696, 776, 840, 904, 1000, 1064, 1128, 1192, 1256, 1480]
TBS_PRB3 = [56, 88, 144, 176, 208, 224, 256, 328, 392, 456, 504, 584, 680, 744, 840, 904,
            968, 1064, 1160, 1288, 1384, 1480, 1608, 1736, 1800, 1864, 2216]
# TS 36.213 Table 7.1.7.2.3-1 (format 1C)
TBS_1C = [40, 56, 72, 120, 136, 144, 176, 208, 224, 256, 280, 296, 328, 336, 392, 488,
          552, 600, 632, 696, 776, 840, 904, 1000, 1064, 1128, 1224, 1288, 1384, 1480,
          1608, 1736]


def dci_tbs(dci: Dci) -> int | None:
    if dci.fmt == "1C":
        return TBS_1C[dci.mcs]
    if dci.fmt == "1A":
        if dci.mcs > 26:
            return None
        return (TBS_PRB3 if dci.n_prb_1a == 3 else TBS_PRB2)[dci.mcs]
    return None


# -------------------------------------------------- VRB -> PRB mapping


def vrb_to_prb(vrbs, n_rb: int, distributed: bool, gap2: bool = False):
    """Returns (PRBs in even slot, PRBs in odd slot)."""
    vrbs = np.asarray(list(vrbs), dtype=np.int64)
    if not distributed:
        return vrbs.copy(), vrbs.copy()
    g1, g2 = gap_values(n_rb)
    n_gap = g2 if gap2 else g1
    n_vrb = n_vrb_dist(n_rb, gap2)
    nt = n_vrb if not gap2 else 2 * n_gap
    p = rbg_size(n_rb)
    n_row = math.ceil(nt / (4 * p)) * p
    n_null = 4 * n_row - nt
    vt = vrbs % nt
    base = nt * (vrbs // nt)
    pp = 2 * n_row * (vt % 2) + vt // 2 + base
    ppp = n_row * (vt % 4) + vt // 4 + base
    prb = ppp.copy()
    if n_null:
        c1 = (vt >= nt - n_null) & (vt % 2 == 1)
        c2 = (vt >= nt - n_null) & (vt % 2 == 0)
        c3 = (vt < nt - n_null) & (vt % 4 >= 2)
        prb = np.where(c1, pp - n_row, prb)
        prb = np.where(c2, pp - n_row + n_null // 2, prb)
        prb = np.where(c3, ppp - n_null // 2, prb)
    even_t = prb
    odd_t = (prb + nt // 2) % nt + base

    def gapmap(t):
        return np.where(t < nt // 2, t, t + n_gap - nt // 2)
    return gapmap(even_t), gapmap(odd_t)


def dci_prbs(dci: Dci, n_rb: int):
    if dci.fmt == "1":
        p = rbg_size(n_rb)
        prbs = [r for i, on in enumerate(dci.bitmap) if on
                for r in range(i * p, min((i + 1) * p, n_rb))] if dci.ra_type == 0 else []
        prbs = np.array(prbs, dtype=np.int64)
        return prbs, prbs
    vrbs = range(dci.rb_start, dci.rb_start + dci.n_vrb)
    return vrb_to_prb(vrbs, n_rb, dci.distributed, dci.gap2)


# ------------------------------------------------------ RE mapping


def _sync_symbols(cfg: CellConfig, sf: int) -> set:
    """Symbols (in subframe) occupied by PSS/SSS/PBCH in the central PRBs."""
    ns = cfg.n_symb
    out = set()
    if cfg.tdd:
        if sf in (0, 5):
            out.add(2 * ns - 1)            # SSS
        if sf in (1, 6):
            out.add(2)                     # PSS
    else:
        if sf in (0, 5):
            out |= {ns - 2, ns - 1}
    if sf == 0:
        out |= {ns, ns + 1, ns + 2, ns + 3}  # PBCH
    return out


def _central_prb(cfg: CellConfig, prb: int) -> bool:
    lo, hi = 6 * cfg.n_rb - 36, 6 * cfg.n_rb + 36
    return 12 * prb < hi and 12 * prb + 12 > lo


@lru_cache(maxsize=256)
def _crs_mask(cfg: CellConfig, sf: int) -> np.ndarray:
    """Boolean (n_sym, n_sc) mask of CRS REs of the configured ports."""
    m = np.zeros((2 * cfg.n_symb, cfg.n_sc), dtype=bool)
    for slot in range(2):
        for l, ports in crs_symbols(cfg.cp_ext):
            for p in ports:
                if p < cfg.n_ports:
                    k0 = crs_shift(p, l, 2 * sf + slot, cfg.n_id, cfg.cp_ext)
                    m[slot * cfg.n_symb + l, k0::6] = True
    m.setflags(write=False)
    return m


def pdsch_re(cfg: CellConfig, sf: int, cfi: int, prb_even, prb_odd):
    """(symbol, subcarrier) arrays of PDSCH REs in mapping order."""
    n_ctrl = n_ctrl_symbols(cfg, cfi)
    crs_m = _crs_mask(cfg, sf)
    sync = _sync_symbols(cfg, sf)
    ls, ks = [], []
    for slot, prbs in ((0, prb_even), (1, prb_odd)):
        prbs = np.sort(np.asarray(prbs))
        for li in range(cfg.n_symb):
            l = slot * cfg.n_symb + li
            if l < n_ctrl:
                continue
            for prb in prbs:
                if l in sync and _central_prb(cfg, prb):
                    continue
                k = np.arange(12 * prb, 12 * prb + 12)
                k = k[~crs_m[l, k]]
                ls.append(np.full(len(k), l))
                ks.append(k)
    if not ls:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ls), np.concatenate(ks)


def n_layers(cfg: CellConfig) -> int:
    return cfg.n_ports


def pdsch_e(n_re: int, cfg: CellConfig) -> int:
    nl = n_layers(cfg)
    return nl * 2 * ((n_re * 2) // (nl * 2))


def pdsch_scrambling(cfg: CellConfig, sf: int, rnti: int, n: int) -> np.ndarray:
    c_init = rnti * (1 << 14) + sf * (1 << 9) + cfg.n_id
    return gold(c_init, n)


def pdsch_symbols(grid, h, cfg: CellConfig, sf: int, cfi: int, dci: Dci):
    """Equalised (transmit-diversity decoded) PDSCH symbols and gains."""
    pe, po = dci_prbs(dci, cfg.n_rb)
    ls, ks = pdsch_re(cfg, sf, cfi, pe, po)
    e = pdsch_e(len(ls), cfg)
    n_sym = e // 2
    ls, ks = ls[:n_sym], ks[:n_sym]
    return td_decode(grid[ls, ks], h[:cfg.n_ports, ls, ks], cfg.n_ports)


def pdsch_llr(grid, h, noise, cfg: CellConfig, sf: int, cfi: int, dci: Dci):
    """Descrambled soft bits of the PDSCH codeword described by `dci`."""
    x, g = pdsch_symbols(grid, h, cfg, sf, cfi, dci)
    llr = coding.qpsk_llr(x, g / max(noise, 1e-12))
    return coding.scramble_llr(llr, pdsch_scrambling(cfg, sf, dci.rnti, len(llr)))


def sib1_rv(sfn: int) -> int:
    k = (sfn // 2) % 4
    return math.ceil(1.5 * k) % 4


def decode_pdsch(grid, h, noise, cfg: CellConfig, sf: int, cfi: int, dci: Dci,
                 rv: int | None = None, softbuf: TbSoftBuffer | None = None):
    """Decode a broadcast PDSCH. Returns (bytes or None, soft buffer)."""
    tbs = dci_tbs(dci)
    if tbs is None:
        return None, softbuf
    llr = pdsch_llr(grid, h, noise, cfg, sf, cfi, dci)
    if rv is None:
        rv = dci.rv
    if softbuf is None or softbuf.tbs != tbs:
        softbuf = TbSoftBuffer(tbs)
    softbuf.add(llr, rv)
    bits = softbuf.decode()
    if bits is None:
        return None, softbuf
    return coding.bits_to_bytes(bits), softbuf
