"""CORESET#0 and the Type0-PDCCH common search space (TS 38.213 13), the
PDCCH of CORESET#0 (TS 38.211 7.3.2, 7.4.1.3) and DCI format 1_0 with CRC
scrambled by SI-RNTI (TS 38.212 7.3)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..coding import bits_to_int, int_to_bits, qpsk_mod
from ..sequences import gold
from . import polar
from .pbch import CRC24C, Mib
from .params import SSB_REF_K

SI_RNTI = 0xFFFF
AGGREGATION_LEVELS = (4, 8, 16)          # Type0-PDCCH CSS (38.213 Table 10.1-1)

# 38.213 Tables 13-1..13-4 (FR1, bands with 5/10 MHz minimum channel
# bandwidth): (multiplexing pattern, N_RB, N_symb, offset in RBs)
_T13_1 = [(1, 24, 2, 0), (1, 24, 2, 2), (1, 24, 2, 4), (1, 24, 3, 0), (1, 24, 3, 2),
          (1, 24, 3, 4), (1, 48, 1, 12), (1, 48, 1, 16), (1, 48, 2, 12), (1, 48, 2, 16),
          (1, 48, 3, 12), (1, 48, 3, 16), (1, 96, 1, 38), (1, 96, 2, 38), (1, 96, 3, 38)]
_T13_2 = [(1, 24, 2, 5), (1, 24, 2, 6), (1, 24, 2, 7), (1, 24, 2, 8), (1, 24, 3, 5),
          (1, 24, 3, 6), (1, 24, 3, 7), (1, 24, 3, 8), (1, 48, 1, 18), (1, 48, 1, 20),
          (1, 48, 2, 18), (1, 48, 2, 20), (1, 48, 3, 18), (1, 48, 3, 20)]
_T13_3 = [(1, 48, 1, 2), (1, 48, 1, 6), (1, 48, 2, 2), (1, 48, 2, 6), (1, 48, 3, 2),
          (1, 48, 3, 6), (1, 96, 1, 28), (1, 96, 2, 28), (1, 96, 3, 28)]
_T13_4 = [(1, 24, 2, 0), (1, 24, 2, 1), (1, 24, 2, 2), (1, 24, 2, 3), (1, 24, 2, 4),
          (1, 24, 3, 0), (1, 24, 3, 1), (1, 24, 3, 2), (1, 24, 3, 3), (1, 24, 3, 4),
          (1, 48, 1, 12), (1, 48, 1, 14), (1, 48, 1, 16), (1, 48, 2, 12), (1, 48, 2, 14),
          (1, 48, 2, 16)]
_TABLES = {(15000, 15000): _T13_1, (15000, 30000): _T13_2, (30000, 15000): _T13_3,
           (30000, 30000): _T13_4}
# 38.213 Table 13-11 (pattern 1, FR1): (O [ms], search space sets per slot,
# M, first symbol(s); None = N_symb of the CORESET)
_T13_11 = [(0, 1, 1.0, (0,)), (0, 2, 0.5, (0, None)), (2, 1, 1.0, (0,)), (2, 2, 0.5, (0, None)),
           (5, 1, 1.0, (0,)), (5, 2, 0.5, (0, None)), (7, 1, 1.0, (0,)), (7, 2, 0.5, (0, None)),
           (0, 1, 2.0, (0,)), (5, 1, 2.0, (0,)), (0, 1, 1.0, (1,)), (0, 1, 1.0, (2,)),
           (2, 1, 1.0, (1,)), (2, 1, 1.0, (2,)), (5, 1, 1.0, (1,)), (5, 1, 1.0, (2,))]


@dataclass(frozen=True)
class Coreset0:
    scs: int                  # subCarrierSpacingCommon (PDCCH/PDSCH SCS) [Hz]
    n_rb: int
    n_symb: int
    offset: int               # RBs from the CORESET start to the CRB of the SSB
    f_low: float              # RF frequency of subcarrier 0 of its lowest RB [Hz]
    o: float                  # search space #0: O [ms]
    m: float                  # M
    first_symbols: tuple      # first symbol per search space set in a slot

    @property
    def n_sc(self) -> int:
        return 12 * self.n_rb

    @property
    def n_cce(self) -> int:
        return self.n_rb * self.n_symb // 6

    @property
    def bandwidth(self) -> float:
        return self.n_sc * self.scs

    @property
    def f_center(self) -> float:
        return self.f_low + self.bandwidth / 2

    @property
    def mu(self) -> int:
        return {15000: 0, 30000: 1}[self.scs]

    def occasions(self, ssb_index: int) -> tuple[int, int, int]:
        """(n0, frame parity, first symbol) of the Type0-PDCCH monitoring
        occasion for an SSB index (38.213 13): slots n0 and n0 + 1 of the
        frames with SFN mod 2 = parity."""
        n_slot = 10 * 2 ** self.mu
        v = int(self.o * 2 ** self.mu + np.floor(ssb_index * self.m))
        fs = self.first_symbols
        sym = fs[0] if len(fs) == 1 else (fs[0] if ssb_index % 2 == 0 else self.n_symb)
        return v % n_slot, (v // n_slot) % 2, sym


def coreset0(mib: Mib, f_ref: float, ssb_scs: int) -> Coreset0 | None:
    """CORESET#0 of a cell from its MIB. f_ref: SS_REF of the SSB [Hz]
    (subcarrier 120 of the block). k_SSB and the CRB grid are in units of
    15 kHz in FR1 (38.211 7.4.3.1)."""
    if not mib.sib1_present:
        return None
    table = _TABLES.get((ssb_scs, mib.scs_common))
    if table is None or mib.coreset0 >= len(table) or mib.search_space0 >= len(_T13_11):
        return None
    pattern, n_rb, n_symb, offset = table[mib.coreset0]
    if pattern != 1:
        return None
    o, _, m, first = _T13_11[mib.search_space0]
    ssb_sc0 = f_ref - SSB_REF_K * ssb_scs
    crb = ssb_sc0 - mib.k_ssb * 15000
    f_low = crb - offset * 12 * mib.scs_common
    return Coreset0(mib.scs_common, n_rb, n_symb, offset, f_low, o, m, first)


# --------------------------------------------------------------- mapping
@lru_cache(maxsize=64)
def cce_regs(n_rb: int, n_symb: int, n_id: int, cce: int, level: int) -> tuple[int, ...]:
    """REG indices (time first numbering) of a PDCCH candidate in CORESET#0:
    interleaved, bundle size 6, R = 2, n_shift = N_ID (38.211 7.3.2.2)."""
    n_reg = n_rb * n_symb
    lb, r_il = 6, 2
    c_il = n_reg // (lb * r_il)
    out = []
    for x in range(cce, cce + level):         # one bundle per CCE (L = 6)
        c, r = divmod(x, r_il)
        f = (r * c_il + c + n_id) % (n_reg // lb)
        out.extend(range(f * lb, (f + 1) * lb))
    return tuple(sorted(out))


def candidates(c0: Coreset0) -> list[tuple[int, int]]:
    """(first CCE, aggregation level) of the Type0 CSS candidates
    (38.213 10.1, Y = 0)."""
    n = c0.n_cce
    out = []
    for lv, m_l in zip(AGGREGATION_LEVELS, (4, 2, 1)):
        if lv > n:
            continue
        for m in range(m_l):
            cce = lv * ((m * n) // (lv * m_l) % (n // lv))
            if (cce, lv) not in out:
                out.append((cce, lv))
    return out


DATA_SC = np.array([k for k in range(12) if k % 4 != 1])
DMRS_SC = np.array([1, 5, 9])


def candidate_res(c0: Coreset0, n_id: int, cce: int, level: int, sym0: int):
    """(l, k) arrays of the data REs of a candidate in mapping order (k
    first, then l) and the RBs it uses."""
    regs = cce_regs(c0.n_rb, c0.n_symb, n_id, cce, level)
    rbs = sorted(set(r // c0.n_symb for r in regs))
    ls, ks = [], []
    for s in range(c0.n_symb):
        for rb in rbs:
            ks.append(12 * rb + DATA_SC)
            ls.append(np.full(len(DATA_SC), sym0 + s))
    return np.concatenate(ls), np.concatenate(ks), rbs


def dmrs(n_rb: int, slot: int, l: int, n_id: int, n_symb_slot: int = 14) -> np.ndarray:
    """PDCCH DMRS of CORESET#0 for one symbol: (n_rb, 3) values at
    subcarriers 12 n + 1, 5, 9 counted from the CORESET's lowest RB."""
    c_init = (2 ** 17 * (n_symb_slot * slot + l + 1) * (2 * n_id + 1) + 2 * n_id) % 2 ** 31
    return qpsk_mod(gold(c_init, 6 * n_rb)).reshape(n_rb, 3)


def scrambling(n_id: int, n: int, rnti: int = 0) -> np.ndarray:
    return gold((rnti * 2 ** 16 + n_id) % 2 ** 31, n)


# ------------------------------------------------------------------ DCI
def dci10_si_size(n_rb: int) -> int:
    """Payload size of DCI 1_0 with SI-RNTI (38.212 7.3.1.2.1)."""
    return int(np.ceil(np.log2(n_rb * (n_rb + 1) / 2))) + 4 + 1 + 5 + 2 + 1 + 15


@dataclass(frozen=True)
class Dci10Si:
    n_rb: int
    riv: int
    tdra: int                 # time domain resource assignment (row - 1)
    vrb_interleaved: bool
    mcs: int
    rv: int
    si_indicator: int         # 0 = SIB1, 1 = other SI message

    @property
    def rb_start_len(self) -> tuple[int, int]:
        """RIV of resource allocation type 1 (38.214 5.1.2.2.2)."""
        n = self.n_rb
        lm1, start = divmod(self.riv, n)
        if lm1 + start + 1 <= n and lm1 <= n // 2:
            return start, lm1 + 1
        return n - 1 - start, n - lm1 + 1


def riv(n_rb: int, start: int, length: int) -> int:
    if length - 1 <= n_rb // 2:
        return n_rb * (length - 1) + start
    return n_rb * (n_rb - length + 1) + (n_rb - 1 - start)


def dci10_si_pack(d: Dci10Si) -> np.ndarray:
    nf = dci10_si_size(d.n_rb) - 28
    b = np.concatenate([int_to_bits(d.riv, nf), int_to_bits(d.tdra, 4), [int(d.vrb_interleaved)],
                        int_to_bits(d.mcs, 5), int_to_bits(d.rv, 2), [d.si_indicator],
                        np.zeros(15, dtype=np.uint8)])
    return b.astype(np.uint8)


def dci10_si_unpack(a: np.ndarray, n_rb: int) -> Dci10Si:
    nf = dci10_si_size(n_rb) - 28
    p = nf
    return Dci10Si(n_rb, bits_to_int(a[:nf]), bits_to_int(a[p:p + 4]), bool(a[p + 4]),
                   bits_to_int(a[p + 5:p + 10]), bits_to_int(a[p + 10:p + 12]), int(a[p + 12]))


def _crc_bits(a: np.ndarray, rnti: int) -> np.ndarray:
    """38.212 7.3.2: CRC24C over 24 ones + payload, RNTI on the last 16."""
    from ..coding import crc
    p = crc(np.concatenate([np.ones(24, dtype=np.uint8), a]), CRC24C)
    return int_to_bits(p ^ rnti, 24)


def dci_encode(a: np.ndarray, e: int, rnti: int = SI_RNTI) -> np.ndarray:
    c = np.concatenate([np.asarray(a, dtype=np.uint8), _crc_bits(a, rnti)])
    return polar.encode_rm(c, e)


def dci_decode(llr: np.ndarray, a_len: int, rnti: int = SI_RNTI,
               list_size: int = 8) -> np.ndarray | None:
    """E soft bits (descrambled) -> the A payload bits, or None."""
    k = a_len + 24
    e = len(llr)
    d = polar.rate_dematch_rm(llr, k)

    def check(c):
        return np.array_equal(c[a_len:], _crc_bits(c[:a_len], rnti))

    c = polar.decode(d, k, check, list_size, e=e)
    return None if c is None else c[:a_len]


def pdcch_symbols(dci: Dci10Si, level: int, n_id: int) -> np.ndarray:
    """QPSK symbols of a DCI 1_0 (SI-RNTI) on CORESET#0."""
    e = 108 * level
    b = dci_encode(dci10_si_pack(dci), e) ^ scrambling(n_id, e)
    return qpsk_mod(b)
