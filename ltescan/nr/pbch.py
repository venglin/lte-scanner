"""NR SS/PBCH block: PSS, SSS and PBCH DMRS sequences (TS 38.211 7.4.2,
7.4.1.4), resource layout of the block (7.4.3.1), PBCH payload generation,
scrambling and coding (38.212 7.1) and MIB parsing (38.331)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..coding import bits_to_int, crc, int_to_bits, qpsk_mod
from ..sequences import gold
from . import polar

CRC24C = (0xB2B117, 24)
A_BAR = 32                       # MIB (24 bits) + 8 timing bits
PBCH_K = A_BAR + 24
PBCH_N = 512
PBCH_E = 864
# 38.212 Table 7.1.1-1
G = (16, 23, 18, 17, 8, 30, 10, 6, 24, 7, 0, 5, 3, 2, 1, 4, 9, 11, 12, 13, 14, 15, 19, 20, 21,
     22, 25, 26, 27, 28, 29, 31)


# --------------------------------------------------------------- sequences
def _mseq(init: tuple[int, ...], taps: tuple[int, int]) -> np.ndarray:
    x = list(init) + [0] * 120
    a, b = taps
    for i in range(127 - 7):
        x[i + 7] = (x[i + a] + x[i + b]) % 2
    return np.array(x[:127], dtype=np.int64)


_X_PSS = _mseq((0, 1, 1, 0, 1, 1, 1), (4, 0))
_X0_SSS = _mseq((1, 0, 0, 0, 0, 0, 0), (4, 0))
_X1_SSS = _mseq((1, 0, 0, 0, 0, 0, 0), (1, 0))


@lru_cache(maxsize=3)
def pss(n_id_2: int) -> np.ndarray:
    n = np.arange(127)
    return 1.0 - 2 * _X_PSS[(n + 43 * n_id_2) % 127]


def sss(n_id: int) -> np.ndarray:
    n_id_1, n_id_2 = divmod(n_id, 3)
    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112
    n = np.arange(127)
    return (1.0 - 2 * _X0_SSS[(n + m0) % 127]) * (1.0 - 2 * _X1_SSS[(n + m1) % 127])


@lru_cache(maxsize=3)
def sss_table(n_id_2: int) -> np.ndarray:
    """(336, 127) SSS sequences for all N_ID1 with this N_ID2."""
    return np.stack([sss(3 * i + n_id_2) for i in range(336)])


def dmrs(n_id: int, i_ssb_bar: int) -> np.ndarray:
    """144 PBCH DMRS symbols. i_ssb_bar: i_SSB + 4 n_hf for L_max = 4,
    else the three LSBs of the SSB index."""
    c_init = (2 ** 11 * (i_ssb_bar + 1) * (n_id // 4 + 1) + 2 ** 6 * (i_ssb_bar + 1)
              + n_id % 4)
    return qpsk_mod(gold(c_init, 288))


# ------------------------------------------------------------------ layout
@lru_cache(maxsize=4)
def layout(v: int):
    """RE positions (l, k) within the 4x240 block, v = N_ID mod 4.
    Returns (pss_k, sss_k, dmrs (l, k) arrays, pbch data (l, k) arrays);
    DMRS and data are in mapping order (k first, then l)."""
    pss_k = np.arange(56, 183)
    dl, dk, pl, pk = [], [], [], []
    for l in (1, 2, 3):
        ks = np.arange(240) if l != 2 else np.r_[0:48, 192:240]
        for k in ks:
            (dl if k % 4 == v else pl).append(l)
            (dk if k % 4 == v else pk).append(k)
    out = pss_k, pss_k, (np.array(dl), np.array(dk)), (np.array(pl), np.array(pk))
    return out


# ------------------------------------------------------------------- MIB
@dataclass
class Mib:
    sfn: int                      # 10 bits
    half_frame: int
    scs_common: int               # Hz (FR1: 15 or 30 kHz)
    k_ssb: int                    # 0..31 (FR1, 5 bits)
    dmrs_type_a_pos: int          # 2 or 3
    coreset0: int                 # controlResourceSetZero 0..15
    search_space0: int            # searchSpaceZero 0..15
    cell_barred: bool
    intra_freq_reselection: bool  # True = allowed
    ssb_index: int = 0            # full SSB index (with DMRS / payload bits)
    l_max: int = 8

    @property
    def sib1_present(self) -> bool:
        """k_SSB > 23 (FR1): no CORESET#0 / Type0-PDCCH for SIB1 on this SSB."""
        return self.k_ssb <= 23


def mib_bits(m: Mib) -> np.ndarray:
    """24-bit BCCH-BCH-Message (UPER: CHOICE bit, then the MIB fields)."""
    b = [0]
    b += list(int_to_bits(m.sfn >> 4, 6))
    b += [0 if m.scs_common in (15000, 60000) else 1]
    b += list(int_to_bits(m.k_ssb & 15, 4))
    b += [0 if m.dmrs_type_a_pos == 2 else 1]
    b += list(int_to_bits(m.coreset0, 4)) + list(int_to_bits(m.search_space0, 4))
    b += [0 if m.cell_barred else 1, 0 if m.intra_freq_reselection else 1, 0]
    return np.array(b, dtype=np.uint8)


def _categories():
    """Category of each a_bar bit for the G interleaving."""
    cat = []
    for i in range(A_BAR):
        if 1 <= i <= 6 or 24 <= i <= 27:
            cat.append("sfn")
        elif i == 28:
            cat.append("hrf")
        elif 29 <= i <= 31:
            cat.append("ssb")
        else:
            cat.append("other")
    return cat


def _interleave_positions() -> np.ndarray:
    """pos[i] = index in a' of a_bar_i (38.212 7.1.1)."""
    j = {"sfn": 0, "hrf": 10, "ssb": 11, "other": 14}
    pos = []
    for c in _categories():
        pos.append(G[j[c]])
        j[c] += 1
    return np.array(pos)


_POS = _interleave_positions()


def _scramble_mask(n_id: int, a_prime: np.ndarray, l_max: int) -> np.ndarray:
    """38.212 7.1.2 scrambling sequence s for a' (depends on the
    unscrambled 2nd/3rd SFN LSBs carried in a')."""
    skip = {G[10], G[7], G[8]} | ({G[11], G[12], G[13]} if l_max == 64 else set())
    m = A_BAR - (6 if l_max == 64 else 3)
    v = 2 * int(a_prime[G[7]]) + int(a_prime[G[8]])
    c = gold(n_id, 4 * m + A_BAR)
    s = np.zeros(A_BAR, dtype=np.uint8)
    j = 0
    for i in range(A_BAR):
        if i not in skip:
            s[i] = c[j + v * m]
            j += 1
    return s


def payload_encode(m: Mib, n_id: int) -> np.ndarray:
    """MIB + timing bits -> 56 bits (scrambled payload + CRC24C)."""
    a = np.zeros(A_BAR, dtype=np.uint8)
    a[:24] = mib_bits(m)
    a[24:28] = int_to_bits(m.sfn & 15, 4)
    a[28] = m.half_frame
    if m.l_max == 64:
        a[29:32] = int_to_bits(m.ssb_index >> 3, 3)
    else:
        a[29] = (m.k_ssb >> 4) & 1
    ap = np.zeros(A_BAR, dtype=np.uint8)
    ap[_POS] = a
    app = ap ^ _scramble_mask(n_id, ap, m.l_max)
    return np.concatenate([app, int_to_bits(crc(app, CRC24C), 24)])


def payload_decode(c: np.ndarray, n_id: int, ssb_bar: int, l_max: int) -> Mib:
    """56 decoded bits (CRC already checked) -> Mib."""
    app = np.asarray(c[:A_BAR], dtype=np.uint8)
    ap = app ^ _scramble_mask(n_id, app, l_max)
    a = ap[_POS]
    sfn = (bits_to_int(a[1:7]) << 4) | bits_to_int(a[24:28])
    k_ssb = bits_to_int(a[8:12])
    if l_max == 64:
        ssb_index = (bits_to_int(a[29:32]) << 3) | (ssb_bar & 7)
    else:
        k_ssb |= int(a[29]) << 4
        ssb_index = ssb_bar & 3 if l_max == 4 else ssb_bar & 7
    return Mib(sfn=sfn, half_frame=int(a[28]), scs_common=15000 if a[7] == 0 else 30000,
               k_ssb=k_ssb, dmrs_type_a_pos=2 + int(a[12]), coreset0=bits_to_int(a[13:17]),
               search_space0=bits_to_int(a[17:21]), cell_barred=bool(a[21] == 0),
               intra_freq_reselection=bool(a[22] == 0), ssb_index=ssb_index, l_max=l_max)


def crc_ok(c: np.ndarray) -> bool:
    return crc(c[:A_BAR], CRC24C) == bits_to_int(c[A_BAR:])


def scrambling_v(ssb_index: int, l_max: int) -> int:
    """v of the 38.211 7.3.3.1 PBCH bit scrambling."""
    return ssb_index & (3 if l_max == 4 else 7)


def pbch_symbols(m: Mib, n_id: int) -> np.ndarray:
    """432 QPSK symbols of the PBCH (encode + scramble + modulate)."""
    e = polar.encode(payload_encode(m, n_id), PBCH_N, PBCH_E)
    v = scrambling_v(m.ssb_index, m.l_max)
    b = e ^ gold(n_id, 8 * PBCH_E)[v * PBCH_E:(v + 1) * PBCH_E]
    return qpsk_mod(b)


def pbch_decode(llr: np.ndarray, n_id: int, ssb_bar: int, l_max: int,
                list_size: int = 8) -> Mib | None:
    """864 soft bits (before descrambling) -> Mib, or None if no list path
    passes the CRC."""
    v = scrambling_v(ssb_bar, l_max)
    c = gold(n_id, 8 * PBCH_E)[v * PBCH_E:(v + 1) * PBCH_E]
    d = polar.rate_dematch(llr * (1 - 2 * c.astype(np.float64)), PBCH_N)
    bits = polar.decode(d, PBCH_K, crc_ok, list_size)
    if bits is None:
        return None
    return payload_decode(bits, n_id, ssb_bar, l_max)


def ssb_grid(n_id: int, m: Mib, beta_pss: float = 1.0) -> np.ndarray:
    """(4, 240) complex resource grid of one SS/PBCH block."""
    g = np.zeros((4, 240), dtype=complex)
    pk, sk, (dl, dk), (pl, pk2) = layout(n_id % 4)
    g[0, pk] = beta_pss * pss(n_id % 3)
    g[2, sk] = sss(n_id)
    ssb_bar = m.ssb_index & 7 if m.l_max != 4 else (m.ssb_index & 3) + 4 * m.half_frame
    g[dl, dk] = dmrs(n_id, ssb_bar)
    g[pl, pk2] = pbch_symbols(m, n_id)
    return g
