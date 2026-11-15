"""PDSCH scheduled by DCI 1_0 in the Type0-PDCCH CSS (SIB1): default time
domain allocation, resource allocation type 1 and VRB-to-PRB interleaving,
DMRS (TS 38.211 7.4.1.1), MCS and TBS (TS 38.214 5.1.3), scrambling,
modulation and soft demapping (TS 38.211 5.1, 7.3.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coding import qpsk_mod
from ..sequences import gold
from .pdcch import SI_RNTI, Dci10Si

# 38.214 Table 5.1.2.1.1-2 (default A, normal CP):
# row -> (mapping, K0, S, L) for dmrs-TypeA-Position 2 and 3
_TDRA_A = {
    1: (("A", 0, 2, 12), ("A", 0, 3, 11)), 2: (("A", 0, 2, 10), ("A", 0, 3, 9)),
    3: (("A", 0, 2, 9), ("A", 0, 3, 8)), 4: (("A", 0, 2, 7), ("A", 0, 3, 6)),
    5: (("A", 0, 2, 5), ("A", 0, 3, 4)), 6: (("B", 0, 9, 4), ("B", 0, 10, 4)),
    7: (("B", 0, 4, 4), ("B", 0, 6, 4)), 8: (("B", 0, 5, 7),) * 2, 9: (("B", 0, 5, 2),) * 2,
    10: (("B", 0, 9, 2),) * 2, 11: (("B", 0, 12, 2),) * 2, 12: (("A", 0, 1, 13),) * 2,
    13: (("A", 0, 1, 6),) * 2, 14: (("A", 0, 2, 4),) * 2, 15: (("B", 0, 4, 7),) * 2,
    16: (("B", 0, 8, 4),) * 2,
}

# 38.214 Table 5.1.3.1-1 (MCS index table 1): (Qm, R x 1024)
MCS_TABLE1 = [(2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526),
              (2, 602), (2, 679), (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616),
              (4, 658), (6, 438), (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719),
              (6, 772), (6, 822), (6, 873), (6, 910), (6, 948)]

# 38.214 Table 5.1.3.2-1
TBS_TABLE = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
             168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384,
             408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
             888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416,
             1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280,
             2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
             3624, 3752, 3824]


def tdra(row_index: int, dmrs_type_a_pos: int) -> tuple[str, int, int, int]:
    """(mapping type, K0, S, L) for the DCI field value (row = value + 1)."""
    return _TDRA_A[row_index + 1][0 if dmrs_type_a_pos == 2 else 1]


def dmrs_symbols(mapping: str, s: int, length: int, type_a_pos: int) -> list[int]:
    """Single-symbol DMRS positions in the slot with dmrs-AdditionalPosition
    pos2 (DCI 1_0, 38.214 5.1.6.2; 38.211 Table 7.4.1.1.2-3)."""
    if mapping == "A":
        ld = s + length
        l0 = type_a_pos
        if ld < 8:
            add = []
        elif ld < 10:
            add = [7]
        elif ld < 13:
            add = [6, 9]
        else:
            add = [7, 11]
        return [l0] + add
    # mapping type B: positions relative to the PDSCH start
    ld = length
    if ld < 5:
        rel = [0]
    elif ld < 8:
        rel = [0, 4]
    elif ld < 10:
        rel = [0, 4]                # pos2 with 8..9 symbols adds nothing more
    elif ld < 12:
        rel = [0, 4, 8]
    else:
        rel = [0, 5, 9]
    return [s + r for r in rel if r < ld]


def cdm_groups_without_data(length: int) -> int:
    return 1 if length == 2 else 2


def vrb_to_prb(vrbs: list[int], n_rb: int, interleaved: bool) -> list[int]:
    """PRBs (relative to the lowest RB of CORESET#0) of the VRBs, in VRB
    order. Interleaved: bundles of 2 RBs, R = 2 (38.211 7.3.1.6)."""
    if not interleaved:
        return list(vrbs)
    n_bundle = -(-n_rb // 2)
    c = n_bundle // 2
    out = []
    for v in vrbs:
        j, o = divmod(v, 2)
        if j == n_bundle - 1:
            f = j
        else:
            cc, r = divmod(j, 2)
            f = r * c + cc
        out.append(2 * f + o)
    return out


def tbs(n_prb: int, n_symb: int, n_dmrs_symb: int, mcs: int, cdm_groups: int = 2) -> int:
    """38.214 5.1.3.2 (one layer, no overhead, no scaling)."""
    qm, r1024 = MCS_TABLE1[mcs]
    rate = r1024 / 1024
    n_re_p = 12 * n_symb - 6 * cdm_groups * n_dmrs_symb
    n_info = min(156, n_re_p) * n_prb * rate * qm
    if n_info <= 0:
        return 0
    if n_info <= 3824:
        n = max(3, int(np.floor(np.log2(n_info))) - 6)
        n_p = max(24, (2 ** n) * int(np.floor(n_info / 2 ** n)))
        return next(t for t in TBS_TABLE if t >= n_p)
    n = int(np.floor(np.log2(n_info - 24))) - 5
    n_p = max(3840, (2 ** n) * int(np.round((n_info - 24) / 2 ** n)))
    if rate <= 0.25:
        c = -(-(n_p + 24) // 3816)
        return 8 * c * -(-(n_p + 24) // (8 * c)) - 24
    if n_p > 8424:
        c = -(-(n_p + 24) // 8424)
        return 8 * c * -(-(n_p + 24) // (8 * c)) - 24
    return 8 * -(-(n_p + 24) // 8) - 24


def dmrs_seq(slot: int, l: int, n_id: int, n_sc: int, n_symb_slot: int = 14) -> np.ndarray:
    """PDSCH DMRS (type 1, port 1000, n_SCID 0) at the even subcarriers
    0, 2, 4, ... counted from the lowest RB of CORESET#0 (SIB1 PDSCH)."""
    c_init = (2 ** 17 * (n_symb_slot * slot + l + 1) * (2 * n_id + 1) + 2 * n_id) % 2 ** 31
    return qpsk_mod(gold(c_init, n_sc))


def scrambling(n_id: int, n: int, rnti: int = SI_RNTI) -> np.ndarray:
    return gold((rnti * 2 ** 15 + n_id) % 2 ** 31, n)


# --------------------------------------------------------------- QAM
def _levels(qm: int):
    """Per axis: amplitude levels and the bits mapped to them (38.211 5.1)."""
    h = qm // 2
    pts = []
    for v in range(2 ** h):
        b = [(v >> (h - 1 - i)) & 1 for i in range(h)]
        if h == 1:
            a = 1 - 2 * b[0]
        elif h == 2:
            a = (1 - 2 * b[0]) * (2 - (1 - 2 * b[1]))
        else:
            a = (1 - 2 * b[0]) * (4 - (1 - 2 * b[1]) * (2 - (1 - 2 * b[2])))
        pts.append((a, b))
    norm = {1: np.sqrt(2), 2: np.sqrt(10), 3: np.sqrt(42)}[h]
    return np.array([p[0] for p in pts]) / norm, np.array([p[1] for p in pts])


def modulate(bits: np.ndarray, qm: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int64).reshape(-1, qm)
    lv, _ = _levels(qm)
    w = 2 ** np.arange(qm // 2 - 1, -1, -1)
    return lv[b[:, 0::2] @ w] + 1j * lv[b[:, 1::2] @ w]


def demap(z: np.ndarray, gain: np.ndarray, qm: int) -> np.ndarray:
    """Max-log soft bits of equalised symbols z (z = y h* / |h|^2) with
    reliability gain = |h|^2 / noise. Bits interleave I and Q (b0 I, b1 Q,
    b2 I, ...) as in 38.211 5.1."""
    lv, lb = _levels(qm)
    h = qm // 2
    out = np.empty((len(z), qm))
    for comp, off in ((z.real, 0), (z.imag, 1)):
        d = (comp[:, None] - lv[None, :]) ** 2
        for i in range(h):
            m0 = d[:, lb[:, i] == 0].min(axis=1)
            m1 = d[:, lb[:, i] == 1].min(axis=1)
            out[:, 2 * i + off] = (m1 - m0) * gain
    return out.reshape(-1)


@dataclass
class Allocation:
    """Where the SIB1 PDSCH of one DCI lies in its slot."""
    dci: Dci10Si
    mapping: str
    s: int
    length: int
    dmrs: list[int]
    prbs: list[int]           # in VRB order
    qm: int
    rate: float
    tbs: int

    @property
    def data_symbols(self) -> list[int]:
        return [l for l in range(self.s, self.s + self.length)
                if not (l in self.dmrs and cdm_groups_without_data(self.length) == 2)]


def allocation(dci: Dci10Si, type_a_pos: int, tdra_list: list | None = None) -> Allocation | None:
    """tdra_list: pdsch-TimeDomainAllocationList of SIB1 [(K0, mapping, S, L)],
    which replaces default table A outside the Type0-PDCCH CSS."""
    if dci.mcs >= len(MCS_TABLE1):
        return None
    if tdra_list:
        if dci.tdra >= len(tdra_list):
            return None
        k0, mapping, s, length = tdra_list[dci.tdra]
    else:
        mapping, k0, s, length = tdra(dci.tdra, type_a_pos)
    if k0 != 0:
        return None
    start, n = dci.rb_start_len
    if n <= 0 or start + n > dci.n_rb:
        return None
    prbs = vrb_to_prb(list(range(start, start + n)), dci.n_rb, dci.vrb_interleaved)
    dm = dmrs_symbols(mapping, s, length, type_a_pos)
    qm, r = MCS_TABLE1[dci.mcs]
    t = tbs(n, length, len(dm), dci.mcs, cdm_groups_without_data(length))
    return Allocation(dci, mapping, s, length, dm, prbs, qm, r / 1024, t)


def data_res(a: Allocation, blocked: set | None = None):
    """(l, k) of the data REs in mapping order (k over the VRBs first,
    then l). blocked: (l, prb) pairs not available (SS/PBCH block)."""
    cdm1 = cdm_groups_without_data(a.length) == 1
    ls, ks = [], []
    for l in range(a.s, a.s + a.length):
        is_dmrs = l in a.dmrs
        if is_dmrs and not cdm1:
            continue
        for p in a.prbs:
            if blocked and (l, p) in blocked:
                continue
            k = 12 * p + np.arange(12)
            if is_dmrs:
                k = k[k % 2 == 1]          # CDM group 1 free for data
            ks.append(k)
            ls.append(np.full(len(k), l))
    if not ks:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(ls), np.concatenate(ks)
