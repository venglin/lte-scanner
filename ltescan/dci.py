"""DCI formats 0/1A, 1C and 1 (sizes and parsing), TS 36.212 5.3.3.1."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .coding import bits_to_int, int_to_bits

SI_RNTI = 0xFFFF
P_RNTI = 0xFFFE
M_RNTI = 0xFFFD
AMBIGUOUS = {12, 14, 16, 20, 24, 26, 32, 40, 44, 56}


def rnti_type(rnti: int) -> str:
    if rnti == SI_RNTI:
        return "SI-RNTI"
    if rnti == P_RNTI:
        return "P-RNTI"
    if rnti == M_RNTI:
        return "M-RNTI"
    if 1 <= rnti <= 60:
        return "RA-RNTI"
    return "C-RNTI"


def rbg_size(n_rb: int) -> int:
    return 1 if n_rb <= 10 else 2 if n_rb <= 26 else 3 if n_rb <= 63 else 4


def gap_values(n_rb: int) -> tuple[int, int | None]:
    if n_rb <= 10:
        return math.ceil(n_rb / 2), None
    if n_rb == 11:
        return 4, None
    if n_rb <= 19:
        return 8, None
    if n_rb <= 26:
        return 12, None
    if n_rb <= 44:
        return 18, None
    if n_rb <= 49:
        return 27, None
    if n_rb <= 63:
        return 27, 9
    if n_rb <= 79:
        return 32, 16
    return 48, 16


def n_vrb_dist(n_rb: int, gap2: bool = False) -> int:
    g1, g2 = gap_values(n_rb)
    if not gap2:
        return 2 * min(g1, n_rb - g1)
    return (n_rb // (2 * g2)) * 2 * g2


def _riv_bits(n: int) -> int:
    return math.ceil(math.log2(n * (n + 1) / 2))


def size_0_1a(n_rb: int, tdd: bool = False) -> int:
    rba = _riv_bits(n_rb)
    s1a = 1 + 1 + rba + 5 + (4 if tdd else 3) + 1 + 2 + 2 + (2 if tdd else 0)
    s0 = 1 + 1 + rba + 5 + 1 + 2 + 3 + 1 + (2 if tdd else 0)
    s = max(s0, s1a)
    if s in AMBIGUOUS:
        s += 1
    return s


def size_1c(n_rb: int) -> int:
    step = 2 if n_rb < 50 else 4
    n = n_vrb_dist(n_rb) // step
    return (1 if n_rb >= 50 else 0) + _riv_bits(n) + 5


def size_1(n_rb: int, tdd: bool = False) -> int:
    p = rbg_size(n_rb)
    s = (1 if n_rb > 10 else 0) + math.ceil(n_rb / p) + 5 + (4 if tdd else 3) + 1 + 2 + 2 \
        + (2 if tdd else 0)
    s1a = size_0_1a(n_rb, tdd)
    while s == s1a or s in AMBIGUOUS:
        s += 1
    return s


def riv_decode(riv: int, n: int) -> tuple[int, int]:
    """RIV -> (start, length) for a span of n VRBs."""
    l1 = riv // n + 1
    start = riv % n
    if l1 + start > n:
        l1 = n - l1 + 2
        start = n - 1 - start
    return start, l1


def riv_encode(start: int, length: int, n: int) -> int:
    if (length - 1) <= n // 2:
        return n * (length - 1) + start
    return n * (n - length + 1) + (n - 1 - start)


@dataclass
class Dci:
    fmt: str
    rnti: int
    distributed: bool = False
    gap2: bool = False
    rb_start: int = 0
    n_vrb: int = 0
    mcs: int = 0
    harq: int = 0
    ndi: int = 0
    rv: int = 0
    tpc: int = 0
    n_prb_1a: int = 2
    bitmap: tuple = ()
    ra_type: int = 0
    raw: str = ""

    def describe(self) -> str:
        if self.fmt == "0":
            return "DCI0 (uplink grant)"
        if self.fmt == "?":
            return f"DCI ({self.raw}, fields not interpreted)"
        alloc = (f"RBG bitmap {''.join(map(str, self.bitmap))}" if self.fmt == "1"
                 else f"VRB {self.rb_start}..{self.rb_start + self.n_vrb - 1}"
                 f"{' dist' if self.distributed else ''}{' gap2' if self.gap2 else ''}")
        return f"DCI{self.fmt} {alloc} MCS={self.mcs} RV={self.rv}"


def parse_1a(bits: np.ndarray, n_rb: int, rnti: int, tdd: bool = False) -> Dci | None:
    b = list(bits)
    if b[0] == 0:
        return Dci("0", rnti, raw=_hex(bits))
    pos = 1
    dist = bool(b[pos]); pos += 1
    rba_n = _riv_bits(n_rb)
    rba = bits_to_int(b[pos:pos + rba_n]); pos += rba_n
    common = rnti_type(rnti) in ("SI-RNTI", "P-RNTI", "RA-RNTI")
    gap2 = False
    if dist and n_rb >= 50 and not common:
        gap2 = bool(rba >> (rba_n - 1))
        rba &= (1 << (rba_n - 1)) - 1
    mcs = bits_to_int(b[pos:pos + 5]); pos += 5
    hn = 4 if tdd else 3
    harq = bits_to_int(b[pos:pos + hn]); pos += hn
    ndi = b[pos]; pos += 1
    rv = bits_to_int(b[pos:pos + 2]); pos += 2
    tpc = bits_to_int(b[pos:pos + 2]); pos += 2
    if dist and n_rb >= 50 and common:
        gap2 = bool(ndi)
    n = n_vrb_dist(n_rb, gap2) if dist else n_rb
    if rba >= n * (n + 1) // 2:
        return None
    start, length = riv_decode(rba, n)
    if start + length > n:
        return None
    return Dci("1A", rnti, dist, gap2, start, length, mcs, harq, ndi, rv, tpc,
               3 if (tpc & 1) else 2, raw=_hex(bits))


def pack_1a(n_rb: int, rb_start: int, n_vrb: int, mcs: int, rv: int,
            distributed: bool = False, n_prb_1a: int = 2, tdd: bool = False) -> np.ndarray:
    """Format 1A for SI/P/RA-RNTI (used by the test signal generator)."""
    n = n_vrb_dist(n_rb) if distributed else n_rb
    out = [1, int(distributed)]
    out += list(int_to_bits(riv_encode(rb_start, n_vrb, n), _riv_bits(n_rb)))
    out += list(int_to_bits(mcs, 5)) + [0] * (4 if tdd else 3) + [0]
    out += list(int_to_bits(rv, 2)) + [0, 1 if n_prb_1a == 3 else 0]
    if tdd:
        out += [0, 0]
    out += [0] * (size_0_1a(n_rb, tdd) - len(out))
    return np.array(out, dtype=np.uint8)


def parse_1c(bits: np.ndarray, n_rb: int, rnti: int) -> Dci | None:
    b = list(bits)
    pos = 0
    gap2 = False
    if n_rb >= 50:
        gap2 = bool(b[0]); pos = 1
    step = 2 if n_rb < 50 else 4
    nv = n_vrb_dist(n_rb, gap2)
    n = n_vrb_dist(n_rb) // step
    rba_n = _riv_bits(n)
    rba = bits_to_int(b[pos:pos + rba_n]); pos += rba_n
    i_tbs = bits_to_int(b[pos:pos + 5])
    if rba >= n * (n + 1) // 2:
        return None
    s, length = riv_decode(rba, n)
    start, length = s * step, length * step
    if start + length > nv:
        return None
    return Dci("1C", rnti, True, gap2, start, length, i_tbs, raw=_hex(bits))


def pack_1c(n_rb: int, rb_start: int, n_vrb: int, i_tbs: int) -> np.ndarray:
    step = 2 if n_rb < 50 else 4
    n = n_vrb_dist(n_rb) // step
    out = [0] if n_rb >= 50 else []
    out += list(int_to_bits(riv_encode(rb_start // step, n_vrb // step, n), _riv_bits(n)))
    out += list(int_to_bits(i_tbs, 5))
    return np.array(out, dtype=np.uint8)


def parse_1(bits: np.ndarray, n_rb: int, rnti: int, tdd: bool = False) -> Dci:
    b = list(bits)
    pos = 0
    ra_type = 0
    if n_rb > 10:
        ra_type = b[0]; pos = 1
    nb = math.ceil(n_rb / rbg_size(n_rb))
    bitmap = tuple(b[pos:pos + nb]); pos += nb
    mcs = bits_to_int(b[pos:pos + 5]); pos += 5
    hn = 4 if tdd else 3
    harq = bits_to_int(b[pos:pos + hn]); pos += hn
    ndi = b[pos]; pos += 1
    rv = bits_to_int(b[pos:pos + 2]); pos += 2
    tpc = bits_to_int(b[pos:pos + 2])
    return Dci("1", rnti, mcs=mcs, harq=harq, ndi=ndi, rv=rv, tpc=tpc,
               bitmap=bitmap, ra_type=ra_type, raw=_hex(bits))


def _hex(bits) -> str:
    v = bits_to_int(bits)
    return f"{v:0{(len(bits) + 3) // 4}x}"
