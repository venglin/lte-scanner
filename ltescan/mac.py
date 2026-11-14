"""MAC-level helpers for broadcast transport blocks: Random Access Response
parsing (TS 36.321 6.1.5/6.2.2-6.2.3) and paging (PCCH, TS 36.331) parsing
with masking of subscriber identities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coding import bits_to_bytes, bytes_to_bits

TA_METERS = 16 * 299792458 / (15000 * 2048) / 2   # one TA step, one way (~78 m)


@dataclass
class RarEntry:
    rapid: int          # random access preamble id
    ta: int             # timing advance command (0..1282)
    ul_grant: int       # 20-bit UL grant
    temp_crnti: int

    @property
    def distance_m(self) -> float:
        return self.ta * TA_METERS


@dataclass
class Rar:
    backoff: int | None = None
    entries: list[RarEntry] = field(default_factory=list)


def parse_rar(data: bytes) -> Rar:
    """MAC RAR PDU: E/T/RAPID (or E/T/R/R/BI) subheaders, then 6-byte
    RAR payloads for the RAPID subheaders. Padding after the last one."""
    out = Rar()
    rapids = []
    i = 0
    while i < len(data):
        b = data[i]
        e, t = b >> 7, (b >> 6) & 1
        if t:
            rapids.append(b & 0x3F)
        else:
            out.backoff = b & 0x0F
        i += 1
        if not e:
            break
    for rapid in rapids:
        if i + 6 > len(data):
            break
        p = data[i:i + 6]
        ta = ((p[0] & 0x7F) << 4) | (p[1] >> 4)
        grant = ((p[1] & 0x0F) << 16) | (p[2] << 8) | p[3]
        out.entries.append(RarEntry(rapid, ta, grant, (p[4] << 8) | p[5]))
        i += 6
    return out


# ------------------------------------------------------------------ paging
@dataclass
class Paging:
    s_tmsi: int = 0                 # number of S-TMSI records
    imsi: int = 0                   # number of IMSI records
    other: int = 0                  # records of unknown (extension) type
    cs_domain: int = 0              # records for the CS domain (CSFB calls)
    si_modification: bool = False
    etws: bool = False
    complete: bool = True           # whole message parsed


class _Bits:
    def __init__(self, b: np.ndarray):
        self.b = b
        self.pos = 0

    def read(self, n: int) -> int:
        if self.pos + n > len(self.b):
            raise ValueError("koniec")
        v = 0
        for x in self.b[self.pos:self.pos + n]:
            v = (v << 1) | int(x)
        self.pos += n
        return v

    def zero(self, start: int, end: int):
        self.b[start:end] = 0


def mask_paging(data: bytes) -> tuple[bytes, Paging]:
    """Parse a PCCH-Message and zero every S-TMSI (MMEC + M-TMSI) and IMSI
    digit in it. Returns (masked bytes, summary). Anything that cannot be
    parsed with certainty is zeroed from that point on, so no identity can
    leak through an unexpected encoding."""
    bits = bytes_to_bits(data).copy()
    r = _Bits(bits)
    info = Paging()
    try:
        if r.read(1):                         # messageClassExtension
            raise ValueError("ext")
        has_list, has_sim, has_etws, _has_nce = (r.read(1) for _ in range(4))
        if has_list:
            n = r.read(4) + 1                 # SIZE (1..16)
            for _ in range(n):
                rec_ext = r.read(1)           # PagingRecord extension marker
                if r.read(1):                 # PagingUE-Identity extension
                    info.other += 1
                    raise ValueError("unknown identity type")
                if r.read(1) == 0:            # s-TMSI
                    a = r.pos
                    r.read(40)                # MMEC (8) + M-TMSI (32)
                    r.zero(a, r.pos)
                    info.s_tmsi += 1
                else:                         # imsi: SIZE (6..21) OF digit
                    nd = r.read(4) + 6
                    a = r.pos
                    r.read(4 * nd)
                    r.zero(a, r.pos)
                    info.imsi += 1
                info.cs_domain += r.read(1)   # cn-Domain {ps, cs}
                if rec_ext:
                    raise ValueError("record extensions")
        info.si_modification = bool(has_sim)
        info.etws = bool(has_etws)
    except ValueError:
        info.complete = False
        r.zero(r.pos, len(bits))
    return bits_to_bytes(bits)[:len(data)], info


# ----------------------------------------------------- test-signal encoders
def encode_rar(rapid: int, ta: int, ul_grant: int, temp_crnti: int) -> bytes:
    """Single-response MAC RAR PDU (used by the test signal generator)."""
    return bytes([0x40 | (rapid & 0x3F), (ta >> 4) & 0x7F,
                  ((ta & 0xF) << 4) | ((ul_grant >> 16) & 0xF),
                  (ul_grant >> 8) & 0xFF, ul_grant & 0xFF,
                  temp_crnti >> 8, temp_crnti & 0xFF])


def encode_paging(records, si_modification: bool = False) -> bytes:
    """PCCH-Message with records [("s-tmsi", mmec, m_tmsi, cs) or
    ("imsi", "digits", cs)] - synthetic identities for tests only."""
    out: list[int] = []

    def w(v, n):
        out.extend((v >> (n - 1 - i)) & 1 for i in range(n))
    w(0, 1)                                    # c1
    w(1 if records else 0, 1)
    w(int(si_modification), 1)
    w(0, 1)
    w(0, 1)
    if records:
        w(len(records) - 1, 4)
        for rec in records:
            w(0, 1)                            # no record extension
            w(0, 1)                            # identity in root
            if rec[0] == "s-tmsi":
                w(0, 1)
                w(rec[1], 8)
                w(rec[2], 32)
            else:
                w(1, 1)
                w(len(rec[1]) - 6, 4)
                for d in rec[1]:
                    w(int(d), 4)
            w(int(rec[-1]), 1)
    return bits_to_bytes(out)
