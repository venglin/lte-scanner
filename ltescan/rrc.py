"""Minimal ASN.1 UPER decoding of BCCH-DL-SCH messages (TS 36.331):
SystemInformationBlockType1 in full (Rel-8 part), SystemInformation:
list of contained SIB types. Also an encoder for SIB1 used by the test
signal generator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coding import bits_to_bytes, bytes_to_bits

SI_PERIODICITY = [8, 16, 32, 64, 128, 256, 512]
SI_WINDOW = [1, 2, 5, 10, 15, 20, 40]
SIB_TYPES = [f"SIB{n}" for n in range(3, 19)]          # root of SIB-Type
SIB_CHOICE = ["SIB2", "SIB3", "SIB4", "SIB5", "SIB6", "SIB7", "SIB8",
              "SIB9", "SIB10", "SIB11"]                 # root of sib-TypeAndInfo
SIB_CHOICE_EXT = ["SIB12", "SIB13", "SIB14", "SIB15", "SIB16", "SIB17",
                  "SIB18", "SIB19", "SIB20", "SIB21"]


class BitReader:
    def __init__(self, data: bytes | np.ndarray):
        self.b = bytes_to_bits(data) if isinstance(data, (bytes, bytearray)) else np.asarray(data)
        self.pos = 0

    def read(self, n: int) -> int:
        if self.pos + n > len(self.b):
            raise ValueError("koniec danych ASN.1")
        v = 0
        for x in self.b[self.pos:self.pos + n]:
            v = (v << 1) | int(x)
        self.pos += n
        return v

    def bit(self) -> bool:
        return bool(self.read(1))

    def constrained(self, lo: int, hi: int) -> int:
        n = (hi - lo).bit_length()
        return lo + (self.read(n) if n else 0)

    def length(self, lo: int, hi: int) -> int:
        return self.constrained(lo, hi)

    def enum(self, n_root: int, ext: bool = False) -> int:
        if ext and self.bit():
            return n_root + self.read(7)   # normally-small extension value
        return self.constrained(0, n_root - 1)


class BitWriter:
    def __init__(self):
        self.bits: list[int] = []

    def write(self, v: int, n: int):
        self.bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]

    def constrained(self, v: int, lo: int, hi: int):
        self.write(v - lo, (hi - lo).bit_length())

    def to_bytes(self) -> bytes:
        return bits_to_bytes(self.bits)


@dataclass
class Plmn:
    mcc: str | None
    mnc: str
    reserved: bool

    def __str__(self):
        return f"{self.mcc or '???'}-{self.mnc}"


@dataclass
class Sib1:
    plmns: list[Plmn]
    tac: int
    cell_identity: int
    cell_barred: bool
    intra_freq_resel_allowed: bool
    csg_indication: bool
    csg_identity: int | None
    q_rx_lev_min_dbm: int
    q_rx_lev_min_offset: int | None
    p_max: int | None
    band: int
    scheduling: list[tuple[int, list[str]]]      # (periodicity rf, [SIBs])
    tdd_config: tuple[int, int] | None
    si_window_ms: int
    value_tag: int
    has_extension: bool = False

    @property
    def enb_id(self) -> int:
        return self.cell_identity >> 8

    @property
    def local_cell_id(self) -> int:
        return self.cell_identity & 0xFF

    def lines(self) -> list[str]:
        out = [
            f"PLMN: {', '.join(str(p) for p in self.plmns)}",
            f"TAC: {self.tac} (0x{self.tac:04x})",
            f"Cell Identity: {self.cell_identity} (eNB {self.enb_id}, sektor {self.local_cell_id})",
            f"Pasmo (freqBandIndicator): {self.band}",
            f"cellBarred: {'TAK' if self.cell_barred else 'nie'}, "
            f"intraFreqReselection: {'allowed' if self.intra_freq_resel_allowed else 'notAllowed'}",
            f"q-RxLevMin: {self.q_rx_lev_min_dbm} dBm"
            + (f", p-Max: {self.p_max} dBm" if self.p_max is not None else ""),
            f"SI window: {self.si_window_ms} ms, systemInfoValueTag: {self.value_tag}",
        ]
        if self.csg_indication:
            out.append(f"CSG: tak, id={self.csg_identity}")
        if self.tdd_config:
            out.append(f"TDD: subframeAssignment sa{self.tdd_config[0]}, "
                       f"specialSubframePatterns ssp{self.tdd_config[1]}")
        for i, (per, sibs) in enumerate(self.scheduling):
            sibs = (["SIB2"] if i == 0 else []) + list(sibs)   # SIB2 always in SI-1
            out.append(f"SI-{i + 1}: co {per} ramek: {', '.join(sibs) or '-'}")
        return out


@dataclass
class SystemInformation:
    sibs: list[str] = field(default_factory=list)
    n_sibs: int = 0


def _plmn(r: BitReader) -> Plmn:
    has_mcc = r.bit()
    mcc = "".join(str(r.read(4)) for _ in range(3)) if has_mcc else None
    n_mnc = r.length(2, 3)
    mnc = "".join(str(r.read(4)) for _ in range(n_mnc))
    return Plmn(mcc, mnc, False)


def decode_sib1(r: BitReader) -> Sib1:
    opt_pmax, opt_tdd, opt_ext = r.bit(), r.bit(), r.bit()
    # cellAccessRelatedInfo
    opt_csg = r.bit()
    n_plmn = r.length(1, 6)
    plmns = []
    last_mcc = None
    for _ in range(n_plmn):
        p = _plmn(r)
        if p.mcc is None:
            p.mcc = last_mcc            # absent MCC = same as previous entry
        last_mcc = p.mcc
        p.reserved = r.read(1) == 0
        plmns.append(p)
    tac = r.read(16)
    cid = r.read(28)
    barred = r.read(1) == 0
    intra = r.read(1) == 0
    csg_ind = r.bit()
    csg_id = r.read(27) if opt_csg else None
    # cellSelectionInfo
    opt_off = r.bit()
    qrx = r.constrained(-70, -22) * 2
    qoff = r.constrained(1, 8) if opt_off else None
    pmax = r.constrained(-30, 33) if opt_pmax else None
    band = r.constrained(1, 64)
    n_si = r.length(1, 32)
    sched = []
    for _ in range(n_si):
        per = SI_PERIODICITY[r.constrained(0, 6)]
        n_sib = r.length(0, 31)
        sibs = []
        for _ in range(n_sib):
            if r.bit():                 # extension value
                sibs.append(f"SIB-ext{r.read(7)}")
            else:
                sibs.append(SIB_TYPES[r.read(4)])
        sched.append((per, sibs))
    tdd = (r.constrained(0, 6), r.constrained(0, 8)) if opt_tdd else None
    win = SI_WINDOW[r.constrained(0, 6)]
    tag = r.read(5)
    return Sib1(plmns, tac, cid, barred, intra, csg_ind, csg_id, qrx, qoff, pmax,
                band, sched, tdd, win, tag, opt_ext)


def decode_si(r: BitReader) -> SystemInformation:
    if r.bit():                        # criticalExtensionsFuture
        return SystemInformation()
    r.bit()                            # nonCriticalExtension present?
    n = r.length(1, 32)
    si = SystemInformation(n_sibs=n)
    # only the type of the first element can be read without decoding it
    if r.bit():
        idx = r.read(7)                # normally small non-negative number
        si.sibs.append(SIB_CHOICE_EXT[idx] if idx < len(SIB_CHOICE_EXT) else f"ext{idx}")
    else:
        si.sibs.append(SIB_CHOICE[r.read(4)])
    return si


def decode_bcch_dlsch(data: bytes):
    """Returns Sib1, SystemInformation or None."""
    r = BitReader(data)
    if r.bit():                        # messageClassExtension
        return None
    if r.bit():
        return decode_sib1(r)
    return decode_si(r)


# --------------------------------------------------------------- encoder


def encode_sib1(mcc: str, mnc: str, tac: int, cell_id: int, band: int,
                q_rx_lev_min: int = -120, sched=((16, []),), si_window: int = 20,
                tag: int = 0, tdd_config=None, p_max: int | None = None) -> bytes:
    w = BitWriter()
    w.write(0, 1)              # c1
    w.write(1, 1)              # systemInformationBlockType1
    w.write(int(p_max is not None), 1)
    w.write(int(tdd_config is not None), 1)
    w.write(0, 1)              # no nonCriticalExtension
    w.write(0, 1)              # no csg-Identity
    w.constrained(1, 1, 6)     # one PLMN
    w.write(1, 1)              # mcc present
    for d in mcc:
        w.write(int(d), 4)
    w.constrained(len(mnc), 2, 3)
    for d in mnc:
        w.write(int(d), 4)
    w.write(1, 1)              # notReserved
    w.write(tac, 16)
    w.write(cell_id, 28)
    w.write(1, 1)              # notBarred
    w.write(0, 1)              # intraFreqReselection allowed
    w.write(0, 1)              # csg-Indication false
    w.write(0, 1)              # no q-RxLevMinOffset
    w.constrained(q_rx_lev_min // 2, -70, -22)
    if p_max is not None:
        w.constrained(p_max, -30, 33)
    w.constrained(band, 1, 64)
    w.constrained(len(sched), 1, 32)
    for per, sibs in sched:
        w.constrained(SI_PERIODICITY.index(per), 0, 6)
        w.constrained(len(sibs), 0, 31)
        for s in sibs:
            w.write(0, 1)
            w.write(SIB_TYPES.index(s), 4)
    if tdd_config is not None:
        w.constrained(tdd_config[0], 0, 6)
        w.constrained(tdd_config[1], 0, 8)
    w.constrained(SI_WINDOW.index(si_window), 0, 6)
    w.write(tag, 5)
    return w.to_bytes()
