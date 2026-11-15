"""Minimal ASN.1 UPER decoding of the NR BCCH-DL-SCH message (TS 38.331):
SIB1 up to servingCellConfigCommon / frequencyInfoDL (cell selection,
PLMNs, TAC, NR cell identity, SI scheduling, bands and carriers). Also a
SIB1 encoder for the test signal generator."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..rrc import BitReader, BitWriter

SI_PERIODICITY = [8, 16, 32, 64, 128, 256, 512]
SI_WINDOW = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
SIB_TYPES = [f"SIB{n}" for n in range(2, 15)] + ["spare3", "spare2", "spare1"]
SCS = [15, 30, 60, 120, 240, 480, 960, 0]
# 38.101-1 Table 5.3.2-1: transmission bandwidth N_RB -> channel bandwidth [MHz]
CHANNEL_BW = {15: {25: 5, 52: 10, 79: 15, 106: 20, 133: 25, 160: 30, 188: 35, 216: 40, 242: 45,
                   270: 50},
              30: {11: 5, 24: 10, 38: 15, 51: 20, 65: 25, 78: 30, 92: 35, 106: 40, 119: 45,
                   133: 50, 162: 60, 189: 70, 217: 80, 245: 90, 273: 100}}


@dataclass
class NrPlmnInfo:
    plmns: list[tuple[str, str]]          # (MCC, MNC); MCC None = same as the previous
    tac: int | None
    ranac: int | None
    cell_identity: int                    # 36-bit NR cell identity
    reserved_for_operator: bool


@dataclass
class ScsCarrier:
    scs_khz: int
    offset_to_carrier: int                # RBs of this SCS from point A
    n_rb: int

    @property
    def channel_mhz(self) -> int | None:
        return CHANNEL_BW.get(self.scs_khz, {}).get(self.n_rb)


@dataclass
class NrSib1:
    q_rx_lev_min: int | None = None       # dBm
    q_qual_min: int | None = None         # dB
    plmn_info: list[NrPlmnInfo] = field(default_factory=list)
    reserved_for_other_use: bool = False
    conn_est_fail_count: int | None = None
    si: list[tuple[int, list[str]]] = field(default_factory=list)   # (period [frames], SIBs)
    si_window_slots: int | None = None
    si_request_config: bool = False
    bands: list[int] = field(default_factory=list)
    offset_to_point_a: int | None = None  # RBs of 15 kHz (FR1)
    carriers: list[ScsCarrier] = field(default_factory=list)
    complete: bool = True                 # parsed up to frequencyInfoDL

    @property
    def plmns(self) -> list[str]:
        out = []
        for pi in self.plmn_info:
            last = None
            for mcc, mnc in pi.plmns:
                mcc = mcc or last
                last = mcc
                out.append(f"{mcc}-{mnc}")
        return out


def _skip_extensions(r: BitReader):
    """Skip the extension additions of a SEQUENCE whose extension bit is set
    (normally small length, presence bitmap, open types)."""
    if r.bit():
        raise ValueError("too many extension additions")
    n = r.read(6) + 1
    present = [r.bit() for _ in range(n)]
    for p in present:
        if p:
            if not r.bit():
                ln = r.read(7)
            elif not r.bit():
                ln = r.read(14)
            else:
                raise ValueError("fragmented open type")
            r.pos += 8 * ln


def _plmn(r: BitReader) -> tuple[str | None, str]:
    mcc = None
    if r.bit():
        mcc = "".join(str(r.read(4)) for _ in range(3))
    n = r.length(2, 3)
    return mcc, "".join(str(r.read(4)) for _ in range(n))


def _plmn_info(r: BitReader) -> NrPlmnInfo:
    ext = r.bit()
    tac_p, ranac_p = r.bit(), r.bit()
    plmns = [_plmn(r) for _ in range(r.length(1, 12))]
    tac = r.read(24) if tac_p else None
    ranac = r.read(8) if ranac_p else None
    cid = r.read(36)
    reserved = r.read(1) == 0
    if ext:
        _skip_extensions(r)
    return NrPlmnInfo(plmns, tac, ranac, cid, reserved)


def decode_sib1(r: BitReader) -> NrSib1:
    s = NrSib1()
    (csi, cefc, sisi, sccc, _ims, _ecall, _uet, _uac, _rid, _late,
     _nce) = (r.bit() for _ in range(11))
    if csi:
        off_p, sul_p, qq_p, qqo_p = r.bit(), r.bit(), r.bit(), r.bit()
        s.q_rx_lev_min = 2 * r.constrained(-70, -22)
        if off_p:
            r.constrained(1, 8)
        if sul_p:
            r.constrained(-70, -22)
        if qq_p:
            s.q_qual_min = r.constrained(-43, -12)
        if qqo_p:
            r.constrained(1, 8)
    ext = r.bit()
    s.reserved_for_other_use = r.bit()
    s.plmn_info = [_plmn_info(r) for _ in range(r.length(1, 12))]
    if ext:
        _skip_extensions(r)
    if cefc:
        off_p = r.bit()
        s.conn_est_fail_count = r.read(2) + 1
        r.read(3)
        if off_p:
            r.read(4)
    if sisi:
        ext = r.bit()
        req, req_sul, area = r.bit(), r.bit(), r.bit()
        for _ in range(r.length(1, 32)):
            r.read(1)                                     # si-BroadcastStatus
            per = SI_PERIODICITY[r.constrained(0, 6)]
            sibs = []
            for _ in range(r.length(1, 32)):
                vt, _area = r.bit(), r.bit()
                sibs.append(SIB_TYPES[r.enum(16, ext=True)])
                if vt:
                    r.read(5)
            s.si.append((per, sibs))
        s.si_window_slots = SI_WINDOW[min(r.read(4), len(SI_WINDOW) - 1)]
        if req or req_sul:
            s.si_request_config = True
            s.complete = False                            # RACH-ConfigCommonSI not parsed
            return s
        if area:
            r.read(24)
        if ext:
            _skip_extensions(r)
    if sccc:
        r.bit()                                           # ServingCellConfigCommonSIB ext
        r.read(4)                                         # its optional fields
        r.bit()                                           # DownlinkConfigCommonSIB ext
        for _ in range(r.length(1, 8)):                   # frequencyBandList
            band_p, pmax_p = r.bit(), r.bit()
            if band_p:
                s.bands.append(r.constrained(1, 1024))
            if pmax_p:
                for _ in range(r.length(1, 8)):
                    if r.bit():
                        r.constrained(-30, 33)
                    r.read(3)
        s.offset_to_point_a = r.constrained(0, 2199)
        for _ in range(r.length(1, 5)):
            ext = r.bit()
            off = r.constrained(0, 2199)
            scs = SCS[r.read(3)]
            n_rb = r.constrained(1, 275)
            s.carriers.append(ScsCarrier(scs, off, n_rb))
            if ext:
                _skip_extensions(r)
    return s


def decode_bcch_dlsch(data: bytes):
    """-> ("SIB1", NrSib1) | ("SI", None) | (other, None). Raises
    ValueError on malformed data."""
    r = BitReader(data)
    if r.bit():
        return "messageClassExtension", None
    if r.bit():
        return "SIB1", decode_sib1(r)
    return "SI", None


def encode_sib1(mcc: str, mnc: str, tac: int, cell_identity: int, band: int,
                offset_to_point_a: int, scs_khz: int, n_rb: int, offset_to_carrier: int = 0,
                q_rx_lev_min: int = -140, si_sibs: tuple = ("SIB2", "SIB3")) -> bytes:
    """BCCH-DL-SCH-Message with a SIB1 (cell selection, one PLMN, SI
    scheduling, frequencyInfoDL; the rest of servingCellConfigCommon is
    left as zeros: enough for decode_sib1, not a complete message)."""
    w = BitWriter()
    w.write(0, 1)                    # c1
    w.write(1, 1)                    # systemInformationBlockType1
    w.write(0b10110000000, 11)       # cellSelectionInfo, si-SchedulingInfo, servingCellConfigCommon
    w.write(0, 4)                    # cellSelectionInfo optionals
    w.constrained(q_rx_lev_min // 2, -70, -22)
    w.write(0, 1)                    # CellAccessRelatedInfo ext
    w.write(0, 1)                    # cellReservedForOtherUse
    w.constrained(1, 1, 12)          # one PLMN-IdentityInfo
    w.write(0, 1)                    # ext
    w.write(1, 1)                    # TAC present
    w.write(0, 1)                    # ranac absent
    w.constrained(1, 1, 12)          # one PLMN
    w.write(1, 1)                    # mcc present
    for d in mcc:
        w.write(int(d), 4)
    w.constrained(len(mnc), 2, 3)
    for d in mnc:
        w.write(int(d), 4)
    w.write(tac, 24)
    w.write(cell_identity, 36)
    w.write(1, 1)                    # notReserved
    # si-SchedulingInfo
    w.write(0, 4)                    # ext, si-RequestConfig, -SUL, systemInformationAreaID
    w.constrained(1, 1, 32)
    w.write(0, 1)                    # broadcasting
    w.constrained(1, 0, 6)           # rf16
    w.constrained(len(si_sibs), 1, 32)
    for name in si_sibs:
        w.write(0, 2)                # valueTag, areaScope absent
        w.write(0, 1)                # type within the root
        w.write(SIB_TYPES.index(name), 4)
    w.write(1, 4)                    # si-WindowLength s10
    # servingCellConfigCommon
    w.write(0, 1)
    w.write(0, 4)
    w.write(0, 1)                    # DownlinkConfigCommonSIB ext
    w.constrained(1, 1, 8)
    w.write(1, 1)                    # freqBandIndicatorNR present
    w.write(0, 1)
    w.constrained(band, 1, 1024)
    w.constrained(offset_to_point_a, 0, 2199)
    w.constrained(1, 1, 5)
    w.write(0, 1)
    w.constrained(offset_to_carrier, 0, 2199)
    w.write(SCS.index(scs_khz), 3)
    w.constrained(n_rb, 1, 275)
    w.write(0, 64)                   # stand-in for the rest
    return w.to_bytes()
