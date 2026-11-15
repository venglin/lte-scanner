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


SS_PERIODS = [1, 2, 4, 5, 8, 10, 16, 20, 40, 80, 160, 320, 640, 1280, 2560]
CANDIDATES = [0, 1, 2, 3, 4, 5, 6, 8]


@dataclass
class SearchSpace:
    id: int
    coreset_id: int | None
    period: int | None                 # slots
    offset: int | None
    duration: int | None
    first_symbols: list[int] | None    # from monitoringSymbolsWithinSlot
    candidates: dict | None            # aggregation level -> number of candidates
    common: bool = True


@dataclass
class CoresetCfg:
    id: int
    freq_bitmap: int                   # 45 bits, one per 6 RBs
    duration: int
    interleaved: bool
    reg_bundle: int | None = None
    interleaver: int | None = None
    shift: int | None = None


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
    # initialDownlinkBWP (when it could be parsed)
    bwp_location: int | None = None       # locationAndBandwidth (RIV over 275 RBs)
    search_spaces: dict = field(default_factory=dict)       # id -> SearchSpace
    common_coreset: CoresetCfg | None = None
    ss_sib1: int | None = None
    ss_osi: int | None = None             # searchSpaceOtherSystemInformation
    ss_paging: int | None = None
    ss_ra: int | None = None
    pdsch_tdra: list = field(default_factory=list)          # (K0, mapping, S, L)

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


def _search_space(r: BitReader) -> SearchSpace:
    cs_p, per_p, dur_p, sym_p, cand_p, type_p = (r.bit() for _ in range(6))
    ss = SearchSpace(r.read(6), None, None, None, None, None, None)
    if cs_p:
        ss.coreset_id = r.constrained(0, 11)
    if per_p:
        i = r.read(4)
        ss.period = SS_PERIODS[i]
        ss.offset = r.constrained(0, ss.period - 1) if ss.period > 1 else 0
    if dur_p:
        ss.duration = r.constrained(2, 2559)
    if sym_p:
        bm = r.read(14)
        ss.first_symbols = [l for l in range(14) if bm >> (13 - l) & 1]
    if cand_p:
        c = [CANDIDATES[r.read(3)] for _ in range(5)]
        ss.candidates = dict(zip((1, 2, 4, 8, 16), c))
    if type_p:
        if r.bit():                                       # ue-Specific
            raise ValueError("UE-specific search space in SIB1")
        f00, f20, f21, f22, f23 = (r.bit() for _ in range(5))
        if f00:
            if r.bit():
                _skip_extensions(r)
        if f20:
            ext = r.bit()
            pres = [r.bit() for _ in range(5)]
            for p in pres:
                if p:
                    r.read(1)
            if ext:
                _skip_extensions(r)
        for f in (f21, f22):
            if f and r.bit():
                _skip_extensions(r)
        if f23:
            ext = r.bit()
            if r.bit():
                r.read(3)
            r.read(1)
            if ext:
                _skip_extensions(r)
    return ss


def _coreset(r: BitReader) -> CoresetCfg:
    ext = r.bit()
    add_p, rel_p, _tci_p, scr_p = r.bit(), r.bit(), r.bit(), r.bit()
    cid = r.constrained(0, 11)
    bm = r.read(45)
    dur = r.constrained(1, 3)
    cs = CoresetCfg(cid, bm, dur, False)
    if r.bit() == 0:                                      # interleaved
        sh_p = r.bit()
        cs.interleaved = True
        cs.reg_bundle = (2, 3, 6)[r.read(2)]
        cs.interleaver = (2, 3, 6)[r.read(2)]
        if sh_p:
            cs.shift = r.constrained(0, 274)
    r.read(1)                                             # precoderGranularity
    for p in (add_p, rel_p):
        if p:
            for _ in range(r.length(1, 64)):
                r.read(7)
    if scr_p:
        r.read(16)
    if ext:
        _skip_extensions(r)
    return cs


def _initial_bwp(r: BitReader, s: NrSib1):
    """BWP-DownlinkCommon: genericParameters, PDCCH-ConfigCommon (search
    spaces) and PDSCH-ConfigCommon (time domain allocation list)."""
    ext, pdcch_p, pdsch_p = r.bit(), r.bit(), r.bit()
    r.bit()                                               # cyclicPrefix present (no bits)
    s.bwp_location = r.constrained(0, 37949)
    r.read(3)                                             # subcarrierSpacing
    if pdcch_p and r.bit():                               # SetupRelease: setup
        pext = r.bit()
        cz, ccs, sz, ssl, s1, osi, pg, ra = (r.bit() for _ in range(8))
        if cz:
            r.read(4)
        if ccs:
            s.common_coreset = _coreset(r)
        if sz:
            r.read(4)
        if ssl:
            for _ in range(r.length(1, 4)):
                ss = _search_space(r)
                s.search_spaces[ss.id] = ss
        if s1:
            s.ss_sib1 = r.constrained(0, 39)
        if osi:
            s.ss_osi = r.constrained(0, 39)
        if pg:
            s.ss_paging = r.constrained(0, 39)
        if ra:
            s.ss_ra = r.constrained(0, 39)
        if pext:
            _skip_extensions(r)
    if pdsch_p and r.bit():
        dext = r.bit()
        if r.bit():
            for _ in range(r.length(1, 16)):
                k0_p = r.bit()
                k0 = r.constrained(0, 32) if k0_p else 0
                mapping = "AB"[r.read(1)]
                a, b = divmod(r.read(7), 14)
                s_, l_ = (b, a + 1) if a + b + 1 <= 14 else (13 - b, 15 - a)
                s.pdsch_tdra.append((k0, mapping, s_, l_))
        if dext:
            _skip_extensions(r)
    if ext:
        _skip_extensions(r)


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
        try:
            _initial_bwp(r, s)
        except (ValueError, IndexError):
            pass                                          # keep what was parsed
    return s


def decode_bcch_dlsch(data: bytes):
    """-> ("SIB1", NrSib1) | ("SI", NrSystemInformation) | (other, None). Raises
    ValueError on malformed data."""
    r = BitReader(data)
    if r.bit():
        return "messageClassExtension", None
    if r.bit():
        return "SIB1", decode_sib1(r)
    return "SI", decode_si(r)


def encode_sib1(mcc: str, mnc: str, tac: int, cell_identity: int, band: int,
                offset_to_point_a: int, scs_khz: int, n_rb: int, offset_to_carrier: int = 0,
                q_rx_lev_min: int = -140, si_sibs: tuple = ("SIB2", "SIB3"),
                si: list | None = None, si_window: int = 10, osi_ss: SearchSpace | None = None,
                pdsch_tdra: list | None = None) -> bytes:
    """BCCH-DL-SCH-Message with a SIB1 (cell selection, one PLMN, SI
    scheduling, frequencyInfoDL; the rest of servingCellConfigCommon is
    left as zeros: enough for decode_sib1, not a complete message).
    si: [(period in frames, [SIB names]), ...] (default: si_sibs every 16
    frames); si_window in slots. osi_ss / pdsch_tdra: a common search space
    on CORESET#0 used as searchSpaceOtherSystemInformation, and the
    pdsch-TimeDomainAllocationList [(K0, mapping, S, L)] of the initial BWP."""
    si = si or [(16, list(si_sibs))]
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
    w.constrained(len(si), 1, 32)
    for period, sibs in si:
        w.write(0, 1)                # broadcasting
        w.constrained(SI_PERIODICITY.index(period), 0, 6)
        w.constrained(len(sibs), 1, 32)
        for name in sibs:
            w.write(0, 2)            # valueTag, areaScope absent
            w.write(0, 1)            # type within the root
            w.write(SIB_TYPES.index(name), 4)
    w.write(SI_WINDOW.index(si_window), 4)
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
    # initialDownlinkBWP
    w.write(0, 1)                    # ext
    w.write(1 if osi_ss else 0, 1)
    w.write(1 if pdsch_tdra else 0, 1)
    w.write(0, 1)                    # cyclicPrefix
    w.constrained(0, 0, 37949)
    w.write(SCS.index(scs_khz), 3)
    if osi_ss:
        ss = osi_ss
        w.write(1, 1)                # setup
        w.write(0, 1)                # ext
        w.write(0b00011100, 8)       # commonSearchSpaceList, searchSpaceSIB1, -OtherSystemInformation
        w.constrained(1, 1, 4)
        w.write(0b110111, 6)         # coreset, periodicity, (no duration), symbols, candidates, type
        w.write(ss.id, 6)
        w.constrained(ss.coreset_id or 0, 0, 11)
        w.write(SS_PERIODS.index(ss.period), 4)
        if ss.period > 1:
            w.constrained(ss.offset, 0, ss.period - 1)
        w.write(sum(1 << (13 - l) for l in ss.first_symbols), 14)
        for lv in (1, 2, 4, 8, 16):
            w.write(CANDIDATES.index((ss.candidates or {}).get(lv, 0)), 3)
        w.write(0, 1)                # common
        w.write(0b10000, 5)          # dci-Format0-0-AndFormat1-0
        w.write(0, 1)
        w.constrained(0, 0, 39)      # searchSpaceSIB1
        w.constrained(ss.id, 0, 39)  # searchSpaceOtherSystemInformation
    if pdsch_tdra:
        w.write(1, 1)                # setup
        w.write(0, 1)
        w.write(1, 1)
        w.constrained(len(pdsch_tdra), 1, 16)
        for k0, mapping, s_, l_ in pdsch_tdra:
            w.write(1 if k0 else 0, 1)
            if k0:
                w.constrained(k0, 0, 32)
            w.write("AB".index(mapping), 1)
            sliv = 14 * (l_ - 1) + s_ if l_ - 1 <= 7 else 14 * (14 - l_ + 1) + (13 - s_)
            w.write(sliv, 7)
    w.write(0, 64)                   # stand-in for the rest
    return w.to_bytes()


# ------------------------------------------------------- SystemInformation
Q_HYST = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
T_EVAL = [30, 60, 120, 180, 240, 0, 0, 0]
SI_CHOICE = ["SIB2", "SIB3", "SIB4", "SIB5", "SIB6", "SIB7", "SIB8", "SIB9"]
SI_CHOICE_EXT = ["SIB10", "SIB11", "SIB12", "SIB13", "SIB14", "SIB15", "SIB16", "SIB17",
                 "SIB18", "SIB19", "SIB20", "SIB21"]


@dataclass
class NrSib2:
    """Head of SIB2: common reselection parameters of the serving frequency."""
    q_hyst_db: int
    thresh_serving_low_p: int        # x 2 dB
    cell_reselection_priority: int
    s_non_intra_search_p: int | None  # x 2 dB
    q_rx_lev_min: int                 # dBm
    s_intra_search_p: int             # x 2 dB
    t_reselection_nr: int             # s


@dataclass
class NrSystemInformation:
    n_sibs: int                       # elements of sib-TypeAndInfo
    first: str                        # type of the first element
    sib2: NrSib2 | None = None


def _sib2(r: BitReader) -> NrSib2:
    r.bit()                                               # SIB2 ext
    r.bit()                                               # cellReselectionInfoCommon ext
    nrof, cons, rng, speed = r.bit(), r.bit(), r.bit(), r.bit()
    if nrof:
        r.constrained(2, 16)
    if cons:
        p = [r.bit() for _ in range(3)]
        for x in p:
            if x:
                r.read(7)
    if rng:
        r.read(5)                                         # Q-OffsetRange
    q_hyst = Q_HYST[r.read(4)]
    if speed:
        r.read(3 + 3 + 4 + 4)                             # MobilityStateParameters
        r.read(2 + 2)                                     # q-HystSF
    r.bit()                                               # cellReselectionServingFreqInfo ext
    snp, snq, tlq, sub = r.bit(), r.bit(), r.bit(), r.bit()
    s_non = r.read(5) if snp else None
    if snq:
        r.read(5)
    tlp = r.read(5)
    if tlq:
        r.read(5)
    prio = r.read(3)
    if sub:
        r.read(2)
    r.bit()                                               # intraFreqCellReselectionInfo ext
    sul, qq, siq = r.bit(), r.bit(), r.bit()
    r.read(6)                                             # band lists, p-Max, smtc, ss-RSSI, ssb-ToMeasure
    q_rx = 2 * r.constrained(-70, -22)
    if sul:
        r.read(6)
    if qq:
        r.read(5)
    s_intra = r.read(5)
    if siq:
        r.read(5)
    t_res = r.read(3)
    return NrSib2(q_hyst, tlp, prio, s_non, q_rx, s_intra, t_res)


def decode_si(r: BitReader) -> NrSystemInformation:
    """SystemInformation (after the BCCH-DL-SCH choice bits): number and
    first type of the contained SIBs, and the head of SIB2 when it is first.
    The SIBs are not open types, so the others need a full parser (the
    Wireshark export shows them)."""
    if r.bit():
        raise ValueError("criticalExtensionsFuture")
    r.read(2)                                             # late / nonCritical extension
    n = r.length(1, 32)
    if r.bit():                                           # extension alternative
        if r.bit():
            raise ValueError("large choice index")
        idx = r.read(6)                                   # normally small number
        first = SI_CHOICE_EXT[idx] if idx < len(SI_CHOICE_EXT) else f"ext{idx}"
        return NrSystemInformation(n, first)
    first = SI_CHOICE[r.read(3)]
    si = NrSystemInformation(n, first)
    if first == "SIB2":
        try:
            si.sib2 = _sib2(r)
        except ValueError:
            pass
    return si


def encode_si(sib2: NrSib2 | None = None, n_sibs: int = 1, pad: int = 8,
              first: str = "SIB2") -> bytes:
    """BCCH-DL-SCH SystemInformation whose first element is SIB2 with the
    given head, or another root SIB type (the rest stand-in bytes; for the
    test generator)."""
    s = sib2 or NrSib2(4, 2, 5, 10, -120, 30, 1)
    w = BitWriter()
    w.write(0, 1)                    # c1
    w.write(0, 1)                    # systemInformation
    w.write(0, 1)                    # criticalExtensions: systemInformation
    w.write(0, 2)
    w.constrained(n_sibs, 1, 32)
    w.write(0, 1)                    # root alternative
    w.write(SI_CHOICE.index(first), 3)
    if first != "SIB2":
        w.write(0, 8 * pad)
        return w.to_bytes()
    w.write(0, 1)
    w.write(0, 1)
    w.write(0, 4)
    w.write(Q_HYST.index(s.q_hyst_db), 4)
    w.write(0, 1)
    w.write(1 if s.s_non_intra_search_p is not None else 0, 1)
    w.write(0, 3)
    if s.s_non_intra_search_p is not None:
        w.write(s.s_non_intra_search_p, 5)
    w.write(s.thresh_serving_low_p, 5)
    w.write(s.cell_reselection_priority, 3)
    w.write(0, 1)
    w.write(0, 9)
    w.constrained(s.q_rx_lev_min // 2, -70, -22)
    w.write(s.s_intra_search_p, 5)
    w.write(s.t_reselection_nr, 3)
    w.write(0, 1)                    # deriveSSB-IndexFromCell
    w.write(0, 8 * pad)
    return w.to_bytes()
