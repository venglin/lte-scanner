"""High level LTE downlink receiver: cell search -> MIB -> control channel
-> broadcast PDSCH (SIB1 / SI) with measurements and statistics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import chest, control, dci as dcimod, mac, mimo, pbch, pdsch, rrc
from .coding import bits_to_bytes
from .dsp import mix, resample
from .ofdm import bin_index, demod_subframe, subframe_len, symbol_offsets
from .params import BASE_RATE, FFT_SIZE, TDD_MI, TDD_UL_DL, CellConfig, freq_to_earfcn, earfcn_to_band, sample_rate
from .sequences import pss
from .sync import CellSync, cell_search


# ------------------------------------------------------------ timing
class Demodulator:
    """Subframe-level access to a CFO-corrected buffer, with timing derived
    from tracking PSS (takes care of sampling clock offset)."""

    def __init__(self, x: np.ndarray, fs: float, cell: CellSync, fft: int):
        self.x = x
        self.fs = fs
        self.fft = fft
        self.cell = cell
        self.sf_len = subframe_len(fft)
        r = fs / BASE_RATE
        self.t0 = cell.frame_start * r
        self.spacing = float(self.sf_len)
        self._track()

    def _pss_template(self):
        f = np.zeros(self.fft, dtype=complex)
        f[bin_index(self.fft, 72)[5:67]] = pss(self.cell.n_id_2)
        t = np.fft.ifft(f)
        return t / np.linalg.norm(t)

    def _track(self):
        cell = self.cell
        n_symb = 6 if cell.cp_ext else 7
        offs = symbol_offsets(self.fft, cell.cp_ext)
        if cell.tdd:
            sf_pss, off = 1, offs[2]
        else:
            sf_pss, off = 0, offs[n_symb - 1]
        t = self._pss_template()
        w = max(8, int(4 * self.fs / BASE_RATE))
        hs, ps = [], []
        h = 0
        while True:
            p = int(round(self.t0 + (sf_pss + 5 * h) * self.spacing + off))
            if p + w + self.fft > len(self.x):
                break
            if p - w >= 0:
                seg = self.x[p - w:p + w + self.fft]
                c = np.abs(signal.correlate(seg, t, mode="valid"))
                i = int(np.argmax(c))
                if 0 < i < len(c) - 1:
                    a, b, cc = c[i - 1], c[i], c[i + 1]
                    den = a - 2 * b + cc
                    frac = 0.5 * (a - cc) / den if den != 0 else 0.0
                else:
                    frac = 0.0
                hs.append(h)
                ps.append(p - w + i + frac - off - sf_pss * self.spacing)
            h += 1
        if len(hs) >= 2:
            slope, icpt = np.polyfit(hs, ps, 1)
            if abs(slope / (5 * self.sf_len) - 1) < 1e-3:
                self.spacing = slope / 5
                self.t0 = icpt
            else:
                self.t0 = float(np.median(np.array(ps) - np.array(hs) * 5 * self.spacing))
        elif ps:
            self.t0 = ps[0]
        # the frame start must not be negative
        while self.t0 < 0:
            self.t0 += 10 * self.spacing
        self.ppm = (self.spacing / self.sf_len - 1) * 1e6

    def n_subframes(self) -> int:
        return int((len(self.x) - self.t0 - self.fft) // self.spacing) - 1

    def grid(self, j: int, n_sc: int) -> np.ndarray:
        start = int(round(self.t0 + j * self.spacing))
        return demod_subframe(self.x, start, self.fft, n_sc, self.cell.cp_ext)


# ------------------------------------------------------------ results
@dataclass
class DciEvent:
    sfn: int
    sf: int
    rnti: int
    fmt: str
    cce: int
    agg: int
    text: str


@dataclass
class CellReport:
    sync: CellSync
    fc: float | None = None
    mib: pbch.Mib | None = None
    rsrp_dbfs: float = float("nan")
    rsrq_db: float = float("nan")
    snr_db: float = float("nan")
    ppm: float = float("nan")
    cfg: CellConfig | None = None
    sib1: rrc.Sib1 | None = None
    sib1_raw: bytes | None = None
    sib1_tx: int = 0
    si: list = field(default_factory=list)
    cfi_hist: Counter = field(default_factory=Counter)
    rntis: Counter = field(default_factory=Counter)
    dcis: list = field(default_factory=list)
    paging: int = 0
    rar: int = 0
    rar_entries: list = field(default_factory=list)      # mac.RarEntry
    paging_msgs: list = field(default_factory=list)      # mac.Paging (masked)
    cce_load: list = field(default_factory=list)
    subframes: int = 0
    notes: list = field(default_factory=list)
    diag: dict = field(default_factory=dict, repr=False)

    @property
    def earfcn(self):
        return freq_to_earfcn(self.fc) if self.fc else None

    @property
    def band(self):
        e = self.earfcn
        return earfcn_to_band(e) if e is not None else None

    @property
    def cfo_hz(self):
        return self.sync.cfo

    def confirmed_rntis(self, min_count: int = 2) -> dict:
        """C-RNTIs seen at least `min_count` times (single hits may be
        false CRC matches)."""
        return {r: n for r, n in self.rntis.most_common()
                if n >= min_count and 0x3D <= r <= 0xFFF3}


# ------------------------------------------------------------ receiver
class LteReceiver:
    def __init__(self, x: np.ndarray, fs: float, fc: float | None = None,
                 cfo_max: float = 40e3):
        self.x = np.asarray(x, dtype=np.complex64)
        self.fs = fs
        self.fc = fc
        self.cfo_max = cfo_max
        # minimum LLR-weighted code agreement per aggregation level; random
        # data reaches at most ~0.955 / 0.83 / 0.73 / 0.67
        self.min_reliability = {1: 0.975, 2: 0.9, 4: 0.82, 8: 0.76}
        self.x192 = resample(self.x - np.mean(self.x), fs, BASE_RATE)

    def search(self, **kw) -> list[CellSync]:
        kw.setdefault("cfo_max", self.cfo_max)
        return cell_search(self.x192, **kw)

    # ------------------------------------------------------------- MIB
    def _mib(self, cell: CellSync, rep: CellReport):
        x = mix(self.x192, -cell.cfo, BASE_RATE)
        dem = Demodulator(x, BASE_RATE, cell, 128)
        cfg6 = CellConfig(cell.n_id, 6, cell.cp_ext, cell.tdd, 4)
        frames, meas = [], []
        n_frames = dem.n_subframes() // 10
        for k in range(min(n_frames, 8)):
            try:
                g = dem.grid(10 * k, 72)
            except IndexError:
                break
            ce = chest.estimate(g, cfg6, 0, 4)
            frames.append((g, ce.h, ce.noise))
            meas.append(ce)
        if not frames:
            rep.notes.append("not enough data to decode the PBCH")
            return None
        rep.rsrp_dbfs = 10 * np.log10(np.mean([m.rsrp for m in meas]) / 128)
        rep.snr_db = float(np.mean([m.snr_db for m in meas]))
        rep.rsrq_db = float(np.mean([m.rsrq_db for m in meas]))
        return pbch.decode_mib(frames, cfg6)

    # ------------------------------------------------------------- full
    def analyze(self, cell: CellSync, decode_sib: bool = True, rnti_discovery: bool = False,
                max_subframes: int | None = None, verbose=None, on_pdu=None,
                decode_rar: bool = True, decode_paging: bool = False) -> CellReport:
        """on_pdu(kind, rnti, sfn, sf, payload, t_offset) is called for every
        decoded transport block: kind is "BCH" (MIB), "SI", "RAR" or "PCH"
        (paging, with subscriber identities masked); t_offset is seconds
        from the start of the buffer. Paging is decoded only when
        decode_paging is set."""
        rep = CellReport(cell, self.fc)
        self._on_pdu = on_pdu
        self._decode_rar = decode_rar
        self._decode_paging = decode_paging
        if self.fc:
            rep.ppm = -cell.cfo / self.fc * 1e6
        mib = self._mib(cell, rep)
        rep.mib = mib
        if mib is None:
            rep.notes.append("MIB could not be decoded")
            return rep
        if on_pdu:
            raw = pbch.mib_pack(mib.n_rb, mib.phich_ext, mib.phich_ng, mib.sfn, mib.spare)
            on_pdu("BCH", 0, mib.sfn, 0, bits_to_bytes(raw), cell.frame_start / BASE_RATE)
        cfg = CellConfig(cell.n_id, mib.n_rb, cell.cp_ext, cell.tdd, mib.n_ports,
                         mib.phich_ng, mib.phich_ext)
        rep.cfg = cfg
        if not decode_sib:
            return rep
        occupied = mib.n_rb * 180e3
        if self.fs < occupied:
            rep.notes.append(f"sample rate {self.fs / 1e6:.2f} Msps < occupied bandwidth "
                             f"{occupied / 1e6:.1f} MHz: edge subcarriers are lost, "
                             "PDCCH/PDSCH decoding may fail")
        fs_lte = sample_rate(mib.n_rb)
        x = resample(self.x - np.mean(self.x), self.fs, fs_lte)
        x = mix(x, -cell.cfo, fs_lte)
        dem = Demodulator(x, fs_lte, cell, FFT_SIZE[mib.n_rb])
        if np.isnan(rep.ppm):
            rep.ppm = dem.ppm
        self._decode_subframes(dem, cfg, mib, rep, rnti_discovery, max_subframes, verbose)
        return rep

    def _decode_subframes(self, dem, cfg, mib, rep, rnti_discovery, max_sf, verbose):
        n_sf = dem.n_subframes()
        if max_sf:
            n_sf = min(n_sf, max_sf)
        size_1a = dcimod.size_0_1a(cfg.n_rb, cfg.tdd)
        size_1c = dcimod.size_1c(cfg.n_rb)
        size_1 = dcimod.size_1(cfg.n_rb, cfg.tdd)
        sib1_buf = None
        tdd_cfg = None
        for j in range(n_sf):
            sfn = (mib.sfn + j // 10) % 1024
            sf = j % 10
            if cfg.tdd:
                pattern = TDD_UL_DL[tdd_cfg] if tdd_cfg is not None else None
                if pattern is None and sf not in (0, 5):
                    continue
                if pattern is not None and pattern[sf] != "D":
                    continue
            try:
                g = dem.grid(j, cfg.n_sc)
            except IndexError:
                break
            rep.subframes += 1
            ce = chest.estimate(g, cfg, sf)
            if j < 10:
                rep.diag.setdefault("frame", []).append(g)
                if sf == 0:
                    ls, ks = pbch.pbch_re(cfg.n_id, cfg.n_rb, cfg.cp_ext)
                    x_eq, _ = mimo.td_decode(g[ls, ks], ce.h[:, ls, ks], cfg.n_ports)
                    rep.diag["pbch"] = x_eq
                    rep.diag["h"] = ce.h
            cfi, _conf = control.decode_pcfich(g, ce.h, ce.noise, cfg, sf)
            rep.cfi_hist[cfi] += 1
            sizes = (size_1a, size_1c, size_1)
            if cfg.tdd and tdd_cfg is None:
                # PHICH size (m_i) unknown until SIB1 gives the TDD config:
                # pick the hypothesis on which a common-search-space DCI
                # decodes; UE-specific discovery only on a confirmed layout
                found, load = [], None
                for m_i in (1, 0, 2):
                    found, load = self._pdcch(g, ce, cfg, sf, sfn, cfi, m_i, sizes, False)
                    if found:
                        if rnti_discovery:
                            found, load = self._pdcch(g, ce, cfg, sf, sfn, cfi, m_i, sizes, True)
                        break
            else:
                m_i = TDD_MI[tdd_cfg][sf] if cfg.tdd else 1
                found, load = self._pdcch(g, ce, cfg, sf, sfn, cfi, m_i, sizes, rnti_discovery)
            self._record(rep, found, load)
            t_sf = (dem.t0 + j * dem.spacing) / dem.fs
            for ev, d in found:
                if d.fmt not in ("1A", "1C"):
                    continue
                if d.rnti != dcimod.SI_RNTI:
                    self._other_broadcast(g, ce, cfg, sf, sfn, cfi, d, rep, t_sf)
                    continue
                is_sib1 = sf == 5 and sfn % 2 == 0
                if is_sib1:
                    if rep.sib1 is not None:
                        continue
                    rv = d.rv if d.fmt == "1A" else pdsch.sib1_rv(sfn)
                    data, sib1_buf = pdsch.decode_pdsch(g, ce.h, ce.noise, cfg, sf, cfi, d,
                                                        rv, sib1_buf)
                    rep.sib1_tx += 1
                    rep.diag["pdsch"] = pdsch.pdsch_symbols(g, ce.h, cfg, sf, cfi, d)[0]
                    if data:
                        self._emit("SI", d.rnti, sfn, sf, data, t_sf)
                        self._handle_bcch(data, rep, sfn, sf, True)
                        if rep.sib1 and rep.sib1.tdd_config and cfg.tdd:
                            tdd_cfg = rep.sib1.tdd_config[0]
                else:
                    rvs = [d.rv] if d.fmt == "1A" else [0, 2, 3, 1]
                    for rv in rvs:
                        data, _ = pdsch.decode_pdsch(g, ce.h, ce.noise, cfg, sf, cfi, d, rv)
                        if data:
                            self._emit("SI", d.rnti, sfn, sf, data, t_sf)
                            self._handle_bcch(data, rep, sfn, sf, False)
                            break
            if verbose:
                verbose(j, n_sf)

    def _emit(self, kind, rnti, sfn, sf, data, t):
        if getattr(self, "_on_pdu", None):
            self._on_pdu(kind, rnti, sfn, sf, data, t)

    def _other_broadcast(self, g, ce, cfg, sf, sfn, cfi, d, rep, t_sf):
        """PDSCH of RA-RNTI (random access response) and P-RNTI (paging)."""
        is_rar = 1 <= d.rnti <= 60
        if is_rar and not self._decode_rar:
            return
        if d.rnti == dcimod.P_RNTI and not self._decode_paging:
            return
        if not is_rar and d.rnti != dcimod.P_RNTI:
            return
        rvs = [d.rv] if d.fmt == "1A" else [0, 2, 3, 1]
        for rv in rvs:
            data, _ = pdsch.decode_pdsch(g, ce.h, ce.noise, cfg, sf, cfi, d, rv)
            if data:
                break
        else:
            return
        if is_rar:
            r = mac.parse_rar(data)
            rep.rar_entries += r.entries
            self._emit("RAR", d.rnti, sfn, sf, data, t_sf)
        else:
            masked, info = mac.mask_paging(data)
            rep.paging_msgs.append(info)
            self._emit("PCH", d.rnti, sfn, sf, masked, t_sf)

    def _handle_bcch(self, data: bytes, rep: CellReport, sfn, sf, is_sib1):
        try:
            msg = rrc.decode_bcch_dlsch(data)
        except Exception as e:                       # malformed / unsupported
            rep.notes.append(f"SFN {sfn}.{sf}: ASN.1 decoding error: {e}")
            return
        if isinstance(msg, rrc.Sib1):
            rep.sib1 = msg
            rep.sib1_raw = data
        elif isinstance(msg, rrc.SystemInformation):
            rep.si.append((sfn, sf, msg, data))

    @staticmethod
    def _record(rep, found, load):
        for ev, d in found:
            rep.dcis.append(ev)
            rep.rntis[ev.rnti] += 1
            if ev.rnti == dcimod.P_RNTI:
                rep.paging += 1
            elif 1 <= ev.rnti <= 60:
                rep.rar += 1
        if load is not None:
            rep.cce_load.append(load)

    def _pdcch(self, g, ce, cfg, sf, sfn, cfi, m_i, sizes, discovery):
        """Blind-decode PDCCH. Returns (list of (DciEvent, Dci), CCE load)."""
        size_1a, size_1c, size_1 = sizes
        try:
            cce_llr, regs = control.pdcch_llr(g, ce.h, ce.noise, cfg, sf, cfi, m_i)
        except (IndexError, ValueError):
            return [], None
        n_cce = cce_llr.shape[0]
        # activity per CCE: received power vs noise
        act = self._cce_activity(g, ce, cfg, cfi, m_i, n_cce)
        used = np.zeros(n_cce, dtype=bool)
        found = []

        def try_cand(cce, agg, size, accept_rnti):
            if used[cce:cce + agg].any() or not act[cce:cce + agg].all():
                return None
            c = control.decode_candidate(cce_llr, cce, agg, size)
            if not accept_rnti(c.rnti) or c.reliability < self.min_reliability[agg]:
                return None
            if size == size_1c:
                d = dcimod.parse_1c(c.bits, cfg.n_rb, c.rnti)
            elif size == size_1:
                d = dcimod.parse_1(c.bits, cfg.n_rb, c.rnti, cfg.tdd)
            else:
                d = dcimod.parse_1a(c.bits, cfg.n_rb, c.rnti, cfg.tdd)
            if d is None:
                if accept_rnti is common:
                    return None
                # C-RNTI found by CRC + strong code agreement; fields of an
                # unknown/other format cannot be interpreted reliably
                d = dcimod.Dci("?", c.rnti, raw=f"{c.size} bits")
            used[cce:cce + agg] = True
            ev = DciEvent(sfn, sf, c.rnti, d.fmt, cce, agg, d.describe())
            return ev, d

        def common(r):
            return r in (dcimod.SI_RNTI, dcimod.P_RNTI) or 1 <= r <= 60

        for cce, agg in control.common_search_space(n_cce):
            for size in (size_1a, size_1c):
                res = try_cand(cce, agg, size, common)
                if res:
                    found.append(res)
                    break
        if discovery:
            def crnti(r):
                return 0x3D <= r <= 0xFFF3
            # low -> high: a DCI sent at level L also decodes from its first
            # CCEs (circular-buffer prefix), while a higher-level candidate
            # spanning two DCIs could otherwise swallow both
            for agg in (1, 2, 4, 8):
                for cce in range(0, n_cce - agg + 1, agg):
                    for size in (size_1a, size_1):
                        res = try_cand(cce, agg, size, crnti)
                        if res:
                            found.append(res)
                            break
        return found, (float(act.mean()) if n_cce else 0.0)

    @staticmethod
    def _cce_activity(g, ce, cfg, cfi, m_i, n_cce):
        regs = control.pdcch_regs(cfg, cfi, m_i)
        n_reg = len(regs)
        ls, ks = control.reg_index(cfg, regs)
        pw = (np.abs(g[ls, ks]) ** 2).reshape(n_reg, 4).mean(axis=1)
        perm = control.quad_permutation(n_reg, cfg.n_id)
        w = np.empty(n_reg)
        w[perm] = pw
        cce_pw = w[:n_cce * 9].reshape(n_cce, 9).mean(axis=1)
        return cce_pw > 2.5 * ce.noise
