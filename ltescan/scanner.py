"""Band / frequency scanner: finds LTE carriers in wideband captures (PSD
based candidate selection on the 100 kHz raster), then runs cell search
and MIB decoding on every candidate."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .dsp import lowpass_decimate, psd
from .params import BANDS, BASE_RATE, auto_cfo_max, band_edges, freq_to_earfcn
from .receiver import CellReport, LteReceiver

OCCUPIED_MHZ = [1.08, 2.7, 4.5, 9.0, 13.5, 18.0]
RASTER = 100e3


@dataclass
class Carrier:
    lo: float              # absolute Hz
    hi: float
    lo_real: bool          # edge observed (not truncated by capture)
    hi_real: bool
    level_db: float

    @property
    def width(self):
        return self.hi - self.lo


@dataclass
class ScanResult:
    reports: list[CellReport] = field(default_factory=list)
    carriers: list[Carrier] = field(default_factory=list)
    psd: list = field(default_factory=list)          # (freqs_abs, dB) per chunk
    tested: int = 0
    unconfirmed: int = 0          # PSS/SSS hits without a decodable MIB


def find_carriers(x: np.ndarray, fs: float, fc: float, thresh_db: float = 8.0,
                  usable: float = 0.8):
    """Occupied spectral regions wider than ~1 MHz. Returns (carriers,
    (abs freqs, psd dB))."""
    nfft = 1 << int(np.ceil(np.log2(fs / 10e3)))
    f, p = psd(x - np.mean(x), fs, nfft)
    lim = usable * fs / 2
    sel = np.abs(f) <= lim
    f, p = f[sel], p[sel]
    floor = np.percentile(p, 10)
    occ = p > floor + thresh_db
    df = f[1] - f[0]
    # close small gaps (DC spike notch, empty subcarriers), drop narrow blips
    k = max(1, int(150e3 / df))
    occ = ndimage.binary_closing(occ, structure=np.ones(k))
    occ = ndimage.binary_opening(occ, structure=np.ones(max(1, int(300e3 / df))))
    lab, n = ndimage.label(occ)
    out = []
    edge = int(400e3 / df)          # an edge this close to the limit may be a cut
    for i in range(1, n + 1):
        idx = np.nonzero(lab == i)[0]
        lo, hi = f[idx[0]], f[idx[-1]]
        if hi - lo < 0.9e6:
            continue
        out.append(Carrier(fc + lo, fc + hi, bool(idx[0] > edge),
                           bool(idx[-1] < len(f) - 1 - edge),
                           float(np.median(p[idx]) - floor)))
    return out, (f + fc, p)


def snap(f: float) -> float:
    return round(f / RASTER) * RASTER


def carrier_candidates(c: Carrier) -> list[float]:
    """Plausible centre frequencies (on the 100 kHz raster) for a region."""
    cands = []
    if c.lo_real and c.hi_real:
        cands.append(0.5 * (c.lo + c.hi))
    for w in OCCUPIED_MHZ:
        w *= 1e6
        if w > c.width + 0.4e6 and c.lo_real and c.hi_real:
            continue
        if c.lo_real:
            cands.append(c.lo + w / 2)
        if c.hi_real:
            cands.append(c.hi - w / 2)
    if not c.lo_real and not c.hi_real:
        # region covers the whole capture: try the raster around the middle
        mid = 0.5 * (c.lo + c.hi)
        cands += [mid + d for d in np.arange(-3e6, 3e6 + 1, RASTER)]
    # most plausible first: exact snapped values, then +-1 raster step
    out = []
    for d in (0, -RASTER, RASTER):
        for f in cands:
            s = snap(f) + d
            if s not in out:
                out.append(s)
    return out


class Scanner:
    def __init__(self, cfo_max: float | None = None, ppm_hint: float | None = None,
                 log=print, decode_sib: bool = False, keep_unconfirmed: bool = False):
        # keep_unconfirmed: also report PSS/SSS detections whose MIB could not
        # be decoded. In a raster scan these are mostly ghosts of a strong
        # carrier probed off-centre, so they are hidden by default.
        self.keep_unconfirmed = keep_unconfirmed
        self.cfo_max = cfo_max
        self.ppm = ppm_hint            # learned receiver clock error
        self.log = log
        self.decode_sib = decode_sib

    def _cfo_window(self, f: float):
        if self.ppm is None:
            return 0.0, self.cfo_max or auto_cfo_max(f), 5e3
        return -self.ppm * 1e-6 * f, 6e3, 3e3

    def probe(self, x192: np.ndarray, freq: float, wide=None) -> list[CellReport]:
        """Cell search + MIB on a 1.92 Msps buffer centred at `freq`."""
        centre, span, step = self._cfo_window(freq)
        rx = LteReceiver(x192, BASE_RATE, fc=freq, cfo_max=span)
        if centre:
            from .dsp import mix
            rx.x192 = mix(rx.x192, -centre, BASE_RATE)
        from .sync import cell_search
        cells = cell_search(rx.x192, cfo_max=span, cfo_step=step)
        reps = []
        for cell in cells:
            cell.cfo += centre
            rx.x192 = x192
            if wide is not None and self.decode_sib:
                xw, fsw = wide
                full = LteReceiver(xw, fsw, fc=freq)
                rep = full.analyze(cell, decode_sib=True)
            else:
                rep = rx.analyze(cell, decode_sib=False)
            if rep.mib is not None and self.ppm is None:
                self.ppm = -cell.cfo / freq * 1e6
                self.log(f"  oszacowany błąd zegara odbiornika: {self.ppm:+.2f} ppm")
            reps.append(rep)
        return reps

    def scan_buffer(self, x: np.ndarray, fs: float, fc: float, band: int | None = None,
                    exhaustive: bool = False, thresh_db: float = 8.0,
                    result: ScanResult | None = None) -> ScanResult:
        res = result or ScanResult()
        carriers, spec = find_carriers(x, fs, fc, thresh_db)
        res.carriers += carriers
        res.psd.append(spec)
        lim = 0.8 * fs / 2 - 0.6e6
        if exhaustive:
            fl = snap(fc - lim) + RASTER
            cands = list(np.arange(fl, fc + lim, RASTER))
        else:
            cands = []
            for c in sorted(carriers, key=lambda c: -c.level_db):
                cands += [f for f in carrier_candidates(c) if f not in cands]
        if band is not None:
            lo, hi = band_edges(band)
            cands = [f for f in cands if lo <= f <= hi]
        cands = [f for f in cands if abs(f - fc) <= lim]
        for f in cands:
            if any(r.mib is not None and _covers(r, f) for r in res.reports):
                continue                       # inside an already found carrier
            res.tested += 1
            x192 = lowpass_decimate(x, fs, BASE_RATE, f_shift=fc - f)
            wide = None
            if self.decode_sib:
                from .dsp import mix
                wide = (mix(x, fc - f, fs), fs)
            for rep in self.probe(x192, f, wide):
                if rep.mib is None and not self.keep_unconfirmed:
                    res.unconfirmed += 1
                    continue
                if not any(r.sync.n_id == rep.sync.n_id and abs(r.fc - rep.fc) < 0.5e6
                           for r in res.reports):
                    res.reports.append(rep)
                    self.log("  " + short_line(rep))
        res.reports = prune(res.reports)
        return res

    def scan_band(self, source, band: int, fs: float = 20e6, dwell: float = 0.045,
                  exhaustive: bool = False, thresh_db: float = 8.0) -> ScanResult:
        lo, hi = band_edges(band)
        lo -= 0.1e6
        hi += 0.1e6
        half = 0.8 * fs / 2 - 0.6e6
        step = 2 * half - 1e6
        n_chunks = max(1, int(np.ceil((hi - lo) / step)))
        centres = [lo + half + i * step for i in range(n_chunks)]
        if (hi - lo) < 2 * half:
            centres = [0.5 * (lo + hi)]
        res = ScanResult()
        for c in centres:
            self.log(f"pasmo {band}: przechwytywanie {c / 1e6:.1f} MHz ({fs / 1e6:.0f} Msps)")
            cap = source.capture(c, fs, int(dwell * fs), bb_filter=0.8 * fs)
            self.scan_buffer(cap.x, cap.fs, cap.fc, band, exhaustive, thresh_db, res)
        return res

    def scan_freqs(self, source, freqs: list[float], fs: float = 7.68e6,
                   dwell: float = 0.045) -> ScanResult:
        """Direct capture at each given carrier frequency."""
        res = ScanResult()
        for f in freqs:
            self.log(f"{f / 1e6:.1f} MHz ...")
            cap = source.capture(f, fs, int(dwell * fs), bb_filter=min(fs, 6e6))
            res.tested += 1
            x192 = lowpass_decimate(cap.x, cap.fs, BASE_RATE)
            wide = (cap.x, cap.fs) if self.decode_sib else None
            for rep in self.probe(x192, f, wide):
                if rep.mib is None and not self.keep_unconfirmed:
                    res.unconfirmed += 1
                    continue
                res.reports.append(rep)
                self.log("  " + short_line(rep))
        res.reports = prune(res.reports)
        return res


def prune(reports: list[CellReport]) -> list[CellReport]:
    """Drop unconfirmed (no MIB) detections that fall inside the bandwidth of
    a confirmed carrier: off-raster probes of a strong carrier can produce
    PSS/SSS ghosts (a frequency shift of a Zadoff-Chu sequence looks like a
    time shift)."""
    good = [r for r in reports if r.mib is not None]
    out = []
    for r in reports:
        if r.mib is None and any(_covers(g, r.fc) for g in good):
            continue
        out.append(r)
    return out


def _covers(rep: CellReport, f: float) -> bool:
    bw = rep.mib.n_rb * 180e3 if rep.mib else 0
    return abs(rep.fc - f) < bw / 2


def short_line(rep: CellReport) -> str:
    e = freq_to_earfcn(rep.fc) if rep.fc else None
    m = rep.mib
    s = (f"{rep.fc / 1e6:9.3f} MHz EARFCN {e if e is not None else '-':>5}  PCI {rep.sync.n_id:3d} "
         f"{rep.sync.duplex} {'ext' if rep.sync.cp_ext else 'norm'}CP")
    if m:
        s += f"  BW {m.n_rb:3d}RB  porty {m.n_ports}  SFN {m.sfn:4d}"
    else:
        s += "  (MIB nieodczytany)"
    s += f"  RSRP {rep.rsrp_dbfs:6.1f} dBFS  SNR {rep.snr_db:5.1f} dB"
    return s


def band_list() -> list[int]:
    return sorted(BANDS)
