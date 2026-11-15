"""NR SS/PBCH block search in a wideband capture and PBCH/MIB decoding.

1. PSS search for every GSCN that fits in the capture: one large FFT of
   the whole capture; for each GSCN and frequency-offset hypothesis the
   sub-band around the PSS is cut out of that spectrum (ideal decimation)
   and correlated with the three PSS templates by a small IFFT.
2. Each PSS peak is cut out of the full-rate signal, shifted to baseband and
   resampled to 512 x SCS; timing and CFO are refined on the PSS, then the
   four SSB symbols are demodulated.
3. SSS -> N_ID1, PBCH DMRS -> SSB index (3 LSBs), DMRS channel estimate,
   PBCH equalisation, polar SCL decoding with CRC -> MIB.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft

from ..dsp import resample
from . import pbch
from .params import (SSB_REF_K, band_of, cp_len, default_scs, freq_to_arfcn, gscn_to_freq,
                     freq_to_gscn, l_max_candidates)

N_DEMOD = 512
SSS_MARGIN_MIN = 1.4
PSS_THRESH = 0.15


@dataclass
class SsbObs:
    """One received SS/PBCH block."""
    gscn: int
    f_ref: float              # nominal SS_REF of the GSCN [Hz]
    cfo: float                # Hz, relative to f_ref
    t: float                  # s, start of the PSS symbol (useful part) in the capture
    n_id: int
    pss_metric: float
    sss_margin: float
    ssb_bar: int              # DMRS hypothesis (3 bits)
    dmrs_margin: float
    rsrp: float               # linear, per RE (FFT / sqrt(N) scaling)
    noise: float
    mib: pbch.Mib | None = None

    @property
    def rsrp_dbfs(self) -> float:
        return 10 * np.log10(max(self.rsrp, 1e-20))

    @property
    def sinr_db(self) -> float:
        return 10 * np.log10(max(self.rsrp, 1e-20) / max(self.noise, 1e-20))


@dataclass
class NrCell:
    n_id: int
    scs: int
    obs: list = field(default_factory=list)

    @property
    def best(self) -> SsbObs:
        dec = [o for o in self.obs if o.mib]
        return max(dec or self.obs, key=lambda o: o.rsrp)

    @property
    def mib(self) -> pbch.Mib | None:
        return self.best.mib

    @property
    def gscn(self) -> int:
        return self.best.gscn

    @property
    def f_ref(self) -> float:
        return self.best.f_ref

    @property
    def cfo(self) -> float:
        dec = [o.cfo for o in self.obs if o.mib]
        return float(np.median(dec)) if dec else self.best.cfo

    @property
    def ppm(self) -> float:
        return self.cfo / self.f_ref * 1e6

    @property
    def arfcn(self) -> int:
        return freq_to_arfcn(self.f_ref)

    @property
    def band(self) -> str:
        b = band_of(self.f_ref)
        return b.name if b else "?"

    @property
    def rsrp_dbfs(self) -> float:
        return self.best.rsrp_dbfs

    @property
    def sinr_db(self) -> float:
        return self.best.sinr_db

    @property
    def beams(self) -> dict[int, float]:
        """SSB index -> best SS-RSRP [dBFS] (only decoded blocks know their
        full index; others are keyed by the DMRS hypothesis)."""
        out: dict[int, float] = {}
        for o in self.obs:
            i = o.mib.ssb_index if o.mib else o.ssb_bar
            out[i] = max(out.get(i, -999.0), o.rsrp_dbfs)
        return dict(sorted(out.items()))

    @property
    def confirmed(self) -> bool:
        return any(o.mib for o in self.obs)


# ------------------------------------------------------------- PSS search
def _template(n_id_2: int, fs_out: float, scs: int) -> np.ndarray:
    nt = int(round(fs_out / scs))
    n = np.arange(nt)
    k = np.arange(56, 183) - SSB_REF_K
    t = np.exp(2j * np.pi * np.outer(n, k) * scs / fs_out) @ pbch.pss(n_id_2)
    return t / np.linalg.norm(t)


def _circ_energy(x: np.ndarray, n: int) -> np.ndarray:
    p = np.abs(x.astype(np.complex128)) ** 2
    c = np.concatenate([[0.0], np.cumsum(np.concatenate([p, p[:n]]))])
    e = c[n:n + len(p)] - c[:len(p)]
    return np.maximum(e, 1e-3 * n * np.mean(p) + 1e-30)


@dataclass
class PssPeak:
    gscn: int
    f_ref: float
    cfo: float
    t: float
    n_id_2: int
    metric: float


def pss_search(x: np.ndarray, fs: float, fc: float, gscns: list[int], scs: int,
               cfo_max: float, max_peaks: int = 24, thresh: float = PSS_THRESH) -> list[PssPeak]:
    nfft = sfft.next_fast_len(len(x))
    X = sfft.fft(np.asarray(x, dtype=np.complex64), nfft, workers=-1)
    df = fs / nfft
    m = sfft.next_fast_len(int(np.ceil(160 * scs / df)))
    fs_out = m * df
    rel = np.fft.fftfreq(m, 1 / m).astype(np.int64)
    tmpl = [np.conj(sfft.fft(_template(u, fs_out, scs), m)).astype(np.complex64) for u in range(3)]
    nt = int(round(fs_out / scs))
    n_valid = int((len(x) / fs) * fs_out) - nt - 1
    step = scs / 3
    cfos = np.arange(-np.ceil(cfo_max / step), np.ceil(cfo_max / step) + 1) * step
    peaks = []
    for g in gscns:
        f_ref = gscn_to_freq(g)
        off = f_ref - fc
        energy = _circ_energy(sfft.ifft(X[(int(round(off / df)) + rel) % nfft], workers=-1), nt)
        best = np.zeros((3, m))
        arg = np.zeros((3, m), dtype=np.int64)
        for ci, cfo in enumerate(cfos):
            xs = X[(int(round((off + cfo) / df)) + rel) % nfft]
            for u in range(3):
                c = sfft.ifft(xs * tmpl[u], workers=-1)
                met = (c.real ** 2 + c.imag ** 2) / energy
                upd = met > best[u]
                best[u][upd] = met[upd]
                arg[u][upd] = ci
        best[:, n_valid:] = 0
        for u in range(3):
            b = best[u]
            thr = max(thresh, 12 * float(np.median(b[:n_valid])))
            for _ in range(max_peaks):
                p = int(np.argmax(b))
                if b[p] < thr:
                    break
                peaks.append(PssPeak(g, f_ref, float(cfos[arg[u][p]]), p / fs_out, u, float(b[p])))
                b[max(0, p - nt):p + nt] = 0
    return peaks


# ----------------------------------------------------------------- demod
def _pss_freq(n_id_2: int) -> np.ndarray:
    f = np.zeros(N_DEMOD, dtype=complex)
    f[(np.arange(56, 183) - SSB_REF_K) % N_DEMOD] = pbch.pss(n_id_2)
    return f


def _demod_grid(y: np.ndarray, q: int, n: int = N_DEMOD) -> np.ndarray | None:
    """(4, 240) grid of the SSB whose PSS useful part starts at sample q."""
    cp = cp_len(n)
    bo = cp // 4
    starts = q + np.arange(4) * (n + cp) - bo
    if starts[0] < 0 or starts[-1] + n > len(y):
        return None
    f = np.fft.fft(y[starts[:, None] + np.arange(n)[None, :]], axis=1) / np.sqrt(n)
    k = np.arange(240) - SSB_REF_K
    return f[:, k % n] * np.exp(2j * np.pi * k * bo / n)[None, :]


def _refine(y: np.ndarray, q0: float, n_id_2: int, fs_d: float, win: int = 12):
    """Integer timing and fine CFO from the PSS."""
    t = np.fft.ifft(_pss_freq(n_id_2))
    t /= np.linalg.norm(t)
    best = (-1.0, 0)
    for q in range(int(round(q0)) - win, int(round(q0)) + win + 1):
        if q < 0 or q + N_DEMOD > len(y):
            continue
        v = abs(np.vdot(t, y[q:q + N_DEMOD]))
        if v > best[0]:
            best = (v, q)
    q = best[1]
    # keep only the PSS subcarriers: data around the SSB is orthogonal to
    # the full template but not to its halves
    f = np.fft.fft(y[q:q + N_DEMOD])
    f[_pss_freq(n_id_2) == 0] = 0
    s = np.fft.ifft(f)
    h = N_DEMOD // 2
    c1 = np.vdot(t[:h], s[:h])
    c2 = np.vdot(t[h:], s[h:])
    return q, float(np.angle(c2 * np.conj(c1)) / (2 * np.pi * h / fs_d))


def _smooth(v: np.ndarray, w: int = 3) -> np.ndarray:
    if len(v) < w:
        return v
    k = np.ones(w) / w
    out = np.convolve(v, k, mode="same")
    out[0] = v[:2].mean()
    out[-1] = v[-2:].mean()
    return out


def _channel(grid: np.ndarray, n_id: int, ssb_bar: int):
    """DMRS-based channel estimate on symbols 1..3 -> (H (4,240), noise)."""
    _, _, (dl, dk), _ = pbch.layout(n_id % 4)
    ls = grid[dl, dk] * np.conj(pbch.dmrs(n_id, ssb_bar))
    # remove the common phase slope (residual timing offset) before smoothing,
    # otherwise the smoothing error of a linear phase looks like noise
    acc = 0j
    for l in (1, 2, 3):
        sel = dl == l
        v, k = ls[sel], dk[sel]
        adj = np.diff(k) == 4
        acc += np.sum(v[1:][adj] * np.conj(v[:-1][adj]))
    slope = np.angle(acc) / 4
    h = np.zeros(grid.shape, dtype=complex)
    res = []
    kk = np.arange(240)
    for l in (1, 2, 3):
        sel = dl == l
        k = dk[sel]
        v = ls[sel] * np.exp(-1j * slope * k)
        segs = np.split(np.arange(len(k)), np.where(np.diff(k) > 4)[0] + 1)
        sm = np.empty_like(v)
        for s in segs:
            sm[s] = _smooth(v[s])
        res.append(v - sm)
        h[l] = (np.interp(kk, k, sm.real) + 1j * np.interp(kk, k, sm.imag)) * np.exp(1j * slope * kk)
    noise = float(np.mean(np.abs(np.concatenate(res)) ** 2)) * 1.5
    return h, max(noise, 1e-12)


def demod_peak(x: np.ndarray, fs: float, fc: float, pk: PssPeak, scs: int,
               l_max: tuple[int, ...] = (8,)) -> SsbObs | None:
    fs_d = N_DEMOD * scs
    sym = (N_DEMOD + cp_len(N_DEMOD)) / fs_d
    pad = int(0.15e-3 * fs)
    p = int(pk.t * fs) - pad
    seg_len = int(4 * sym * fs) + 2 * pad
    if p < 0 or p + seg_len > len(x):
        return None
    off = pk.f_ref - fc + pk.cfo
    n = np.arange(p, p + seg_len)
    seg = x[p:p + seg_len] * np.exp(-2j * np.pi * off * n / fs)
    y = resample(seg.astype(np.complex64), fs, fs_d)
    q0 = (pk.t * fs - p) * fs_d / fs
    q, dcfo = _refine(y, q0, pk.n_id_2, fs_d)
    cfo = pk.cfo + dcfo
    y = y * np.exp(-2j * np.pi * dcfo * np.arange(len(y)) / fs_d)
    q, dcfo2 = _refine(y, q, pk.n_id_2, fs_d, win=2)
    if abs(dcfo2) > 50:
        cfo += dcfo2
        y = y * np.exp(-2j * np.pi * dcfo2 * np.arange(len(y)) / fs_d)
    grid = _demod_grid(y, q)
    if grid is None:
        return None
    # SSS -> N_ID1 (equalised with the PSS channel)
    hp = grid[0, 56:183] * pbch.pss(pk.n_id_2)
    z = grid[2, 56:183] * np.conj(hp)
    sc = np.abs(pbch.sss_table(pk.n_id_2) @ z)
    i1 = int(np.argmax(sc))
    second = np.partition(sc, -2)[-2]
    margin = float(sc[i1] / max(second, 1e-12))
    if margin < SSS_MARGIN_MIN:
        return None
    n_id = 3 * i1 + pk.n_id_2
    # DMRS -> 3 LSBs of the SSB index (+ half frame for L_max = 4)
    _, _, (dl, dk), (pl, pk2) = pbch.layout(n_id % 4)
    yd = grid[dl, dk]
    met = []
    for ib in range(8):
        ls = yd * np.conj(pbch.dmrs(n_id, ib))
        met.append(abs(np.sum(ls[1:] * np.conj(ls[:-1]))))
    met = np.array(met)
    ib = int(np.argmax(met))
    dmrs_margin = float(met[ib] / max(np.partition(met, -2)[-2], 1e-12))
    # residual CFO from the DMRS phase between symbols 1 and 3
    ls = yd * np.conj(pbch.dmrs(n_id, ib))
    rot = np.sum(ls[dl == 3] * np.conj(ls[dl == 1]))
    cfo += float(np.angle(rot) / (2 * np.pi * 2 * (N_DEMOD + cp_len(N_DEMOD)) / fs_d))
    h, noise = _channel(grid, n_id, ib)
    hs = h[pl, pk2]
    zz = grid[pl, pk2] * np.conj(hs)
    llr = np.empty(2 * len(zz))
    llr[0::2] = zz.real
    llr[1::2] = zz.imag
    llr *= 2 * np.sqrt(2) / noise
    mib = None
    tried = set()
    for lm in l_max:
        v = pbch.scrambling_v(ib, lm)
        if v in tried and lm != 64:
            continue
        tried.add(v)
        mib = pbch.pbch_decode(llr, n_id, ib, lm)
        if mib:
            break
    sss_pow = np.abs(grid[2, 56:183]) ** 2
    rsrp = max(float(np.mean(sss_pow)) - noise, 1e-3 * noise)
    # below 3 GHz the raster entries are only 100 kHz apart: report the one
    # nearest to the measured frequency (receiver offset well below 50 kHz)
    f_true = pk.f_ref + cfo
    g = freq_to_gscn(f_true) if pk.f_ref < 3000e6 else pk.gscn
    f_ref = gscn_to_freq(g)
    return SsbObs(g, f_ref, f_true - f_ref, (p + q * fs / fs_d) / fs, n_id, pk.metric, margin,
                  ib, dmrs_margin, rsrp, noise, mib)


# --------------------------------------------------------------- top level
def capture_gscns(fc: float, fs: float, scs: int, cfo_max: float,
                  usable: float = 0.4) -> list[int]:
    """GSCNs whose SSB (plus the CFO range) lies inside the usable part of
    the capture bandwidth."""
    reach = usable * fs - 120 * scs - cfo_max
    if reach <= 0:
        return []
    g0, g1 = freq_to_gscn(fc - reach) - 1, freq_to_gscn(fc + reach) + 1
    out = []
    for g in range(max(g0, 2), g1 + 1):
        f = gscn_to_freq(g)
        if abs(f - fc) <= reach:
            b = band_of(f)
            if b is None or b.scs == scs or b.name in ("n38", "n40", "n41"):
                out.append(g)
    return out


def ssb_search(x: np.ndarray, fs: float, fc: float, scs: int | None = None,
               gscns: list[int] | None = None, cfo_max: float | None = None,
               max_peaks: int = 24, log=None) -> list[NrCell]:
    """Find and decode all SS/PBCH blocks in a capture centred on fc."""
    scs = scs or default_scs(fc)
    if cfo_max is None:
        cfo_max = max(15e3, 20e-6 * fc)
    if gscns is None:
        gscns = capture_gscns(fc, fs, scs, cfo_max)
    if not gscns:
        return []
    peaks = pss_search(x, fs, fc, gscns, scs, cfo_max, max_peaks)
    if log:
        log(f"{len(gscns)} GSCNs, {len(peaks)} PSS peaks")
    obs: list[SsbObs] = []
    for pk in sorted(peaks, key=lambda p: -p.metric):
        # the same block seen from a neighbouring GSCN/CFO hypothesis
        f_true = pk.f_ref + pk.cfo
        if any(abs(o.t - pk.t) < 30e-6 and abs(o.f_ref + o.cfo - f_true) < scs
               and o.n_id % 3 == pk.n_id_2 for o in obs):
            continue
        o = demod_peak(x, fs, fc, pk, scs, l_max_candidates(pk.f_ref))
        if o:
            obs.append(o)
    cells: list[NrCell] = []
    for o in sorted(obs, key=lambda o: abs(o.cfo)):
        c = next((c for c in cells if c.n_id == o.n_id
                  and abs(c.f_ref + c.cfo - (o.f_ref + o.cfo)) < scs), None)
        if c is None:
            c = NrCell(o.n_id, scs)
            cells.append(c)
        c.obs.append(o)
    # report the GSCN with the smallest offset (raster ambiguity < 3 GHz)
    for c in cells:
        c.obs.sort(key=lambda o: abs(o.cfo))
    cells.sort(key=lambda c: (not c.confirmed, -c.rsrp_dbfs))
    return cells
