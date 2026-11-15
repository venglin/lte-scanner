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
from scipy.ndimage import uniform_filter1d

from ..dsp import resample
from . import pbch
from .params import (SSB_REF_K, band_of, cp_len, default_scs, freq_to_arfcn, gscn_to_freq,
                     freq_to_gscn, l_max_candidates)

N_DEMOD = 512
SSS_MARGIN_MIN = 1.4
SSS_MARGIN_WEAK = 1.05      # multi-block tracks: PBCH CRC decides
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
    combined: int = 1         # blocks soft-combined for this MIB
    ssb_index: int | None = None   # known from the MIB of any block of the track
    llr: np.ndarray | None = field(default=None, repr=False)

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
            i = o.ssb_index if o.ssb_index is not None else o.ssb_bar
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
    metric: float              # detection metric (folded over periods for tracks)
    track: int = -1            # occurrences of one periodic SSB share a track id
    raw: float = 0.0           # single-occurrence metric at this position


SSB_PERIOD = 20e-3            # periodicity assumed for folding (initial access default)


def _nms(b: np.ndarray, thr: float, radius: int, max_peaks: int, circular: bool = False):
    b = b.copy()
    out = []
    for _ in range(max_peaks):
        p = int(np.argmax(b))
        if b[p] < thr:
            break
        out.append((p, float(b[p])))
        if circular:
            b[np.arange(p - radius, p + radius) % len(b)] = 0
        else:
            b[max(0, p - radius):p + radius] = 0
    return out


def pss_search(x: np.ndarray, fs: float, fc: float, gscns: list[int], scs: int,
               cfo_max: float, max_peaks: int = 24, thresh: float = PSS_THRESH) -> list[PssPeak]:
    """PSS peaks of single occurrences, plus (when the capture holds at
    least two SSB periods) peaks of the correlation averaged over periods,
    which find SSBs too weak to stand out in one period. Every period of a
    folded peak is returned as one occurrence of a track."""
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
    n_per = int(round(SSB_PERIOD * fs_out))
    k_per = n_valid // n_per
    peaks = []
    track = 0
    for g in gscns:
        f_ref = gscn_to_freq(g)
        off = f_ref - fc
        energy = _circ_energy(sfft.ifft(X[(int(round(off / df)) + rel) % nfft], workers=-1), nt)
        best = np.zeros((3, m))
        arg = np.zeros((3, m), dtype=np.int64)
        fold = np.zeros((3, n_per))
        farg = np.zeros((3, n_per), dtype=np.int64)
        fmean = np.zeros((3, n_per))      # average over CFO hypotheses (noise level)
        for ci, cfo in enumerate(cfos):
            xs = X[(int(round((off + cfo) / df)) + rel) % nfft]
            for u in range(3):
                c = sfft.ifft(xs * tmpl[u], workers=-1)
                met = (c.real ** 2 + c.imag ** 2) / energy
                met[n_valid:] = 0
                upd = met > best[u]
                best[u][upd] = met[upd]
                arg[u][upd] = ci
                if k_per >= 2:
                    f = met[:k_per * n_per].reshape(k_per, n_per).mean(axis=0)
                    fmean[u] += f / len(cfos)
                    upd = f > fold[u]
                    fold[u][upd] = f[upd]
                    farg[u][upd] = ci
        for u in range(3):
            b = best[u]
            thr = max(thresh, 12 * float(np.median(b[:n_valid])))
            for p, v in _nms(b, thr, nt, max_peaks):
                peaks.append(PssPeak(g, f_ref, float(cfos[arg[u][p]]), p / fs_out, u, v,
                                     raw=v))
            if k_per < 2:
                continue
            # noise-only folded metric ~ Gamma(K): threshold relative to the
            # median of one hypothesis (the maximum over CFO would inflate it)
            thr_f = max(0.02, (1 + 12 / np.sqrt(k_per)) * float(np.median(fmean[u])))
            for p, v in _nms(fold[u], thr_f, nt, max_peaks, circular=True):
                track += 1
                for k in range((n_valid - p - 1) // n_per + 1):
                    q = p + k * n_per
                    peaks.append(PssPeak(g, f_ref, float(cfos[farg[u][p]]), q / fs_out, u, v,
                                         track, float(b[q] if q < len(b) else 0)))
    return peaks


# ----------------------------------------------------------------- demod
def _pss_freq(n_id_2: int) -> np.ndarray:
    f = np.zeros(N_DEMOD, dtype=complex)
    f[(np.arange(56, 183) - SSB_REF_K) % N_DEMOD] = pbch.pss(n_id_2)
    return f


def _demod_grid(y: np.ndarray, q: int, f_ref: float = 0.0, scs: int = 30000,
                n: int = N_DEMOD) -> np.ndarray | None:
    """(4, 240) grid of the SSB whose PSS useful part starts at sample q.

    TS 38.211 5.4 restarts the upconversion phase at every symbol
    (exp(j2 pi f0 (t - t_start,l - N_CP,l Tc))). A receiver that mixes the
    SSB to baseband therefore sees an extra phase of -2 pi f_SSB t_l per
    symbol, which looks like a frequency offset between symbols but not
    within one. It is removed here (relative to the PSS symbol)."""
    cp = cp_len(n)
    bo = cp // 4
    starts = q + np.arange(4) * (n + cp) - bo
    if starts[0] < 0 or starts[-1] + n > len(y):
        return None
    f = np.fft.fft(y[starts[:, None] + np.arange(n)[None, :]], axis=1) / np.sqrt(n)
    k = np.arange(240) - SSB_REF_K
    t_sym = (n + cp) / (n * scs)
    sym_phase = np.exp(2j * np.pi * f_ref * t_sym * np.arange(4))
    return f[:, k % n] * np.exp(2j * np.pi * k * bo / n)[None, :] * sym_phase[:, None]


def _pss_time(n_id_2: int) -> np.ndarray:
    t = np.fft.ifft(_pss_freq(n_id_2))
    return t / np.linalg.norm(t)


def _timing(y: np.ndarray, q0: float, n_id_2: int, win: int) -> int:
    """Integer start of the PSS useful part (correlation peak near q0)."""
    t = _pss_time(n_id_2)
    qs = [q for q in range(int(round(q0)) - win, int(round(q0)) + win + 1)
          if q >= 0 and q + N_DEMOD <= len(y)]
    if not qs:
        return int(round(q0))
    return qs[int(np.argmax([abs(np.vdot(t, y[q:q + N_DEMOD])) for q in qs]))]


def _cfo_product(y: np.ndarray, q: int, n_id_2: int) -> complex:
    """c2 * conj(c1) of the two PSS halves. Only the PSS subcarriers are
    kept: data around the SSB is orthogonal to the full template but not to
    its halves."""
    t = _pss_time(n_id_2)
    f = np.fft.fft(y[q:q + N_DEMOD])
    f[_pss_freq(n_id_2) == 0] = 0
    sg = np.fft.ifft(f)
    h = N_DEMOD // 2
    return complex(np.vdot(t[h:], sg[h:]) * np.conj(np.vdot(t[:h], sg[:h])))


def _smooth(v: np.ndarray, w: int) -> np.ndarray:
    """Moving average over w samples after removing the linear phase
    (residual timing offset), which would otherwise be smeared."""
    if len(v) < 2 or w <= 1:
        return v
    # only trust the slope when neighbouring estimates are coherent; at low
    # SNR a random slope would cancel the signal inside the window
    c = np.sum(v[1:] * np.conj(v[:-1]))
    coh = abs(c) / max(float(np.sum(np.abs(v[1:] * v[:-1]))), 1e-30)
    slope = float(np.angle(c)) if coh > 0.3 else 0.0
    k = np.arange(len(v))
    r = v * np.exp(-1j * slope * k)
    sm = uniform_filter1d(r.real, w, mode="nearest") + 1j * uniform_filter1d(r.imag, w,
                                                                          mode="nearest")
    return sm * np.exp(1j * slope * k)


def _channel(grid: np.ndarray, n_id: int, ssb_bar: int, w: int = 3):
    """PBCH channel estimate from the DMRS of symbols 1..3, averaged over
    the three symbols (with the phase progression of a residual CFO) and
    smoothed over w DMRS. Returns (H (4, 240), noise per RE, phase change
    from symbol 1 to symbol 3)."""
    v = n_id % 4
    _, _, (dl, dk), _ = pbch.layout(v)
    ls = grid[dl, dk] * np.conj(pbch.dmrs(n_id, ssb_bar))
    s1, s3 = dl == 1, dl == 3                      # identical subcarrier sets
    phi = float(np.angle(np.sum(ls[s3] * np.conj(ls[s1]))))
    rot = np.exp(-1j * phi * (dl - 2) / 2)         # undo the per-symbol phase
    kd = np.arange(v, 240, 4)
    acc = np.zeros(len(kd), dtype=complex)
    cnt = np.zeros(len(kd))
    np.add.at(acc, dk // 4, ls * rot)
    np.add.at(cnt, dk // 4, 1)
    h0 = _smooth(acc / cnt, w)
    res = ls * rot - h0[dk // 4]
    n_eff = w * float(np.mean(cnt))
    noise = float(np.mean(np.abs(res) ** 2)) / max(1 - 1 / n_eff, 0.3)
    kk = np.arange(240)
    hk = np.interp(kk, kd, h0.real) + 1j * np.interp(kk, kd, h0.imag)
    h = np.zeros(grid.shape, dtype=complex)
    for l in (1, 2, 3):
        h[l] = hk * np.exp(1j * phi * (l - 2) / 2)
    return h, max(noise, 1e-12), phi


@dataclass
class _Occ:
    pk: PssPeak
    y: np.ndarray             # SSB segment at 512 x SCS, shifted to baseband
    q: int                    # start of the PSS useful part in y
    cfo: float                # frequency offset removed so far (vs f_ref)
    t0: float                 # capture time of y[0] [s]
    fs_d: float
    grid: np.ndarray | None = None

    @property
    def t(self) -> float:
        return self.t0 + self.q / self.fs_d


def _demod_occ(x: np.ndarray, fs: float, fc: float, pk: PssPeak, scs: int) -> _Occ | None:
    """Cut one SSB out of the capture, shift it to baseband with the search
    CFO, resample to 512 x SCS and find the PSS timing."""
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
    q = _timing(y, (pk.t * fs - p) * fs_d / fs, pk.n_id_2, 12)
    return _Occ(pk, y, q, pk.cfo, p / fs, fs_d)


def _sync_track(occs: list[_Occ], scs: int) -> list[_Occ]:
    """Common fine CFO of all occurrences (PSS halves summed over them),
    then timing and demodulation of each."""
    n2 = occs[0].pk.n_id_2
    fs_d = occs[0].fs_d
    prod = sum(_cfo_product(o.y, o.q, n2) for o in occs)
    d = float(np.angle(prod) / (2 * np.pi * (N_DEMOD // 2) / fs_d))
    if abs(d) > scs / 4:
        d = 0.0
    out = []
    for o in occs:
        o.y = o.y * np.exp(-2j * np.pi * d * np.arange(len(o.y)) / fs_d)
        o.cfo += d
        o.q = _timing(o.y, o.q, n2, 2)
        o.grid = _demod_grid(o.y, o.q, o.pk.f_ref, scs)
        if o.grid is not None:
            out.append(o)
    return out


def _ss_channel(grid: np.ndarray, n_id_2: int, n_id: int | None = None, w: int = 11):
    """Smoothed channel on subcarriers 56..182 from the PSS (and the SSS
    once the cell id is known)."""
    h = grid[0, 56:183] * pbch.pss(n_id_2)
    if n_id is not None:
        h = (h + grid[2, 56:183] * pbch.sss(n_id)) / 2
    return _smooth(h, w)


def _sss_scores(o: _Occ, w: int = 11) -> np.ndarray:
    z = o.grid[2, 56:183] * np.conj(_ss_channel(o.grid, o.pk.n_id_2, w=w))
    return np.abs(pbch.sss_table(o.pk.n_id_2) @ z)


def _dmrs_scores(o: _Occ, n_id: int, w: int = 11) -> np.ndarray:
    """Coherent DMRS correlation (8 hypotheses) on the subcarriers covered
    by the PSS/SSS channel estimate."""
    _, _, (dl, dk), _ = pbch.layout(n_id % 4)
    hs = _ss_channel(o.grid, n_id % 3, n_id, w)
    sel = (dk >= 56) & (dk <= 182)
    y = o.grid[dl[sel], dk[sel]] * np.conj(hs[dk[sel] - 56])
    met = np.empty(8)
    for ib in range(8):
        c = y * np.conj(pbch.dmrs(n_id, ib)[sel])
        met[ib] = abs(np.sum(c[dl[sel] == 1])) + abs(np.sum(c[dl[sel] == 3]))
    return met


def _margin(v: np.ndarray) -> tuple[int, float]:
    i = int(np.argmax(v))
    return i, float(v[i] / max(np.partition(v, -2)[-2], 1e-12))


def _decode_cell(occs: list[_Occ], scs: int, n_id: int, margin: float,
                 lms: list[int], w: int = 11) -> list[SsbObs]:
    """DMRS hypothesis, channel estimate and PBCH decoding (alone, then
    soft-combined) of the occurrences of one SSB for a given cell id."""
    ib, dmrs_margin = _margin(sum(_dmrs_scores(o, n_id, w) for o in occs))
    _, _, _, (pl, pk2) = pbch.layout(n_id % 4)
    fs_d = occs[0].fs_d
    lms = [lm for i, lm in enumerate(lms)
           if all(pbch.scrambling_v(ib, lm) != pbch.scrambling_v(ib, o) for o in lms[:i])]
    obs = []
    rot = 0j
    for o in occs:
        h, noise, phi = _channel(o.grid, n_id, ib, 3)
        rsrp = max(float(np.mean(np.abs(o.grid[2, 56:183]) ** 2)) - noise, 1e-3 * noise)
        if rsrp / noise < 3:              # low SNR: smooth the estimate harder
            h, noise, phi = _channel(o.grid, n_id, ib, 9)
            rsrp = max(float(np.mean(np.abs(o.grid[2, 56:183]) ** 2)) - noise, 1e-3 * noise)
        _, _, (dl, dk), _ = pbch.layout(n_id % 4)
        ls = o.grid[dl, dk] * np.conj(pbch.dmrs(n_id, ib))
        rot += np.sum(ls[dl == 3] * np.conj(ls[dl == 1]))
        zz = o.grid[pl, pk2] * np.conj(h[pl, pk2])
        llr = np.empty(2 * len(zz))
        llr[0::2] = zz.real
        llr[1::2] = zz.imag
        llr *= 2 * np.sqrt(2) / noise
        mib = None
        for lm in lms:
            mib = pbch.pbch_decode(llr, n_id, ib, lm)
            if mib:
                break
        obs.append(SsbObs(o.pk.gscn, o.pk.f_ref, o.cfo, o.t, n_id, o.pk.metric, margin, ib,
                          dmrs_margin, rsrp, noise, mib, llr=llr))
    # residual CFO from the DMRS phase between symbols 1 and 3 (whole track)
    dcfo = float(np.angle(rot) / (2 * np.pi * 2 * (N_DEMOD + cp_len(N_DEMOD)) / fs_d))
    for ob in obs:
        # below 3 GHz the raster entries are only 100 kHz apart: report the
        # one nearest to the measured frequency (receiver offset < 50 kHz)
        f_true = ob.f_ref + ob.cfo + dcfo
        if ob.f_ref < 3000e6:
            ob.gscn = freq_to_gscn(f_true)
            ob.f_ref = gscn_to_freq(ob.gscn)
        ob.cfo = f_true - ob.f_ref
    if len(obs) >= 2 and not any(ob.mib for ob in obs):
        t0 = obs[0].t
        half = [int(round((ob.t - t0) / 5e-3)) for ob in obs]
        for lm in lms:
            # a clear SSS makes a real cell likely: afford trying every hypothesis
            r = pbch.combine_decode([ob.llr for ob in obs], half, n_id, ib, lm,
                                    n_try=300 if margin >= 1.25 else 32)
            if r:
                obs[0].mib, obs[0].combined = r
                break
    idx = next((ob.mib.ssb_index for ob in obs if ob.mib), None)
    for ob in obs:
        ob.ssb_index = idx
    return obs


def _decode_track(occs: list[_Occ], scs: int, l_max: tuple[int, ...]) -> list[SsbObs]:
    """Occurrences of one SSB (same cell and SSB index, one per period):
    common CFO, SSS decided on the sum over all occurrences, then PBCH. With
    several occurrences and an unclear SSS the next best cell ids are tried
    too: the PBCH CRC (with the cell id in its scrambling) decides."""
    if not occs:
        return []
    occs = _sync_track(occs, scs)
    if not occs:
        return []
    n_id_2 = occs[0].pk.n_id_2
    # weak SSBs: average the PSS/SSS channel over ~0.9 MHz (30 kHz SCS)
    w = 11 if max(o.pk.raw for o in occs) >= PSS_THRESH else 31
    sc = sum(_sss_scores(o, w) for o in occs)
    i1, margin = _margin(sc)
    cands = [i1]
    if margin < SSS_MARGIN_MIN:
        if len(occs) < 3 or margin < SSS_MARGIN_WEAK:
            return []
        cands = [int(i) for i in np.argsort(sc)[::-1][:3]]
    lms = list(l_max)
    for i in cands:
        obs = _decode_cell(occs, scs, 3 * i + n_id_2, margin, lms, w)
        if any(ob.mib for ob in obs) or margin >= SSS_MARGIN_MIN:
            return obs
    return []


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
    # group into tracks; single-occurrence peaks are tracks of their own
    tracks: dict[int, list[PssPeak]] = {}
    for i, pk in enumerate(peaks):
        tracks.setdefault(pk.track if pk.track >= 0 else -1 - i, []).append(pk)
    if log:
        log(f"{len(gscns)} GSCNs, {len(peaks)} PSS peaks in {len(tracks)} tracks")
    done: list[PssPeak] = []

    def seen(pk):
        # the same block from another track / GSCN / CFO hypothesis
        f = pk.f_ref + pk.cfo
        return any(abs(d.t - pk.t) < 30e-6 and abs(d.f_ref + d.cfo - f) < scs
                   and d.n_id_2 == pk.n_id_2 for d in done)

    obs: list[SsbObs] = []
    order = sorted(tracks.values(), key=lambda tr: -max(max(p.raw, p.metric) for p in tr))
    for tr in order:
        tr = [p for p in tr if not seen(p)]
        if not tr:
            continue
        # a clearly visible SSB with a longer period than the folding period:
        # leave out the empty positions
        top = max(p.raw for p in tr)
        if top > 2 * PSS_THRESH:
            tr = [p for p in tr if p.raw > 0.2 * top]
        occs = [o for o in (_demod_occ(x, fs, fc, p, scs) for p in tr) if o]
        new = _decode_track(occs, scs, l_max_candidates(tr[0].f_ref))
        if new:
            done.extend(tr)
            obs.extend(new)
    cells: list[NrCell] = []
    for o in sorted(obs, key=lambda o: abs(o.cfo)):
        c = next((c for c in cells if c.n_id == o.n_id
                  and abs(c.f_ref + c.cfo - (o.f_ref + o.cfo)) < scs), None)
        if c is None:
            c = NrCell(o.n_id, scs)
            cells.append(c)
        c.obs.append(o)
    for c in cells:
        c.obs.sort(key=lambda o: o.t)
    cells.sort(key=lambda c: (not c.confirmed, -c.rsrp_dbfs))
    return cells
