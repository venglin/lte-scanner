"""NR downlink resource occupancy from energy, without decoding user data.

The resource grid of a band (any 20 MHz piece of a carrier) is demodulated
symbol by symbol on the carrier's PRB grid. In a TDD cell the uplink
symbols carry almost nothing at a receiver away from the phones, so they
give the noise floor of every subcarrier; a PRB of a downlink slot is
counted as used when its power over the slot's downlink symbols is clearly
above that floor. The power of a PRB is the median of its 12 subcarriers,
which ignores a DC spike or a narrowband receiver spur.

Frame timing comes from the SS/PBCH blocks when the band contains them. A
band without SSB (other parts of a wide carrier) is aligned by correlating
its downlink/uplink power envelope with the pattern measured where the SSB
was: the TDD pattern repeats every 5 or 10 ms, which is all the alignment
needs (slot parity is the same)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dsp import mix, resample
from .sib1 import _sym_durations, cp_len_s

USE_RATIO = 1.3              # PRB power / noise floor that counts as used (3.5 sigma
                             # of a slot average of 12 x 12 noise-only REs)
ENV_STEP = 32                # envelope resolution [samples at N x SCS]


@dataclass(frozen=True)
class Band:
    f_low: float             # RF frequency of subcarrier 0 of PRB 0 of the band [Hz]
    n_rb: int
    scs: int
    prb0: int = 0            # index of the band's first PRB in the carrier

    @property
    def f_center(self) -> float:
        return self.f_low + 6 * self.n_rb * self.scs

    @property
    def f_high(self) -> float:
        return self.f_low + 12 * self.n_rb * self.scs


def _n_fft(scs: int) -> int:
    return 1024 if scs == 30000 else 2048


def frame_symbols(scs: int) -> int:
    return 140 * (scs // 15000)


def _symbol_times(scs: int, n_sym: int) -> np.ndarray:
    """Start of the useful part of the first n_sym symbols after a frame
    start [s] (long CP every 0.5 ms)."""
    d = _sym_durations(scs)
    per_sf = len(d)
    out = np.empty(n_sym)
    t = 0.0
    for s in range(n_sym):
        i = s % per_sf
        out[s] = t + cp_len_s(scs, i)
        t += d[i]
    return out


def _band_signal(x: np.ndarray, fs: float, fc: float, band: Band, cfo: float = 0.0):
    """Band shifted to DC and resampled to N x SCS."""
    n = _n_fft(band.scs)
    fs_d = n * band.scs
    y = mix(np.asarray(x, dtype=np.complex64), -(band.f_center - fc + cfo), fs)
    return resample(y.astype(np.complex64), fs, fs_d), fs_d


def subcarrier_power(y: np.ndarray, fs_d: float, band: Band, t_frame: float, n_frames: int,
                     fc: float | None = None) -> np.ndarray:
    """|X|^2 of every subcarrier of the band in every symbol of n_frames
    frames starting at t_frame (capture time of a frame start, >= 0).
    Returns (n_frames * symbols per frame, 12 n_rb). The subcarriers next
    to the receiver's DC (fc) are set to NaN."""
    n = _n_fft(band.scs)
    n_sym = n_frames * frame_symbols(band.scs)
    t_u = t_frame + _symbol_times(band.scs, n_sym)
    bo = (144 * n // 2048) // 4
    starts = np.round(t_u * fs_d).astype(np.int64) - bo
    ok = (starts >= 0) & (starts + n <= len(y))
    k = np.arange(12 * band.n_rb) - 6 * band.n_rb
    out = np.full((n_sym, len(k)), np.nan)
    idx = np.nonzero(ok)[0]
    for a in range(0, len(idx), 512):
        sel = idx[a:a + 512]
        seg = y[starts[sel][:, None] + np.arange(n)[None, :]]
        f = np.fft.fft(seg, axis=1) / np.sqrt(n)
        out[sel] = np.abs(f[:, k % n]) ** 2
    if fc is not None:
        dc = (fc - band.f_low) / band.scs - 0.5          # subcarrier index of the DC
        bad = np.abs(np.arange(len(k)) - dc) < 1.5
        out[:, bad] = np.nan
    return out


def _prb_median(p: np.ndarray, n_rb: int) -> np.ndarray:
    """(..., 12 n_rb) -> (..., n_rb): median over the PRB's subcarriers."""
    return np.nanmedian(p.reshape(*p.shape[:-1], n_rb, 12), axis=-1)


@dataclass
class TddPattern:
    dl: np.ndarray           # bool per symbol of a frame
    level: np.ndarray        # mean power per symbol of a frame (relative)

    def slots(self, scs: int) -> str:
        """D / S / U per slot of a frame (D: >= 12 downlink symbols)."""
        out = []
        for s in range(len(self.dl) // 14):
            n = int(self.dl[14 * s:14 * s + 14].sum())
            out.append("D" if n >= 12 else "U" if n <= 2 else "S")
        txt = "".join(out)
        half = len(txt) // 2
        return txt[:half] if txt[:half] == txt[half:] else txt


def tdd_pattern(p: np.ndarray, scs: int) -> TddPattern:
    """Downlink symbols of the frame from the power of every symbol
    (averaged over frames and over the band): two clearly separated levels
    in TDD; everything is downlink in FDD."""
    fs_ = frame_symbols(scs)
    per_sym = np.nanmean(p, axis=1)
    n_frames = len(per_sym) // fs_
    lv = np.nanmean(per_sym[:n_frames * fs_].reshape(n_frames, fs_), axis=0)
    lg = np.log10(np.maximum(lv, 1e-20))
    lo, hi = np.percentile(lg, 10), np.percentile(lg, 90)
    all_dl = TddPattern(np.ones(fs_, dtype=bool), lv)
    if hi - lo < 0.1:                                    # < 1 dB: nothing to tell apart
        return all_dl
    dl = lg > (lo + hi) / 2
    # a TDD pattern has whole uplink slots and 10-60 % uplink symbols;
    # a few strong broadcast symbols (SSB, SIB1) in an idle FDD-like
    # picture do not make one
    ul_slots = sum(1 for s in range(fs_ // 14) if not dl[14 * s:14 * s + 14].any())
    if ul_slots == 0 or not 0.1 <= 1 - dl.mean() <= 0.6:
        return all_dl
    return TddPattern(dl, lv)


@dataclass
class BandLoad:
    band: Band
    aligned: bool
    align_quality: float
    use: np.ndarray          # per PRB: share of downlink slots in which it is used
    noise: np.ndarray        # per PRB noise floor (median subcarrier power)
    n_slots: int             # downlink slots measured
    pattern: TddPattern | None = None
    excess_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    threshold_db: float = 0.0  # level above the floor that counted as used

    @property
    def mean_use(self) -> float:
        return float(np.mean(self.use)) if len(self.use) else 0.0


FIRST_PDSCH_SYMBOL = 2        # symbols 0-1 carry the PDCCH (CORESETs), not counted
LEAK_DB = 15.0                # transmitter leakage margin below the slot's strong PRBs


def band_load(p: np.ndarray, band: Band, pattern: TddPattern, aligned: bool = True,
              quality: float = 1.0, ratio: float = USE_RATIO) -> BandLoad:
    """PDSCH occupancy of the band from symbol powers p (whole frames,
    frame aligned) and the downlink symbol map. Only downlink symbols from
    FIRST_PDSCH_SYMBOL on count: the control region at the start of every
    slot is busy whenever any PDCCH is sent."""
    fs_ = frame_symbols(band.scs)
    n_frames = len(p) // fs_
    pr = _prb_median(p[:n_frames * fs_], band.n_rb)            # (n_sym, n_rb)
    dl = np.tile(pattern.dl, n_frames)
    n_slot_frame = fs_ // 14
    if (~pattern.dl).sum() >= 4:
        # same statistic as for the slots below (median over subcarriers,
        # then a typical value over symbols): an empty PRB gives ~1
        noise = np.nanmedian(pr[~dl], axis=0)
    else:
        # no uplink gaps (FDD or an idle cell): quietest tenth of the slots
        slot_lvl = np.nanmean(pr[:n_frames * fs_].reshape(-1, 14, band.n_rb), axis=1)
        # the slot average of noise-only PRBs spreads by ~8 %, so its 10th
        # percentile lies ~1.28 sigma below the noise level itself
        noise = np.nanpercentile(slot_lvl, 10, axis=0) * 1.1
    noise = np.maximum(noise, 1e-20)
    used, n = np.zeros(band.n_rb), 0
    exc = []
    for f in range(n_frames):
        for s in range(n_slot_frame):
            a = f * fs_ + 14 * s
            m = pattern.dl[14 * s:14 * s + 14].copy()
            m[:FIRST_PDSCH_SYMBOL] = False
            if m.sum() < 5:
                continue
            lvl = np.nanmean(pr[a:a + 14][m], axis=0)
            exc.append(lvl / noise)
            n += 1
    if not exc:
        return BandLoad(band, aligned, quality, np.zeros(band.n_rb), noise, 0, pattern)
    r = np.array(exc)                                           # (slots, PRB), vs floor
    # a strong cell's empty PRBs are not at the thermal floor: its
    # transmitter noise leaks into them about 25-30 dB below the PDSCH of
    # the same slot (more in busy slots). Per slot the threshold is
    # therefore `ratio` above the floor, but never lower than LEAK_DB below
    # the slot's strong PRBs (90th percentile).
    hi = np.nanpercentile(r, 90, axis=1, keepdims=True)
    thr = np.maximum(ratio, hi / 10 ** (LEAK_DB / 10))
    use = np.nanmean(r > thr, axis=0)
    exc_db = 10 * np.log10(np.maximum(np.nanmean(r, axis=0), 1e-3))
    return BandLoad(band, aligned, quality, use, noise, n, pattern, exc_db,
                    float(10 * np.log10(np.median(thr))))


# ------------------------------------------------------------- alignment
def _envelope(y: np.ndarray, fs_d: float, scs: int) -> np.ndarray:
    """Power folded over 10 ms at ENV_STEP resolution, smoothed over half
    a symbol."""
    per = int(round(10e-3 * fs_d)) // ENV_STEP
    pw = np.abs(y) ** 2
    m = len(pw) // ENV_STEP
    e = pw[:m * ENV_STEP].reshape(m, ENV_STEP).mean(axis=1)
    k = max(1, int(round(0.5 / scs * fs_d / ENV_STEP)))
    e = np.convolve(e, np.ones(k) / k, mode="same")
    n_per = len(e) // per
    return e[:n_per * per].reshape(n_per, per).mean(axis=0)


def _mask_envelope(pattern: TddPattern, fs_d: float, scs: int, length: int) -> np.ndarray:
    fs_ = frame_symbols(scs)
    t = _symbol_times(scs, fs_ + 1)
    n = _n_fft(scs)
    mask = np.zeros(length)
    for s in range(fs_):
        a = int((t[s] - cp_len_s(scs, s % len(_sym_durations(scs)))) * fs_d) // ENV_STEP
        b = int((t[s] + n / fs_d) * fs_d) // ENV_STEP
        mask[a:min(b, length)] = 1.0 if pattern.dl[s] else 0.0
    return mask


def align(y: np.ndarray, fs_d: float, scs: int, pattern: TddPattern) -> tuple[float, float]:
    """Frame start (mod 10 ms, as a capture time >= 0) of a band from its
    power envelope and the downlink pattern. Returns (t_frame, quality =
    correlation coefficient at the best lag)."""
    env = _envelope(y, fs_d, scs)
    mask = _mask_envelope(pattern, fs_d, scs, len(env))
    a = (env - env.mean()) / (env.std() + 1e-20)
    b = (mask - mask.mean()) / (mask.std() + 1e-20)
    c = np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))) / len(a)
    lag = int(np.argmax(c))
    return lag * ENV_STEP / fs_d, float(c[lag])


def refine_cp(y: np.ndarray, fs_d: float, scs: int, t_frame: float, pattern: TddPattern,
              search: int = 96, step: int = 2) -> float:
    """Symbol timing to the sample from the cyclic prefix: the CP repeats
    the end of the symbol, so |sum y[CP] conj(y[CP + N])| peaks when the
    window is right. Errors beyond the CP (a few us from the envelope) leak
    every symbol into its neighbours, which looks like occupancy."""
    n = _n_fft(scs)
    fs_ = frame_symbols(scs)
    n_frames = max(1, min(8, int((len(y) / fs_d - t_frame) // 10e-3)))
    t_u = t_frame + _symbol_times(scs, n_frames * fs_)
    dl = np.tile(pattern.dl, n_frames)
    cp = 144 * n // 2048
    base = np.round(t_u[dl] * fs_d).astype(np.int64) - cp     # CP starts
    best, best_d = -1.0, 0
    for d in range(-search, search + 1, step):
        st = base + d
        st = st[(st >= 0) & (st + cp + n <= len(y))]
        if not len(st):
            continue
        idx = st[:, None] + np.arange(cp)[None, :]
        c = np.sum(y[idx] * np.conj(y[idx + n]), axis=1)
        e = np.sum(np.abs(y[idx]) ** 2 + np.abs(y[idx + n]) ** 2, axis=1) / 2
        m = float(np.sum(np.abs(c)) / max(np.sum(e), 1e-20))
        if m > best:
            best, best_d = m, d
    return t_frame + best_d / fs_d


def measure(x: np.ndarray, fs: float, fc: float, band: Band, t_frame: float | None,
            pattern: TddPattern | None = None, cfo: float = 0.0,
            min_quality: float = 0.3) -> BandLoad:
    """Occupancy of a band of a capture. With t_frame (from the SSBs) the
    TDD pattern is measured here (unless given); without it the band is
    aligned to `pattern` by its envelope."""
    y, fs_d = _band_signal(x, fs, fc, band, cfo)
    q = 1.0
    aligned = True
    if t_frame is None:
        if pattern is None or pattern.dl.all():
            t_frame, q, aligned = 0.0, 0.0, pattern is not None
        else:
            t_frame, q = align(y, fs_d, band.scs, pattern)
            aligned = q >= min_quality
            if aligned:
                t_frame = refine_cp(y, fs_d, band.scs, t_frame, pattern)
    t_frame = t_frame % 10e-3
    n_frames = int((len(y) / fs_d - t_frame) // 10e-3)
    p = subcarrier_power(y, fs_d, band, t_frame, n_frames, fc=fc)
    if pattern is None:
        pattern = tdd_pattern(p, band.scs)
    return band_load(p, band, pattern, aligned, q)


def bar(v: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)
