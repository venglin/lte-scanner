"""Cell search: PSS detection (with coarse CFO search), fine CFO, SSS
detection (cell id, CP length, FDD/TDD, frame timing).

All processing here runs at 1.92 Msps (128-point FFT, central 6 RB)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import signal

from .ofdm import bin_index, cp_lengths, symbol_offsets
from .params import BASE_RATE
from .sequences import pss, sss, sss_table

FS = BASE_RATE
N = 128
HALF_FRAME = 9600
SUBFRAME = 1920
FRAME = 19200
SSS_MARGIN_MIN = 1.5


@lru_cache(maxsize=3)
def pss_time(n_id_2: int) -> np.ndarray:
    f = np.zeros(N, dtype=complex)
    f[bin_index(N, 72)[5:67]] = pss(n_id_2)
    t = np.fft.ifft(f)
    return t / np.linalg.norm(t)


@dataclass
class PssHit:
    n_id_2: int
    pos: int          # start of PSS useful part (first occurrence in buffer)
    cfo: float        # Hz
    metric: float     # folded, normalised correlation (0..1)
    pmr: float        # peak to median ratio of folded metric


@dataclass
class CellSync:
    n_id: int
    cp_ext: bool
    tdd: bool
    cfo: float                 # Hz, total (coarse + fine)
    frame_start: int           # sample (1.92 Msps) of subframe 0 start, 0..19199
    pss_metric: float
    pss_pmr: float
    sss_metric: float
    pss_positions: list = field(default_factory=list)
    sss_margin: float = 0.0

    @property
    def n_id_1(self):
        return self.n_id // 3

    @property
    def n_id_2(self):
        return self.n_id % 3

    @property
    def duplex(self):
        return "TDD" if self.tdd else "FDD"

    @property
    def cp(self):
        return "extended" if self.cp_ext else "normal"


def _sliding_energy(x: np.ndarray, n: int) -> np.ndarray:
    # float64 cumsum: in float32 the difference of two large partial sums
    # loses all precision and produces bogus (near zero) window energies
    p = np.abs(x.astype(np.complex128)) ** 2
    c = np.concatenate([[0.0], np.cumsum(p)])
    e = c[n:] - c[:-n]
    return np.maximum(e, 1e-3 * n * np.mean(p) + 1e-30)


def pss_search(x: np.ndarray, cfo_max: float = 40e3, cfo_step: float = 5e3,
               peaks_per_root: int = 2, cfo_center: float = 0.0) -> list[PssHit]:
    """Search PSS over all three roots and a grid of frequency offsets.
    x must be sampled at 1.92 Msps and be at least ~10 ms long."""
    x = np.asarray(x, dtype=np.complex64)
    x = x - np.mean(x)
    n_valid = len(x) - N + 1
    n_fold = n_valid // HALF_FRAME
    if n_fold < 1:
        raise ValueError("buffer too short for PSS search (min 5 ms)")
    energy = _sliding_energy(x, N)[:n_valid]
    n = np.arange(N)
    cfos = cfo_center + np.arange(-cfo_max, cfo_max + 1e-6, cfo_step)
    hits = []
    for u in range(3):
        t = pss_time(u)
        best = None
        best_metric_arr = None
        for f in cfos:
            tf = t * np.exp(2j * np.pi * f * n / FS)
            c = signal.fftconvolve(x, np.conj(tf[::-1]), mode="valid")
            m = np.abs(c) ** 2 / energy
            folded = m[:n_fold * HALF_FRAME].reshape(n_fold, HALF_FRAME).mean(axis=0)
            pk = int(np.argmax(folded))
            if best is None or folded[pk] > best[2]:
                best = (f, pk, folded[pk])
                best_metric_arr = folded
        f, pk, val = best
        med = float(np.median(best_metric_arr)) + 1e-12
        hits.append(PssHit(u, pk, f, float(val), float(val / med)))
        # additional peaks for the same root (different timing)
        arr = best_metric_arr.copy()
        for _ in range(peaks_per_root - 1):
            lo, hi = pk - 64, pk + 64
            idx = np.arange(lo, hi) % HALF_FRAME
            arr[idx] = 0
            pk = int(np.argmax(arr))
            hits.append(PssHit(u, pk, f, float(arr[pk]), float(arr[pk] / med)))
    hits.sort(key=lambda h: -h.metric)
    return hits


def fine_cfo_pss(x: np.ndarray, hit: PssHit) -> float:
    """Residual CFO from phase difference between PSS halves."""
    t = pss_time(hit.n_id_2)
    acc = 0j
    for p in range(hit.pos, len(x) - N, HALF_FRAME):
        seg = x[p:p + N] * np.exp(-2j * np.pi * hit.cfo * np.arange(p, p + N) / FS)
        c1 = np.sum(seg[:N // 2] * np.conj(t[:N // 2]))
        c2 = np.sum(seg[N // 2:] * np.conj(t[N // 2:]))
        acc += c2 * np.conj(c1)
    return float(np.angle(acc) / (2 * np.pi * (N / 2) / FS))


def fine_cfo_cp(x: np.ndarray, frame_start: int, cp_ext: bool) -> float:
    """CFO from cyclic-prefix correlation over all symbols in the buffer
    (unambiguous range +-7.5 kHz). x must already be coarse-corrected."""
    offs = symbol_offsets(N, cp_ext)
    cps = cp_lengths(N, cp_ext)
    acc = 0j
    sf0 = frame_start % SUBFRAME
    for s in range(sf0, len(x) - SUBFRAME, SUBFRAME):
        for o, cp in zip(offs, cps):
            a = s + o - cp
            c = x[a:a + cp]
            d = x[a + N:a + N + cp]
            acc += np.sum(np.conj(c) * d)
    return float(np.angle(acc) / (2 * np.pi * N / FS))


# FFT-window distance between SSS and PSS for (tdd, cp_ext)
_SSS_DIST = {(False, False): 137, (False, True): 160,
             (True, False): 412, (True, True): 480}


def _fft_bins(x, p):
    y = np.fft.fft(x[p:p + N])
    return y[bin_index(N, 72)[5:67]]


def sss_detect(x: np.ndarray, hit: PssHit, cfo: float, hyps=None) -> CellSync | None:
    """x: raw 1.92 Msps buffer, hit: PSS detection, cfo: total CFO.
    hyps: optional list of allowed (tdd, cp_ext) hypotheses."""
    nn = np.arange(len(x))
    xc = x * np.exp(-2j * np.pi * cfo * nn / FS)
    u = hit.n_id_2
    table = sss_table(u)
    d_pss = pss(u)
    best = None
    for (tdd, ext), dist in _SSS_DIST.items():
        if hyps is not None and (tdd, ext) not in hyps:
            continue
        acc = np.zeros(336)
        energy = 0.0
        positions = []
        k = 0
        for p in range(hit.pos, len(x) - N, HALF_FRAME):
            s = p - dist
            if s < 0:
                k += 1
                continue
            h = _fft_bins(xc, p) * np.conj(d_pss)
            y = _fft_bins(xc, s)
            z = y * np.conj(h) / (np.abs(h) ** 2 + 1e-3 * np.mean(np.abs(h) ** 2))
            c = table @ z.real
            if k & 1:                      # swap sf0/sf5 interpretation
                c = c.reshape(168, 2)[:, ::-1].reshape(-1)
            acc += c
            energy += np.sqrt(62 * np.sum(np.abs(z.real) ** 2))
            positions.append(p)
            k += 1
        if not positions:
            continue
        j = int(np.argmax(acc))
        metric = acc[j] / (energy + 1e-12)
        if best is None or metric > best[0]:
            # margin: best candidate vs best one with a different N_ID_1
            others = np.delete(acc.reshape(168, 2), j // 2, axis=0)
            margin = acc[j] / max(others.max(), 1e-12)
            best = (metric, tdd, ext, j, positions, margin)
    if best is None:
        return None
    metric, tdd, ext, j, positions, margin = best
    n1 = j // 2
    first_is_sf5 = bool(j & 1)
    # PSS position of the first occurrence in the buffer (parity of the
    # SSS hypothesis always refers to this occurrence)
    p0 = hit.pos
    offs = symbol_offsets(N, ext)
    n_symb = 6 if ext else 7
    if not tdd:
        sf_start = p0 - offs[n_symb - 1]           # last symbol of slot 0
        sf_index = 5 if first_is_sf5 else 0
    else:
        sf_start = p0 - offs[2]                    # 3rd symbol of subframe 1/6
        sf_index = 6 if first_is_sf5 else 1
    frame_start = (sf_start - sf_index * SUBFRAME) % FRAME
    return CellSync(n_id=3 * n1 + u, cp_ext=ext, tdd=tdd, cfo=cfo,
                    frame_start=int(frame_start), pss_metric=hit.metric,
                    pss_pmr=hit.pmr, sss_metric=float(metric),
                    pss_positions=positions, sss_margin=float(margin))


def _smooth_sc(h: np.ndarray, w: int) -> np.ndarray:
    """Moving average over w subcarriers (each half of the DC gap separately,
    edges with the available neighbours)."""
    out = np.empty_like(h)
    half = len(h) // 2
    ker = np.ones(w)
    for sl in (slice(0, half), slice(half, len(h))):
        v = h[sl]
        num = np.convolve(v, ker, mode="same")
        den = np.convolve(np.ones(len(v)), ker, mode="same")
        out[sl] = num / den
    return out


def cancel_sync(xc: np.ndarray, cell: CellSync) -> np.ndarray:
    """Subtract the PSS and SSS of `cell` (channel estimated per occurrence)
    from a CFO-corrected 1.92 Msps buffer (successive interference
    cancellation). Returns a new array."""
    y = xc.astype(np.complex128).copy()
    cp = 32 if cell.cp_ext else 9
    dist = _SSS_DIST[(cell.tdd, cell.cp_ext)]
    bins = bin_index(N, 72)[5:67]
    d_pss = pss(cell.n_id_2)
    for p in cell.pss_positions:
        sf5 = ((p - cell.frame_start) % FRAME) >= HALF_FRAME   # second half-frame
        for start, d in ((p, d_pss), (p - dist, sss(cell.n_id_1, cell.n_id_2, sf5))):
            if start - cp < 0 or start + N > len(y):
                continue
            yf = np.fft.fft(y[start:start + N])
            # smoothed over neighbouring subcarriers: a per-subcarrier y/d
            # would also remove the PSS/SSS of a time-aligned neighbour
            # (synchronised networks put every cell's PSS in the same symbol)
            h = _smooth_sc(yf[bins] / d, 9)
            f = np.zeros(N, dtype=complex)
            f[bins] = h * d
            t = np.fft.ifft(f)
            y[start - cp:start + N] -= np.concatenate([t[-cp:], t])
    return y.astype(np.complex64)


def cell_search(x: np.ndarray, cfo_max: float = 40e3, cfo_step: float = 5e3,
                pmr_min: float = 6.0, sss_min: float | None = None,
                max_cells: int = 6) -> list[CellSync]:
    """Full cell search on a 1.92 Msps buffer. Returns detected cells,
    strongest first.

    All cells are received through the same local oscillator, so their
    frequency offsets agree to within a few hundred Hz. Once the strongest
    cell is found, its PSS/SSS are cancelled from the signal and the search
    is repeated only around its CFO. This suppresses the cross-root and
    time/frequency ambiguity sidelobes of a strong cell's Zadoff-Chu PSS
    that would otherwise show up as phantom neighbour cells."""
    cells: list[CellSync] = []
    hits = pss_search(x, cfo_max, cfo_step)
    first = None
    for hit in hits:
        if hit.pmr < pmr_min:
            continue
        first = _try_hit(x, hit, sss_min, None, cfo_max)
        if first is not None:
            break
    if first is None:
        return cells
    cells.append(first)
    ref = first.cfo
    nn = np.arange(len(x))
    xc = (x * np.exp(-2j * np.pi * ref * nn / FS)).astype(np.complex64)
    xc = cancel_sync(xc, _relative(first, ref))
    while len(cells) < max_cells:
        found = None
        for hit in pss_search(xc, cfo_step / 2, cfo_step / 2, peaks_per_root=2):
            if hit.pmr < pmr_min:
                continue
            cell = _try_hit(xc, hit, sss_min, [(first.tdd, first.cp_ext)], 0.0)
            if cell is None or any(c.n_id == cell.n_id for c in cells)                     or abs(cell.cfo) > 1.5e3:
                continue
            found = cell
            break
        if found is None:
            break
        xc = cancel_sync(xc, found)
        found.cfo += ref
        cells.append(found)
    return cells


def _relative(cell: CellSync, ref: float) -> CellSync:
    from dataclasses import replace
    return replace(cell, cfo=cell.cfo - ref)


def _refine_pss(x: np.ndarray, n_id_2: int, pos: int, cfo: float, win: int = 80):
    """Re-locate the PSS (folded over half-frames) near `pos` assuming
    frequency offset `cfo`. Returns (pos, folded metric)."""
    t = pss_time(n_id_2)
    n = np.arange(N)
    tf = np.conj((t * np.exp(2j * np.pi * cfo * n / FS))[::-1])
    acc = np.zeros(2 * win + 1)
    cnt = 0
    for p in range(pos % HALF_FRAME, len(x) - N - win, HALF_FRAME):
        a = p - win
        if a < 0:
            continue
        seg = x[a:a + 2 * win + N]
        c = np.abs(signal.fftconvolve(seg, tf, mode="valid")) ** 2
        e = _sliding_energy(seg, N)
        acc += c / e
        cnt += 1
    if not cnt:
        return pos, 0.0
    i = int(np.argmax(acc))
    p0 = pos % HALF_FRAME - win + i
    while p0 < 0:
        p0 += HALF_FRAME
    return p0, float(acc[i] / cnt)


def _try_hit(x, hit: PssHit, sss_min, hyps=None, cfo_max: float = 40e3) -> CellSync | None:
    # A Zadoff-Chu PSS shifted by an integer number of subcarriers looks
    # like a time-shifted PSS, so the coarse search may lock k*15 kHz away
    # from the true offset. Test those alternatives and let the SSS (which
    # has no such ambiguity) decide.
    best = None
    for k in range(-3, 4):
        c0 = hit.cfo + k * 15e3
        if abs(c0) > cfo_max + 7.5e3:
            continue
        pos, metric = (hit.pos, hit.metric) if k == 0 else             _refine_pss(x, hit.n_id_2, hit.pos, c0)
        if metric < 0.3 * hit.metric:
            continue
        h = PssHit(hit.n_id_2, pos, c0, metric, hit.pmr)
        cfo = c0 + fine_cfo_pss(x, h)
        cell = sss_detect(x, h, cfo, hyps)
        if cell is not None and (best is None or cell.sss_metric > best.sss_metric):
            best = cell
    cell = best
    if cell is None:
        return None
    cfo = cell.cfo
    # noise-only SSS metric has std ~ 1/sqrt(62*K) for K half-frames
    k = max(len(cell.pss_positions), 1)
    thr = sss_min if sss_min is not None else max(0.12, 4.5 / np.sqrt(62 * k))
    if cell.sss_metric < thr:
        return None
    # the winning N_ID_1 must clearly beat every other one: genuine cells
    # give a margin of ~2 (>=1.58 even at -6 dB SNR), sidelobes and
    # interference stay below ~1.35
    if cell.sss_margin < SSS_MARGIN_MIN:
        return None
    # refine CFO with CP correlation (only meaningful for a dominant cell)
    if cell.pss_metric > 0.3:
        nn = np.arange(len(x))
        xc = x * np.exp(-2j * np.pi * cfo * nn / FS)
        cell.cfo = cfo + fine_cfo_cp(xc, cell.frame_start, cell.cp_ext)
    return cell
