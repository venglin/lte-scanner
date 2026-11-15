"""SIB1 reception of an NR cell found by the SS/PBCH block search.

1. CORESET#0 and search space #0 from the MIB give the frequency span of
   the initial downlink BWP and the slots of the Type0-PDCCH monitoring
   occasions of every SSB index (38.213 13).
2. Frame timing and SFN come from the received SSBs (PSS position of a
   known SSB index, SFN and half frame of the decoded MIBs).
3. Each monitored slot is demodulated over CORESET#0 (with the 38.211 5.4
   per-symbol phase removed), the PDCCH candidates of aggregation levels
   4/8/16 are blind-decoded for DCI 1_0 with SI-RNTI.
4. The scheduled PDSCH (default time allocation table A, type 1 frequency
   allocation, DMRS type 1 with up to two additional positions) is
   equalised with its DMRS, demapped and descrambled. All transmissions of
   the same transport block (SIB1 repeats every 20 ms or so, with varying
   redundancy versions) are soft-combined in the LDPC circular buffer.
5. The transport block is the BCCH-DL-SCH message (transparent MAC)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..dsp import resample
from . import ldpc, pdcch, pdsch, polar
from .pdcch import Coreset0, Dci10Si, coreset0
from .rrc import NrSib1, decode_bcch_dlsch
from .search import NrCell
from .synth import SSB_SYMBOLS


# ------------------------------------------------------------------ timing
def _sym_durations(scs: int) -> np.ndarray:
    """Durations [s] of the 14 * 2^mu symbols of a subframe (incl. CP)."""
    mu = {15000: 0, 30000: 1}[scs]
    n = 14 * 2 ** mu
    d = np.full(n, (1 + 144 / 2048) / scs)
    d[np.arange(n) % (7 * 2 ** mu) == 0] += 16 / (2048 * 15000)
    return d


def cp_len_s(scs: int, s: int) -> float:
    """CP duration [s] of symbol s of a subframe."""
    mu = {15000: 0, 30000: 1}[scs]
    cp = 144 / 2048 / scs
    return cp + (16 / (2048 * 15000) if s % (7 * 2 ** mu) == 0 else 0.0)


def sym_start(scs: int, s_in_frame: int) -> float:
    """Start [s] of symbol s (counted from the frame start, CP included)."""
    d = _sym_durations(scs)
    sf, s = divmod(s_in_frame, len(d))
    return sf * 1e-3 + float(np.sum(d[:s]))


def pss_offset(scs_ssb: int, ssb_index: int, half_frame: int) -> float:
    """Time from the frame start to the useful part of the PSS of an SSB
    (case A / C positions)."""
    s = SSB_SYMBOLS[ssb_index] + half_frame * 70 * (scs_ssb // 15000)
    d = _sym_durations(scs_ssb)
    return sym_start(scs_ssb, s) + cp_len_s(scs_ssb, s % len(d))


@dataclass
class FrameRef:
    t: float                  # capture time of the frame start [s]
    sfn: int


def frame_refs(cell: NrCell, scs_ssb: int) -> list[FrameRef]:
    """Frame starts and SFNs from every received SSB whose index and half
    frame are known (SFN from the decoded MIBs)."""
    dec = [o for o in cell.obs if o.mib]
    if not dec:
        return []
    hf = {o.mib.ssb_index: o.mib.half_frame for o in dec}
    r0 = dec[0]
    t0 = r0.t - pss_offset(scs_ssb, r0.mib.ssb_index, r0.mib.half_frame)
    out = []
    for o in cell.obs:
        if o.ssb_index is None or o.ssb_index not in hf:
            continue
        t = o.t - pss_offset(scs_ssb, o.ssb_index, hf[o.ssb_index])
        n = int(round((t - t0) / 10e-3))
        if abs(t - t0 - n * 10e-3) > 20e-6:
            continue
        out.append(FrameRef(t, (r0.mib.sfn + n) % 1024))
    return sorted(out, key=lambda f: f.t)


def frames_in(refs: list[FrameRef], t_end: float) -> list[FrameRef]:
    """All frames of the capture [0, t_end). Frame starts come from a
    straight line fitted to all received SSBs (frame period incl. the
    receiver clock error), which is far more accurate than any single SSB."""
    if not refs:
        return []
    first = refs[0]
    n = np.array([round((r.t - first.t) / 10e-3) for r in refs], dtype=float)
    t = np.array([r.t for r in refs])
    if len(set(n)) >= 2:
        period, t0 = np.polyfit(n, t, 1)
    else:
        period, t0 = 10e-3, float(np.mean(t - n * 10e-3))
    k0 = -int(np.floor(first.t / 10e-3)) - 1
    k1 = int(np.ceil((t_end - first.t) / 10e-3)) + 1
    out = []
    for k in range(k0, k1):
        tk = t0 + k * period
        if -1e-3 < tk < t_end:
            out.append(FrameRef(float(tk), (first.sfn + k) % 1024))
    return out


# --------------------------------------------------------- slot demodulation
def slot_grid(x: np.ndarray, fs: float, fc: float, c0: Coreset0, t_slot: float, slot: int,
              cfo: float, n_fft: int | None = None) -> np.ndarray | None:
    """(14, n_sc) resource grid of CORESET#0's band for slot `slot` (of
    the frame) starting at capture time t_slot. cfo: receiver frequency
    offset at the CORESET centre [Hz]."""
    scs = c0.scs
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(fs / scs)))
    fs_d = n_fft * scs
    mu = c0.mu
    slot_dur = 1e-3 / 2 ** mu
    pad = 50e-6
    p0 = int(np.floor((t_slot - pad) * fs))
    p1 = int(np.ceil((t_slot + slot_dur + pad) * fs))
    if p0 < 0 or p1 > len(x):
        return None
    n = np.arange(p0, p1)
    seg = x[p0:p1] * np.exp(-2j * np.pi * (c0.f_center - fc + cfo) * n / fs)
    y = resample(seg.astype(np.complex64), fs, fs_d)
    t_seg = p0 / fs
    d = _sym_durations(scs)
    slot_in_sf = slot % 2 ** mu
    k_rel = np.arange(c0.n_sc) - c0.n_sc // 2
    grid = np.empty((14, c0.n_sc), dtype=complex)
    t = 0.0
    cp_n = 144 * n_fft // 2048
    bo = cp_n // 4
    for l in range(14):
        s = slot_in_sf * 14 + l
        cp = cp_len_s(scs, s)
        ideal = (t_slot - t_seg + t + cp) * fs_d           # useful part start
        w = int(round(ideal)) - bo
        if w < 0 or w + n_fft > len(y):
            return None
        f = np.fft.fft(y[w:w + n_fft]) / np.sqrt(n_fft)
        delta = ideal - w
        ph = np.exp(2j * np.pi * k_rel * delta / n_fft)
        # TS 38.211 5.4: per-symbol upconversion phase at the mixing frequency
        sym_ph = np.exp(2j * np.pi * c0.f_center * (t + cp))
        grid[l] = f[k_rel % n_fft] * ph * sym_ph
        t += d[s]
    return grid


# ------------------------------------------------------------------ PDCCH
def _rb_channel(grid: np.ndarray, c0: Coreset0, n_id: int, slot: int, sym0: int):
    """Per-RB channel of the CORESET from its DMRS (averaged over the
    CORESET symbols and over neighbouring RBs of the same REG bundle) and
    a noise estimate per bundle."""
    rbs = np.arange(c0.n_rb)
    ls = np.zeros((c0.n_symb, c0.n_rb, 3), dtype=complex)
    for s in range(c0.n_symb):
        l = sym0 + s
        ref = pdcch.dmrs(c0.n_rb, slot, l, n_id)
        ls[s] = grid[l][12 * rbs[:, None] + pdcch.DMRS_SC[None, :]] * np.conj(ref)
    per_rb = ls.mean(axis=(0, 2))
    h = np.empty(c0.n_rb, dtype=complex)
    noise = np.empty(c0.n_rb)
    bundle_rbs = 6 // c0.n_symb
    for b0 in range(0, c0.n_rb, bundle_rbs):
        sl = slice(b0, b0 + bundle_rbs)
        hb = per_rb[sl]
        # neighbour average inside the bundle (3 RBs)
        sm = np.convolve(hb, np.ones(3) / 3, mode="same")
        sm[0] = hb[:2].mean()
        sm[-1] = hb[-2:].mean()
        h[sl] = sm
        res = ls[:, sl, :] - sm[None, :, None]
        n_pil = ls[:, sl, :].size
        noise[sl] = float(np.mean(np.abs(res) ** 2)) * n_pil / max(n_pil - len(hb), 1)
    return h, np.maximum(noise, 1e-12)


@dataclass
class DciRx:
    t: float                  # slot start in the capture [s]
    sfn: int
    slot: int
    level: int
    cce: int
    dci: Dci10Si
    snr_db: float
    how: str = "decoded"      # decoded | combined (PDCCH soft-combined) | matched (known DCI)


def decode_pdcch(grid: np.ndarray, c0: Coreset0, n_id: int, slot: int, sym0: int):
    """Blind decoding of the Type0 CSS candidates. Returns a list of
    (level, cce, Dci10Si, snr_db)."""
    h, noise = _rb_channel(grid, c0, n_id, slot, sym0)
    a_len = pdcch.dci10_si_size(c0.n_rb)
    out = []
    used: set = set()
    for cce, lv in sorted(pdcch.candidates(c0), key=lambda c: -c[1]):
        l, k, rbs = pdcch.candidate_res(c0, n_id, cce, lv, sym0)
        if used & set(rbs):
            continue
        hk = h[k // 12]
        nk = noise[k // 12]
        z = grid[l, k] * np.conj(hk)
        llr = np.empty(2 * len(z))
        llr[0::2] = z.real
        llr[1::2] = z.imag
        llr *= 2 * np.sqrt(2) / np.repeat(nk, 2)
        llr *= 1 - 2 * pdcch.scrambling(n_id, len(llr)).astype(np.float64)
        a = pdcch.dci_decode(llr, a_len)
        if a is None:
            continue
        snr = float(np.mean(np.abs(hk) ** 2) / np.mean(nk))
        out.append((lv, cce, pdcch.dci10_si_unpack(a, c0.n_rb), 10 * np.log10(max(snr, 1e-9))))
        used |= set(rbs)
    return out


def _candidate_pilots(grid: np.ndarray, c0: Coreset0, n_id: int, slot: int, sym0: int,
                      rbs: list[int]):
    """LS estimates at the DMRS of the given RBs: (values, subcarriers)."""
    v, k = [], []
    for s_ in range(c0.n_symb):
        l = sym0 + s_
        ref = pdcch.dmrs(c0.n_rb, slot, l, n_id)
        for rb in rbs:
            kk = 12 * rb + pdcch.DMRS_SC
            v.append(grid[l, kk] * np.conj(ref[rb]))
            k.append(kk)
    return np.concatenate(v), np.concatenate(k)


WEAK_DMRS_METRIC = 6.0       # coherent DMRS energy / noise-only expectation
_SLOPES = np.linspace(-0.04, 0.04, 33)       # rad per subcarrier


def decode_pdcch_combined(slots: list, c0: Coreset0, n_id: int):
    """PDCCH of a weak cell: soft-combine one candidate over the monitoring
    occasions of one SSB index (same CCEs and scrambling every period).

    slots: list of (grid, slot number) of the same beam and slot offset.
    Per occasion, a phase slope over frequency and a common phase are fitted
    to the candidate's DMRS; the occasions whose DMRS stands out of the
    noise are kept and the channel response is averaged over them. The DCIs
    are assumed identical except for the redundancy version: the codeword
    difference of an RV change is known (polar code and CRC are linear), so
    each occasion is aligned with the running sum by the RV difference that
    correlates best. Returns (level, cce, Dci10Si of the first kept
    occasion, [(index into slots, RV)], snr_db) or None."""
    a_len = pdcch.dci10_si_size(c0.n_rb)
    rv_pos = a_len - 18                       # RV field: 2 bits before SI ind. + 15 reserved
    for cce, lv in sorted(pdcch.candidates(c0), key=lambda c: -c[1]):
        l_re, k_re, rbs = pdcch.candidate_res(c0, n_id, cce, lv, sym0=0)
        occ = []
        for idx, (g, slot, sym0) in enumerate(slots):
            v, kp = _candidate_pilots(g, c0, n_id, slot, sym0, rbs)
            # residual timing -> phase slope over frequency; searched over a
            # small range (+-0.2 us) since a free estimate is noise at low SNR
            e = np.abs(np.exp(-1j * _SLOPES[:, None] * kp[None, :]) @ v)
            slope = float(_SLOPES[int(np.argmax(e))])
            r = v * np.exp(-1j * slope * kp)
            phi = float(np.angle(np.sum(r)))
            al = r * np.exp(-1j * phi)
            p = float(np.mean(np.abs(v) ** 2))
            metric = abs(np.sum(al)) ** 2 / max(len(v) * p, 1e-30)
            occ.append((idx, slope, phi, al, kp, metric))
        kept = [o for o in occ if o[5] >= WEAK_DMRS_METRIC]
        if len(kept) < 2:
            continue
        kp = kept[0][4]
        h_avg = np.mean([o[3] for o in kept], axis=0)
        # per RB average, smoothed over neighbouring RBs of the candidate
        rb_of = kp // 12
        h_rb = {rb: np.mean(h_avg[rb_of == rb]) for rb in rbs}
        noise = float(np.mean([np.mean(np.abs(o[3] - h_avg) ** 2) for o in kept]))
        noise *= len(kept) / max(len(kept) - 1, 1)
        llrs = []
        for idx, slope, phi, al, _, _ in kept:
            g = slots[idx][0]
            l = l_re + slots[idx][2]          # candidate_res was built for symbol 0
            hk = np.array([h_rb[k // 12] for k in k_re]) * np.exp(1j * (phi + slope * k_re))
            z = g[l, k_re] * np.conj(hk)
            llr = np.empty(2 * len(z))
            llr[0::2] = z.real
            llr[1::2] = z.imag
            llr *= 2 * np.sqrt(2) / max(noise, 1e-20)
            llr *= 1 - 2 * pdcch.scrambling(n_id, len(llr)).astype(np.float64)
            llrs.append(llr)
        e = 108 * lv
        diffs = []
        for dv in range(4):
            a = np.zeros(a_len, dtype=np.uint8)
            a[rv_pos:rv_pos + 2] = (dv >> 1) & 1, dv & 1
            c = np.concatenate([a, pdcch._crc_bits(a, 0) ^ pdcch._crc_bits(np.zeros(a_len, np.uint8), 0)])
            diffs.append(1 - 2.0 * polar.encode_rm(c, e))
        order = sorted(range(len(kept)), key=lambda i: -kept[i][5])
        acc = llrs[order[0]].copy()
        rel = {order[0]: 0}
        for i in order[1:]:
            dv = int(np.argmax([np.dot(acc, llrs[i] * df) for df in diffs]))
            acc += llrs[i] * diffs[dv]
            rel[i] = dv
        a = pdcch.dci_decode(acc, a_len)
        if a is None:
            continue
        dci = pdcch.dci10_si_unpack(a, c0.n_rb)
        first = kept[order[0]][0]
        rvs = [(kept[i][0], dci.rv ^ rel[i]) for i in order]
        snr = float(np.mean(np.abs(h_avg) ** 2) / max(noise, 1e-20))
        return lv, cce, dci, rvs, 10 * np.log10(max(snr, 1e-9)), first
    return None


MATCH_SIGMA = 6.0            # known-DCI correlation threshold (noise: N(0, 1))


def _flat_llr(g: np.ndarray, c0: Coreset0, n_id: int, slot: int, sym0: int, cce: int, lv: int):
    """Descrambled soft bits of a candidate with a flat channel model
    (amplitude, common phase and a timing slope fitted to its DMRS)."""
    l_re, k_re, rbs = pdcch.candidate_res(c0, n_id, cce, lv, sym0)
    v, kp = _candidate_pilots(g, c0, n_id, slot, sym0, rbs)
    e = np.abs(np.exp(-1j * _SLOPES[:, None] * kp[None, :]) @ v)
    slope = float(_SLOPES[int(np.argmax(e))])
    phi = float(np.angle(np.sum(v * np.exp(-1j * slope * kp))))
    z = g[l_re, k_re] * np.exp(-1j * (phi + slope * k_re))
    llr = np.empty(2 * len(z))
    llr[0::2] = z.real
    llr[1::2] = z.imag
    return llr * (1 - 2 * pdcch.scrambling(n_id, len(llr)).astype(np.float64))


def match_known_dci(g: np.ndarray, c0: Coreset0, n_id: int, slot: int, sym0: int,
                    known: dict, cache: dict):
    """Look for a DCI already decoded elsewhere (any RV) in this slot: the
    correlation of the soft bits with the known codewords, normalised to
    N(0, 1) under noise. Returns (level, cce, Dci10Si, sigma) or None."""
    best = None
    for dci0, lv in known.items():
        for cce, lv2 in pdcch.candidates(c0):
            if lv2 != lv:
                continue
            llr = _flat_llr(g, c0, n_id, slot, sym0, cce, lv)
            nrm = float(np.sqrt(np.dot(llr, llr))) or 1.0
            for rv in range(4):
                d = replace(dci0, rv=rv)
                key = (d.riv, d.tdra, d.vrb_interleaved, d.mcs, rv, d.si_indicator, lv)
                x = cache.get(key)
                if x is None:
                    x = 1 - 2.0 * pdcch.dci_encode(pdcch.dci10_si_pack(d), 108 * lv)
                    cache[key] = x
                sig = float(np.dot(llr, x)) / nrm
                if best is None or sig > best[3]:
                    best = (lv, cce, d, sig)
    return best if best and best[3] >= MATCH_SIGMA else None


# ------------------------------------------------------------------ PDSCH
def _ssb_blocked(cell: NrCell, c0: Coreset0, t_slot: float, slot: int, scs_ssb: int) -> set:
    """(symbol, PRB) pairs of the slot that carry a received SS/PBCH block."""
    out = set()
    d = _sym_durations(c0.scs)
    slot_in_sf = slot % 2 ** c0.mu
    starts, t = [], t_slot
    for l in range(14):
        dl = d[slot_in_sf * 14 + l]
        starts.append((t, t + dl))
        t += dl
    t_ssb_sym = (1 + 144 / 2048) / scs_ssb
    lo = (cell.f_ref - 120 * scs_ssb - c0.f_low) / c0.scs
    hi = lo + 240 * scs_ssb / c0.scs
    prbs = [p for p in range(c0.n_rb) if 12 * p < hi and 12 * (p + 1) > lo]
    for o in cell.obs:
        a = o.t - 144 / 2048 / scs_ssb
        b = a + 4 * t_ssb_sym
        if b < t_slot or a > t:
            continue
        for l, (s0, s1) in enumerate(starts):
            if s0 < b - 1e-7 and s1 > a + 1e-7:
                out |= {(l, p) for p in prbs}
    return out


def pdsch_llr(grid: np.ndarray, alloc: pdsch.Allocation, n_id: int, slot: int,
              blocked: set | None = None) -> tuple[np.ndarray, float]:
    """Soft bits (descrambled) of the PDSCH of one DCI and its SINR [dB]."""
    pil_k, pil_v = [], []
    for l in alloc.dmrs:
        ref = pdsch.dmrs_seq(slot, l, n_id, 6 * alloc.dci.n_rb * 2)
        for p in sorted(alloc.prbs):
            k = 12 * p + np.arange(0, 12, 2)
            pil_k.append(k)
            pil_v.append(grid[l, k] * np.conj(ref[k // 2]))
    pil_k = np.concatenate(pil_k)
    pil_v = np.concatenate(pil_v)
    nd = len(alloc.dmrs)
    kk = pil_k.reshape(nd, -1)[0]
    vv = pil_v.reshape(nd, -1).mean(axis=0)
    w = 5
    sm = np.convolve(vv, np.ones(w) / w, mode="same")
    edge = np.convolve(np.ones(len(vv)), np.ones(w) / w, mode="same")
    sm = sm / edge
    res = pil_v.reshape(nd, -1) - sm[None, :]
    n_pil = res.size
    noise = float(np.mean(np.abs(res) ** 2)) * n_pil / max(n_pil - len(vv) / w * nd, 1)
    beta = np.sqrt(2) if pdsch.cdm_groups_without_data(alloc.length) == 2 else 1.0
    l_re, k_re = pdsch.data_res(alloc, blocked)
    hk = (np.interp(k_re, kk, sm.real) + 1j * np.interp(k_re, kk, sm.imag)) / beta
    noise_d = noise / beta ** 2 * 1.0
    y = grid[l_re, k_re]
    g2 = np.abs(hk) ** 2
    z = y * np.conj(hk) / np.maximum(g2, 1e-20)
    llr = pdsch.demap(z, g2 / max(noise_d, 1e-20), alloc.qm)
    llr *= 1 - 2 * pdsch.scrambling(n_id, len(llr)).astype(np.float64)
    sinr = float(np.mean(g2) / max(noise_d, 1e-20))
    return llr, 10 * np.log10(max(sinr, 1e-9))


# --------------------------------------------------------------- top level
@dataclass
class Sib1Result:
    n_id: int
    coreset: Coreset0 | None
    dcis: list[DciRx] = field(default_factory=list)
    alloc: pdsch.Allocation | None = None
    pdsch_sinr_db: float | None = None
    combined: int = 0                       # transmissions in the decoded TB
    payload: bytes | None = None
    sib1: NrSib1 | None = None
    error: str | None = None
    slots_tried: int = 0
    rvs: list[int] = field(default_factory=list)

    @property
    def point_a(self) -> float | None:
        """RF frequency of point A [Hz] (from offsetToPointA)."""
        if not self.sib1 or self.sib1.offset_to_point_a is None or self.crb_ssb is None:
            return None
        return self.crb_ssb - self.sib1.offset_to_point_a * 12 * 15000

    crb_ssb: float | None = None

    def carriers(self) -> list[tuple[int, int, float, float]]:
        """(SCS kHz, N_RB, lowest subcarrier [Hz], centre [Hz]) per carrier."""
        pa = self.point_a
        if pa is None:
            return []
        out = []
        for c in self.sib1.carriers:
            lo = pa + c.offset_to_carrier * 12 * c.scs_khz * 1e3
            out.append((c.scs_khz, c.n_rb, lo, lo + c.n_rb * 12 * c.scs_khz * 1e3 / 2))
        return out


def coreset_in_capture(c0: Coreset0, fs: float, fc: float, usable: float = 0.49) -> bool:
    return (c0.f_low >= fc - usable * fs) and (c0.f_low + c0.bandwidth <= fc + usable * fs)


def decode_sib1(x: np.ndarray, fs: float, fc: float, cell: NrCell, log=None) -> Sib1Result:
    m = cell.mib
    res = Sib1Result(cell.n_id, None)
    if m is None:
        res.error = "MIB not decoded"
        return res
    if not m.sib1_present:
        res.error = "no CORESET#0 (k_SSB > 23)"
        return res
    scs_ssb = cell.scs
    c0 = coreset0(m, cell.f_ref, scs_ssb)
    res.coreset = c0
    if c0 is None:
        res.error = "CORESET#0 configuration not supported"
        return res
    res.crb_ssb = cell.f_ref - 120 * scs_ssb - m.k_ssb * 15000
    if not coreset_in_capture(c0, fs, fc):
        res.error = (f"CORESET#0 ({c0.f_low / 1e6:.3f}-{(c0.f_low + c0.bandwidth) / 1e6:.3f} MHz) "
                     f"is not inside the capture; tune to {c0.f_center / 1e6:.3f} MHz")
        return res
    refs = frame_refs(cell, scs_ssb)
    frames = frames_in(refs, len(x) / fs)
    cfo = cell.cfo * c0.f_center / cell.f_ref
    n_slot = 10 * 2 ** c0.mu
    beams = sorted({o.ssb_index for o in cell.obs if o.ssb_index is not None})
    occ = {}
    for i in beams:
        n0, parity, sym0 = c0.occasions(i)
        for f in frames:
            if f.sfn % 2 != parity:
                continue
            for ds in (0, 1):
                s = n0 + ds
                t = f.t + (s // n_slot) * 10e-3 + sym_start(c0.scs, (s % n_slot) * 14)
                occ[(round(t * 1e7), sym0)] = (t, (f.sfn + s // n_slot) % 1024, s % n_slot, sym0,
                                               i, ds)
    rx: list[tuple[pdsch.Allocation, np.ndarray, float, DciRx]] = []
    grids = []
    for t, sfn, slot, sym0, beam, ds in sorted(occ.values()):
        g = slot_grid(x, fs, fc, c0, t, slot, cfo)
        if g is None:
            continue
        grids.append((t, sfn, slot, sym0, beam, ds, g))
        res.slots_tried += 1
        for lv, cce, dci, snr in decode_pdcch(g, c0, cell.n_id, slot, sym0):
            d = DciRx(t, sfn, slot, lv, cce, dci, snr)
            res.dcis.append(d)
            if dci.si_indicator != 0:
                continue
            a = pdsch.allocation(dci, m.dmrs_type_a_pos)
            if a is None or a.tbs == 0:
                continue
            llr, sinr = pdsch_llr(g, a, cell.n_id, slot, _ssb_blocked(cell, c0, t, slot, scs_ssb))
            rx.append((a, llr, sinr, d))
    def add(item, d: DciRx):
        t, sfn, slot, sym0, _, _, g = item
        res.dcis.append(d)
        if d.dci.si_indicator != 0:
            return
        a = pdsch.allocation(d.dci, m.dmrs_type_a_pos)
        if a is None or a.tbs == 0:
            return
        llr, sinr = pdsch_llr(g, a, cell.n_id, slot, _ssb_blocked(cell, c0, t, slot, scs_ssb))
        rx.append((a, llr, sinr, d))

    if not res.dcis:
        # weak cell: combine the PDCCH over the occasions of each beam / slot
        groups_w: dict = {}
        for item in grids:
            groups_w.setdefault((item[4], item[5]), []).append(item)
        for items in groups_w.values():
            r = decode_pdcch_combined([(it[6], it[2], it[3]) for it in items], c0, cell.n_id)
            if r is None:
                continue
            lv, cce, dci, rvs, snr, _ = r
            for idx, rv in rvs:
                it = items[idx]
                add(it, DciRx(it[0], it[1], it[2], lv, cce, replace(dci, rv=rv), snr, "combined"))
    # the same DCI (any RV) in the other occasions: correlation with the
    # known codewords works far below the PDCCH decoding threshold
    known = {replace(d.dci, rv=0): d.level for d in res.dcis}
    if known:
        have = {round(d.t * 1e7) for d in res.dcis}
        cache: dict = {}
        for it in grids:
            if round(it[0] * 1e7) in have:
                continue
            mk = match_known_dci(it[6], c0, cell.n_id, it[2], it[3], known, cache)
            if mk:
                lv, cce, dci, sig = mk
                add(it, DciRx(it[0], it[1], it[2], lv, cce, dci, float(sig), "matched"))
    if log:
        n_dec = sum(1 for d in res.dcis if d.how == "decoded")
        log(f"PCI {cell.n_id}: {res.slots_tried} Type0-PDCCH slots, {len(res.dcis)} DCIs "
            f"(SI-RNTI; {n_dec} decoded alone, the others by soft combining / matching)")
    if not rx:
        res.error = "no DCI 1_0 with SI-RNTI found" if not res.dcis else "no SIB1 PDSCH"
        return res
    groups: dict = {}
    for item in rx:
        a = item[0]
        groups.setdefault((a.tbs, a.qm, a.rate), []).append(item)
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tbs, qm, rate = key
        # single transmissions (self-decodable RV 0 / 3 first), then all combined
        order = sorted(items, key=lambda it: (it[0].dci.rv not in (0, 3), -it[2]))
        tries = [[it] for it in order[:4]] + ([order] if len(order) > 1 else [])
        for sel in tries:
            bits = ldpc.dlsch_decode([(it[1], it[0].dci.rv) for it in sel], tbs, rate, qm)
            if bits is None:
                continue
            payload = np.packbits(bits).tobytes()
            res.alloc = sel[0][0]
            res.pdsch_sinr_db = float(np.mean([it[2] for it in sel]))
            res.combined = len(sel)
            res.rvs = [it[0].dci.rv for it in sel]
            res.payload = payload
            try:
                kind, sib = decode_bcch_dlsch(payload)
            except (ValueError, IndexError):
                kind, sib = "undecodable", None
            res.sib1 = sib if kind == "SIB1" else None
            if res.sib1 is None:
                res.error = f"transport block decoded but it is not a SIB1 ({kind})"
            return res
    res.alloc = rx[0][0]
    res.pdsch_sinr_db = float(np.mean([it[2] for it in rx]))
    res.error = f"PDSCH CRC failed ({len(rx)} transmissions combined)"
    return res
