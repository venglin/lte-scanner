"""Synthetic NR downlink with SS/PBCH block bursts (for tests and demos).

SSBs are placed at the case A/C positions {2, 8} + 14 n of the first half
frame, every `period_ms`, each SSB index with its own (beam) gain. The rest
of the carrier is filled with random QPSK like a loaded PDSCH. Optionally a
SIB1 is broadcast: DCI 1_0 with SI-RNTI on CORESET#0 in the Type0-PDCCH
monitoring occasion (slot n0) of every SSB index, and its PDSCH, every
20 ms with the redundancy versions 0, 2, 3, 1."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..dsp import mix, resample
from . import ldpc, pdcch, pdsch
from .params import SSB_REF_K, cp_len
from .pbch import Mib, ssb_grid

SSB_SYMBOLS = (2, 8, 16, 22, 30, 36, 44, 50)


@dataclass
class NrSynthConfig:
    n_id: int = 500
    scs: int = 30000
    l_max: int = 8
    ssb_indices: tuple = (0, 1, 2, 3)
    beam_gain_db: tuple = (0.0, -3.0, -6.0, -9.0, -12.0, -12.0, -12.0, -12.0)
    period_ms: int = 20
    sfn0: int = 100
    k_ssb: int = 6
    coreset0: int = 12
    search_space0: int = 0
    cell_barred: bool = False
    carrier_sc: int = 600            # filled subcarriers around the SSB (0 = none)
    fill_db: float = 0.0
    seed: int = 0
    dmrs_type_a_pos: int = 2
    sib1: bytes | None = None        # BCCH-DL-SCH payload to broadcast
    sib1_mcs: int = 2
    sib1_rbs: tuple = (0, 24)        # (first VRB, number of RBs)
    sib1_tdra: int = 0               # default table A row - 1
    sib1_level: int = 8
    sib1_interleaved: bool = False
    sib1_rvs: tuple = (0, 2, 3, 1)
    # SI messages (BCCH-DL-SCH payloads) in their SI-windows; the schedule
    # must match the one in the SIB1 payload
    si: tuple = ()                   # payload per SI message n = 1, 2, ...
    si_periods: tuple = (16,)        # si-Periodicity [frames] per message
    si_window: int = 10              # si-WindowLength [slots]
    si_slots: tuple = (1, 5)         # transmissions at these window slots
    si_mcs: int = 4
    si_rbs: tuple = (0, 24)
    si_coreset0_crb: int = 0         # common RB of CORESET#0 (SI DMRS start at point A)
    si_tdra: int = 0                 # DCI time domain field of the SI messages
    si_tdra_list: tuple = ()         # pdsch-TimeDomainAllocationList (as in SIB1)

    def mib(self, sfn: int, ssb_index: int) -> Mib:
        return Mib(sfn=sfn, half_frame=0, scs_common=self.scs, k_ssb=self.k_ssb,
                   dmrs_type_a_pos=self.dmrs_type_a_pos, coreset0=self.coreset0, search_space0=self.search_space0,
                   cell_barred=self.cell_barred, intra_freq_reselection=True,
                   ssb_index=ssb_index, l_max=self.l_max)


def generate(cfg: NrSynthConfig, duration: float, fs_out: float = 20e6,
             f_offset: float = 0.0, f_ref: float = 0.0) -> np.ndarray:
    """Baseband signal at fs_out whose SS_REF lies at f_offset. f_ref is
    the RF frequency of SS_REF: like a real gNB (38.211 5.4) every symbol
    then carries the phase -2 pi f_ref (t_start,l + N_CP,l Tc), with t
    counted from the start of the subframe."""
    rng = np.random.default_rng(cfg.seed)
    mu = {15000: 0, 30000: 1}[cfg.scs]
    n = 1024
    fs = n * cfg.scs
    cp = cp_len(n)
    extra = 16 * 2 ** mu * n // 2048
    per_half_ms = 7 * 2 ** mu                 # symbols per 0.5 ms
    sym_per_frame = 140 * 2 ** mu
    n_sym = int(np.ceil(duration * 1000 * 14 * 2 ** mu)) + 1
    k_rel = np.arange(240) - SSB_REF_K
    fill_k = np.arange(-cfg.carrier_sc // 2, cfg.carrier_sc // 2)
    fill_amp = 10 ** (cfg.fill_db / 20)
    out = []
    pos = 0                                    # samples since the subframe start
    sym_per_sf = 14 * 2 ** mu
    sib = _Sib1Tx(cfg, f_ref) if cfg.sib1 is not None else None
    for s in range(n_sym):
        frame, l = divmod(s, sym_per_frame)
        sfn = (cfg.sfn0 + frame) % 1024
        f = np.zeros(n, dtype=complex)
        if cfg.carrier_sc:
            q = rng.integers(0, 2, (2, len(fill_k)))
            f[fill_k % n] = fill_amp * ((1 - 2 * q[0]) + 1j * (1 - 2 * q[1])) / np.sqrt(2)
        if sib is not None:
            sib.apply(f, frame, sfn, l)
        if (frame * 10) % cfg.period_ms == 0:
            for i in cfg.ssb_indices:
                s0 = SSB_SYMBOLS[i]
                if s0 <= l < s0 + 4:
                    g = _ssb_cached(cfg, sfn, i)
                    f[k_rel % n] = g[l - s0] * 10 ** (cfg.beam_gain_db[i] / 20)
        t = np.fft.ifft(f) * np.sqrt(n)
        c = cp + (extra if l % per_half_ms == 0 else 0)
        if l % sym_per_sf == 0:
            pos = 0
        if f_ref:
            t = t * np.exp(-2j * np.pi * f_ref * (pos + c) / fs)
        pos += c + n
        out.append(np.concatenate([t[-c:], t]))
    x = np.concatenate(out)[:int(duration * fs)]
    if f_offset:
        x = mix(x, f_offset, fs)
    return resample(x.astype(np.complex64), fs, fs_out)


class _Sib1Tx:
    """PDCCH + PDSCH of the SIB1 broadcast, slot by slot."""

    def __init__(self, cfg: NrSynthConfig, f_ref: float):
        self.cfg = cfg
        f_ref = f_ref or 3e9
        self.c0 = pdcch.coreset0(cfg.mib(0, 0), f_ref, cfg.scs)
        if self.c0 is None:
            raise ValueError("SIB1 needs a supported CORESET#0 (k_SSB <= 23)")
        self.k_low = int(round((self.c0.f_low - f_ref) / cfg.scs))
        self.slots: dict = {}
        self.occ = {i: self.c0.occasions(i) for i in cfg.ssb_indices}
        self.count = 0
        start, length = cfg.sib1_rbs
        self.dci = pdcch.Dci10Si(self.c0.n_rb, pdcch.riv(self.c0.n_rb, start, length),
                                 cfg.sib1_tdra, cfg.sib1_interleaved, cfg.sib1_mcs, 0, 0)
        a = pdsch.allocation(self.dci, cfg.dmrs_type_a_pos)
        if a is None or a.tbs < 8 * len(cfg.sib1):
            raise ValueError(f"SIB1 of {len(cfg.sib1)} bytes does not fit "
                             f"(TBS {a.tbs if a else 0})")
        self.tb = self._tb(cfg.sib1, a.tbs)
        self.si = []                 # (dci template, TB bits, transmissions so far)
        for n, payload in enumerate(cfg.si):
            st, ln = cfg.si_rbs
            d = pdcch.Dci10Si(self.c0.n_rb, pdcch.riv(self.c0.n_rb, st, ln), cfg.si_tdra, False,
                              cfg.si_mcs, 0, 1)
            a = pdsch.allocation(d, cfg.dmrs_type_a_pos, list(cfg.si_tdra_list) or None)
            self.si.append([d, self._tb(payload, a.tbs), 0])

    @staticmethod
    def _tb(payload: bytes, tbs: int) -> np.ndarray:
        if tbs < 8 * len(payload):
            raise ValueError(f"payload of {len(payload)} bytes does not fit (TBS {tbs})")
        pad = bytes(tbs // 8 - len(payload))
        return np.unpackbits(np.frombuffer(payload + pad, dtype=np.uint8))

    def _si_message(self, sfn: int, slot: int) -> int | None:
        """Index of the SI message transmitted in this slot (38.331 5.2.2.3.2)."""
        cfg = self.cfg
        nsl = 10 * 2 ** self.c0.mu
        for n in range(len(cfg.si)):
            per = cfg.si_periods[n % len(cfg.si_periods)]
            xw = n * cfg.si_window
            for j in cfg.si_slots:
                s = xw % nsl + j
                if s % nsl == slot and (sfn - s // nsl) % per == (xw // nsl) % per:
                    return n
        return None

    def _ssb_blocked(self, slot: int) -> set:
        """(symbol, PRB) of the SSBs of this slot (first half frame)."""
        lo = -SSB_REF_K - self.k_low
        prbs = [p for p in range(self.c0.n_rb) if 12 * p < lo + 240 and 12 * (p + 1) > lo]
        out = set()
        for i in self.cfg.ssb_indices:
            s0 = SSB_SYMBOLS[i]
            if s0 // 14 == slot:
                for l in range(s0 % 14, s0 % 14 + 4):
                    out |= {(l, p) for p in prbs}
        return out

    def _build(self, frame: int, slot: int, sym0: int, si: int | None = None):
        cfg = self.cfg
        c0 = self.c0
        n_id = cfg.n_id
        g = np.zeros((14, c0.n_sc), dtype=complex)
        own = np.zeros((14, c0.n_sc), dtype=bool)
        if si is None:
            rv = cfg.sib1_rvs[self.count % len(cfg.sib1_rvs)]
            self.count += 1
            dci, tb = replace(self.dci, rv=rv), self.tb
        else:
            st = self.si[si]
            rv = cfg.sib1_rvs[st[2] % len(cfg.sib1_rvs)]
            st[2] += 1
            dci, tb = replace(st[0], rv=rv), st[1]
        # PDCCH on the first candidate of the configured aggregation level
        cce, lv = next(c for c in pdcch.candidates(c0) if c[1] == cfg.sib1_level)
        ll, kk, rbs = pdcch.candidate_res(c0, n_id, cce, lv, sym0)
        g[ll, kk] = pdcch.pdcch_symbols(dci, lv, n_id)
        for sidx in range(c0.n_symb):
            l = sym0 + sidx
            ref = pdcch.dmrs(c0.n_rb, slot, l, n_id)
            for rb in rbs:
                g[l, 12 * rb + pdcch.DMRS_SC] = ref[rb]
                own[l, 12 * rb:12 * rb + 12] = True
        # PDSCH (K0 = 0)
        a = pdsch.allocation(dci, cfg.dmrs_type_a_pos,
                             list(cfg.si_tdra_list) or None if si is not None else None)
        blocked = self._ssb_blocked(slot) if (frame * 10) % cfg.period_ms == 0 else set()
        l_re, k_re = pdsch.data_res(a, blocked)
        gbits = len(l_re) * a.qm
        bits = ldpc.dlsch_encode(tb, a.rate, a.qm, gbits, rv) ^ pdsch.scrambling(n_id, gbits)
        g[l_re, k_re] = pdsch.modulate(bits, a.qm)
        beta = np.sqrt(2) if pdsch.cdm_groups_without_data(a.length) == 2 else 1.0
        crb0 = 0 if si is None else cfg.si_coreset0_crb
        for l in a.dmrs:
            ref = pdsch.dmrs_seq(slot, l, n_id, c0.n_sc + 12 * crb0)
            for p in a.prbs:
                if (l, p) not in blocked:
                    k = 12 * p + np.arange(0, 12, 2)
                    g[l, k] = beta * ref[k // 2 + 6 * crb0]
        for l in range(a.s, a.s + a.length):
            for p in a.prbs:
                if (l, p) not in blocked:
                    own[l, 12 * p:12 * p + 12] = True
        return g, own

    def apply(self, f: np.ndarray, frame: int, sfn: int, l_frame: int):
        """Write the SIB1 REs of symbol l_frame of this frame into f."""
        slot, l = divmod(l_frame, 14)
        key = (frame, slot)
        if key not in self.slots:
            self.slots[key] = None
            for n0, parity, sym0 in self.occ.values():
                if n0 == slot and sfn % 2 == parity:
                    self.slots[key] = self._build(frame, slot, sym0)
                    break
            if self.slots[key] is None and self.si:
                n = self._si_message(sfn, slot)
                if n is not None:
                    self.slots[key] = self._build(frame, slot, 0, n)
        v = self.slots[key]
        if v is None:
            return
        g, own = v
        idx = (self.k_low + np.arange(self.c0.n_sc)) % len(f)
        f[idx[own[l]]] = g[l][own[l]]


_CACHE: dict = {}


def _ssb_cached(cfg: NrSynthConfig, sfn: int, i: int) -> np.ndarray:
    key = (cfg.n_id, cfg.l_max, cfg.k_ssb, cfg.coreset0, cfg.search_space0, cfg.cell_barred,
           cfg.scs, sfn, i)
    g = _CACHE.get(key)
    if g is None:
        g = ssb_grid(cfg.n_id, cfg.mib(sfn, i))
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[key] = g
    return g


__all__ = ["NrSynthConfig", "generate", "replace"]
