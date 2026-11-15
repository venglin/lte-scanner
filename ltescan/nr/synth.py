"""Synthetic NR downlink with SS/PBCH block bursts (for tests and demos).

SSBs are placed at the case A/C positions {2, 8} + 14 n of the first half
frame, every `period_ms`, each SSB index with its own (beam) gain. The rest
of the carrier is filled with random QPSK like a loaded PDSCH."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..dsp import mix, resample
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

    def mib(self, sfn: int, ssb_index: int) -> Mib:
        return Mib(sfn=sfn, half_frame=0, scs_common=self.scs, k_ssb=self.k_ssb,
                   dmrs_type_a_pos=2, coreset0=self.coreset0, search_space0=self.search_space0,
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
    for s in range(n_sym):
        frame, l = divmod(s, sym_per_frame)
        sfn = (cfg.sfn0 + frame) % 1024
        f = np.zeros(n, dtype=complex)
        if cfg.carrier_sc:
            q = rng.integers(0, 2, (2, len(fill_k)))
            f[fill_k % n] = fill_amp * ((1 - 2 * q[0]) + 1j * (1 - 2 * q[1])) / np.sqrt(2)
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
