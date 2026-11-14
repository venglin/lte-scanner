"""CRS based channel estimation and RSRP/RSRQ/SINR measurements."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import CellConfig
from .sequences import crs, crs_symbols


@dataclass
class ChannelEstimate:
    h: np.ndarray            # (n_ports, n_sym, n_sc)
    noise: float             # noise variance per RE
    rsrp: float              # linear, per RE, port 0
    rssi_re: float           # mean received power per RE in CRS symbols
    port_power: np.ndarray   # mean |h|^2 per port (to guess #ports)

    @property
    def snr_db(self) -> float:
        return 10 * np.log10(max(self.rsrp, 1e-20) / max(self.noise, 1e-20))

    @property
    def rsrq_db(self) -> float:
        return 10 * np.log10(max(self.rsrp, 1e-20) / max(12 * self.rssi_re, 1e-20))


def crs_positions(cfg: CellConfig, sf: int, port: int):
    """Yield (symbol index in subframe, k array, reference values)."""
    for slot in range(2):
        ns = 2 * sf + slot
        for l, ports in crs_symbols(cfg.cp_ext):
            if port in ports:
                k, r = crs(ns, l, port, cfg.n_id, cfg.n_rb, cfg.cp_ext)
                yield slot * cfg.n_symb + l, k, r


def _smooth(v: np.ndarray, taps: int) -> np.ndarray:
    if taps <= 1 or len(v) < taps:
        return v
    ker = np.ones(taps) / taps
    pad = taps // 2
    vp = np.concatenate([v[pad:0:-1], v, v[-2:-pad - 2:-1]])
    return np.convolve(vp, ker, mode="valid")


def estimate(grid: np.ndarray, cfg: CellConfig, sf: int, n_ports: int | None = None,
             smooth: int = 3) -> ChannelEstimate:
    n_ports = n_ports or cfg.n_ports
    n_sym, n_sc = grid.shape
    h = np.zeros((n_ports, n_sym, n_sc), dtype=complex)
    kk = np.arange(n_sc)
    noise_acc, noise_n = 0.0, 0
    rsrp_acc, rsrp_n = 0.0, 0
    port_power = np.zeros(n_ports)
    for p in range(n_ports):
        syms, est = [], []
        for l, k, r in crs_positions(cfg, sf, p):
            ls = grid[l, k] * np.conj(r)
            sm = _smooth(ls, smooth)
            if p == 0:
                d = ls - sm
                noise_acc += np.sum(np.abs(d) ** 2) / (1 - 1 / smooth if smooth > 1 else 1)
                noise_n += len(d)
                rsrp_acc += np.sum(np.abs(ls) ** 2)
                rsrp_n += len(ls)
            port_power[p] += np.mean(np.abs(sm) ** 2)
            hf = np.interp(kk, k, sm.real) + 1j * np.interp(kk, k, sm.imag)
            syms.append(l)
            est.append(hf)
        syms = np.array(syms)
        est = np.array(est)
        order = np.argsort(syms)
        syms, est = syms[order], est[order]
        port_power[p] /= len(syms)
        for l in range(n_sym):
            if l <= syms[0]:
                h[p, l] = est[0]
            elif l >= syms[-1]:
                h[p, l] = est[-1]
            else:
                j = np.searchsorted(syms, l)
                a = (l - syms[j - 1]) / (syms[j] - syms[j - 1])
                h[p, l] = (1 - a) * est[j - 1] + a * est[j]
    noise = noise_acc / max(noise_n, 1)
    rsrp = max(rsrp_acc / max(rsrp_n, 1) - noise, 1e-20)
    crs_l = [l for l, _, _ in crs_positions(cfg, sf, 0)]
    rssi = float(np.mean(np.abs(grid[crs_l]) ** 2))
    return ChannelEstimate(h, noise, rsrp, rssi, port_power)
