"""Transmit-diversity (SFBC) precoding/decoding, TS 36.211 6.3.3.3/6.3.4.3."""
from __future__ import annotations

import numpy as np

_S2 = 1 / np.sqrt(2)


def td_encode(d: np.ndarray, n_ports: int) -> np.ndarray:
    """Symbols d -> per-port RE values, shape (n_ports, len(d))."""
    n = len(d)
    if n_ports == 1:
        return d[None, :].copy()
    y = np.zeros((n_ports, n), dtype=complex)
    if n_ports == 2:
        x0, x1 = d[0::2], d[1::2]
        y[0, 0::2], y[1, 0::2] = x0 * _S2, -np.conj(x1) * _S2
        y[0, 1::2], y[1, 1::2] = x1 * _S2, np.conj(x0) * _S2
        return y
    x0, x1, x2, x3 = d[0::4], d[1::4], d[2::4], d[3::4]
    y[0, 0::4], y[2, 0::4] = x0 * _S2, -np.conj(x1) * _S2
    y[0, 1::4], y[2, 1::4] = x1 * _S2, np.conj(x0) * _S2
    y[1, 2::4], y[3, 2::4] = x2 * _S2, -np.conj(x3) * _S2
    y[1, 3::4], y[3, 3::4] = x3 * _S2, np.conj(x2) * _S2
    return y


def _alamouti(r0, r1, ha, hb):
    """Pair decoder. r0,r1 received on the two REs; ha/hb channels of the
    two ports (averaged over the pair)."""
    g = np.abs(ha) ** 2 + np.abs(hb) ** 2 + 1e-12
    x0 = (np.conj(ha) * r0 + hb * np.conj(r1)) / g / _S2
    x1 = (np.conj(ha) * r1 - hb * np.conj(r0)) / g / _S2
    return x0, x1, g * 0.5


def td_decode(r: np.ndarray, h: np.ndarray, n_ports: int):
    """r: received REs (N,), h: channel per port (n_ports, N).
    Returns (equalised symbols, per-symbol gain |h|^2 for LLR weighting)."""
    if n_ports == 1:
        g = np.abs(h[0]) ** 2 + 1e-12
        return r * np.conj(h[0]) / g, g
    n = len(r)
    x = np.zeros(n, dtype=complex)
    gain = np.zeros(n)
    if n_ports == 2:
        ha = 0.5 * (h[0, 0::2] + h[0, 1::2])
        hb = 0.5 * (h[1, 0::2] + h[1, 1::2])
        x[0::2], x[1::2], g = _alamouti(r[0::2], r[1::2], ha, hb)
        gain[0::2] = gain[1::2] = g
        return x, gain
    # 4 ports: REs 4i,4i+1 on ports (0,2); 4i+2,4i+3 on ports (1,3)
    ha = 0.5 * (h[0, 0::4] + h[0, 1::4])
    hb = 0.5 * (h[2, 0::4] + h[2, 1::4])
    x[0::4], x[1::4], g = _alamouti(r[0::4], r[1::4], ha, hb)
    gain[0::4] = gain[1::4] = g
    ha = 0.5 * (h[1, 2::4] + h[1, 3::4])
    hb = 0.5 * (h[3, 2::4] + h[3, 3::4])
    x[2::4], x[3::4], g = _alamouti(r[2::4], r[3::4], ha, hb)
    gain[2::4] = gain[3::4] = g
    return x, gain
