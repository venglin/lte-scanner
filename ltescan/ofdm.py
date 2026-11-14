"""OFDM modulation / demodulation of LTE downlink subframes."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def cp_lengths(fft: int, cp_ext: bool) -> tuple[int, ...]:
    """CP length of each of the symbols in one subframe."""
    if cp_ext:
        return tuple([512 * fft // 2048] * 12)
    first, other = 160 * fft // 2048, 144 * fft // 2048
    return tuple(([first] + [other] * 6) * 2)


@lru_cache(maxsize=64)
def symbol_offsets(fft: int, cp_ext: bool) -> tuple[int, ...]:
    """Offset (in samples, from subframe start) of every FFT window."""
    out, pos = [], 0
    for cp in cp_lengths(fft, cp_ext):
        out.append(pos + cp)
        pos += cp + fft
    return tuple(out)


def subframe_len(fft: int) -> int:
    return 15 * fft


@lru_cache(maxsize=64)
def bin_index(fft: int, n_sc: int) -> np.ndarray:
    """FFT bin of each resource-grid subcarrier k=0..n_sc-1 (DC skipped)."""
    k = np.arange(n_sc)
    idx = np.where(k < n_sc // 2, fft - n_sc // 2 + k, k - n_sc // 2 + 1)
    idx.setflags(write=False)
    return idx


def demod_symbol(x: np.ndarray, start: int, fft: int, n_sc: int,
                 backoff: int = 0) -> np.ndarray:
    """FFT of one symbol whose (CP-free) window begins at `start`.
    `backoff` samples are taken from the CP to stay clear of ISI; the
    resulting linear phase is removed."""
    seg = x[start - backoff:start - backoff + fft]
    if len(seg) < fft:
        raise IndexError("symbol outside buffer")
    f = np.fft.fft(seg) / np.sqrt(fft)
    idx = bin_index(fft, n_sc)
    y = f[idx]
    if backoff:
        freq = np.where(idx < fft // 2, idx, idx - fft)
        y = y * np.exp(2j * np.pi * freq * backoff / fft)
    return y


def demod_subframe(x: np.ndarray, start: int, fft: int, n_sc: int,
                   cp_ext: bool, backoff: int | None = None) -> np.ndarray:
    """Resource grid (n_symbols, n_sc) of the subframe beginning at `start`."""
    offs = symbol_offsets(fft, cp_ext)
    if backoff is None:
        backoff = max(1, cp_lengths(fft, cp_ext)[1] // 4)
    if start - backoff < 0 or start + subframe_len(fft) > len(x):
        raise IndexError("subframe outside buffer")
    idx = bin_index(fft, n_sc)
    starts = np.array(offs) + start - backoff
    segs = x[starts[:, None] + np.arange(fft)[None, :]]
    f = np.fft.fft(segs, axis=1) / np.sqrt(fft)
    y = f[:, idx]
    freq = np.where(idx < fft // 2, idx, idx - fft)
    return y * np.exp(2j * np.pi * freq * backoff / fft)[None, :]


def mod_subframe(grid: np.ndarray, fft: int, cp_ext: bool) -> np.ndarray:
    """Time-domain samples of one subframe from its resource grid."""
    n_sym, n_sc = grid.shape
    idx = bin_index(fft, n_sc)
    out = []
    for l, cp in enumerate(cp_lengths(fft, cp_ext)):
        f = np.zeros(fft, dtype=complex)
        f[idx] = grid[l]
        t = np.fft.ifft(f) * np.sqrt(fft)
        out.append(t[-cp:])
        out.append(t)
    return np.concatenate(out)
