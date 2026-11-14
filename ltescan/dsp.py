"""Generic DSP helpers: resampling, mixing, spectra."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational resampling fs_in -> fs_out (polyphase FIR)."""
    if abs(fs_in - fs_out) < 1e-6 * fs_out:
        return x
    fr = Fraction(fs_out / fs_in).limit_denominator(2000)
    up, down = fr.numerator, fr.denominator
    return signal.resample_poly(x, up, down).astype(np.complex64)


def mix(x: np.ndarray, f_shift: float, fs: float, t0: int = 0) -> np.ndarray:
    """Multiply by exp(j*2*pi*f_shift*t). Positive f_shift moves spectrum up."""
    n = np.arange(t0, t0 + len(x))
    return (x * np.exp(2j * np.pi * f_shift / fs * n)).astype(np.complex64)


def lowpass_decimate(x: np.ndarray, fs_in: float, fs_out: float,
                     f_shift: float = 0.0) -> np.ndarray:
    """Shift by f_shift then resample to fs_out (anti-alias included)."""
    if f_shift:
        x = mix(x, f_shift, fs_in)
    return resample(x, fs_in, fs_out)


def stretch(x: np.ndarray, ratio: float) -> np.ndarray:
    """Fractional resampling by a ratio very close to 1 (sample clock ppm
    correction). Output sample n is input at n*ratio (cubic interpolation)."""
    if abs(ratio - 1) < 1e-9:
        return x
    n_out = int((len(x) - 4) / ratio)
    t = np.arange(n_out) * ratio
    i = np.clip(np.floor(t).astype(np.int64), 1, len(x) - 3)
    mu = t - i
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    # Catmull-Rom cubic
    a = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
    b = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
    c = -0.5 * xm1 + 0.5 * x1
    return (((a * mu + b) * mu + c) * mu + x0).astype(np.complex64)


def psd(x: np.ndarray, fs: float, nfft: int = 4096):
    """Welch PSD, returns (freqs centred on 0, power dB)."""
    f, p = signal.welch(x, fs=fs, nperseg=min(nfft, len(x)),
                        return_onesided=False, scaling="density")
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-20)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))
