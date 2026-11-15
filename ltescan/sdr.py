"""IQ sources: HackRF (via hackrf_transfer or SoapySDR) and IQ files."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# HackRF MAX2837 baseband filter bandwidths (Hz)
HACKRF_BB_FILTERS = [1.75e6, 2.5e6, 3.5e6, 5e6, 5.5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 12e6,
                     14e6, 15e6, 20e6, 24e6, 28e6]


def pick_bb_filter(bw: float) -> float:
    for f in HACKRF_BB_FILTERS:
        if f >= bw:
            return f
    return HACKRF_BB_FILTERS[-1]


def _hackrf_candidates() -> list[Path]:
    """Usual install locations of hackrf_transfer that are often not on PATH."""
    env = os.environ
    dirs = []
    if os.name == "nt":
        for pf in {env.get("ProgramFiles"), env.get("ProgramFiles(x86)"),
                   env.get("ProgramW6432"), r"C:\Program Files"}:
            if pf:
                dirs += [Path(pf) / "PothosSDR" / "bin", Path(pf) / "HackRF" / "bin",
                         Path(pf) / "hackrf" / "bin"]
        home = Path.home()
        for base in ("radioconda", "miniforge3", "mambaforge", "miniconda3", "anaconda3"):
            dirs.append(home / base / "Library" / "bin")
        if env.get("CONDA_PREFIX"):
            dirs.append(Path(env["CONDA_PREFIX"]) / "Library" / "bin")
        names = ["hackrf_transfer.exe"]
    else:
        dirs += [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/bin"),
                 Path("/opt/local/bin")]
        if env.get("CONDA_PREFIX"):
            dirs.append(Path(env["CONDA_PREFIX"]) / "bin")
        names = ["hackrf_transfer"]
    return [d / n for d in dirs for n in names]


def find_hackrf_transfer() -> str | None:
    """$HACKRF_TRANSFER, then PATH, then the usual install locations."""
    p = os.environ.get("HACKRF_TRANSFER")
    if p:
        if Path(p).is_file():
            return p
        raise RuntimeError(f"HACKRF_TRANSFER={p} does not exist")
    p = shutil.which("hackrf_transfer")
    if p:
        return p
    return next((str(c) for c in _hackrf_candidates() if c.is_file()), None)


HACKRF_NOT_FOUND = (
    "'hackrf_transfer' not found (looked in PATH and the usual PothosSDR/radioconda/"
    "Homebrew locations). Install the HackRF host tools, add their directory to PATH, "
    "or point to the program with --hackrf-transfer PATH or the HACKRF_TRANSFER "
    "environment variable.")


@dataclass
class Capture:
    x: np.ndarray       # complex64 samples
    fs: float           # nominal sample rate
    fc: float           # nominal centre frequency


class IqSource:
    """Base class. `ppm` is the known clock error of the receiver: the
    source tunes so that the *real* frequency/rate equal the requested."""
    ppm: float = 0.0

    def capture(self, fc: float, fs: float, n: int, bb_filter: float | None = None) -> Capture:
        raise NotImplementedError

    def _corr(self, v: float) -> float:
        return v / (1 + self.ppm * 1e-6)


class HackRfTransfer(IqSource):
    """Uses the `hackrf_transfer` command line tool (part of hackrf host
    tools; on Windows e.g. from PothosSDR or the official release zip)."""

    def __init__(self, lna: int = 32, vga: int = 30, amp: bool = False, serial: str | None = None,
                 ppm: float = 0.0, exe: str | list | None = None, settle_ms: float = 20.0):
        if isinstance(exe, str) and not (Path(exe).is_file() or shutil.which(exe)):
            raise RuntimeError(f"hackrf_transfer program {exe} does not exist")
        exe = exe or find_hackrf_transfer()
        # exe may also be a command prefix list (e.g. [python, script])
        self.exe = list(exe) if isinstance(exe, (list, tuple)) else ([exe] if exe else None)
        if not self.exe:
            raise RuntimeError(HACKRF_NOT_FOUND)
        self.lna = int(lna) // 8 * 8
        self.vga = int(vga) // 2 * 2
        self.amp = amp
        self.serial = serial
        self.ppm = ppm
        self.settle_ms = settle_ms

    def capture(self, fc, fs, n, bb_filter=None) -> Capture:
        skip = int(self.settle_ms * 1e-3 * fs)
        total = n + skip
        fd, path = tempfile.mkstemp(suffix=".cs8")
        os.close(fd)
        try:
            cmd = [*self.exe, "-r", path, "-f", str(int(round(self._corr(fc)))),
                   "-s", str(int(round(self._corr(fs)))), "-n", str(total),
                   "-l", str(self.lna), "-g", str(self.vga), "-a", "1" if self.amp else "0"]
            if bb_filter:
                cmd += ["-b", str(int(pick_bb_filter(bb_filter)))]
            if self.serial:
                cmd += ["-d", self.serial]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30 + n / fs * 4)
            raw = np.fromfile(path, dtype=np.int8)
            if raw.size < 2 * total:
                raise RuntimeError(f"hackrf_transfer returned {raw.size // 2} of {total} samples:\n"
                                   + (res.stderr or res.stdout)[-800:])
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        x = cs8_to_complex(raw[2 * skip:2 * total])
        return Capture(x, fs, fc)


class SoapySource(IqSource):
    """HackRF via SoapySDR Python bindings (SoapySDR + SoapyHackRF)."""

    def __init__(self, lna: int = 32, vga: int = 30, amp: bool = False, serial: str | None = None,
                 ppm: float = 0.0, args: str = "driver=hackrf"):
        import SoapySDR  # noqa: F401  (optional dependency)
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX
        self.SoapySDR = SoapySDR
        self.RX, self.CF32 = SOAPY_SDR_RX, SOAPY_SDR_CF32
        if serial:
            args += f",serial={serial}"
        self.dev = SoapySDR.Device(args)
        self.dev.setGain(self.RX, 0, "LNA", int(lna) // 8 * 8)
        self.dev.setGain(self.RX, 0, "VGA", int(vga) // 2 * 2)
        self.dev.setGain(self.RX, 0, "AMP", 14 if amp else 0)
        self.ppm = ppm

    def capture(self, fc, fs, n, bb_filter=None) -> Capture:
        d = self.dev
        d.setSampleRate(self.RX, 0, self._corr(fs))
        d.setFrequency(self.RX, 0, self._corr(fc))
        if bb_filter:
            d.setBandwidth(self.RX, 0, pick_bb_filter(bb_filter))
        st = d.setupStream(self.RX, self.CF32)
        d.activateStream(st)
        skip = int(0.02 * fs)
        buf = np.empty(n + skip, dtype=np.complex64)
        got = 0
        chunk = np.empty(65536, dtype=np.complex64)
        try:
            while got < n + skip:
                r = d.readStream(st, [chunk], len(chunk), timeoutUs=1_000_000)
                if r.ret <= 0:
                    raise RuntimeError(f"SoapySDR readStream error {r.ret}")
                m = min(r.ret, n + skip - got)
                buf[got:got + m] = chunk[:m]
                got += m
        finally:
            d.deactivateStream(st)
            d.closeStream(st)
        return Capture(buf[skip:], fs, fc)


def open_sdr(driver: str = "auto", exe: str | None = None, **kw) -> IqSource:
    if driver in ("auto", "hackrf_transfer"):
        try:
            return HackRfTransfer(exe=exe, **kw)
        except RuntimeError as e:
            if driver != "auto" or exe:
                raise
            why = str(e)
    else:
        why = ""
    try:
        return SoapySource(**kw)
    except ImportError as e:
        raise RuntimeError((why + "\n" if why else "") + "SoapySDR Python bindings are not "
                           "available either (alternative to hackrf_transfer: SoapySDR with "
                           "the SoapyHackRF module).") from e


# ------------------------------------------------------------------ files
def cs8_to_complex(raw: np.ndarray) -> np.ndarray:
    f = raw.astype(np.float32) / 128.0
    return (f[0::2] + 1j * f[1::2]).astype(np.complex64)


FORMATS = {".cs8": "cs8", ".iq": "cs8", ".bin": "cs8", ".cs16": "cs16", ".sc16": "cs16",
           ".cf32": "cf32", ".fc32": "cf32", ".cfile": "cf32", ".raw": "cf32", ".npy": "npy"}


def meta_path(path: str | Path) -> Path:
    return Path(str(path) + ".json")


def read_iq(path: str | Path, fmt: str | None = None, max_samples: int | None = None,
            offset: int = 0):
    """Returns (samples, metadata dict). Metadata comes from a sidecar
    '<file>.json' (written by `record` / `synth`) if present."""
    path = Path(path)
    fmt = fmt or FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unknown file format {path.suffix}; give --format")
    meta = {}
    mp = meta_path(path)
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    count = -1 if max_samples is None else max_samples
    if fmt == "npy":
        x = np.load(path).astype(np.complex64)[offset:]
        if max_samples:
            x = x[:max_samples]
    elif fmt == "cs8":
        raw = np.fromfile(path, dtype=np.int8, count=2 * count if count > 0 else -1,
                          offset=2 * offset)
        x = cs8_to_complex(raw)
    elif fmt == "cs16":
        raw = np.fromfile(path, dtype=np.int16, count=2 * count if count > 0 else -1,
                          offset=4 * offset).astype(np.float32) / 32768.0
        x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    else:
        x = np.fromfile(path, dtype=np.complex64, count=count, offset=8 * offset)
    return x, meta


def write_iq(path: str | Path, x: np.ndarray, fs: float, fc: float, fmt: str | None = None,
             extra: dict | None = None):
    path = Path(path)
    fmt = fmt or FORMATS.get(path.suffix.lower(), "cs8")
    if fmt == "cs8":
        peak = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)), 1e-12)
        scale = 127.0 / peak if peak > 1.0 else 127.0
        iq = np.empty(2 * len(x), dtype=np.int8)
        iq[0::2] = np.clip(np.round(x.real * scale), -127, 127)
        iq[1::2] = np.clip(np.round(x.imag * scale), -127, 127)
        iq.tofile(path)
    elif fmt == "cs16":
        iq = np.empty(2 * len(x), dtype=np.int16)
        iq[0::2] = np.clip(np.round(x.real * 32767), -32767, 32767)
        iq[1::2] = np.clip(np.round(x.imag * 32767), -32767, 32767)
        iq.tofile(path)
    elif fmt == "npy":
        np.save(path, x.astype(np.complex64))
    else:
        x.astype(np.complex64).tofile(path)
    meta = {"sample_rate": fs, "center_freq": fc, "format": fmt}
    meta.update(extra or {})
    meta_path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
