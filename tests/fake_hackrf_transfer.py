"""Stand-in for `hackrf_transfer -r FILE -f FREQ -s RATE -n N ...` used by
the tests: writes int8 IQ of a synthetic 10 MHz LTE cell (PCI 99) on
806 MHz (band 20, EARFCN 6300), as seen by a receiver with a +3 ppm clock."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltescan.dsp import mix, resample  # noqa: E402
from ltescan.params import CellConfig  # noqa: E402
from ltescan.synth import EnbSimulator, SynthConfig, channel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-r")
ap.add_argument("-f", type=float)
ap.add_argument("-s", type=float)
ap.add_argument("-n", type=int)
ap.add_argument("-l")
ap.add_argument("-g")
ap.add_argument("-a")
ap.add_argument("-b")
ap.add_argument("-d")
a = ap.parse_args()

F_CELL, PPM = 806e6, 3.0
cell = CellConfig(n_id=99, n_rb=50, n_ports=2)
n_frames = int(np.ceil(a.n / a.s / 0.01)) + 1
tx = EnbSimulator(SynthConfig(cell, sfn0=500)).generate(n_frames)
y = channel(tx, cell.fs, snr_db=60, delay=777, seed=1)
y = resample(y, cell.fs, a.s)
lo = a.f * (1 + PPM * 1e-6)                 # receiver LO is off by +3 ppm
y = mix(y, F_CELL - lo, a.s)
if abs(F_CELL - lo) > 0.45 * a.s + 5e6:
    y = 0 * y                                 # cell outside this capture
rng = np.random.default_rng(int(a.f) % 1000)
y = 0.05 * y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-30) * (np.abs(F_CELL - lo) < 0.45 * a.s + 5e6)
y = y + 0.003 * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
y = y[:a.n]
iq = np.empty(2 * len(y), dtype=np.int8)
iq[0::2] = np.clip(np.round(y.real * 127), -127, 127)
iq[1::2] = np.clip(np.round(y.imag * 127), -127, 127)
iq.tofile(a.r)
