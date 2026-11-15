"""NR downlink occupancy from energy: TDD pattern, noise floor from the
uplink symbols, envelope alignment of a band without SSB, CLI."""
import json

import numpy as np
import pytest

from ltescan.cli import main
from ltescan.dsp import mix
from ltescan.nr import load as L
from ltescan.nr import sib1 as S1
from ltescan.nr.params import gscn_to_freq
from ltescan.nr.search import ssb_search
from ltescan.nr.synth import NrSynthConfig, generate
from ltescan.sdr import write_iq
from ltescan.synth import channel

F_REF = gscn_to_freq(7906)
FC = F_REF + 1e6


def _capture(load, snr=8, taps=2, tdd="DDDSUUDDDD"):
    cfg = NrSynthConfig(n_id=494, ssb_indices=(0, 1, 2, 3), k_ssb=0, carrier_sc=600, tdd=tdd,
                        load=load, sfn0=300)
    y = generate(cfg, 0.15, 20e6, 0.0, F_REF)
    x = channel(mix(y, F_REF - FC, 20e6)[None, :], 20e6, snr_db=snr, cfo=1e3, delay=500,
                taps=taps, seed=1)
    return x[0] if x.ndim > 1 else x


def _timing(x):
    cell = [c for c in ssb_search(x, 20e6, FC, gscns=[7906], cfo_max=5e3) if c.confirmed][0]
    frames = S1.frames_in(S1.frame_refs(cell, 30000), len(x) / 20e6)
    return cell, next(f.t for f in frames if f.t >= 0)


BAND = L.Band(F_REF - 300 * 30e3, 50, 30000)      # the generator's filled PRBs


@pytest.mark.parametrize("load", [0.1, 0.4, 0.9])
def test_occupancy_with_ssb_timing_and_envelope(load):
    x = _capture(load)
    cell, t0 = _timing(x)
    r = L.measure(x, 20e6, FC, BAND, t0, cfo=cell.cfo)
    assert r.pattern.slots(30000) == "DDDSUUDDDD"
    assert abs(r.mean_use - load) < 0.06          # + the SSB's own PRBs
    # the same band without SSB timing: aligned by its DL/UL envelope
    r2 = L.measure(x, 20e6, FC, BAND, None, pattern=r.pattern, cfo=cell.cfo)
    assert r2.aligned and abs(r2.mean_use - r.mean_use) < 0.03


def test_idle_cell_and_fdd():
    x = _capture(0.0, tdd="")
    cell, t0 = _timing(x)
    r = L.measure(x, 20e6, FC, BAND, t0, cfo=cell.cfo)
    assert r.pattern.dl.all() and r.mean_use < 0.08     # only the SSB
    x = _capture(0.5, tdd="", snr=20, taps=1)
    cell, t0 = _timing(x)
    r = L.measure(x, 20e6, FC, BAND, t0, cfo=cell.cfo)
    assert abs(r.mean_use - 0.5) < 0.08


def test_cli_nrload(tmp_path, capsys):
    f = tmp_path / "n78.cs8"
    write_iq(f, _capture(0.3), 20e6, FC)
    assert main(["nrload", "--file", str(f), "--gscn", "7906", "--cfo-max", "5000",
                 "--csv", str(tmp_path / "l.csv"), "--json", str(tmp_path / "l.json")]) == 0
    out = capsys.readouterr().out
    assert "TDD DDDSUUDDDD" in out and "downlink PRB use" in out
    use = [u for u in json.loads((tmp_path / "l.json").read_text())["prb_use"] if u is not None]
    assert len(use) > 30 and 0.2 < np.mean(use) < 0.45
    assert (tmp_path / "l.csv").read_text().count("\n") == 2
