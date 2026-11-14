"""End-to-end tests: synthetic eNB signal -> channel -> receiver."""
import numpy as np
import pytest

from ltescan.cli import main
from ltescan.dsp import mix, resample
from ltescan.params import CellConfig
from ltescan.receiver import LteReceiver
from ltescan.scanner import Scanner
from ltescan.sdr import read_iq, write_iq
from ltescan.sync import cell_search
from ltescan.synth import EnbSimulator, SynthConfig, channel


def run(cell, frames=5, snr=15, cfo=12e3, taps=1, sro=0.0, fs=None, rnti=True, **sk):
    sim = EnbSimulator(SynthConfig(cell, sfn0=101, **sk))
    y = channel(sim.generate(frames), cell.fs, snr_db=snr, cfo=cfo, delay=4321, taps=taps,
                sro_ppm=sro, seed=3)
    rate = cell.fs
    if fs:
        y, rate = resample(y, cell.fs, fs), fs
    rx = LteReceiver(y, rate, fc=1.8e9)
    cells = rx.search()
    assert cells and cells[0].n_id == cell.n_id
    assert cells[0].tdd == cell.tdd and cells[0].cp_ext == cell.cp_ext
    rep = rx.analyze(cells[0], rnti_discovery=rnti)
    assert rep.mib is not None
    assert (rep.mib.n_rb, rep.mib.n_ports, rep.mib.sfn) == (cell.n_rb, cell.n_ports, 101)
    assert rep.sib1 is not None, rep.notes
    assert str(rep.sib1.plmns[0]) == "260-03" and rep.sib1.tac == 0x1A2B
    return rep


CASES = [
    dict(n_id=0, n_rb=6, n_ports=1),
    dict(n_id=301, n_rb=25, n_ports=2),
    dict(n_id=17, n_rb=15, n_ports=4),
    dict(n_id=503, n_rb=50, n_ports=2, phich_ng=3),
    dict(n_id=88, n_rb=25, n_ports=2, cp_ext=True),
    dict(n_id=222, n_rb=25, n_ports=2, tdd=True),
    dict(n_id=260, n_rb=75, n_ports=2, phich_ng=3, phich_ext=True),
]


@pytest.mark.parametrize("kw", CASES, ids=lambda k: "-".join(f"{a}{b}" for a, b in k.items()))
def test_configurations(kw):
    cell = CellConfig(**kw)
    rep = run(cell, cfi=3 if cell.phich_ext else 2)
    assert rep.cfi_hist.most_common(1)[0][0] == (3 if cell.phich_ext else 2)
    if cell.n_rb >= 25:
        assert set(rep.confirmed_rntis()) == {0x1234, 0x4321}


def test_dci_1c_and_distributed():
    run(CellConfig(n_id=11, n_rb=15, n_ports=2), use_1c=True, sib1_rb=(2, 4))
    run(CellConfig(n_id=12, n_rb=50, n_ports=2), distributed=True, sib1_rb=(3, 8))


def test_impairments():
    cell = CellConfig(n_id=42, n_rb=50, n_ports=2)
    run(cell, snr=8, taps=4)                          # frequency selective channel
    run(cell, frames=10, cfo=-36e3, sro=20.0)         # 20 ppm clock: CFO + sample drift
    run(cell, snr=-4, frames=10, rnti=False)          # low SNR (SIB1 soft combining)


def test_non_native_sample_rates():
    run(CellConfig(n_id=13, n_rb=50, n_ports=2), fs=12.5e6)
    run(CellConfig(n_id=401, n_rb=100, n_ports=2), fs=20e6)     # HackRF max rate


def test_two_cells_and_no_phantoms():
    c1 = CellConfig(n_id=100, n_rb=6, n_ports=2)
    c2 = CellConfig(n_id=101, n_rb=6, n_ports=1)
    t1 = EnbSimulator(SynthConfig(c1)).generate(6)
    t2 = EnbSimulator(SynthConfig(c2, sfn0=7), seed=3).generate(6)
    y1 = channel(t1, c1.fs, snr_db=60, cfo=0, delay=5000, seed=1)
    y2 = channel(t2, c1.fs, snr_db=60, cfo=0, delay=7333, seed=2)[:len(y1)]
    y = channel((y1 + 0.1 * y2)[None, :], c1.fs, snr_db=25, cfo=9e3, seed=5)
    assert [c.n_id for c in cell_search(y)] == [100, 101]


def test_noise_gives_no_cells():
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = (rng.normal(size=19200 * 4) + 1j * rng.normal(size=19200 * 4)).astype(np.complex64)
        assert cell_search(x) == []


def test_wideband_scan():
    fs, fc = 20e6, 1840e6

    def carrier(n_id, n_rb, f, seed):
        cell = CellConfig(n_id=n_id, n_rb=n_rb, n_ports=2)
        tx = EnbSimulator(SynthConfig(cell, sfn0=seed * 10), seed=seed).generate(5)
        y = channel(tx, cell.fs, snr_db=60, delay=1234 * seed, seed=seed)
        return mix(resample(y, cell.fs, fs), f - fc, fs)

    a, b = carrier(301, 50, 1842.5e6, 1), carrier(77, 25, 1832.6e6, 2)
    n = min(len(a), len(b))
    y = channel((a[:n] + 0.3 * b[:n])[None, :], fs, snr_db=20, cfo=-14e3, seed=9)
    res = Scanner(log=lambda *a: None).scan_buffer(y, fs, fc, band=3)
    found = sorted((round(r.fc / 1e5) / 10, r.sync.n_id, r.mib.n_rb) for r in res.reports)
    assert found == [(1832.6, 77, 25), (1842.5, 301, 50)]


def test_cli_synth_and_cell(tmp_path, capsys):
    f = tmp_path / "x.cs8"
    assert main(["synth", "--out", str(f), "--nrb", "25", "--pci", "77", "--duration", "0.08",
                 "--rate", "10e6"]) == 0
    x, meta = read_iq(f)
    assert meta["sample_rate"] == 10e6 and len(x) > 0
    assert main(["cell", "--file", str(f), "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "PCI 77" in out and "260-03" in out
    write_iq(tmp_path / "y.cf32", x, 10e6, 1842.5e6)
    assert read_iq(tmp_path / "y.cf32")[0].shape == x.shape


def test_hackrf_transfer_path_and_band_scan():
    """Drive HackRfTransfer + Scanner.scan_band with a fake hackrf_transfer."""
    import sys
    from pathlib import Path
    from ltescan.sdr import HackRfTransfer
    fake = [sys.executable, str(Path(__file__).with_name("fake_hackrf_transfer.py"))]
    src = HackRfTransfer(exe=fake, settle_ms=0)
    cap = src.capture(806e6, 15.36e6, 300000)
    assert cap.x.dtype == np.complex64 and len(cap.x) == 300000
    res = Scanner(log=lambda *a: None).scan_band(src, 20)
    assert [(r.earfcn, r.sync.n_id, r.mib.n_rb) for r in res.reports] == [(6300, 99, 50)]
    assert abs(res.reports[0].ppm - 3.0) < 0.3
