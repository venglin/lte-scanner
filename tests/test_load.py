"""LTE load monitoring: DCI formats 2/2A/0, resource allocation types 0/1,
C-RNTI discovery on TM3/TM4 traffic, PRB use and the monitor CSV."""
import numpy as np
import pytest

from ltescan import dci
from ltescan.cli import main
from ltescan.params import CellConfig
from ltescan.receiver import LteReceiver
from ltescan.sdr import write_iq
from ltescan.synth import EnbSimulator, SynthConfig, channel


def test_dci2_sizes():
    # 36.212 5.3.3.1.5/5A, FDD: header + RBG bitmap + TPC 2 + HARQ 3 + swap 1
    # + 2 x (MCS 5, NDI 1, RV 2) + precoding (2: 3/6 bits, 2A: 0/2 bits)
    assert dci.size_2(50, 2) == 1 + 17 + 2 + 3 + 1 + 16 + 3
    assert dci.size_2(100, 4) == 1 + 25 + 2 + 3 + 1 + 16 + 6
    assert dci.size_2a(50, 2) == 41                 # 40 is ambiguous: one zero bit added
    assert dci.size_2a(75, 4) == 45                 # 44 likewise
    for n in (6, 15, 25, 50, 75, 100):
        for p in (2, 4):
            s = {dci.size_0_1a(n), dci.size_1(n), dci.size_2(n, p), dci.size_2a(n, p)}
            assert len(s) == 4


def test_allocation_types():
    assert dci.ra_type01_prbs(0, [1, 0, 1], 6) == [0, 2]
    # type 1, 50 RB (P = 3): subset 1, no shift
    assert dci.ra_type01_prbs(1, [0, 1, 0] + [1] * 14, 50)[:6] == [3, 4, 5, 12, 13, 14]
    # format 2: second TB disabled (MCS 0, RV 1)
    b = [0] + [1] * 17 + [0, 0] + [0, 0, 0] + [0] + [0, 1, 0, 1, 0, 1, 0, 0] + \
        [0, 0, 0, 0, 0, 0, 0, 1] + [0, 0, 0]
    d = dci.parse_2(np.array(b), 50, 0x1234, 2)
    assert d.n_tb == 1 and d.mcs == 10 and len(d.prbs) == 50
    # format 0: RIV of 10 RBs from RB 5
    riv = dci.riv_encode(5, 10, 50)
    bits = [0, 0] + [int(c) for c in format(riv, f"0{dci._riv_bits(50)}b")] + [0] * 20
    u = dci.parse_0(np.array(bits), 50, 0x1234)
    assert (u.fmt, u.rb_start, u.n_vrb) == ("0", 5, 10)


@pytest.mark.parametrize("fmt,ports,n_rb", [("2", 4, 50), ("2A", 2, 25)])
def test_rnti_discovery_tm3_tm4(fmt, ports, n_rb):
    """TM4 / TM3 assignments at level 2 in the UE search space: both C-RNTIs
    found, no phantom RNTI from the second CCE (the real span is resolved),
    PRB use from the RBG pattern 1100 (about half of the band)."""
    cell = CellConfig(n_id=77, n_rb=n_rb, n_ports=ports, phich_ng=1)
    tx = EnbSimulator(SynthConfig(cell, extra_fmt=fmt, extra_rbg=0b1100)).generate(4)
    y = channel(tx, cell.fs, snr_db=12, cfo=3e3, delay=1000, seed=1)
    rx = LteReceiver(y[0] if y.ndim > 1 else y, cell.fs, 1842.5e6)
    rep = rx.analyze(rx.search()[0], rnti_discovery=True)
    assert set(rep.confirmed_rntis()) == {0x1234, 0x4321}
    assert {d.fmt for d in rep.dcis if d.rnti == 0x1234} == {fmt}
    assert 0.5 <= np.mean(rep.prb_dl) <= 0.6
    assert np.mean(rep.prb_energy) < 0.1           # the generator sends no user PDSCH


def test_monitor_csv(tmp_path, capsys):
    cell = CellConfig(n_id=77, n_rb=25, n_ports=2, phich_ng=1)
    tx = EnbSimulator(SynthConfig(cell, extra_fmt="2A")).generate(30)
    y = channel(tx, cell.fs, snr_db=12, cfo=3e3, delay=1000, seed=1)
    f = tmp_path / "m.cs8"
    write_iq(f, y[0] if y.ndim > 1 else y, cell.fs, 1842.5e6)
    csv = tmp_path / "load.csv"
    assert main(["monitor", "--file", str(f), "--rnti", "--burst", "0.1",
                 "--csv", str(csv)]) == 0
    lines = csv.read_text().strip().splitlines()
    assert lines[0].startswith("time,burst,pci") and len(lines) == 4
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert row["pci"] == "77" and row["c_rnti"] == "2" and float(row["prb_dl_dci"]) > 0.4
    assert "PRB DL" in capsys.readouterr().out


def test_time_aligned_cells():
    """Two cells of a synchronised network on one carrier, almost equally
    strong, with their PSS/SSS in the same symbols: cancelling the first
    must not remove the second (live case: Play PCI 112 and 494 in band 20)."""
    cells = []
    for pci, gain in ((112, 1.0), (494, 0.9)):
        cell = CellConfig(n_id=pci, n_rb=25, n_ports=2, phich_ng=1)
        tx = EnbSimulator(SynthConfig(cell, extra_rntis=[]), seed=pci).generate(3)
        cells.append(gain * tx)
    y = channel(cells[0] + cells[1], cell.fs, snr_db=15, cfo=2e3, delay=700, seed=2)
    rx = LteReceiver(y[0] if y.ndim > 1 else y, cell.fs, 803.5e6)
    assert {c.n_id for c in rx.search()} == {112, 494}
