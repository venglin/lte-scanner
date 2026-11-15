"""5G NR: raster, polar code, PBCH chain and SS/PBCH block search."""
import numpy as np
import pytest

from ltescan.cli import main
from ltescan.nr import pbch, polar
from ltescan.nr.params import (NR_BANDS, arfcn_to_freq, freq_to_arfcn, freq_to_gscn,
                               gscn_to_freq, l_max_candidates)
from ltescan.nr.search import ssb_search
from ltescan.nr.synth import NrSynthConfig, generate
from ltescan.sdr import write_iq
from ltescan.synth import channel


def test_raster():
    assert gscn_to_freq(7711) == pytest.approx(3305.28e6)
    assert gscn_to_freq(5279) == pytest.approx(2112.05e6)        # N=1760, M=1
    assert gscn_to_freq(5280) == pytest.approx(2112.15e6)        # M=3
    for g in (2, 1000, 5279, 5280, 5281, 7498, 7499, 8051):
        assert freq_to_gscn(gscn_to_freq(g)) == g
    assert freq_to_arfcn(3600e6) == 640000 and arfcn_to_freq(640000) == 3600e6
    assert freq_to_arfcn(2112.15e6) == 422430
    g = NR_BANDS["n78"].gscns()
    assert 7711 in g and 8051 in g and g[0] >= 7710 and g[-1] <= 8052
    assert l_max_candidates(3.6e9) == (8,) and l_max_candidates(2.1e9) == (4,)


def test_polar_tables():
    assert sorted(polar.Q_MAX) == list(range(1024))
    assert sorted(polar.PI_IL_MAX) == list(range(164))
    assert sorted(pbch.G) == list(range(32))
    # cross-checked against an independent implementation (srsRAN)
    r = {v: i for i, v in enumerate(polar.Q_MAX)}
    assert (r[247], r[251], r[352], r[608]) == (695, 690, 261, 260)


def test_pss_sss_properties():
    for u in range(3):
        assert np.all(np.abs(pbch.pss(u)) == 1)
    a, b = pbch.pss(0), pbch.pss(1)
    assert abs(a @ b) < 20 and a @ a == 127
    t = pbch.sss_table(1)
    c = np.abs(t @ t.T)
    np.fill_diagonal(c, 0)
    assert c.max() < 0.3 * 127


@pytest.mark.parametrize("l_max,idx", [(4, 3), (8, 6), (64, 45)])
def test_payload_roundtrip(l_max, idx):
    m = pbch.Mib(sfn=777, half_frame=1, scs_common=30000, k_ssb=17 if l_max < 64 else 9,
                 dmrs_type_a_pos=3, coreset0=5, search_space0=9, cell_barred=True,
                 intra_freq_reselection=False, ssb_index=idx, l_max=l_max)
    c = pbch.payload_encode(m, 333)
    assert pbch.crc_ok(c)
    bar = (idx & 3) + 4 * m.half_frame if l_max == 4 else idx & 7
    assert pbch.payload_decode(c, 333, bar, l_max) == m


def test_pbch_decodes_at_low_snr():
    rng = np.random.default_rng(5)
    m = pbch.Mib(sfn=1, half_frame=0, scs_common=15000, k_ssb=31, dmrs_type_a_pos=2,
                 coreset0=0, search_space0=0, cell_barred=False, intra_freq_reselection=True,
                 ssb_index=2, l_max=8)
    s = pbch.pbch_symbols(m, 99)
    sig = np.sqrt(10 ** (4 / 10) / 2)                      # SNR -4 dB per RE
    r = s + sig * (rng.normal(size=432) + 1j * rng.normal(size=432))
    llr = np.empty(864)
    llr[0::2], llr[1::2] = r.real, r.imag
    assert pbch.pbch_decode(llr / sig ** 2, 99, 2, 8) == m
    assert pbch.pbch_decode(llr / sig ** 2, 98, 2, 8) is None      # wrong PCI


def _cells(cfg, gscn, f_off, snr=6, cfo=15e3, duration=0.045, taps=3):
    f_ref = gscn_to_freq(gscn)
    y = channel(generate(cfg, duration, 20e6, f_off)[None, :], 20e6, snr_db=snr, cfo=cfo,
                delay=4321, taps=taps, seed=1)
    return ssb_search(y, 20e6, f_ref - f_off), y, f_ref - f_off


def test_search_n78_beams():
    cfg = NrSynthConfig(n_id=617, scs=30000, l_max=8, ssb_indices=(0, 1, 2, 3),
                        beam_gain_db=(0, -2, -4, -6, 0, 0, 0, 0), sfn0=200)
    cells, _, _ = _cells(cfg, 7963, 2.3e6)
    assert len(cells) == 1
    c = cells[0]
    assert (c.n_id, c.gscn, c.band, c.mib.l_max, c.mib.k_ssb) == (617, 7963, "n78", 8, 6)
    assert c.mib.sfn in (200, 202, 204) and c.mib.sib1_present and not c.mib.cell_barred
    assert set(c.beams) == {0, 1, 2, 3}
    b = c.beams
    assert b[0] > b[1] > b[2] > b[3]
    assert abs(c.cfo - 15e3) < 300


def test_search_n1_raster_and_lmax4():
    cfg = NrSynthConfig(n_id=1001, scs=15000, l_max=4, ssb_indices=(0, 1), k_ssb=31,
                        cell_barred=True, carrier_sc=1000)
    cells, _, _ = _cells(cfg, 5281, -3.1e6, cfo=-22e3, taps=1)   # M = 5 raster entry
    c = [c for c in cells if c.confirmed][0]
    assert (c.n_id, c.gscn, c.mib.l_max) == (1001, 5281, 4)
    assert not c.mib.sib1_present and c.mib.cell_barred
    assert {o.mib.ssb_index for o in c.obs if o.mib} == {0, 1}


def test_cli_nr_file(tmp_path, capsys):
    cfg = NrSynthConfig(n_id=42, ssb_indices=(0, 1))
    _, y, fc = _cells(cfg, 7900, -1.5e6, snr=10)
    f = tmp_path / "nr.cs8"
    write_iq(f, y, 20e6, fc)
    assert main(["nr", "--file", str(f), "--json", str(tmp_path / "nr.json")]) == 0
    out = capsys.readouterr().out
    assert "NR cell PCI 42" in out and "GSCN 7900" in out and "SIB1: scheduled" in out
    assert '"pci": 42' in (tmp_path / "nr.json").read_text()
