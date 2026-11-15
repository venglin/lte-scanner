"""5G NR SIB1: LDPC, DCI polar coding, CORESET#0, TBS, RRC and the whole
PDCCH -> PDSCH -> SIB1 chain on synthetic gNB signals."""
import numpy as np
import pytest

from ltescan.cli import main
from ltescan.dsp import mix
from ltescan.nr import ldpc, pdcch, pdsch, polar, rrc
from ltescan.nr.ldpc_tables import BG1, BG2
from ltescan.nr.params import gscn_to_freq
from ltescan.nr.pbch import Mib
from ltescan.nr.search import ssb_search
from ltescan.nr.sib1 import decode_si, decode_sib1
from ltescan.nr.synth import NrSynthConfig, generate
from ltescan.sdr import write_iq
from ltescan.synth import channel

PAYLOAD = rrc.encode_sib1("260", "03", 0x00ABCD, 0x123456789, 78, 1234, 30, 273, 0)


def test_ldpc_tables_and_codec():
    assert len(BG1) == 316 and len(BG2) == 197
    rng = np.random.default_rng(0)
    for bg, z in ((1, 64), (2, 104), (2, 384)):
        rows, n_rows, n_cols, kc = ldpc.graph(bg, z)
        cw = ldpc.encode(rng.integers(0, 2, kc * z), bg, z)
        assert ldpc.syndrome_ok(cw, bg, z)
    seg = ldpc.segment(984, 2)
    assert (seg.c, seg.k_prime, seg.z, seg.k) == (1, 1000, 104, 1040)
    seg = ldpc.segment(20000, 1)
    assert seg.c == 3 and seg.cb_crc == 24
    assert ldpc.select_bg(984, 0.19) == 2 and ldpc.select_bg(8000, 0.7) == 1


def test_ldpc_rv_combining():
    """RV 1 alone carries no systematic bits (no decoding, and never the
    all-zero word that passes the CRC); RV 0 + RV 2 below the single-RV
    threshold do."""
    rng = np.random.default_rng(1)
    tb = rng.integers(0, 2, 3240).astype(np.uint8)
    rate, qm, g = 0.37, 4, 8640
    rx = {}
    for rv in (0, 1, 2):
        f = ldpc.dlsch_encode(tb, rate, qm, g, rv)
        sig = 10 ** (3 / 20)                            # -3 dB per bit
        r = (1 - 2.0 * f) + sig * rng.normal(size=g)
        rx[rv] = (2 * r / sig ** 2, rv)
    assert ldpc.dlsch_decode([rx[1]], 3240, rate, qm) is None
    assert ldpc.dlsch_decode([rx[0]], 3240, rate, qm) is None
    out = ldpc.dlsch_decode([rx[0], rx[2]], 3240, rate, qm)
    assert out is not None and np.array_equal(out, tb)


@pytest.mark.parametrize("k,e", [(63, 432), (63, 864), (63, 1728), (60, 120), (39, 54)])
def test_polar_rate_matching(k, e):
    """Puncturing (K/E <= 7/16), repetition and shortening."""
    rng = np.random.default_rng(k + e)
    c = rng.integers(0, 2, k).astype(np.uint8)
    y = polar.encode_rm(c, e)
    assert len(y) == e
    llr = (1 - 2.0 * y) * 4 + rng.normal(size=e)
    d = polar.rate_dematch_rm(llr, k)
    assert np.array_equal(polar.decode(d, k, lambda b: np.array_equal(b, c), e=e), c)


def test_dci_si_rnti():
    d = pdcch.Dci10Si(48, pdcch.riv(48, 3, 20), 1, False, 4, 2, 0)
    assert pdcch.dci10_si_size(48) == 39 and d.rb_start_len == (3, 20)
    assert pdcch.riv(48, 0, 48) == 95 and pdcch.Dci10Si(
        48, 95, 0, 0, 0, 0, 0).rb_start_len == (0, 48)
    a = pdcch.dci10_si_pack(d)
    assert pdcch.dci10_si_unpack(a, 48) == d
    rng = np.random.default_rng(2)
    for lv in (4, 8, 16):
        e = pdcch.dci_encode(a, 108 * lv)
        llr = (1 - 2.0 * e) * 2 + rng.normal(size=len(e))
        assert np.array_equal(pdcch.dci_decode(llr, 39), a)
        assert pdcch.dci_decode(llr, 39, rnti=0xFFFE) is None     # other RNTI


def test_coreset0_and_search_space0():
    m = Mib(sfn=0, half_frame=0, scs_common=30000, k_ssb=22, dmrs_type_a_pos=2, coreset0=10,
            search_space0=9, cell_barred=True, intra_freq_reselection=True)
    f_ref = gscn_to_freq(7923)
    c0 = pdcch.coreset0(m, f_ref, 30000)
    assert (c0.n_rb, c0.n_symb, c0.offset, c0.n_cce) == (48, 1, 12, 8)
    # CRB grid: SSB subcarrier 0 minus k_SSB x 15 kHz, then 12 RBs lower
    assert c0.f_low == pytest.approx(f_ref - 120 * 30e3 - 22 * 15e3 - 12 * 12 * 30e3)
    assert [c0.occasions(i) for i in range(4)] == [(10, 0, 0), (12, 0, 0), (14, 0, 0),
                                                   (16, 0, 0)]
    # 15 kHz: Table 13-1, search space #0 index 1 (two sets per slot)
    m15 = Mib(0, 0, 15000, 4, 2, 6, 1, False, True)
    c15 = pdcch.coreset0(m15, gscn_to_freq(1902), 15000)
    assert (c15.n_rb, c15.n_symb, c15.offset) == (48, 1, 12)
    assert c15.occasions(1) == (0, 0, 1) and c15.occasions(2) == (1, 0, 0)
    assert pdcch.coreset0(Mib(0, 0, 30000, 31, 2, 0, 0, False, True), f_ref, 30000) is None
    # interleaved REG bundles of CORESET#0 cover the CORESET exactly once
    regs = sorted(r for cce in range(8) for r in pdcch.cce_regs(48, 1, 703, cce, 1))
    assert regs == list(range(48))


def test_tbs_and_time_allocation():
    assert pdsch.tdra(0, 2) == ("A", 0, 2, 12) and pdsch.tdra(0, 3) == ("A", 0, 3, 11)
    assert pdsch.dmrs_symbols("A", 2, 12, 2) == [2, 7, 11]
    assert pdsch.dmrs_symbols("A", 2, 9, 2) == [2, 6, 9]
    # hand computed from 38.214 5.1.3.2
    assert pdsch.tbs(24, 12, 3, 2) == 984
    assert pdsch.tbs(48, 12, 3, 9) == 6912
    assert pdsch.vrb_to_prb(list(range(48)), 48, True)[:6] == [0, 1, 24, 25, 2, 3]


def test_sib1_rrc_roundtrip():
    kind, s = rrc.decode_bcch_dlsch(PAYLOAD)
    assert kind == "SIB1"
    assert s.plmns == ["260-03"] and s.plmn_info[0].tac == 0xABCD
    assert s.plmn_info[0].cell_identity == 0x123456789
    assert s.si == [(16, ["SIB2", "SIB3"])] and s.bands == [78]
    assert s.carriers[0].n_rb == 273 and s.carriers[0].channel_mhz == 100


def _capture(cfg, gscn, snr, duration, taps=1, offset=0.4e6):
    f_ref = gscn_to_freq(gscn)
    c0 = pdcch.coreset0(cfg.mib(0, 0), f_ref, cfg.scs)
    fc = c0.f_center - offset
    y = mix(generate(cfg, duration, 20e6, 0.0, f_ref), f_ref - fc, 20e6)
    y = channel(y[None, :], 20e6, snr_db=snr, cfo=-2e3, delay=777, taps=taps, seed=3)
    cells = ssb_search(y, 20e6, fc, gscns=[gscn], cfo_max=5e3)
    cell = next(c for c in cells if c.n_id == cfg.n_id and c.confirmed)
    return y, fc, cell


def test_sib1_n78_beams():
    """CORESET#0 10 / search space #0 9: one Type0-PDCCH slot pair per beam."""
    cfg = NrSynthConfig(n_id=703, ssb_indices=(0, 1, 2, 3), k_ssb=22, coreset0=10,
                        search_space0=9, sib1=PAYLOAD, sfn0=500, carrier_sc=0)
    y, fc, cell = _capture(cfg, 7923, 10, 0.05)
    r = decode_sib1(y, 20e6, fc, cell)
    assert r.error is None and r.sib1.plmns == ["260-03"]
    assert {d.slot for d in r.dcis} == {10, 12, 14, 16}
    assert r.alloc.tbs == 984 and r.payload.startswith(PAYLOAD)
    (scs, n_rb, lo, ctr), = r.carriers()
    assert (scs, n_rb) == (30, 273)


def test_sib1_ssb_overlap_interleaved_16qam():
    """PDSCH in the SSB slots (rate-matched around the SSB PRBs),
    interleaved VRBs, 16QAM, a loaded carrier and multipath; 2-symbol
    CORESET with aggregation level 16 on another configuration."""
    cfg = NrSynthConfig(n_id=55, ssb_indices=(0, 1, 2, 3), k_ssb=6, coreset0=12,
                        search_space0=0, sib1=PAYLOAD, sfn0=17, carrier_sc=1200,
                        sib1_rbs=(10, 20), sib1_mcs=12, sib1_interleaved=True)
    y, fc, cell = _capture(cfg, 7900, 15, 0.05, taps=3)
    r = decode_sib1(y, 20e6, fc, cell)
    assert r.error is None and r.alloc.qm == 4 and r.sib1.plmn_info[0].tac == 0xABCD
    cfg = NrSynthConfig(n_id=1, ssb_indices=(0, 1), k_ssb=10, coreset0=13, search_space0=4,
                        sib1=PAYLOAD, sfn0=1020, carrier_sc=1500, sib1_rbs=(30, 12),
                        sib1_mcs=10, sib1_level=16, dmrs_type_a_pos=3, sib1_tdra=1)
    y, fc, cell = _capture(cfg, 7900, 10, 0.05)
    r = decode_sib1(y, 20e6, fc, cell)
    assert r.error is None and {d.level for d in r.dcis} == {16}


def test_sib1_soft_combining():
    """Too weak for any single PDSCH: the repetitions (RV 0, 2, 3, 1 over
    four beams and several frames) are combined."""
    cfg = NrSynthConfig(n_id=703, ssb_indices=(0, 1, 2, 3), k_ssb=22, coreset0=10,
                        search_space0=9, sib1=PAYLOAD, sfn0=500, carrier_sc=0, sib1_mcs=9)
    y, fc, cell = _capture(cfg, 7923, -17, 0.15)
    r = decode_sib1(y, 20e6, fc, cell)
    assert r.error is None and r.combined > 1 and len(set(r.rvs)) >= 2


def test_cli_nr_sib1(tmp_path, capsys):
    cfg = NrSynthConfig(n_id=874, scs=15000, l_max=4, ssb_indices=(0, 1), k_ssb=4, coreset0=6,
                        search_space0=4, sib1=PAYLOAD, sfn0=88, carrier_sc=0,
                        sib1_rbs=(4, 30), sib1_mcs=11, sib1_tdra=2)
    y, fc, _ = _capture(cfg, 1902, 10, 0.05)
    f = tmp_path / "n28.cs8"
    write_iq(f, y, 20e6, fc)
    assert main(["nr", "--file", str(f), "--sib1", "--cfo-max", "5000",
                 "--json", str(tmp_path / "nr.json"), "--pcap", str(tmp_path / "nr.pcap")]) == 0
    out = capsys.readouterr().out
    assert "NR cell PCI 874" in out and "SIB1: decoded" in out
    assert "PLMN 260-03: TAC 43981" in out and "16QAM" in out
    assert '"nr_cell_identity": 4886718345' in (tmp_path / "nr.json").read_text()
    pc = (tmp_path / "nr.pcap").read_bytes()
    assert b"nr-rrc.bcch.dl.sch" in pc and PAYLOAD in pc


def test_sib1_pdcch_combining_and_matching():
    """No PDCCH decodes alone: the occasions of one beam are soft-combined
    (RV differences aligned), the DCI is then matched in the remaining
    occasions and all PDSCH transmissions are combined."""
    cfg = NrSynthConfig(n_id=703, ssb_indices=(0, 1, 2, 3), k_ssb=22, coreset0=10,
                        search_space0=9, sib1=PAYLOAD, sfn0=500, carrier_sc=0)
    y, fc, cell = _capture(cfg, 7923, -25, 0.3)
    r = decode_sib1(y, 20e6, fc, cell)
    assert r.error is None and r.sib1.plmns == ["260-03"]
    assert not any(d.how == "decoded" for d in r.dcis)
    assert any(d.how == "combined" for d in r.dcis) and r.combined >= 8


def test_si_rrc_roundtrip():
    b = rrc.NrSib2(6, 3, 6, 12, -118, 31, 2)
    kind, si = rrc.decode_bcch_dlsch(rrc.encode_si(b, n_sibs=2))
    assert kind == "SI" and si.n_sibs == 2 and si.first == "SIB2" and si.sib2 == b
    kind, si = rrc.decode_bcch_dlsch(rrc.encode_si(first="SIB4"))
    assert si.first == "SIB4" and si.sib2 is None
    s = rrc.decode_bcch_dlsch(rrc.encode_sib1("001", "01", 1, 2, 78, 0, 30, 273,
                                              si=[(16, ["SIB2"]), (64, ["SIB4", "SIB5"])],
                                              si_window=20))[1]
    assert s.si == [(16, ["SIB2"]), (64, ["SIB4", "SIB5"])] and s.si_window_slots == 20


def _si_cell(osi: bool):
    """Two SI messages (160 and 320 ms) in their SI-windows. With osi the
    SIB1 also carries searchSpaceOtherSystemInformation (slot 9 of every 10)
    and a pdsch-TimeDomainAllocationList used by the SI PDSCH (whose DMRS
    then starts at point A)."""
    ss = rrc.SearchSpace(8, 0, 10, 9, None, [0], {4: 4, 8: 2, 16: 1}) if osi else None
    tdra = [(0, "A", 2, 12), (0, "A", 2, 5)] if osi else None
    sib1 = rrc.encode_sib1("001", "01", 0x123, 0xABC001, 78, 244, 30, 273,
                           si=[(16, ["SIB2", "SIB3"]), (32, ["SIB4"])], si_window=10,
                           osi_ss=ss, pdsch_tdra=tdra)
    si1 = rrc.encode_si(rrc.NrSib2(6, 3, 6, 12, -118, 31, 2), n_sibs=2, pad=20)
    si2 = rrc.encode_si(first="SIB4", pad=30)
    # CORESET#0 starts at common RB 244 / 2 - 12 = 110 (SI DMRS reference)
    extra = dict(si_slots=(9,), si_tdra=1, si_tdra_list=tuple(tdra), si_rbs=(0, 36),
                 si_mcs=2) if osi else {}
    cfg = NrSynthConfig(n_id=703, ssb_indices=(0, 1, 2, 3), k_ssb=22, coreset0=10,
                        search_space0=9, sib1=sib1, sfn0=500, carrier_sc=600, si=(si1, si2),
                        si_periods=(16, 32), si_coreset0_crb=110, **extra)
    return cfg, si1, si2


def test_si_whole_windows():
    """No search space in SIB1: every slot of the SI-windows is searched."""
    cfg, si1, si2 = _si_cell(False)
    y, fc, cell = _capture(cfg, 7923, 6, 0.35)
    r = decode_sib1(y, 20e6, fc, cell)
    msgs = decode_si(y, 20e6, fc, cell, r)
    assert [m.payload is not None for m in msgs] == [True, True]
    assert msgs[0].info.sib2.cell_reselection_priority == 6 and msgs[1].info.first == "SIB4"


def test_cli_nr_si_messages(tmp_path, capsys):
    """searchSpaceOtherSystemInformation and the SIB1 time allocation list;
    report, JSON and pcap."""
    cfg, si1, si2 = _si_cell(True)
    y, fc, _ = _capture(cfg, 7923, 6, 0.35)
    f = tmp_path / "si.cs8"
    write_iq(f, y, 20e6, fc)
    assert main(["nr", "--file", str(f), "--si", "--cfo-max", "5000",
                 "--json", str(tmp_path / "nr.json"), "--pcap", str(tmp_path / "nr.pcap")]) == 0
    out = capsys.readouterr().out
    assert "SI 1 (SIB2+SIB3, every 160 ms): decoded" in out and "36 PRB" in out
    assert "SIB2: q-Hyst 6 dB, cellReselectionPriority 6" in out
    assert "SI 2 (SIB4, every 320 ms): decoded" in out
    assert '"first_sib": "SIB4"' in (tmp_path / "nr.json").read_text()
    pc = (tmp_path / "nr.pcap").read_bytes()
    assert si1[:20] in pc and si2[:20] in pc


def test_spur_subcarriers_erased():
    """A strong comb on every 4th subcarrier (receiver spur) is detected."""
    from ltescan.nr.sib1 import _spur_subcarriers
    rng = np.random.default_rng(0)
    g = (rng.normal(size=(14, 576)) + 1j * rng.normal(size=(14, 576))) / np.sqrt(2)
    g[:, ::4] += 3
    sp = _spur_subcarriers(g, [2, 3, 4, 5, 6], list(range(36)))
    assert sp == set(range(0, 432, 4))
