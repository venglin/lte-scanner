"""Unit tests of the coding, sequence, DCI and RRC building blocks."""
import numpy as np
import pytest

from ltescan import coding, dci, params, rrc, turbo
from ltescan.control import common_search_space, pdcch_regs, phich_regs, pcfich_regs
from ltescan.pdsch import vrb_to_prb
from ltescan.sequences import gold, pss, sss_table


def test_crc16_xmodem_check_value():
    bits = coding.bytes_to_bits(b"123456789")
    assert coding.crc(bits, coding.CRC16) == 0x31C3


@pytest.mark.parametrize("spec", [coding.CRC24A, coding.CRC24B, coding.CRC16, coding.CRC8])
def test_crc_attach_check(spec):
    rng = np.random.default_rng(1)
    for n in (1, 7, 40, 333):
        b = rng.integers(0, 2, n).astype(np.uint8)
        ok, res = coding.crc_check(coding.crc_attach(b, spec, mask=0x5), spec)
        assert res == 0x5 and not ok


def test_viterbi_tailbiting_roundtrip():
    rng = np.random.default_rng(2)
    for k in (40, 43, 64):
        b = rng.integers(0, 2, k).astype(np.uint8)
        e = coding.conv_rate_match(coding.conv_encode(b), 3 * k + 50)
        llr = (1 - 2 * e.astype(float)) + rng.normal(0, 0.7, len(e))
        assert np.array_equal(coding.viterbi_tb(coding.conv_rate_dematch(llr, k)), b)


def test_qpp_tables_are_permutations():
    assert len(turbo.TURBO_K) == 188
    for k in turbo.TURBO_K:
        assert len(np.unique(turbo.qpp(k))) == k


@pytest.mark.parametrize("tbs,rv", [(120, 0), (1000, 2), (2216, 3), (40, 1)])
def test_turbo_dlsch_roundtrip(tbs, rv):
    rng = np.random.default_rng(tbs)
    tb = rng.integers(0, 2, tbs).astype(np.uint8)
    e = 3 * tbs + 100
    cw = turbo.tb_encode(tb, e, rv)
    llr = (1 - 2 * cw.astype(float)) * 2 + rng.normal(0, 1.2, e)
    out = turbo.tb_decode(llr, tbs, rv)
    assert out is not None and np.array_equal(out, tb)


def test_soft_combining_of_redundancy_versions():
    rng = np.random.default_rng(5)
    tbs = 600
    tb = rng.integers(0, 2, tbs).astype(np.uint8)
    e = 500                                   # far too few bits for one shot
    sb = turbo.TbSoftBuffer(tbs)
    for rv in (0, 2, 3, 1):
        cw = turbo.tb_encode(tb, e, rv)
        sb.add((1 - 2 * cw.astype(float)) * 3 + rng.normal(0, 1.0, e), rv)
    out = sb.decode()
    assert out is not None and np.array_equal(out, tb)


def test_gold_sequence_properties():
    c = gold(12345, 10000)
    assert set(np.unique(c)) <= {0, 1}
    assert abs(c.mean() - 0.5) < 0.02
    assert not np.array_equal(gold(1, 100), gold(2, 100))


def test_pss_sss_orthogonality():
    p = [pss(i) for i in range(3)]
    for i in range(3):
        assert abs(np.vdot(p[i], p[i])) == pytest.approx(62)
        for j in range(i + 1, 3):
            # 62-of-63 truncated ZC roots: |cross-corr| ~ 23 (=> ~0.14 in power)
            assert abs(np.vdot(p[i], p[j])) < 30
    t = sss_table(1)
    assert len({tuple(r) for r in t}) == 336
    g = t @ t.T
    np.fill_diagonal(g, 0)
    assert np.max(np.abs(g)) < 40


@pytest.mark.parametrize("n_rb,s1a,s1c,s1", [(6, 21, 8, 19), (15, 22, 10, 23), (25, 25, 12, 27),
                                            (50, 27, 13, 31), (75, 27, 14, 33), (100, 28, 15, 39)])
def test_dci_sizes(n_rb, s1a, s1c, s1):
    assert dci.size_0_1a(n_rb) == s1a
    assert dci.size_1c(n_rb) == s1c
    assert dci.size_1(n_rb) == s1


def test_riv_roundtrip():
    for n in (6, 25, 100):
        for start in range(n):
            for length in range(1, n - start + 1):
                assert dci.riv_decode(dci.riv_encode(start, length, n), n) == (start, length)


def test_dci_1a_pack_parse():
    b = dci.pack_1a(50, 3, 7, 9, 2, distributed=False, n_prb_1a=3)
    d = dci.parse_1a(b, 50, dci.SI_RNTI)
    assert (d.rb_start, d.n_vrb, d.mcs, d.rv, d.n_prb_1a) == (3, 7, 9, 2, 3)


@pytest.mark.parametrize("n_rb", [6, 15, 25, 50, 75, 100])
def test_distributed_vrb_mapping_is_injective(n_rb):
    for gap2 in ((False, True) if n_rb >= 50 else (False,)):
        n = dci.n_vrb_dist(n_rb, gap2)
        even, odd = vrb_to_prb(range(n), n_rb, True, gap2)
        assert len(set(even)) == n and len(set(odd)) == n
        assert max(even.max(), odd.max()) < n_rb


@pytest.mark.parametrize("n_rb,ports,ng", [(6, 1, 0), (25, 2, 1), (50, 4, 3), (100, 2, 2)])
def test_control_regions_are_disjoint(n_rb, ports, ng):
    cfg = params.CellConfig(n_id=123, n_rb=n_rb, n_ports=ports, phich_ng=ng)
    pc = {(0, k) for k in pcfich_regs(cfg)}
    ph = set(phich_regs(cfg))
    assert len(pc) == 4 and not (pc & ph)
    for cfi in (1, 2, 3):
        pd = set(pdcch_regs(cfg, cfi))
        assert not (pd & pc) and not (pd & ph)
    assert common_search_space(len(pdcch_regs(cfg, 3)) // 9)


def test_sib1_roundtrip():
    data = rrc.encode_sib1("260", "06", tac=0xBEEF, cell_id=0x1234567, band=20,
                           sched=((16, []), (64, ["SIB3", "SIB5", "SIB13"])), si_window=40,
                           tag=31, tdd_config=(2, 7), p_max=23)
    m = rrc.decode_bcch_dlsch(data)
    assert isinstance(m, rrc.Sib1)
    assert str(m.plmns[0]) == "260-06" and m.tac == 0xBEEF and m.cell_identity == 0x1234567
    assert m.band == 20 and m.p_max == 23 and m.tdd_config == (2, 7)
    assert m.scheduling == [(16, []), (64, ["SIB3", "SIB5", "SIB13"])]
    assert m.si_window_ms == 40 and m.value_tag == 31
    assert m.enb_id == 0x12345 and m.local_cell_id == 0x67


def test_earfcn_conversion():
    assert params.earfcn_to_freq(1575) == pytest.approx(1842.5e6)
    assert params.earfcn_to_freq(6300) == pytest.approx(806e6)
    assert params.freq_to_earfcn(2655e6) == 3100
    assert params.earfcn_to_band(38000) == 38
