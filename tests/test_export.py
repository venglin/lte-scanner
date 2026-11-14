"""RAR/paging decoding, identity masking, Wireshark export and the JIT
kernels."""
import shutil
import socket
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ltescan import coding, mac, turbo
from ltescan.cli import main
from ltescan.export import (MAC_LTE_SIGNATURE, Exporter, PcapSink, UdpSink, mac_lte_frame,
                            SI_RNTI)
from ltescan.params import CellConfig
from ltescan.receiver import LteReceiver
from ltescan.synth import EnbSimulator, SynthConfig, channel


def _tshark():
    for p in (shutil.which("tshark"), r"C:\Program Files\Wireshark\tshark.exe"):
        if p and Path(p).exists():
            return p
    return None


TSHARK = _tshark()


def test_rar_roundtrip():
    r = mac.parse_rar(mac.encode_rar(17, 25, 0x12345, 0x4602) + b"\0\0")
    assert len(r.entries) == 1
    e = r.entries[0]
    assert (e.rapid, e.ta, e.ul_grant, e.temp_crnti) == (17, 25, 0x12345, 0x4602)
    assert 1900 < e.distance_m < 2000


def test_paging_identities_are_masked():
    msg = mac.encode_paging([("s-tmsi", 0xAB, 0xDEADBEEF, 0), ("imsi", "260031234567890", 1)],
                            si_modification=True)
    masked, info = mac.mask_paging(msg)
    assert (info.s_tmsi, info.imsi, info.cs_domain, info.si_modification, info.complete) == \
        (1, 1, 1, True, True)
    assert len(masked) == len(msg) and masked != msg
    # nothing of the identities survives
    bits = coding.bytes_to_bits(masked)
    s = "".join(map(str, bits))
    assert format(0xDEADBEEF, "032b") not in s
    again, info2 = mac.mask_paging(masked)          # still a valid message
    assert again == masked and info2.complete


def test_paging_unknown_encoding_is_zeroed():
    masked, info = mac.mask_paging(b"\x80\xff\xff\xff")   # messageClassExtension
    # everything after the point where parsing stopped being certain is zeroed
    assert not info.complete and masked == b"\x80\0\0\0"


def _decode_synth(**kw):
    cell = CellConfig(n_id=301, n_rb=25, n_ports=2)
    y = channel(EnbSimulator(SynthConfig(cell, sfn0=101)).generate(6), cell.fs, snr_db=15,
                cfo=5e3, delay=3000)
    rx = LteReceiver(y, cell.fs, 1842.5e6)
    pdus = []
    rep = rx.analyze(rx.search()[0], on_pdu=lambda *a: pdus.append(a), **kw)
    return rep, pdus


def test_broadcast_pdus_and_paging_opt_in():
    rep, pdus = _decode_synth()
    kinds = [p[0] for p in pdus]
    assert kinds[0] == "BCH" and "SI" in kinds and "RAR" in kinds and "PCH" not in kinds
    assert rep.rar_entries and rep.rar_entries[0].temp_crnti == 0x4602
    rep, pdus = _decode_synth(decode_paging=True)
    pch = [p for p in pdus if p[0] == "PCH"]
    assert pch and all(mac.mask_paging(p[4])[0] == p[4] for p in pch)
    assert rep.paging_msgs[0].imsi == 1


@pytest.mark.skipif(TSHARK is None, reason="tshark not installed")
def test_pcap_dissected_by_wireshark(tmp_path):
    f = tmp_path / "x.pcap"
    ex = Exporter([PcapSink(str(f))])
    rep, pdus = _decode_synth(decode_paging=True)
    for p in pdus:
        ex(*p)
    ex.close()
    out = subprocess.run([TSHARK, "-r", str(f)], capture_output=True, text=True).stdout
    assert "MasterInformationBlock" in out
    assert "SystemInformationBlockType1" in out
    assert "RAR (RA-RNTI=3" in out and "Temp C-RNTI=17922" in out
    assert "Paging (2 PagingRecords)" in out
    detail = subprocess.run([TSHARK, "-r", str(f), "-V", "-Y", "lte-rrc.PCCH_Message_element"],
                            capture_output=True, text=True).stdout
    assert "m-TMSI: 00000000" in detail


@pytest.mark.skipif(TSHARK is None, reason="tshark not installed")
def test_udp_framing_dissected_by_heuristic(tmp_path):
    """Write the exact UDP payload of UdpSink into a raw-IP pcap and let
    Wireshark's mac_lte_udp heuristic decode it."""
    payload = MAC_LTE_SIGNATURE + mac_lte_frame(
        bytes.fromhex("684c42821910f66e82119246a80000000000"), SI_RNTI, 0xFFFF, 100, 5)
    udp = struct.pack(">HHHH", 40000, 5847, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0,
                     bytes([127, 0, 0, 1]), bytes([127, 0, 0, 1])) + udp
    f = tmp_path / "u.pcap"
    f.write_bytes(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 101)
                  + struct.pack("<IIII", 0, 0, len(ip), len(ip)) + ip)
    out = subprocess.run([TSHARK, "-r", str(f), "--enable-heuristic", "mac_lte_udp"],
                         capture_output=True, text=True).stdout
    assert "MAC-LTE" in out or "LTE RRC" in out


def test_udp_sink_sends_signature():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    s = UdpSink("127.0.0.1", rx.getsockname()[1])
    s.send(mac_lte_frame(b"\x01\x02", SI_RNTI, 0xFFFF, 1, 5), 0.0)
    data = rx.recv(1000)
    s.close()
    rx.close()
    assert data.startswith(b"mac-lte\x01\x01\x04") and data.endswith(b"\x01\x01\x02")


def test_monitor_on_file(tmp_path, capsys):
    f = tmp_path / "s.cs8"
    assert main(["synth", "--out", str(f), "--nrb", "25", "--pci", "77", "--duration", "0.25",
                 "--rate", "7.68e6"]) == 0
    pc = tmp_path / "m.pcap"
    assert main(["monitor", "--file", str(f), "--burst", "0.1", "--paging",
                 "--pcap", str(pc)]) == 0
    out = capsys.readouterr().out
    assert out.count("PCI 77") == 2 and "RAR" in out
    assert pc.stat().st_size > 200


def test_jit_matches_numpy():
    rng = np.random.default_rng(3)
    k = 43
    b = rng.integers(0, 2, k).astype(np.uint8)
    llr = (1 - 2 * coding.conv_rate_match(coding.conv_encode(b), 144).astype(float)) \
        + rng.normal(0, 1.0, 144)
    d = coding.conv_rate_dematch(llr, k)
    jit = coding.viterbi_tb(d)
    saved = coding.HAVE_NUMBA, turbo.HAVE_NUMBA
    try:
        coding.HAVE_NUMBA = turbo.HAVE_NUMBA = False
        ref = coding.viterbi_tb(d)
        tb = rng.integers(0, 2, 400).astype(np.uint8)
        cw = turbo.tb_encode(tb, 1300, 0)
        l2 = (1 - 2 * cw.astype(float)) * 2 + rng.normal(0, 1.2, 1300)
        ref_t = turbo.tb_decode(l2, 400, 0)
    finally:
        coding.HAVE_NUMBA, turbo.HAVE_NUMBA = saved
    assert np.array_equal(jit, ref)
    jit_t = turbo.tb_decode(l2, 400, 0)
    assert np.array_equal(jit_t, ref_t)
