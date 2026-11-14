"""Export of decoded transport blocks to Wireshark.

Packets use Wireshark's "mac-lte-framed" format (the one srsRAN and OAI
use): a few context bytes/tags (radio type,
direction, RNTI type and value, SFN/subframe) and then the MAC PDU. Wireshark
dissects the MAC layer and hands BCH/BCCH/PCCH payloads to its LTE RRC
dissector, so every SIB is decoded in full.

Sinks:
- UdpSink: live, UDP to 127.0.0.1:5847, prefixed with the ASCII signature
  "mac-lte" (capture on the loopback adapter and enable the "mac_lte_udp"
  heuristic in Analyze > Enabled Protocols);
- PcapSink: a .pcap file (or "-" for stdout, e.g. piped into
  `wireshark -k -i -`). Records use the "exported PDU" link type naming the
  mac-lte-framed dissector, so they decode without any Wireshark setting.
"""
from __future__ import annotations

import socket
import struct
import sys
import time

# RNTI types of packet-mac-lte.h
NO_RNTI, P_RNTI, RA_RNTI, C_RNTI, SI_RNTI = 0, 1, 2, 3, 4
FDD_RADIO, TDD_RADIO = 1, 2
DIRECTION_DOWNLINK = 1
_TAG_PAYLOAD, _TAG_RNTI, _TAG_FRAME_SUBFRAME, _TAG_CRC_STATUS = 0x01, 0x02, 0x04, 0x07

MAC_LTE_UDP_PORT = 5847
MAC_LTE_SIGNATURE = b"mac-lte"

LINKTYPE_WIRESHARK_UPPER_PDU = 252
_EXP_PDU_TAG_END = 0
_EXP_PDU_TAG_DISSECTOR_NAME = 12


def mac_lte_frame(payload: bytes, rnti_type: int, rnti: int, sfn: int, sf: int,
                  tdd: bool = False) -> bytes:
    """One mac-lte-framed PDU (downlink, CRC OK), without the "mac-lte"
    signature (which only the UDP heuristic needs)."""
    b = bytearray()
    b += bytes([TDD_RADIO if tdd else FDD_RADIO, DIRECTION_DOWNLINK, rnti_type])
    b += bytes([_TAG_RNTI]) + struct.pack(">H", rnti & 0xFFFF)
    b += bytes([_TAG_FRAME_SUBFRAME]) + struct.pack(">H", ((sfn & 0x3FF) << 4) | (sf & 0xF))
    b += bytes([_TAG_CRC_STATUS, 1])
    b += bytes([_TAG_PAYLOAD]) + payload
    return bytes(b)


def rnti_type_of(kind: str) -> int:
    return {"BCH": NO_RNTI, "SI": SI_RNTI, "RAR": RA_RNTI, "PCH": P_RNTI}.get(kind, C_RNTI)


class UdpSink:
    def __init__(self, host: str = "127.0.0.1", port: int = MAC_LTE_UDP_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.count = 0

    def send(self, frame: bytes, ts: float):
        self.sock.sendto(MAC_LTE_SIGNATURE + frame, self.addr)
        self.count += 1

    def close(self):
        self.sock.close()


class PcapSink:
    """Classic pcap with the exported-PDU link type."""

    def __init__(self, path: str):
        if path == "-":
            # the original stdout: the CLI redirects text output to stderr
            self.f = sys.__stdout__.buffer
            self.own = False
        else:
            self.f = open(path, "wb")
            self.own = True
        self.f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535,
                                 LINKTYPE_WIRESHARK_UPPER_PDU))
        self.f.flush()
        self.count = 0

    @staticmethod
    def _exported_pdu(frame: bytes) -> bytes:
        name = b"mac-lte-framed"
        padded = name + b"\0" * (-len(name) % 4 or 4)
        hdr = struct.pack(">HH", _EXP_PDU_TAG_DISSECTOR_NAME, len(padded)) + padded
        hdr += struct.pack(">HH", _EXP_PDU_TAG_END, 0)
        return hdr + frame

    def send(self, frame: bytes, ts: float):
        rec = self._exported_pdu(frame)
        sec = int(ts)
        usec = int(round((ts - sec) * 1e6))
        if usec >= 1_000_000:
            sec, usec = sec + 1, usec - 1_000_000
        self.f.write(struct.pack("<IIII", sec, usec, len(rec), len(rec)) + rec)
        self.f.flush()
        self.count += 1

    def close(self):
        if self.own:
            self.f.close()


class Exporter:
    """Receives decoded PDUs from the receiver and fans them out to sinks."""

    def __init__(self, sinks, tdd: bool = False, t0: float | None = None):
        self.sinks = list(sinks)
        self.tdd = tdd
        self.t0 = time.time() if t0 is None else t0
        self.count = 0

    def __call__(self, kind: str, rnti: int, sfn: int, sf: int, payload: bytes,
                 t_offset: float = 0.0):
        frame = mac_lte_frame(payload, rnti_type_of(kind), rnti, sfn, sf, self.tdd)
        for s in self.sinks:
            s.send(frame, self.t0 + t_offset)
        self.count += 1

    def close(self):
        for s in self.sinks:
            s.close()
