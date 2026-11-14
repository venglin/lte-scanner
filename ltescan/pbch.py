"""PBCH / MIB: TS 36.211 6.6 and TS 36.212 5.3.1, TS 36.331 MIB."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import coding
from .mimo import td_decode, td_encode
from .params import MIB_BW, PHICH_NG_NAME, CellConfig
from .sequences import gold

CRC_MASK = {1: 0x0000, 2: 0xFFFF, 4: 0x5555}


@dataclass
class Mib:
    n_rb: int
    phich_ext: bool
    phich_ng: int
    sfn: int              # full 10-bit SFN of the frame the MIB was read in
    spare: int
    n_ports: int

    def describe(self) -> str:
        return (f"BW={self.n_rb} RB, PHICH duration={'ext' if self.phich_ext else 'normal'},"
                f" Ng={PHICH_NG_NAME[self.phich_ng]}, SFN={self.sfn}, CRS ports={self.n_ports}")


def mib_pack(n_rb: int, phich_ext: bool, phich_ng: int, sfn: int, spare: int = 0):
    v = (MIB_BW.index(n_rb) << 21) | (int(phich_ext) << 20) | (phich_ng << 18) \
        | ((sfn >> 2) << 10) | spare
    return coding.int_to_bits(v, 24)


def mib_unpack(bits, quarter: int, n_ports: int) -> Mib | None:
    v = coding.bits_to_int(bits)
    bw = v >> 21
    if bw >= len(MIB_BW):
        return None
    sfn = (((v >> 10) & 0xFF) << 2) | quarter
    return Mib(MIB_BW[bw], bool((v >> 20) & 1), (v >> 18) & 3, sfn, v & 0x3FF, n_ports)


@lru_cache(maxsize=64)
def pbch_re(n_id: int, n_rb: int, cp_ext: bool):
    """(symbol indices, subcarrier indices) of PBCH REs in subframe 0,
    in mapping order."""
    n_symb = 6 if cp_ext else 7
    k0 = n_rb * 6 - 36
    vs = n_id % 3
    crs_syms = {0, 1, 3} if cp_ext else {0, 1}
    ls, ks = [], []
    for l in range(4):
        for k in range(72):
            if l in crs_syms and k % 3 == vs:
                continue
            ls.append(n_symb + l)
            ks.append(k0 + k)
    return np.array(ls), np.array(ks)


def pbch_bits_per_frame(cp_ext: bool) -> int:
    return 432 if cp_ext else 480


def pbch_encode(mib_bits, n_ports: int, n_id: int, cp_ext: bool) -> np.ndarray:
    """Return 4 frames worth of scrambled PBCH bits, shape (4, E/4)."""
    e = 4 * pbch_bits_per_frame(cp_ext)
    c = coding.crc_attach(mib_bits, coding.CRC16, CRC_MASK[n_ports])
    d = coding.conv_encode(c)
    b = coding.conv_rate_match(d, e)
    b = b ^ gold(n_id, e)
    return b.reshape(4, -1)


def pbch_symbols(bits_frame: np.ndarray, n_ports: int) -> np.ndarray:
    """Per-port RE values for one frame's PBCH."""
    return td_encode(coding.qpsk_mod(bits_frame), n_ports)


def pbch_llr(grid: np.ndarray, h: np.ndarray, noise: float, cfg: CellConfig,
             n_ports: int) -> np.ndarray:
    ls, ks = pbch_re(cfg.n_id, cfg.n_rb, cfg.cp_ext)
    r = grid[ls, ks]
    hh = h[:n_ports, ls, ks]
    x, g = td_decode(r, hh, n_ports)
    return coding.qpsk_llr(x, g / max(noise, 1e-12))


def pbch_decode_llr(llr_frames: list[np.ndarray], n_id: int, cp_ext: bool,
                    n_ports: int, first_quarter: int) -> Mib | None:
    """Combine consecutive frames (same 40 ms period) and try to decode."""
    q_len = pbch_bits_per_frame(cp_ext)
    e = 4 * q_len
    c = gold(n_id, e)
    acc = np.zeros(3 * 40)
    for i, llr in enumerate(llr_frames):
        q = first_quarter + i
        if q > 3:
            break
        seg = coding.scramble_llr(llr, c[q * q_len:(q + 1) * q_len])
        # rate-matching index of this quarter
        idx = coding.conv_rm_index(40, e)[q * q_len:(q + 1) * q_len]
        np.add.at(acc, idx, seg)
    bits = coding.viterbi_tb(acc.reshape(3, 40))
    ok, res = coding.crc_check(bits, coding.CRC16)
    if res != CRC_MASK[n_ports]:
        return None
    return mib_unpack(bits[:24], first_quarter, n_ports)


def decode_mib(frames: list[tuple[np.ndarray, np.ndarray, float]], cfg: CellConfig):
    """frames: list of (subframe-0 grid, channel estimate h (4 ports), noise)
    for consecutive radio frames. Returns Mib or None. The returned SFN
    refers to the first frame in the list."""
    llrs = {p: [pbch_llr(g, h, nz, cfg, p) for g, h, nz in frames] for p in (1, 2, 4)}
    # 1) single frames
    for i in range(len(frames)):
        for p in (1, 2, 4):
            for q in range(4):
                mib = pbch_decode_llr([llrs[p][i]], cfg.n_id, cfg.cp_ext, p, q)
                if mib is not None:
                    mib.sfn = (mib.sfn - i) % 1024
                    return mib
    # 2) soft combining of consecutive frames
    for p in (1, 2, 4):
        for q in range(4):
            n = min(4 - q, len(frames))
            if n < 2:
                continue
            mib = pbch_decode_llr(llrs[p][:n], cfg.n_id, cfg.cp_ext, p, q)
            if mib is not None:
                return mib
    return None
