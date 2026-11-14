"""Synthetic LTE downlink generator (eNB side) used for self-tests and
demos: PSS/SSS, CRS, PBCH/MIB, PCFICH, PHICH (dummy), PDCCH with DCI and
PDSCH carrying SIB1, plus a simple radio channel model."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import coding
from .control import (CFI_CODE, pcfich_regs, pcfich_scrambling, pdcch_regs,
                      pdcch_scrambling, phich_regs, quad_permutation, reg_res)
from .dci import P_RNTI, SI_RNTI, pack_1a, pack_1c, size_0_1a, size_1c
from .mimo import td_encode
from .ofdm import mod_subframe
from .params import TDD_MI, TDD_UL_DL, CellConfig
from .pbch import mib_pack, pbch_encode, pbch_re
from .pdsch import (TBS_1C, TBS_PRB2, TBS_PRB3, dci_prbs, pdsch_e, pdsch_re,
                    pdsch_scrambling, sib1_rv)
from .dci import Dci
from .sequences import crs, crs_symbols, pss, sss
from .turbo import tb_encode
from . import mac, rrc



@dataclass
class SynthConfig:
    cell: CellConfig
    sfn0: int = 100
    cfi: int = 2
    use_1c: bool = False
    sib1: bytes = b""
    sib1_rb: tuple = (0, 4)          # (first VRB, number of VRBs)
    distributed: bool = False
    extra_rntis: list = field(default_factory=lambda: [0x1234, 0x4321])
    tdd_config: int = 1
    rar: bool = True                 # RA-RNTI 3 response in subframe 0 of odd frames
    paging: bool = True              # P-RNTI paging in subframe 9 of odd frames


def default_sib1(cell: CellConfig) -> bytes:
    return rrc.encode_sib1("260", "03", tac=0x1A2B, cell_id=(12345 << 8) | 7, band=3,
                           sched=((16, []), (32, ["SIB3", "SIB5"])), si_window=20,
                           tag=5, tdd_config=(1, 7) if cell.tdd else None)


class EnbSimulator:
    def __init__(self, sc: SynthConfig, seed: int = 0):
        self.sc = sc
        self.cfg = sc.cell
        self.rng = np.random.default_rng(seed)
        if not sc.sib1:
            sc.sib1 = default_sib1(self.cfg)
        self.payload = coding.bytes_to_bits(sc.sib1)
        self._pick_mcs()
        # like a real eNB: enlarge the control region until the common
        # search space (4 CCEs at least) fits in every downlink subframe
        while sc.cfi < 3 and min(len(pdcch_regs(self.cfg, sc.cfi, self.m_i(sf))) // 9
                                 for sf in range(10) if self.is_dl(sf) == "D") < 4:
            sc.cfi += 1

    def cfi(self, sf: int) -> int:
        """TS 36.211 Table 6.7-1: in TDD special subframes (which carry the
        PSS in symbol 2) the control region is at most 2 OFDM symbols."""
        if self.is_dl(sf) == "S":
            return min(self.sc.cfi, 2 if self.cfg.n_rb > 10 else 1)
        return self.sc.cfi

    def _pick_mcs(self):
        need = len(self.payload)
        sc = self.sc
        tables = [(2, TBS_1C)] if sc.use_1c else [(2, TBS_PRB2), (3, TBS_PRB3)]
        for col, tab in tables:
            for i, t in enumerate(tab):
                if t >= need:
                    self.mcs, self.tbs, self.n_prb_1a = i, t, col
                    return
        raise ValueError("SIB1 too large")

    def is_dl(self, sf: int) -> str:
        if not self.cfg.tdd:
            return "D"
        return TDD_UL_DL[self.sc.tdd_config][sf]

    def m_i(self, sf: int) -> int:
        return TDD_MI[self.sc.tdd_config][sf] if self.cfg.tdd else 1

    # -------------------------------------------------------------- mapping
    def _pdcch(self, grid, sf, dcis):
        cfg = self.cfg
        regs = pdcch_regs(cfg, self.cfi(sf), self.m_i(sf))
        n_reg = len(regs)
        n_bits = 8 * n_reg
        bits = np.zeros(n_bits, dtype=np.uint8)
        used = np.zeros(n_bits, dtype=bool)
        for payload, rnti, cce, agg in dcis:
            if 72 * (cce + agg) > n_bits:
                continue
            c = coding.crc_attach(payload, coding.CRC16, rnti)
            e = coding.conv_rate_match(coding.conv_encode(c), 72 * agg)
            bits[72 * cce:72 * (cce + agg)] = e
            used[72 * cce:72 * (cce + agg)] = True
        bits ^= pdcch_scrambling(cfg, sf, n_bits)
        sym = coding.qpsk_mod(bits)
        sym[~used[0::2]] = 0
        y = td_encode(sym, cfg.n_ports).reshape(cfg.n_ports, n_reg, 4)
        perm = quad_permutation(n_reg, cfg.n_id)
        for i, (l, k0) in enumerate(regs):
            k = reg_res(cfg, l, k0)
            grid[:, l, k] = y[:, perm[i], :]

    def _subframe_grid(self, sfn: int, sf: int) -> np.ndarray:
        cfg = self.cfg
        n_sym = 2 * cfg.n_symb
        grid = np.zeros((cfg.n_ports, n_sym, cfg.n_sc), dtype=complex)
        kind = self.is_dl(sf)
        if kind == "U":
            return grid
        n_sym_dl = n_sym if kind == "D" else 9      # DwPTS length
        # CRS
        for slot in range(2):
            for l, ports in crs_symbols(cfg.cp_ext):
                if slot * cfg.n_symb + l >= n_sym_dl:
                    continue
                for p in ports:
                    if p < cfg.n_ports:
                        k, r = crs(2 * sf + slot, l, p, cfg.n_id, cfg.n_rb, cfg.cp_ext)
                        grid[p, slot * cfg.n_symb + l, k] = r
        kc = np.arange(62) - 31 + cfg.n_sc // 2
        # PSS / SSS
        if cfg.tdd:
            if sf in (1, 6):
                grid[0, 2, kc] = pss(cfg.n_id_2)
            if sf in (0, 5):
                grid[0, n_sym - 1, kc] = sss(cfg.n_id_1, cfg.n_id_2, sf == 5)
        elif sf in (0, 5):
            grid[0, cfg.n_symb - 1, kc] = pss(cfg.n_id_2)
            grid[0, cfg.n_symb - 2, kc] = sss(cfg.n_id_1, cfg.n_id_2, sf == 5)
        # PBCH
        if sf == 0:
            mib = mib_pack(cfg.n_rb, cfg.phich_ext, cfg.phich_ng, sfn)
            b = pbch_encode(mib, cfg.n_ports, cfg.n_id, cfg.cp_ext)[sfn % 4]
            y = td_encode(coding.qpsk_mod(b), cfg.n_ports)
            ls, ks = pbch_re(cfg.n_id, cfg.n_rb, cfg.cp_ext)
            grid[:, ls, ks] = y
        # PCFICH
        cfi = self.cfi(sf)
        b = CFI_CODE[cfi] ^ pcfich_scrambling(cfg, sf)
        y = td_encode(coding.qpsk_mod(b), cfg.n_ports).reshape(cfg.n_ports, 4, 4)
        for i, k0 in enumerate(pcfich_regs(cfg)):
            grid[:, 0, reg_res(cfg, 0, k0)] = y[:, i, :]
        # PHICH (random content, just occupies its REGs)
        for l, k0 in phich_regs(cfg, self.m_i(sf)):
            k = reg_res(cfg, l, k0)
            s = coding.qpsk_mod(self.rng.integers(0, 2, 8))
            grid[:, l, k] = td_encode(s, cfg.n_ports)
        # PDCCH + PDSCH
        dcis = []
        if sf == 5 and sfn % 2 == 0:
            rv = sib1_rv(sfn)
            start, n_vrb = self.sc.sib1_rb
            if self.sc.use_1c:
                payload = pack_1c(cfg.n_rb, start, n_vrb, self.mcs)
                dci = Dci("1C", SI_RNTI, True, False, start, n_vrb, self.mcs)
            else:
                payload = pack_1a(cfg.n_rb, start, n_vrb, self.mcs, rv, self.sc.distributed,
                                  self.n_prb_1a, cfg.tdd)
                dci = Dci("1A", SI_RNTI, self.sc.distributed, False, start, n_vrb,
                          self.mcs, rv=rv, n_prb_1a=self.n_prb_1a)
            dcis.append((payload, SI_RNTI, 0, 4))
            pe, po = dci_prbs(dci, cfg.n_rb)
            ls, ks = pdsch_re(cfg, sf, cfi, pe, po)
            e = pdsch_e(len(ls), cfg)
            tb = np.zeros(self.tbs, dtype=np.uint8)
            tb[:len(self.payload)] = self.payload
            cw = tb_encode(tb, e, rv) ^ pdsch_scrambling(cfg, sf, SI_RNTI, e)
            y = td_encode(coding.qpsk_mod(cw), cfg.n_ports)
            grid[:, ls[:e // 2], ks[:e // 2]] = y
        if sfn % 2 == 1 and self.sc.rar and sf == 0:
            self._broadcast(grid, sf, cfi, 3, mac.encode_rar(17, 25, 0x12345, 0x4602), dcis)
        if sfn % 2 == 1 and self.sc.paging and sf == 9 and self.is_dl(9) == "D":
            msg = mac.encode_paging([("s-tmsi", 0xAB, 0xDEADBEEF, 0),
                                     ("imsi", "260031234567890", 1)], si_modification=True)
            self._broadcast(grid, sf, cfi, P_RNTI, msg, dcis)
        # some UE-specific traffic for RNTI discovery
        n_cce = len(pdcch_regs(cfg, cfi, self.m_i(sf))) // 9
        size = size_0_1a(cfg.n_rb, cfg.tdd)
        for j, rnti in enumerate(self.sc.extra_rntis):
            cce = 8 + 2 * j
            if cce + 2 <= n_cce:
                payload = self.rng.integers(0, 2, size).astype(np.uint8)
                payload[0] = 1
                dcis.append((payload, rnti, cce, 2))
        self._pdcch(grid, sf, dcis)
        if n_sym_dl < n_sym:
            grid[:, n_sym_dl:] = 0
        return grid

    def _broadcast(self, grid, sf, cfi, rnti, data: bytes, dcis, rb=(0, 4)):
        """DCI 1A in the common search space (CCE 0, L4) + QPSK PDSCH."""
        cfg = self.cfg
        bits = coding.bytes_to_bits(data)
        for col, tab in ((2, TBS_PRB2), (3, TBS_PRB3)):
            hit = [(i, t) for i, t in enumerate(tab) if t >= len(bits)]
            if hit:
                mcs, tbs = hit[0]
                break
        payload = pack_1a(cfg.n_rb, rb[0], rb[1], mcs, 0, False, col, cfg.tdd)
        dci = Dci("1A", rnti, False, False, rb[0], rb[1], mcs, rv=0, n_prb_1a=col)
        dcis.append((payload, rnti, 0, 4))
        pe, po = dci_prbs(dci, cfg.n_rb)
        ls, ks = pdsch_re(cfg, sf, cfi, pe, po)
        e = pdsch_e(len(ls), cfg)
        tb = np.zeros(tbs, dtype=np.uint8)
        tb[:len(bits)] = bits
        cw = tb_encode(tb, e, 0) ^ pdsch_scrambling(cfg, sf, rnti, e)
        grid[:, ls[:e // 2], ks[:e // 2]] = td_encode(coding.qpsk_mod(cw), cfg.n_ports)

    def generate(self, n_frames: int) -> np.ndarray:
        """Per-port time signals, shape (n_ports, n_samples)."""
        cfg = self.cfg
        out = []
        for f in range(n_frames):
            sfn = (self.sc.sfn0 + f) % 1024
            for sf in range(10):
                g = self._subframe_grid(sfn, sf)
                out.append(np.stack([mod_subframe(g[p], cfg.fft, cfg.cp_ext)
                                     for p in range(cfg.n_ports)]))
        return np.concatenate(out, axis=1)


def channel(tx: np.ndarray, fs: float, snr_db: float = 20.0, cfo: float = 0.0,
            delay: int = 0, sro_ppm: float = 0.0, taps: int = 1, seed: int = 0,
            rx_ports_seed: int = 1) -> np.ndarray:
    """Pass per-port signals through independent random channels, sum,
    then add CFO, sampling-rate offset, delay and AWGN."""
    rng = np.random.default_rng(seed)
    n_ports, n = tx.shape
    y = np.zeros(n, dtype=complex)
    for p in range(n_ports):
        if taps == 1:
            h = np.array([np.exp(2j * np.pi * rng.random())])
        else:
            pdp = np.exp(-np.arange(taps) / 1.5)
            h = (rng.normal(size=taps) + 1j * rng.normal(size=taps)) * np.sqrt(pdp / 2 / pdp.sum())
        y += np.convolve(tx[p], h)[:n]
    if sro_ppm:
        from .dsp import stretch
        y = stretch(y, 1 + sro_ppm * 1e-6)
    y = np.concatenate([np.zeros(delay, dtype=complex), y])
    nn = np.arange(len(y))
    y = y * np.exp(2j * np.pi * cfo * nn / fs)
    p_sig = np.mean(np.abs(y[delay:]) ** 2)
    noise = np.sqrt(p_sig / 10 ** (snr_db / 10) / 2)
    y = y + noise * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
    return y.astype(np.complex64)
