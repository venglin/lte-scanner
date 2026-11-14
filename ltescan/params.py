"""LTE numerology, bandwidth table and EARFCN <-> frequency conversion."""
from __future__ import annotations

from dataclasses import dataclass

N_SC_RB = 12              # subcarriers per resource block
SC_SPACING = 15e3         # Hz
N_RB_MAX = 110
BASE_RATE = 1.92e6        # sample rate of a 128-point FFT

# N_RB -> FFT size
FFT_SIZE = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1536, 100: 2048}
BANDWIDTH_MHZ = {6: 1.4, 15: 3, 25: 5, 50: 10, 75: 15, 100: 20}
MIB_BW = [6, 15, 25, 50, 75, 100]         # dl-Bandwidth enum in MIB
PHICH_NG = [1 / 6, 1 / 2, 1, 2]           # phich-Resource enum in MIB
PHICH_NG_NAME = ["1/6", "1/2", "1", "2"]


def auto_cfo_max(fc: float | None, ppm: float = 20.0) -> float:
    """CFO search range covering a +-`ppm` receiver clock (HackRF One's
    crystal is specified at +-20 ppm, i.e. +-52 kHz at 2.6 GHz)."""
    if not fc:
        return 40e3
    return float(min(max(ppm * 1e-6 * fc, 15e3), 60e3))


def sample_rate(n_rb: int) -> float:
    return FFT_SIZE[n_rb] * SC_SPACING


@dataclass(frozen=True)
class CellConfig:
    """Everything the PHY needs to know about a cell."""
    n_id: int                 # physical cell id 0..503
    n_rb: int = 6
    cp_ext: bool = False      # extended cyclic prefix
    tdd: bool = False
    n_ports: int = 1          # CRS antenna ports 1/2/4
    phich_ng: int = 0         # index into PHICH_NG
    phich_ext: bool = False   # extended PHICH duration

    @property
    def n_id_1(self) -> int:
        return self.n_id // 3

    @property
    def n_id_2(self) -> int:
        return self.n_id % 3

    @property
    def n_symb(self) -> int:
        """OFDM symbols per slot."""
        return 6 if self.cp_ext else 7

    @property
    def n_sc(self) -> int:
        return self.n_rb * N_SC_RB

    @property
    def fft(self) -> int:
        return FFT_SIZE[self.n_rb]

    @property
    def fs(self) -> float:
        return sample_rate(self.n_rb)

    def replace(self, **kw) -> "CellConfig":
        d = self.__dict__.copy()
        d.update(kw)
        return CellConfig(**d)


# TDD uplink-downlink configurations (TS 36.211 Table 4.2-2)
TDD_UL_DL = ["DSUUUDSUUU", "DSUUDDSUUD", "DSUDDDSUDD", "DSUUUDDDDD",
             "DSUUDDDDDD", "DSUDDDDDDD", "DSUUUDSUUD"]
# PHICH m_i factor for TDD (TS 36.211 Table 6.9-1)
TDD_MI = [[2, 1, 0, 0, 0, 2, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
          [0, 0, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          [1, 1, 0, 0, 0, 1, 1, 0, 0, 1]]

# ---------------------------------------------------------------------------
# EARFCN table, 3GPP TS 36.101 Table 5.7.3-1 (downlink)
# band: (F_DL_low MHz, N_Offs-DL, N_DL min, N_DL max, duplex)
BANDS = {
    1: (2110.0, 0, 0, 599, "FDD"),
    2: (1930.0, 600, 600, 1199, "FDD"),
    3: (1805.0, 1200, 1200, 1949, "FDD"),
    4: (2110.0, 1950, 1950, 2399, "FDD"),
    5: (869.0, 2400, 2400, 2649, "FDD"),
    7: (2620.0, 2750, 2750, 3449, "FDD"),
    8: (925.0, 3450, 3450, 3799, "FDD"),
    12: (729.0, 5010, 5010, 5179, "FDD"),
    13: (746.0, 5180, 5180, 5279, "FDD"),
    17: (734.0, 5730, 5730, 5849, "FDD"),
    20: (791.0, 6150, 6150, 6449, "FDD"),
    25: (1930.0, 8040, 8040, 8689, "FDD"),
    26: (859.0, 8690, 8690, 9039, "FDD"),
    28: (758.0, 9210, 9210, 9659, "FDD"),
    32: (1452.0, 9920, 9920, 10359, "SDL"),
    38: (2570.0, 37750, 37750, 38249, "TDD"),
    40: (2300.0, 38650, 38650, 39649, "TDD"),
    41: (2496.0, 39650, 39650, 41589, "TDD"),
    42: (3400.0, 41590, 41590, 43589, "TDD"),
    43: (3600.0, 43590, 43590, 45589, "TDD"),
    66: (2110.0, 66436, 66436, 67335, "FDD"),
    71: (617.0, 68586, 68586, 68935, "FDD"),
}


def earfcn_to_band(earfcn: int) -> int:
    for band, (_, _, lo, hi, _) in BANDS.items():
        if lo <= earfcn <= hi:
            return band
    raise ValueError(f"EARFCN {earfcn} does not belong to any known band")


def earfcn_to_freq(earfcn: int) -> float:
    """Downlink carrier frequency in Hz."""
    f_low, n_offs, _, _, _ = BANDS[earfcn_to_band(earfcn)]
    return (f_low + 0.1 * (earfcn - n_offs)) * 1e6


def freq_to_earfcn(freq_hz: float, band: int | None = None) -> int | None:
    bands = [band] if band else list(BANDS)
    for b in bands:
        f_low, n_offs, lo, hi, _ = BANDS[b]
        n = round((freq_hz / 1e6 - f_low) / 0.1) + n_offs
        if lo <= n <= hi and abs(earfcn_to_freq(n) - freq_hz) < 1e3:
            return n
    return None


def band_earfcns(band: int) -> range:
    _, _, lo, hi, _ = BANDS[band]
    return range(lo, hi + 1)


def band_edges(band: int) -> tuple[float, float]:
    _, _, lo, hi, _ = BANDS[band]
    return earfcn_to_freq(lo), earfcn_to_freq(hi)
