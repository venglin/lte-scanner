"""NR FR1 numerology, synchronization raster (GSCN), NR-ARFCN and the
operating bands relevant here (TS 38.101-1, 38.104, 38.211, 38.213)."""
from __future__ import annotations

from dataclasses import dataclass

SSB_SC = 240                 # subcarriers of an SS/PBCH block (20 RB)
SSB_REF_K = 120              # SS_REF is subcarrier 0 of RB 10 of the block


def cp_len(n_fft: int) -> int:
    """Normal cyclic prefix in samples (all SSB symbols use the normal,
    not the 0.5 ms-boundary, CP)."""
    return 144 * n_fft // 2048


# ------------------------------------------------------------------ raster
def gscn_to_freq(gscn: int) -> float:
    """SS_REF [Hz] of a global synchronization channel number (FR1)."""
    if gscn < 2 or gscn > 22255:
        raise ValueError("GSCN outside FR1")
    if gscn <= 7498:
        n = (gscn + 1) // 3
        m = 3 + 2 * (gscn - 3 * n)
        return n * 1.2e6 + m * 50e3
    return 3000e6 + (gscn - 7499) * 1.44e6


def freq_to_gscn(f: float) -> int:
    """Nearest GSCN to a frequency (FR1)."""
    if f < 3000e6:
        n = int(round((f - 150e3) / 1.2e6))
        best = min(((n, m) for n in (n - 1, n, n + 1) for m in (1, 3, 5)),
                   key=lambda nm: abs(nm[0] * 1.2e6 + nm[1] * 50e3 - f))
        return 3 * best[0] + (best[1] - 3) // 2
    return 7499 + int(round((f - 3000e6) / 1.44e6))


def freq_to_arfcn(f: float) -> int:
    """NR-ARFCN on the global raster (what RRC calls absoluteFrequencySSB)."""
    if f < 3000e6:
        return int(round(f / 5e3))
    return 600000 + int(round((f - 3000e6) / 15e3))


def arfcn_to_freq(n: int) -> float:
    if n < 600000:
        return n * 5e3
    return 3000e6 + (n - 600000) * 15e3


# ------------------------------------------------------------------- bands
@dataclass(frozen=True)
class NrBand:
    name: str
    dl_lo: float             # Hz
    dl_hi: float
    scs: int                 # SSB subcarrier spacing [Hz]
    tdd: bool
    gscn_step: int = 1

    def gscns(self) -> list[int]:
        """Synchronization raster entries whose whole SSB fits in the band."""
        half = SSB_SC / 2 * self.scs
        g0, g1 = freq_to_gscn(self.dl_lo + half), freq_to_gscn(self.dl_hi - half)
        out = [g for g in range(g0, g1 + 1)
               if self.dl_lo + half - 1 <= gscn_to_freq(g) <= self.dl_hi - half + 1]
        return out[::self.gscn_step]


NR_BANDS = {b.name: b for b in [
    NrBand("n1", 2110e6, 2170e6, 15000, False),
    NrBand("n3", 1805e6, 1880e6, 15000, False),
    NrBand("n7", 2620e6, 2690e6, 15000, False),
    NrBand("n8", 925e6, 960e6, 15000, False),
    NrBand("n20", 791e6, 821e6, 15000, False),
    NrBand("n28", 758e6, 803e6, 15000, False),
    NrBand("n38", 2570e6, 2620e6, 30000, True),
    NrBand("n40", 2300e6, 2400e6, 30000, True),
    NrBand("n41", 2496e6, 2690e6, 30000, True, 3),
    NrBand("n77", 3300e6, 4200e6, 30000, True),
    NrBand("n78", 3300e6, 3800e6, 30000, True),
]}


def band_of(f: float) -> NrBand | None:
    """Most specific band containing f (n78 before n77)."""
    hits = [b for b in NR_BANDS.values() if b.dl_lo <= f <= b.dl_hi]
    return min(hits, key=lambda b: b.dl_hi - b.dl_lo) if hits else None


def default_scs(f: float) -> int:
    b = band_of(f)
    if b:
        return b.scs
    return 30000 if f >= 3000e6 else 15000


def l_max_candidates(f: float) -> tuple[int, ...]:
    """Maximum number of SSBs per half frame (38.213 4.1). Up to 3 GHz it
    is 4, except that Rel-16 lowered the limit for unpaired (TDD) bands to
    2.4 GHz, so a TDD band between 2.4 and 3 GHz may use 4 or 8."""
    b = band_of(f)
    if f > 3000e6:
        return (8,)
    if b and b.tdd and f > 2400e6:
        return (8, 4)
    return (4,)
