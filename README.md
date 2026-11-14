# lte-scanner — LTE PHY scanner and analyzer for HackRF

*Polish version: [README.pl.md](README.pl.md)*

A passive LTE downlink receiver written in pure Python (NumPy/SciPy). It is not a full
3GPP stack. It demodulates the physical layer and understands the broadcast and control
channels:

| Layer | What is implemented |
|---|---|
| Synchronization | PSS (3 Zadoff-Chu roots, CFO search), SSS (168 groups), FDD/TDD and normal/extended CP detection, frame boundary, coarse/fine CFO (PSS halves + CP), sample clock drift tracking |
| Multiple cells | iterative PSS/SSS cancellation (SIC), phantom rejection (SSS margin, common CFO, ±k·15 kHz ZC ambiguity resolution) |
| OFDM | 1.4/3/5/10/15/20 MHz (6–100 RB), CRS for 1/2/4 ports, LS channel estimation + interpolation |
| Measurements | RSRP (dBFS/RE), RSRQ, SINR on CRS, CFO and receiver clock error in ppm |
| PBCH → **MIB** | bandwidth, number of ports (CRC mask), PHICH configuration, full SFN; single-frame decoding or soft combining of 4 frames |
| PCFICH | CFI (1–3) |
| PHICH | REG allocation (needed for PDCCH; including m_i for TDD) |
| PDCCH | quadruplet de-interleaving, CCEs, blind decoding of the common search space (DCI 0/1A, 1C) for SI-/P-/RA-RNTI; optional discovery of active C-RNTIs (DCI 0/1A, 1) and CCE load |
| PDSCH | type 2 allocation (localized and distributed VRB→PRB), TBS, QPSK, SFBC transmit diversity, rate matching, **turbo decoder** (max-log-MAP), CRC24A, HARQ combining of redundancy versions |
| RRC (ASN.1 UPER) | **SIB1**: PLMN (MCC-MNC), TAC, Cell Identity (eNB ID + sector), band, cellBarred, q-RxLevMin, p-Max, SI scheduling, TDD configuration; SI messages: list of contained SIBs |
| Scanner | fast 20 MHz captures, spectrum-based carrier detection, candidate testing on the 100 kHz raster, learning the HackRF ppm error |

It also includes a built-in **eNB signal generator** (`ltescan synth`) with a channel
model (noise, CFO, clock drift, multipath). The tests are built on it, and it lets you
run the whole chain end to end without hardware.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e .[plot,test]
```

### HackRF

`ltescan` drives the HackRF in one of two ways:

1. **`hackrf_transfer`** (default): the program from the *HackRF host tools* must be on
   `PATH`. On Windows the easiest way to get it is **radioconda** or the **PothosSDR**
   installer. The device needs the WinUSB driver (installed by those packages or by
   Zadig). Check with `hackrf_info`.
2. **SoapySDR** (`--driver soapy`): requires the `SoapySDR` and `SoapyHackRF` Python
   modules.

Without a HackRF, every command works on IQ files (`--file`).

## Quick start

```bash
ltescan bands                         # supported bands and EARFCN ranges
ltescan earfcn 1575 6300 2655M        # EARFCN <-> frequency

# band scan (e.g. the bands used in Poland: 1, 3, 7, 8, 20, 28, 38)
ltescan scan --band 3 --band 20 --json scan.json --plot spectrum.png

# scan known carriers (+ SIB1: PLMN, TAC, Cell ID)
ltescan scan --earfcn 1575 --earfcn 6300 --sib

# detailed analysis of one carrier
ltescan cell --earfcn 1575 --rnti --dci 20 --plot plots --json cell.json

# record IQ and analyze offline later
ltescan record --earfcn 1575 --rate 15.36e6 --duration 0.2 --out b3.cs8
ltescan cell --file b3.cs8
```

Without hardware:

```bash
ltescan synth --out test.cs8 --nrb 50 --pci 260 --ports 2 --snr 10 --cfo 23e3 --rate 15.36e6
ltescan cell --file test.cs8 --rnti --dci 6 --plot plots
```

Example output of `cell`:

```
Cell PCI 260 (N_ID1=86, N_ID2=2) @ 1842.500 MHz, EARFCN 1575, band 3
Duplex: FDD, CP: normal
Sync: PSS metric 0.98 (PMR 133), SSS metric 1.00 (margin 2.07)
CFO: +22976 Hz, receiver clock error ~ -12.47 ppm
RSRP: -36.9 dBFS/RE, RSRQ: -8.0 dB, SINR(CRS): 25.9 dB
MIB: bandwidth 10 MHz (50 RB), CRS ports: 2, PHICH: normal, Ng=1/2, SFN=0
Analyzed 118 subframes; CFI2: 118; PDCCH CCE load ~ 16%
SIB1:
  PLMN: 260-03
  TAC: 6699 (0x1a2b)
  Cell Identity: 3160327 (eNB 12345, sector 7)
  Band (freqBandIndicator): 3
  ...
  SI-1: every 16 frames: SIB2
  SI-2: every 32 frames: SIB3, SIB5
Active C-RNTIs (2): 0x1234x118, 0x4321x118
```

### Practical HackRF tips

- **Gain**: start with `--lna 24 --vga 20`. `record` warns about clipping. A signal that
  is too strong (e.g. close to a base station) saturates the 8-bit ADC; one that is too
  weak does not pass the PSS threshold.
- **Clock**: the HackRF One crystal has a ±20 ppm tolerance, i.e. up to ±52 kHz at
  2.6 GHz. The default CFO search range is 20 ppm of the carrier frequency. The report
  gives the estimated clock error. Pass it later as `--ppm` (tuning correction) or
  `--ppm-hint` (narrower search) to make scans faster and more sensitive.
- **Sample rate**: `cell` uses 15.36 Msps by default, which is enough for cells up to
  10 MHz. When the MIB shows 15/20 MHz, the capture is repeated automatically at
  20 Msps (the HackRF maximum). A 20 MHz cell occupies 18 MHz, so the outermost
  subcarriers sit on the filter slope. PDCCH usually still decodes, because it is spread
  over the whole band and protected by a convolutional code.
- **Antenna**: an omnidirectional antenna for the LTE bands. The telescopic antenna that
  ships with the HackRF works where coverage is good.

## How it works

```
IQ (HackRF / file)
 └─ resample to 1.92 Msps (central 6 RB)
     ├─ PSS: correlation with 3 templates × CFO grid, accumulated every 5 ms
     │     → CFO hypotheses ±k·15 kHz (ZC ambiguity) resolved by the SSS
     ├─ fine CFO: phase between the PSS halves, then CP correlation
     ├─ SSS: equalized with the PSS channel, 336 candidates × {FDD,TDD}×{normal,ext CP}
     │     → PCI, frame boundary; SIC and a new search for weaker cells
     └─ PBCH: CRS → channel estimate, SFBC, descrambling (Gold, c_init=PCI),
           rate de-matching, tail-biting Viterbi (wrap-around),
           CRC16 with the port mask → MIB
 └─ resample to the cell's rate (e.g. 15.36 Msps for 50 RB)
     ├─ PSS tracking every 5 ms → linear regression of position (sample clock drift)
     └─ per subframe: FFT → CRS/channel → PCFICH (CFI) → PDCCH:
           REGs (minus PCFICH/PHICH) → undo quadruplet cyclic shift and interleaving
           → CCEs → blind decoding (Viterbi + CRC16 XOR RNTI)
           → DCI 1A/1C for SI-RNTI → PDSCH: VRB→PRB, RE mapping (excluding CRS,
           PBCH, PSS/SSS and the control region), SFBC, descrambling, turbo, CRC24A
           → ASN.1 UPER → SIB1 / SI
```

C-RNTI discovery decodes every CCE position at every aggregation level. The RNTI is
derived from the CRC remainder, so "every" position matches some RNTI. Genuine
candidates are separated from noise by re-encoding the decoded bits and measuring the
|LLR|-weighted agreement with the received soft bits. Random data reaches at most
~0.95 / 0.83 / 0.73 / 0.67 for aggregation levels 1/2/4/8. The report only lists RNTIs
seen at least twice.

Weaker cells found after interference cancellation whose MIB cannot be decoded are
listed in one line as unconfirmed (they can be residues of the strongest cell); `--all`
prints full reports for them.

## Layout

```
ltescan/
  params.py     numerology, bands, EARFCN, TDD configurations
  sequences.py  Gold sequence, PSS, SSS, CRS
  ofdm.py       OFDM (de)modulation, CP
  dsp.py        resampling, mixing, PSD
  sync.py       cell search: PSS/SSS/CFO/SIC
  chest.py      channel estimation, RSRP/RSRQ/SINR
  mimo.py       SFBC (2 and 4 ports)
  coding.py     CRC, convolutional code + Viterbi, sub-block interleaver, rate matching
  turbo.py      turbo code (QPP), max-log-MAP decoder, rate matching, HARQ buffer
  pbch.py       PBCH/MIB
  control.py    REGs, PCFICH, PHICH, PDCCH, search spaces
  dci.py        DCI formats 0/1A/1C/1, RIV
  pdsch.py      TBS, VRB→PRB, RE mapping, PDSCH decoding
  rrc.py        ASN.1 UPER: SIB1, SystemInformation
  receiver.py   receiver: MIB → PDCCH → SIB, statistics
  scanner.py    band scanner
  sdr.py        HackRF (hackrf_transfer / SoapySDR), IQ files (.cs8/.cs16/.cf32/.npy)
  synth.py      eNB signal generator + channel model
  report.py     text/JSON reports, plots
  cli.py        command-line interface
tests/          49 tests (unit + end-to-end on synthetic signals)
```

Tests: `pytest -q` (~35 s).

## Verification and limitations

**Verified independently:**

- the CRC16 check value;
- the turbo QPP tables: srsRAN sources plus a check that each of the 188 entries is a
  permutation;
- the TBS columns for N_PRB = 2/3 (srsRAN);
- DCI sizes for all bandwidths;
- EARFCN conversions.

**Tested on synthetic signals:**

- all bandwidths, 1/2/4 ports, FDD and TDD, both CP lengths;
- DCI 1A and 1C, localized and distributed allocation;
- SNR down to −8 dB, CFO ±40 kHz, drift ±20 ppm, multipath;
- non-native sample rates (10–20 Msps);
- two cells on one carrier (neighbor 26 dB weaker);
- a random sweep of 50 configurations, all correct, no phantom cells;
- a band scan through a fake `hackrf_transfer`.

**Not tested on a live network.** The generator and the receiver share the same reading
of the specification, so a mistake in the less common details could cancel out in the
tests. This applies to:

- PHICH REG placement (especially extended duration);
- distributed VRB mapping;
- PDSCH RE exclusions around PSS/SSS/PBCH for odd numbers of RBs (15/25/75);
- the TBS table for DCI 1C (from memory, no second source).

Synchronization, MIB and PCFICH are not affected by these. If on a real signal the MIB
decodes but SIB1 does not, start debugging in those places.

Other limitations:

- only transport blocks with a single code block (≤ 6120 bits), which is enough for SI;
- no decoding of user PDSCH, PMCH/MBSFN or PHICH (ACK/NACK);
- SIB2 and later: only the type is recognized, not the contents;
- TDD: the UL/DL configuration is only known from SIB1. Before that, only subframes 0
  and 5 are analyzed, and special subframes are skipped;
- pure-Python performance: about 10–20 ms per subframe (slower with C-RNTI blind
  decoding); this is offline analysis, not real time;
- RSRP is reported in dBFS. A value in dBm needs calibration of the receive chain.

## Legal notes

The tool passively receives and decodes only **broadcast** information (MIB/SIB) that is
transmitted openly to all terminals. It does not decode user traffic or the contents of
paging messages (it only shows that paging occurred). The discovered C-RNTIs are
temporary radio identifiers that are not linked to a subscriber's identity. Receiving
and analyzing radio signals is subject to local law (for example, rules on the secrecy
of communications). The tool is intended for educational, research and measurement use.

## Authors

- Przemysław Frasunek <przemyslaw@frasunek.com>
- Claude (Anthropic), via Claude Code

## License

MIT, see [LICENSE](LICENSE).
