# lte-scanner — LTE and 5G NR cell scanner and analyzer for HackRF

A passive **LTE and 5G NR** downlink receiver for the HackRF One, written in pure Python
(NumPy/SciPy). It finds cells, demodulates the physical layer and decodes the broadcast
channels. It is not a full 3GPP stack.

- **5G NR (FR1):** SS/PBCH block search over whole bands (n78, n38, n28, n1, …),
  polar-coded PBCH, MIB, SSB beams with SS-RSRP, standalone vs non-standalone cells,
  and soft combining that decodes weak cells down to about −10 dB SNR.
- **LTE:** cell search, MIB, PDCCH blind decoding, turbo-coded PDSCH, SIB1 (PLMN, TAC,
  Cell ID), random access responses, paging, and live export to Wireshark.
- Built-in eNB and gNB signal generators, 72 tests, and live verification on four
  Polish operators' networks.

The 20 MHz HackRF is enough for NR too: an SS/PBCH block occupies 7.2 MHz even when
the carrier is 100 MHz wide (see [5G NR](#5g-nr)).

### 5G NR

| Part | What is implemented |
|---|---|
| Raster | synchronization raster (GSCN) and NR-ARFCN for FR1, band tables (n1, n3, n7, n8, n20, n28, n38, n40, n41, n77, n78), band scans that tune to every raster entry |
| Synchronization | PSS search over all GSCNs in a 20 MHz capture with one FFT (sub-band cut-out per GSCN and CFO hypothesis), averaging over SSB periods, SSS (1008 PCIs), fine CFO, TS 38.211 per-symbol phase compensation |
| SS/PBCH block | PBCH DMRS (SSB index, half frame), DMRS channel estimation, 15 and 30 kHz SCS, L_max 4 and 8 |
| PBCH → **MIB** | **polar code** (reliability sequence, interleavers, rate matching) with a CRC-aided successive-cancellation list decoder, 38.212 payload interleaving and scrambling; SFN, half frame, SSB index, k_SSB, CORESET#0 and search space #0, cellBarred, intraFreqReselection |
| Weak cells | all repetitions of one SSB are used together, and PBCHs from different frames are **soft-combined** (SFN and carry hypotheses); about 6 dB better than single blocks |
| Measurements | SS-RSRP per SSB index (beam), SS-SINR, CFO and receiver clock error |
| Interpretation | SIB1 present or not (k_SSB): standalone-capable vs non-standalone (EN-DC) cells |

### LTE

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
| RRC (ASN.1 UPER) | **SIB1**: PLMN (MCC-MNC), TAC, Cell Identity (eNB ID + sector), band, cellBarred, q-RxLevMin, p-Max, SI scheduling, TDD configuration; SI messages: list of contained SIBs (full contents in Wireshark) |
| MAC | **random access responses** (RA-RNTI: timing advance → UE distance, UL grant, temporary C-RNTI); **paging** (P-RNTI, opt-in, subscriber identities always masked) |
| **Wireshark export** | every decoded transport block (MIB, SIB1, SI, RAR, paging) as MAC-LTE frames: live over UDP, into a pcap file, or piped into Wireshark; Wireshark then decodes all SIBs in full |
| Monitor | continuous capture → decode → export loop on one carrier |
| Scanner | fast 20 MHz captures, spectrum-based carrier detection, candidate testing on the 100 kHz raster, learning the HackRF ppm error |

It also includes a built-in **eNB signal generator** (`ltescan synth`) with a channel
model (noise, CFO, clock drift, multipath), and a **gNB SS/PBCH burst generator**
(`ltescan.nr.synth`: beams, data around the SSB, the per-symbol phase of a real gNB).
The tests are built on them, and they let you run the whole chain end to end without
hardware.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e .[plot,test,fast]
```

`fast` installs numba, which compiles the Viterbi and turbo decoders (see
[Performance](#performance)). Everything also works without it, just slower.

### HackRF

`ltescan` drives the HackRF in one of two ways:

1. **`hackrf_transfer`** (default): the program from the *HackRF host tools*. On Windows
   the easiest way to get it is **radioconda** or the **PothosSDR** installer. The device
   needs the WinUSB driver (installed by those packages or by Zadig). Check with
   `hackrf_info`. `ltescan` looks for `hackrf_transfer` in this order: `--hackrf-transfer
   PATH`, the `HACKRF_TRANSFER` environment variable, `PATH`, then the usual install
   directories (`C:\Program Files\PothosSDR\bin`, `%USERPROFILE%\radioconda\Library\bin`
   and other conda prefixes, `/opt/homebrew/bin`, `/usr/local/bin`). The installers do not
   always add their directory to `PATH`.
2. **SoapySDR** (`--driver soapy`): requires the `SoapySDR` and `SoapyHackRF` Python
   modules.

Without a HackRF, every command works on IQ files (`--file`).

## Quick start

5G NR:

```bash
ltescan nr --band n78 --ppm 13 --lna 40 --vga 40     # scan a whole band
ltescan nr --gscn 7880 --duration 0.3                # one raster entry, weak cell
ltescan nr --file n78.cs8 --json nr.json             # offline
```

Example output of `nr` (synthetic cell, four beams):

```
NR cell PCI 412 (N_ID1=137, N_ID2=1) @ 3548.640 MHz, GSCN 7880, SSB ARFCN 636576, band n78
------------------------------------------------------------------------------------------
SSB: SCS 30 kHz, L_max 8, blocks received: 39, SS-RSRP per SSB index [dBFS]: #0 -17.0, #1 -14.4, #2 -16.1, #3 -20.5
Sync: PSS metric 0.39, SSS margin 4.48, DMRS margin 3.77
CFO: +1408 Hz, receiver clock error ~ +0.40 ppm
SS-RSRP: -14.4 dBFS/RE, SS-SINR: 4.1 dB
MIB: SFN 518, half frame 0, SSB index 1, subCarrierSpacingCommon 30 kHz, dmrs-TypeA-Position pos2
     k_SSB 8, controlResourceSetZero 11, searchSpaceZero 0, cellBarred: notBarred, intraFreqReselection: allowed
SIB1: scheduled (CORESET#0 present) - standalone-capable cell
```

LTE:

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

# follow one carrier and stream everything to Wireshark (see below)
ltescan monitor --earfcn 1575 --paging --rnti --pcap - | wireshark -k -i -
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
  weak does not pass the PSS threshold. In the live tests 800/900 MHz worked with
  `--lna 24 --vga 20` (the defaults clipped), while 1.8–2.6 GHz needed
  `--lna 40 --vga 30 --amp`.
- **Check the RF amplifier**: the HackRF's RF amplifier is easily damaged (static, a
  strong signal at the input). A dead amplifier does not just stop amplifying, it
  attenuates: compare `ltescan cell` on a known cell with and without `--amp`. With a
  working amplifier RSRP rises by about 14 dB; on the test unit it dropped by 14 dB,
  so all later live tests ran without `--amp` (`--lna 40 --vga 40` for 3.5 GHz).
- **Weak carriers**: if the spectrum shows a carrier but `scan` does not test it, lower
  the spectral edge threshold, e.g. `--thresh 4`, or use `--exhaustive`.
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

## Wireshark

Decoded transport blocks are exported in Wireshark's *MAC-LTE* format (the one srsRAN
and OpenAirInterface use). Wireshark dissects the MAC layer and hands BCH, BCCH and
PCCH payloads to its LTE RRC dissector, so SIB2, SIB3, SIB5 and every other SIB are
decoded in full: RACH and paging configuration, reference signal power, reselection
thresholds and priorities, inter-frequency neighbours, and so on.

| Option | Use |
|---|---|
| `--pcap FILE` | write a pcap file; open it in Wireshark, no settings needed |
| `--pcap -` | pcap to stdout for live viewing: `... --pcap - \| wireshark -k -i -` |
| `--udp [HOST:PORT]` | send UDP datagrams (default `127.0.0.1:5847`). In Wireshark capture on the loopback adapter and enable the `mac_lte_udp` heuristic (Analyze → Enabled Protocols) |
| `--paging` | also decode paging. S-TMSI and IMSI digits are always zeroed before export; record counts, CS/PS domain and the systemInfoModification/ETWS flags are kept |
| `--no-rar` | do not decode random access responses |

With `--pcap -` all text output goes to stderr, so stdout carries only the pcap stream.
On Windows give Wireshark's full path if it is not on `PATH`:
`... --pcap - | "C:\Program Files\Wireshark\Wireshark.exe" -k -i -`.

The options work with `cell` (one capture) and `monitor`. `monitor` repeats short
captures (`--burst`, default 0.2 s) and prints one status line per burst:

```
21:08:31 burst 4: PCI 112 SFN 1010 RSRP  -50.5 dBFS SINR  10.3 dB CFI 2 CCE  61% | SI 25 RAR 0 PCH 17 C-RNTI 6
```

`monitor` is not continuous reception. Each burst is captured, then decoded, so there
are gaps between bursts (about 3 s per cycle for a 0.3 s burst with HackRF, mostly
`hackrf_transfer` start-up and decoding).

## 5G NR

`ltescan nr` finds NR cells (FR1) by their SS/PBCH blocks and decodes the MIB:

```
ltescan nr --band n78                # whole band, one 20 Msps capture per ~9 MHz
ltescan nr --freq 3650               # every raster entry (GSCN) inside one capture
ltescan nr --gscn 7880               # one raster entry
ltescan nr --arfcn 643296            # SSB ARFCN (absoluteFrequencySSB from RRC)
ltescan nr --file n78.cs8 --json nr.json
```

Known bands: n1, n3, n7, n8, n20, n28, n38, n40, n41, n77, n78. A 100 MHz n78 carrier
is far wider than the HackRF's 20 MHz, but that does not matter here: the SS/PBCH block
occupies only 20 RB (7.2 MHz at 30 kHz) at one raster position inside the carrier, and
the band scan tunes to every raster position in turn. The SSB subcarrier
spacing comes from the band (`--scs 15|30` overrides it, e.g. for n38/n41 which allow
both). The default capture is 45 ms, two SSB periods of 20 ms. Cells that send SSBs
less often (NSA cells may use up to 160 ms) need a longer `--duration`.

**Weak cells: capture longer.** Every SSB index of a cell repeats each period, and the
receiver uses all repetitions in the capture (a "track"): the PSS correlation is
averaged over periods, SSS and DMRS are decided on the sum, and when no single PBCH
decodes the PBCHs are soft-combined. On synthetic signals, with `--duration 0.2` (ten
blocks per beam) the MIB decodes down to about −10 dB SNR per resource element, against
about −4 dB from single blocks. `--duration 0.3` was enough to decode a weak n78 cell in
every attempt; analysing a 0.3 s capture of one GSCN takes a few seconds.

Per cell the report shows PCI, SSB frequency, GSCN and ARFCN, the SSB indices (beams)
received with their SS-RSRP, SS-SINR, frequency offset, and the MIB: SFN, half frame,
SSB index, subCarrierSpacingCommon, k_SSB, controlResourceSetZero, searchSpaceZero,
dmrs-TypeA-Position, cellBarred and intraFreqReselection. k_SSB tells whether the SSB
has a CORESET#0 for SIB1. Without one (k_SSB = 31) the cell is typically a
non-standalone (EN-DC) cell anchored on LTE.

How it works:

1. **PSS search.** One FFT of the whole capture; for every GSCN and frequency-offset
   hypothesis the PSS sub-band is cut out of that spectrum (ideal decimation) and
   correlated with the three PSS m-sequences by a small IFFT.
   With two or more periods in the capture the correlation is also averaged over
   periods (20 ms), and each averaged peak becomes a track of one block per period.
2. **SSS and SSB index.** Each block is cut out of the full-rate signal, shifted to
   baseband and resampled to 512 × SCS. The CFO is refined on the PSS halves summed
   over the track (data around the SSB filtered out). The per-symbol phase of
   TS 38.211 5.4 is removed: a gNB restarts the upconversion phase at every symbol, so
   a receiver mixing the SSB to baseband sees −2π·f_SSB·t_l per symbol. Not removing it
   looks like a CFO between symbols only, and can make the PSS and SSS channel
   estimates cancel (at 3610.56 MHz neighbouring symbols differ by 90°). The SSS
   (equalized with the smoothed PSS channel) gives N_ID1 and the PBCH DMRS (8
   hypotheses, coherent on the PSS/SSS channel) the 3 LSBs of the SSB index, both
   decided on the sum over the track.
3. **PBCH.** DMRS channel estimate averaged over the three PBCH symbols and smoothed
   (wider at low SNR), equalization, descrambling, repetition combining, polar SCL
   decoding (list 8, CRC24C) and the 38.212 payload de-interleaving/descrambling give
   the MIB. Below 3 GHz both L_max = 4 and 8 interpretations are tried where the band
   allows them.
4. **Soft combining.** Between blocks of one SSB index only the SFN, the half-frame bit
   and the 38.212 scrambling that depends on them change. The coding is linear, so for
   each hypothesis of the first block's SFN mod 16 and half frame (and, past a 16-frame
   boundary, the carry into the MIB's SFN bits) the coded-bit difference is known and
   the soft bits can be aligned and added. Hypotheses are ranked by how coherently
   they add up, the best are decoded, and the CRC plus the SFN consistency decide. With
   a weak SSS, the next best cell ids are tried as well.

## How it works (LTE)

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

## Performance

Profiling a 15 MHz cell showed where the time goes. With numba (the `fast` extra), per
subframe of signal:

| | NumPy only | with numba |
|---|---|---|
| without C-RNTI discovery | 9.1 ms | 4.3 ms |
| with C-RNTI discovery | 23.3 ms | 5.7 ms |

Without numba, Viterbi decoding of hundreds of PDCCH candidates dominated. The rest is
interpreter overhead spread over many small operations (channel estimation, REG
gathering, small FFTs).

**A GPU would not help here.** Each work item is tiny (a Viterbi run over 72–288 soft
bits, FFTs of 128–2048 points) and the trellis recursions are strictly sequential. Kernel
launch and transfer latency would exceed the actual computation. A GPU only pays off for
large batches, for example decoding all PDCCH candidates of many subframes at once or
correlating PSS over every raster point of a wide scan. Real-time operation (1 ms per
subframe) would need the remaining per-subframe Python code to move into compiled code.

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
  mac.py        MAC RAR parsing, paging parsing and identity masking
  export.py     MAC-LTE framing, UDP and pcap sinks for Wireshark
  _jit.py       optional numba kernels (Viterbi, turbo max-log-MAP)
  cli.py        command-line interface
  nr/           5G NR
    params.py   GSCN / NR-ARFCN raster, bands, L_max
    polar.py    polar code: reliability sequence, interleavers, encoder, SCL decoder
    pbch.py     PSS/SSS/DMRS, SSB layout, PBCH payload coding, MIB
    search.py   SS/PBCH block search, demodulation, MIB decoding
    synth.py    gNB SSB burst generator
    report.py   NR text/JSON reports
tests/          72 tests (unit, end-to-end on synthetic signals, Wireshark dissection)
```

Tests: `pytest -q` (~55 s). The Wireshark tests run when `tshark` is installed.

## Verification and limitations

**Verified independently:**

- the CRC16 check value;
- the turbo QPP tables: srsRAN sources plus a check that each of the 188 entries is a
  permutation;
- the TBS columns for N_PRB = 2/3 (srsRAN);
- NR: the polar reliability sequence, the input and sub-block interleavers, the PBCH
  payload interleaver G and the PBCH payload scrambling rules (srsRAN);
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

**Tested on live networks** (HackRF One, Poland):

- cells of four operators found and decoded up to SIB1 in bands 3, 7, 8 and 20;
- 5, 10, 15 and 20 MHz carriers, 2 and 4 CRS ports, PHICH Ng=1, DCI 1A for SIB1,
  multi-PLMN SIB1 (network sharing), SI messages;
- random access responses and paging, exported to Wireshark and decoded there
  (SIB2, SIB3, SIB5, paging records);
- a 20 MHz cell decoded from a 20 Msps capture;
- co-channel cells resolved on the same carrier;
- the receiver clock error (here +13 ppm) estimated and then corrected with `--ppm`;
- 5G NR: MIB decoded from a non-standalone TDD cell in n38 (30 kHz SSB, k_SSB = 31),
  from two SSBs with CORESET#0 in n28 (15 kHz, L_max = 4) and from an n78 cell
  (100 MHz TDD carrier, 4 beams). The n78 cell is weak at the test location (no working
  RF amplifier, SS-SINR around −4 dB) and decodes reliably only with soft combining over
  a 0.3 s capture;
- the 38.211 per-symbol phase: without compensation the live cells showed a 180°
  PSS→SSS phase and inconsistent CFOs; with it the NR CFOs agree with LTE (≈ 0 ppm
  after `--ppm`).

Parts of the specification that the live tests did not exercise are checked only
against the built-in generator, which shares the receiver's reading of the
specification:

- LTE TDD (at the test location band 38 carries only NR);
- NR L_max = 64 (FR2, not supported);
- extended PHICH duration and extended CP;
- distributed VRB allocation and DCI 1C;
- PDSCH RE exclusions around PSS/SSS/PBCH for 25-RB cells;
- the TBS table for DCI 1C (from memory, no second source).

Other limitations:

- only transport blocks with a single code block (≤ 6120 bits), which is enough for SI;
- no decoding of user PDSCH, PMCH/MBSFN or PHICH (ACK/NACK);
- SIB2 and later: the built-in report only names the SIB types; the full contents need
  the Wireshark export;
- TDD: the UL/DL configuration is only known from SIB1. Before that, only subframes 0
  and 5 are analyzed, and special subframes are skipped;
- not real time: about 4–6 ms per subframe with numba (see Performance), so `monitor`
  works in bursts with gaps;
- RSRP is reported in dBFS. A value in dBm needs calibration of the receive chain;
- NR: only the SS/PBCH block and the MIB. SIB1 (PDCCH in CORESET#0, LDPC-coded PDSCH)
  is not decoded yet; FR1 only.

## Legal notes

The tool passively receives and decodes only **broadcast** information (MIB, SIBs,
random access responses) that is transmitted openly to all terminals. It does not
decode user traffic. Paging is decoded only on request (`--paging`), and the subscriber
identities in it (S-TMSI, IMSI) are always zeroed before anything is reported or
exported. The discovered C-RNTIs and temporary C-RNTIs are short-lived radio
identifiers that are not linked to a subscriber's identity. Receiving
and analyzing radio signals is subject to local law (for example, rules on the secrecy
of communications). The tool is intended for educational, research and measurement use.

## Authors

- Przemysław Frasunek <przemyslaw@frasunek.com>
- Claude (Anthropic), via Claude Code

## License

MIT, see [LICENSE](LICENSE).
