"""Command line interface: python -m ltescan <command> ..."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import report
from .params import BANDS, BANDWIDTH_MHZ, auto_cfo_max, band_edges, earfcn_to_band, earfcn_to_freq, freq_to_earfcn


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_freq(s: str) -> float:
    s = s.strip().lower()
    mult = 1.0
    for suf, m in (("ghz", 1e9), ("mhz", 1e6), ("khz", 1e3), ("g", 1e9), ("m", 1e6), ("k", 1e3)):
        if s.endswith(suf):
            s, mult = s[:-len(suf)], m
            break
    v = float(s) * mult
    if v < 1e5:          # bare number like 1842.5 -> MHz
        v *= 1e6
    return v


def add_sdr_args(p):
    g = p.add_argument_group("SDR (HackRF)")
    g.add_argument("--driver", default="auto", choices=["auto", "hackrf_transfer", "soapy"])
    g.add_argument("--lna", type=int, default=32, help="LNA/IF gain 0-40 dB (step 8)")
    g.add_argument("--vga", type=int, default=30, help="VGA/baseband gain 0-62 dB (step 2)")
    g.add_argument("--amp", action="store_true", help="enable the RF amplifier (+14 dB)")
    g.add_argument("--serial", help="HackRF serial number")
    g.add_argument("--ppm", type=float, default=0.0,
                   help="known HackRF clock error in ppm (tuning correction)")


def add_file_args(p):
    g = p.add_argument_group("IQ file instead of SDR")
    g.add_argument("--file", help="IQ file (.cs8 from hackrf_transfer, .cs16, .cf32, .npy)")
    g.add_argument("--format", choices=["cs8", "cs16", "cf32", "npy"])
    g.add_argument("--skip", type=float, default=0.0, help="skip the start of the file [s]")


def open_source(a):
    from .sdr import open_sdr
    return open_sdr(a.driver, lna=a.lna, vga=a.vga, amp=a.amp, serial=a.serial, ppm=a.ppm)


def load_file(a, duration=None):
    from .sdr import read_iq
    _, meta = read_iq(a.file, a.format, max_samples=1)
    fs = a.rate or meta.get("sample_rate")
    if not fs:
        raise SystemExit("give --rate (no .json metadata file)")
    fc = meta.get("center_freq")
    freq, earfcn = getattr(a, "freq", None), getattr(a, "earfcn", None)
    if isinstance(freq, list):                     # scan: --freq may repeat
        freq = freq[0]
    if isinstance(earfcn, list):
        earfcn = earfcn[0]
    if freq:
        fc = parse_freq(freq)
    elif earfcn is not None:
        fc = earfcn_to_freq(earfcn)
    n = int(duration * fs) if duration else None
    x, _ = read_iq(a.file, a.format, max_samples=n, offset=int(a.skip * fs))
    return x, float(fs), (float(fc) if fc else None)


# --------------------------------------------------------------- commands
def cmd_bands(a):
    print(f"{'Band':>5} {'Duplex':>7} {'DL from MHz':>11} {'DL to MHz':>10} {'EARFCN':>13}")
    for b, (fl, off, lo, hi, dup) in sorted(BANDS.items()):
        f0, f1 = band_edges(b)
        print(f"{b:5d} {dup:>7} {f0 / 1e6:11.1f} {f1 / 1e6:10.1f} {lo:6d}-{hi:<6d}")


def cmd_earfcn(a):
    for v in a.values:
        if "." in v or v.lower().endswith(("m", "mhz", "g", "ghz")) or float(v) > 100000:
            f = parse_freq(v)
            e = freq_to_earfcn(f)
            print(f"{f / 1e6:.1f} MHz -> EARFCN {e if e is not None else 'none'}"
                  + (f" (band {earfcn_to_band(e)})" if e is not None else ""))
        else:
            e = int(v)
            print(f"EARFCN {e} -> {earfcn_to_freq(e) / 1e6:.1f} MHz (band {earfcn_to_band(e)})")


def cmd_record(a):
    from .sdr import write_iq
    fc = earfcn_to_freq(a.earfcn) if a.earfcn is not None else parse_freq(a.freq)
    src = open_source(a)
    n = int(a.duration * a.rate)
    log(f"recording {a.duration:.3f} s @ {fc / 1e6:.3f} MHz, {a.rate / 1e6:.2f} Msps ...")
    cap = src.capture(fc, a.rate, n, bb_filter=a.bw or 0.75 * a.rate)
    write_iq(a.out, cap.x, cap.fs, cap.fc, extra={"ppm_correction": a.ppm})
    p = 10 * np.log10(np.mean(np.abs(cap.x) ** 2) + 1e-20)
    peak = np.max(np.abs(np.concatenate([cap.x.real, cap.x.imag])))
    log(f"saved {a.out} ({len(cap.x)} samples), power {p:.1f} dBFS, peak {peak:.2f}"
        + ("  <-- CLIPPING, reduce the gain" if peak > 0.98 else ""))


def cmd_scan(a):
    from .scanner import Scanner
    sc = Scanner(cfo_max=a.cfo_max, ppm_hint=a.ppm_hint, log=log, decode_sib=a.sib,
                 keep_unconfirmed=a.all)
    t0 = time.time()
    if a.file:
        x, fs, fc = load_file(a, a.duration)
        if fc is None:
            raise SystemExit("give --freq (centre frequency of the recording)")
        res = sc.scan_buffer(x, fs, fc, a.band[0] if a.band else None, a.exhaustive, a.thresh)
    else:
        src = open_source(a)
        if a.band:
            from .scanner import ScanResult
            res = ScanResult()
            for b in a.band:
                r = sc.scan_band(src, b, a.rate, a.dwell, a.exhaustive, a.thresh)
                res.reports += r.reports
                res.carriers += r.carriers
                res.psd += r.psd
                res.tested += r.tested
                res.unconfirmed += r.unconfirmed
        else:
            freqs = [earfcn_to_freq(e) for e in (a.earfcn or [])] + [parse_freq(f) for f in (a.freq or [])]
            if not freqs:
                raise SystemExit("give --band, --earfcn or --freq")
            # MIB only needs the central 1.08 MHz -> 7.68 Msps unless SIB wanted
            rate = a.rate if (a.rate != 20e6 or a.sib) else 7.68e6
            res = sc.scan_freqs(src, freqs, rate, a.dwell)
    log(f"tested {res.tested} frequencies in {time.time() - t0:.1f} s"
        + (f"; skipped {res.unconfirmed} PSS/SSS detections without MIB (show: --all)"
           if res.unconfirmed else ""))
    print(report.scan_table(res.reports))
    if a.sib or a.verbose:
        for r in res.reports:
            print()
            print(report.cell_text(r))
    if a.json:
        report.save_json(a.json, res.reports)
        log(f"saved {a.json}")
    if a.plot and res.psd:
        report.plot_spectrum(a.plot, res.psd, res.carriers, res.reports)
        log(f"saved {a.plot}")


def cmd_cell(a):
    from .receiver import LteReceiver
    live = not a.file
    if live:
        fc = earfcn_to_freq(a.earfcn) if a.earfcn is not None else parse_freq(a.freq)
        src = open_source(a)
        fs = a.rate or 15.36e6
        log(f"capturing {a.duration:.3f} s @ {fc / 1e6:.3f} MHz, {fs / 1e6:.2f} Msps ...")
        cap = src.capture(fc, fs, int(a.duration * fs), bb_filter=0.75 * fs)
        x, fs, fc = cap.x, cap.fs, cap.fc
    else:
        x, fs, fc = load_file(a, a.duration)
    cfo_max = a.cfo_max or auto_cfo_max(fc)
    rx = LteReceiver(x, fs, fc, cfo_max=cfo_max)
    cells = rx.search()
    if a.pci is not None:
        cells = [c for c in cells if c.n_id == a.pci]
    if not cells:
        print("No LTE cell found.")
        return 1
    log("cells found: " + ", ".join(f"PCI {c.n_id} (PSS {c.pss_metric:.2f})" for c in cells))
    reports, unconfirmed = [], []
    for i, cell in enumerate(cells[:a.max_cells]):
        t0 = time.time()
        rep = rx.analyze(cell, decode_sib=not a.no_sib, rnti_discovery=a.rnti,
                         max_subframes=a.max_subframes)
        # live capture too narrow for this cell: recapture once at 20 Msps
        if (live and rep.mib and rep.mib.n_rb * 180e3 > 0.8 * fs and fs < 20e6
                and not a.no_sib):
            log(f"cell is {BANDWIDTH_MHZ[rep.mib.n_rb]} MHz wide - recapturing @ 20 Msps")
            cap = src.capture(fc, 20e6, int(a.duration * 20e6), bb_filter=20e6)
            rx2 = LteReceiver(cap.x, cap.fs, fc, cfo_max=cfo_max)
            c2 = [c for c in rx2.search() if c.n_id == cell.n_id]
            if c2:
                rep = rx2.analyze(c2[0], decode_sib=True, rnti_discovery=a.rnti,
                                  max_subframes=a.max_subframes)
        log(f"analysis of PCI {cell.n_id}: {time.time() - t0:.1f} s")
        if i and rep.mib is None and not a.all:
            # weaker PSS/SSS detection that no PBCH CRC confirms: may be a
            # residue of the strongest cell, so keep it to a single line
            unconfirmed.append(cell)
            continue
        reports.append(rep)
        if reports[1:]:
            print()
        print(report.cell_text(rep, show_dci=a.dci))
        if a.plot:
            Path(a.plot).mkdir(parents=True, exist_ok=True)
            d = rep.diag
            if d.get("frame"):
                report.plot_cell(str(Path(a.plot) / f"pci{cell.n_id}"),
                                 np.concatenate(d["frame"], axis=0),
                                 d.get("pbch"), d.get("pdsch"), d.get("h"))
                log(f"plots saved in {a.plot}")
    if unconfirmed:
        print()
        print("Unconfirmed weaker detections (PSS/SSS only, MIB not decoded; show with --all): "
              + ", ".join(f"PCI {c.n_id} (PSS {c.pss_metric:.2f}, SSS margin {c.sss_margin:.2f})"
                          for c in unconfirmed))
    if a.json:
        report.save_json(a.json, reports)
        log(f"saved {a.json}")
    return 0


def cmd_synth(a):
    from .dsp import resample
    from .params import CellConfig
    from .sdr import write_iq
    from .synth import EnbSimulator, SynthConfig, channel
    cell = CellConfig(n_id=a.pci, n_rb=a.nrb, n_ports=a.ports, cp_ext=a.ext_cp, tdd=a.tdd,
                      phich_ng=1)
    sim = EnbSimulator(SynthConfig(cell, sfn0=a.sfn, use_1c=a.dci1c))
    n_frames = int(np.ceil(a.duration / 0.01))
    tx = sim.generate(n_frames)
    y = channel(tx, cell.fs, snr_db=a.snr, cfo=a.cfo, delay=int(a.delay * cell.fs),
                taps=a.taps, seed=a.seed)
    fs = a.rate or cell.fs
    if fs != cell.fs:
        y = resample(y, cell.fs, fs)
    y = y / (4 * np.sqrt(np.mean(np.abs(y) ** 2)))          # ~ -12 dBFS
    fc = parse_freq(a.freq)
    write_iq(a.out, y, fs, fc, extra={"synthetic": True, "pci": a.pci, "n_rb": a.nrb})
    log(f"saved {a.out}: PCI {a.pci}, {a.nrb} RB, {a.ports} port(s), {len(y)} samples @ "
        f"{fs / 1e6:.2f} Msps")


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(prog="ltescan",
                                description="LTE PHY scanner and analyzer for HackRF")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("bands", help="list supported bands")
    s.set_defaults(func=cmd_bands)

    s = sub.add_parser("earfcn", help="convert EARFCN <-> frequency")
    s.add_argument("values", nargs="+")
    s.set_defaults(func=cmd_earfcn)

    s = sub.add_parser("record", help="record IQ from the HackRF to a file")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--freq")
    g.add_argument("--earfcn", type=int)
    s.add_argument("--rate", type=float, default=15.36e6)
    s.add_argument("--duration", type=float, default=0.1)
    s.add_argument("--bw", type=float, help="HackRF baseband filter [Hz]")
    s.add_argument("--out", required=True)
    add_sdr_args(s)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("scan", help="scan bands / frequencies for cells")
    s.add_argument("--band", type=int, action="append", help="band number (repeatable)")
    s.add_argument("--earfcn", type=int, action="append")
    s.add_argument("--freq", action="append")
    s.add_argument("--rate", type=float, default=20e6, help="sample rate for band scans")
    s.add_argument("--dwell", type=float, default=0.045, help="capture time per step [s]")
    s.add_argument("--duration", type=float, default=None, help="(file) how many seconds to use")
    s.add_argument("--exhaustive", action="store_true", help="test every 100 kHz raster point")
    s.add_argument("--thresh", type=float, default=8.0, help="spectrum occupancy threshold [dB]")
    s.add_argument("--cfo-max", type=float, default=None,
                   help="CFO search range [Hz] (default: 20 ppm of the frequency)")
    s.add_argument("--ppm-hint", type=float, default=None, help="known clock error in ppm (narrows the CFO search)")
    s.add_argument("--sib", action="store_true", help="also decode SIB1 (slower)")
    s.add_argument("--all", action="store_true",
                   help="also show PSS/SSS detections without a decoded MIB (may be ghosts)")
    s.add_argument("--verbose", "-v", action="store_true")
    s.add_argument("--json")
    s.add_argument("--plot", help="save the spectrum to a PNG file")
    add_sdr_args(s)
    add_file_args(s)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("cell", help="detailed cell analysis (MIB, PCFICH, PDCCH, SIB1)")
    s.add_argument("--freq")
    s.add_argument("--earfcn", type=int)
    s.add_argument("--rate", type=float, default=None, help="sample rate (default 15.36 Msps)")
    s.add_argument("--duration", type=float, default=0.2)
    s.add_argument("--pci", type=int, help="analyze only this cell")
    s.add_argument("--max-cells", type=int, default=3)
    s.add_argument("--max-subframes", type=int, default=None)
    s.add_argument("--no-sib", action="store_true", help="synchronization + MIB only")
    s.add_argument("--rnti", action="store_true", help="discover active C-RNTIs in the PDCCH")
    s.add_argument("--all", action="store_true",
                   help="full reports also for weaker cells whose MIB could not be decoded")
    s.add_argument("--dci", type=int, default=0, metavar="N", help="show the first N DCIs")
    s.add_argument("--cfo-max", type=float, default=None,
                   help="CFO search range [Hz] (default: 20 ppm of the frequency)")
    s.add_argument("--json")
    s.add_argument("--plot", metavar="DIR", help="save diagnostic plots")
    add_sdr_args(s)
    add_file_args(s)
    s.set_defaults(func=cmd_cell)

    s = sub.add_parser("synth", help="generate a synthetic LTE signal for testing")
    s.add_argument("--out", required=True)
    s.add_argument("--pci", type=int, default=123)
    s.add_argument("--nrb", type=int, default=25, choices=[6, 15, 25, 50, 75, 100])
    s.add_argument("--ports", type=int, default=2, choices=[1, 2, 4])
    s.add_argument("--ext-cp", action="store_true")
    s.add_argument("--tdd", action="store_true")
    s.add_argument("--dci1c", action="store_true", help="schedule SIB1 with DCI 1C")
    s.add_argument("--sfn", type=int, default=0)
    s.add_argument("--duration", type=float, default=0.1)
    s.add_argument("--snr", type=float, default=15.0)
    s.add_argument("--cfo", type=float, default=5e3)
    s.add_argument("--delay", type=float, default=0.0021, help="initial delay [s]")
    s.add_argument("--taps", type=int, default=1, help="number of channel taps")
    s.add_argument("--rate", type=float, default=None, help="resample to this rate")
    s.add_argument("--freq", default="1842.5M", help="frequency stored in the metadata")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    try:
        return a.func(a) or 0
    except RuntimeError as e:
        log(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
