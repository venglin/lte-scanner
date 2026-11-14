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
    g.add_argument("--lna", type=int, default=32, help="wzmocnienie LNA/IF 0-40 dB (krok 8)")
    g.add_argument("--vga", type=int, default=30, help="wzmocnienie VGA/BB 0-62 dB (krok 2)")
    g.add_argument("--amp", action="store_true", help="włącz wzmacniacz RF (+14 dB)")
    g.add_argument("--serial", help="numer seryjny HackRF")
    g.add_argument("--ppm", type=float, default=0.0,
                   help="znany błąd zegara HackRF w ppm (korekcja strojenia)")


def add_file_args(p):
    g = p.add_argument_group("plik IQ zamiast SDR")
    g.add_argument("--file", help="plik IQ (.cs8 hackrf_transfer, .cs16, .cf32, .npy)")
    g.add_argument("--format", choices=["cs8", "cs16", "cf32", "npy"])
    g.add_argument("--skip", type=float, default=0.0, help="pomiń początek pliku [s]")


def open_source(a):
    from .sdr import open_sdr
    return open_sdr(a.driver, lna=a.lna, vga=a.vga, amp=a.amp, serial=a.serial, ppm=a.ppm)


def load_file(a, duration=None):
    from .sdr import read_iq
    _, meta = read_iq(a.file, a.format, max_samples=1)
    fs = a.rate or meta.get("sample_rate")
    if not fs:
        raise SystemExit("podaj --rate (brak pliku metadanych .json)")
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
    print(f"{'Pasmo':>5} {'Dupleks':>7} {'DL od MHz':>10} {'DL do MHz':>10} {'EARFCN':>13}")
    for b, (fl, off, lo, hi, dup) in sorted(BANDS.items()):
        f0, f1 = band_edges(b)
        print(f"{b:5d} {dup:>7} {f0 / 1e6:10.1f} {f1 / 1e6:10.1f} {lo:6d}-{hi:<6d}")


def cmd_earfcn(a):
    for v in a.values:
        if "." in v or v.lower().endswith(("m", "mhz", "g", "ghz")) or float(v) > 100000:
            f = parse_freq(v)
            e = freq_to_earfcn(f)
            print(f"{f / 1e6:.1f} MHz -> EARFCN {e if e is not None else 'brak'}"
                  + (f" (pasmo {earfcn_to_band(e)})" if e is not None else ""))
        else:
            e = int(v)
            print(f"EARFCN {e} -> {earfcn_to_freq(e) / 1e6:.1f} MHz (pasmo {earfcn_to_band(e)})")


def cmd_record(a):
    from .sdr import write_iq
    fc = earfcn_to_freq(a.earfcn) if a.earfcn is not None else parse_freq(a.freq)
    src = open_source(a)
    n = int(a.duration * a.rate)
    log(f"nagrywanie {a.duration:.3f} s @ {fc / 1e6:.3f} MHz, {a.rate / 1e6:.2f} Msps ...")
    cap = src.capture(fc, a.rate, n, bb_filter=a.bw or 0.75 * a.rate)
    write_iq(a.out, cap.x, cap.fs, cap.fc, extra={"ppm_correction": a.ppm})
    p = 10 * np.log10(np.mean(np.abs(cap.x) ** 2) + 1e-20)
    peak = np.max(np.abs(np.concatenate([cap.x.real, cap.x.imag])))
    log(f"zapisano {a.out} ({len(cap.x)} próbek), moc {p:.1f} dBFS, szczyt {peak:.2f}"
        + ("  <-- PRZESTEROWANIE, zmniejsz wzmocnienie" if peak > 0.98 else ""))


def cmd_scan(a):
    from .scanner import Scanner
    sc = Scanner(cfo_max=a.cfo_max, ppm_hint=a.ppm_hint, log=log, decode_sib=a.sib,
                 keep_unconfirmed=a.all)
    t0 = time.time()
    if a.file:
        x, fs, fc = load_file(a, a.duration)
        if fc is None:
            raise SystemExit("podaj --freq (częstotliwość środkowa nagrania)")
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
                raise SystemExit("podaj --band, --earfcn albo --freq")
            # MIB only needs the central 1.08 MHz -> 7.68 Msps unless SIB wanted
            rate = a.rate if (a.rate != 20e6 or a.sib) else 7.68e6
            res = sc.scan_freqs(src, freqs, rate, a.dwell)
    log(f"przetestowano {res.tested} częstotliwości w {time.time() - t0:.1f} s"
        + (f"; pominięto {res.unconfirmed} detekcji PSS/SSS bez MIB (pokaż: --all)"
           if res.unconfirmed else ""))
    print(report.scan_table(res.reports))
    if a.sib or a.verbose:
        for r in res.reports:
            print()
            print(report.cell_text(r))
    if a.json:
        report.save_json(a.json, res.reports)
        log(f"zapisano {a.json}")
    if a.plot and res.psd:
        report.plot_spectrum(a.plot, res.psd, res.carriers, res.reports)
        log(f"zapisano {a.plot}")


def cmd_cell(a):
    from .receiver import LteReceiver
    live = not a.file
    if live:
        fc = earfcn_to_freq(a.earfcn) if a.earfcn is not None else parse_freq(a.freq)
        src = open_source(a)
        fs = a.rate or 15.36e6
        log(f"przechwytywanie {a.duration:.3f} s @ {fc / 1e6:.3f} MHz, {fs / 1e6:.2f} Msps ...")
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
        print("Nie znaleziono komórki LTE.")
        return 1
    log("znalezione komórki: " + ", ".join(f"PCI {c.n_id} (PSS {c.pss_metric:.2f})" for c in cells))
    reports = []
    for i, cell in enumerate(cells[:a.max_cells]):
        t0 = time.time()
        rep = rx.analyze(cell, decode_sib=not a.no_sib, rnti_discovery=a.rnti,
                         max_subframes=a.max_subframes)
        # live capture too narrow for this cell: recapture once at 20 Msps
        if (live and rep.mib and rep.mib.n_rb * 180e3 > 0.8 * fs and fs < 20e6
                and not a.no_sib):
            log(f"komórka ma {BANDWIDTH_MHZ[rep.mib.n_rb]} MHz - ponowne przechwycenie @ 20 Msps")
            cap = src.capture(fc, 20e6, int(a.duration * 20e6), bb_filter=20e6)
            rx2 = LteReceiver(cap.x, cap.fs, fc, cfo_max=cfo_max)
            c2 = [c for c in rx2.search() if c.n_id == cell.n_id]
            if c2:
                rep = rx2.analyze(c2[0], decode_sib=True, rnti_discovery=a.rnti,
                                  max_subframes=a.max_subframes)
        log(f"analiza PCI {cell.n_id}: {time.time() - t0:.1f} s")
        reports.append(rep)
        if i:
            print()
        print(report.cell_text(rep, show_dci=a.dci))
        if a.plot:
            Path(a.plot).mkdir(parents=True, exist_ok=True)
            d = rep.diag
            if d.get("frame"):
                report.plot_cell(str(Path(a.plot) / f"pci{cell.n_id}"),
                                 np.concatenate(d["frame"], axis=0),
                                 d.get("pbch"), d.get("pdsch"), d.get("h"))
                log(f"zapisano wykresy w {a.plot}")
    if a.json:
        report.save_json(a.json, reports)
        log(f"zapisano {a.json}")
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
    log(f"zapisano {a.out}: PCI {a.pci}, {a.nrb} RB, {a.ports} port(y), {len(y)} próbek @ "
        f"{fs / 1e6:.2f} Msps")


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(prog="ltescan",
                                description="Skaner i analizator LTE (warstwa PHY) dla HackRF")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("bands", help="lista obsługiwanych pasm")
    s.set_defaults(func=cmd_bands)

    s = sub.add_parser("earfcn", help="konwersja EARFCN <-> częstotliwość")
    s.add_argument("values", nargs="+")
    s.set_defaults(func=cmd_earfcn)

    s = sub.add_parser("record", help="nagraj IQ z HackRF do pliku")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--freq")
    g.add_argument("--earfcn", type=int)
    s.add_argument("--rate", type=float, default=15.36e6)
    s.add_argument("--duration", type=float, default=0.1)
    s.add_argument("--bw", type=float, help="filtr pasmowy HackRF [Hz]")
    s.add_argument("--out", required=True)
    add_sdr_args(s)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("scan", help="skanuj pasma / częstotliwości w poszukiwaniu komórek")
    s.add_argument("--band", type=int, action="append", help="numer pasma (można wielokrotnie)")
    s.add_argument("--earfcn", type=int, action="append")
    s.add_argument("--freq", action="append")
    s.add_argument("--rate", type=float, default=20e6, help="próbkowanie przy skanie pasma")
    s.add_argument("--dwell", type=float, default=0.045, help="czas przechwytu na krok [s]")
    s.add_argument("--duration", type=float, default=None, help="(plik) ile sekund użyć")
    s.add_argument("--exhaustive", action="store_true", help="testuj każdy punkt rastra 100 kHz")
    s.add_argument("--thresh", type=float, default=8.0, help="próg zajętości widma [dB]")
    s.add_argument("--cfo-max", type=float, default=None,
                   help="zakres szukania CFO [Hz] (domyślnie 20 ppm częstotliwości)")
    s.add_argument("--ppm-hint", type=float, default=None, help="znany błąd zegara (zawęża CFO)")
    s.add_argument("--sib", action="store_true", help="dekoduj też SIB1 (wolniej)")
    s.add_argument("--all", action="store_true",
                   help="pokaż też detekcje PSS/SSS bez zdekodowanego MIB (mogą być duchami)")
    s.add_argument("--verbose", "-v", action="store_true")
    s.add_argument("--json")
    s.add_argument("--plot", help="zapisz widmo do pliku PNG")
    add_sdr_args(s)
    add_file_args(s)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("cell", help="szczegółowa analiza komórki (MIB, PCFICH, PDCCH, SIB1)")
    s.add_argument("--freq")
    s.add_argument("--earfcn", type=int)
    s.add_argument("--rate", type=float, default=None, help="próbkowanie (domyślnie 15.36 Msps)")
    s.add_argument("--duration", type=float, default=0.2)
    s.add_argument("--pci", type=int, help="analizuj tylko tę komórkę")
    s.add_argument("--max-cells", type=int, default=3)
    s.add_argument("--max-subframes", type=int, default=None)
    s.add_argument("--no-sib", action="store_true", help="tylko synchronizacja + MIB")
    s.add_argument("--rnti", action="store_true", help="wykrywanie aktywnych C-RNTI w PDCCH")
    s.add_argument("--dci", type=int, default=0, metavar="N", help="pokaż N pierwszych DCI")
    s.add_argument("--cfo-max", type=float, default=None,
                   help="zakres szukania CFO [Hz] (domyślnie 20 ppm częstotliwości)")
    s.add_argument("--json")
    s.add_argument("--plot", metavar="KATALOG", help="zapisz wykresy diagnostyczne")
    add_sdr_args(s)
    add_file_args(s)
    s.set_defaults(func=cmd_cell)

    s = sub.add_parser("synth", help="wygeneruj syntetyczny sygnał LTE do testów")
    s.add_argument("--out", required=True)
    s.add_argument("--pci", type=int, default=123)
    s.add_argument("--nrb", type=int, default=25, choices=[6, 15, 25, 50, 75, 100])
    s.add_argument("--ports", type=int, default=2, choices=[1, 2, 4])
    s.add_argument("--ext-cp", action="store_true")
    s.add_argument("--tdd", action="store_true")
    s.add_argument("--dci1c", action="store_true", help="SIB1 przydzielany DCI 1C")
    s.add_argument("--sfn", type=int, default=0)
    s.add_argument("--duration", type=float, default=0.1)
    s.add_argument("--snr", type=float, default=15.0)
    s.add_argument("--cfo", type=float, default=5e3)
    s.add_argument("--delay", type=float, default=0.0021, help="opóźnienie początku [s]")
    s.add_argument("--taps", type=int, default=1, help="liczba ścieżek kanału")
    s.add_argument("--rate", type=float, default=None, help="przepróbkuj do tej częstotliwości")
    s.add_argument("--freq", default="1842.5M", help="częstotliwość zapisana w metadanych")
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
        log(f"błąd: {e}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
