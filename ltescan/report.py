"""Text / JSON reports and diagnostic plots."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dci import rnti_type
from .params import BANDWIDTH_MHZ, PHICH_NG_NAME
from .receiver import CellReport


def cell_text(rep: CellReport, show_dci: int = 0) -> str:
    s = rep.sync
    L = []
    hdr = f"Cell PCI {s.n_id} (N_ID1={s.n_id_1}, N_ID2={s.n_id_2})"
    if rep.fc:
        hdr += f" @ {rep.fc / 1e6:.3f} MHz"
        if rep.earfcn is not None:
            hdr += f", EARFCN {rep.earfcn}, band {rep.band}"
    L.append(hdr)
    L.append("-" * len(hdr))
    L.append(f"Duplex: {s.duplex}, CP: {s.cp}")
    L.append(f"Sync: PSS metric {s.pss_metric:.2f} (PMR {s.pss_pmr:.0f}), "
             f"SSS metric {s.sss_metric:.2f} (margin {s.sss_margin:.2f})")
    ppm = f", receiver clock error ~ {rep.ppm:+.2f} ppm" if not np.isnan(rep.ppm) else ""
    L.append(f"CFO: {s.cfo:+.0f} Hz{ppm}")
    L.append(f"RSRP: {rep.rsrp_dbfs:.1f} dBFS/RE, RSRQ: {rep.rsrq_db:.1f} dB, "
             f"SINR(CRS): {rep.snr_db:.1f} dB")
    m = rep.mib
    if m:
        L.append(f"MIB: bandwidth {BANDWIDTH_MHZ[m.n_rb]} MHz ({m.n_rb} RB), CRS ports: {m.n_ports}, "
                 f"PHICH: {'extended' if m.phich_ext else 'normal'}, Ng={PHICH_NG_NAME[m.phich_ng]}, "
                 f"SFN={m.sfn}")
    else:
        L.append("MIB: not decoded")
    if rep.subframes:
        cfi = ", ".join(f"CFI{k}: {v}" for k, v in sorted(rep.cfi_hist.items()))
        load = np.mean(rep.cce_load) * 100 if rep.cce_load else 0
        L.append(f"Analyzed {rep.subframes} subframes; {cfi}; PDCCH CCE load ~ {load:.0f}%")
    if rep.sib1:
        L.append("SIB1:" + (f" (after {rep.sib1_tx} transmissions)" if rep.sib1_tx > 1 else ""))
        L += ["  " + x for x in rep.sib1.lines()]
    elif rep.sib1_tx:
        L.append(f"SIB1: decoding failed ({rep.sib1_tx} transmissions)")
    if rep.si:
        seen = Counter()
        for sfn, sf, si, _ in rep.si:
            seen[", ".join(si.sibs) + (f" (+{si.n_sibs - 1})" if si.n_sibs > 1 else "")] += 1
        L.append("SI messages: " + "; ".join(f"{k} x{v}" for k, v in seen.items()))
    if rep.paging or rep.rar:
        L.append(f"Paging (P-RNTI): {rep.paging} subframes, random access responses (RA-RNTI): {rep.rar}")
    if rep.rar_entries:
        d = sorted(e.distance_m for e in rep.rar_entries)
        L.append(f"RAR: {len(rep.rar_entries)} responses, timing advance -> UE distance "
                 f"{d[0]:.0f}..{d[-1]:.0f} m (median {d[len(d) // 2]:.0f} m)")
    if rep.paging_msgs:
        pm = rep.paging_msgs
        L.append(f"Paging decoded: {len(pm)} messages, records: {sum(p.s_tmsi for p in pm)} S-TMSI, "
                 f"{sum(p.imsi for p in pm)} IMSI (identities masked), "
                 f"{sum(p.cs_domain for p in pm)} CS domain; "
                 f"systemInfoModification: {'yes' if any(p.si_modification for p in pm) else 'no'}"
                 + (", ETWS" if any(p.etws for p in pm) else ""))
    conf = rep.confirmed_rntis()
    if conf:
        top = ", ".join(f"0x{r:04x}x{n}" for r, n in list(conf.items())[:20])
        L.append(f"Active C-RNTIs ({len(conf)}): {top}")
    if show_dci and rep.dcis:
        L.append("DCI:")
        for ev in rep.dcis[:show_dci]:
            L.append(f"  SFN {ev.sfn:4d}.{ev.sf} {rnti_type(ev.rnti):8s} 0x{ev.rnti:04x} "
                     f"CCE {ev.cce:3d} L{ev.agg} {ev.text}")
    for n in rep.notes:
        L.append(f"Note: {n}")
    return "\n".join(L)


def burst_line(stamp: str, n: int, rep: CellReport, counts: dict) -> str:
    """One status line per monitor burst."""
    m = rep.mib
    if m is None:
        return f"{stamp} burst {n}: PCI {rep.sync.n_id} MIB not decoded"
    cfi = rep.cfi_hist.most_common(1)[0][0] if rep.cfi_hist else "-"
    load = np.mean(rep.cce_load) * 100 if rep.cce_load else 0
    s = (f"{stamp} burst {n}: PCI {rep.sync.n_id} SFN {m.sfn:4d} "
         f"RSRP {rep.rsrp_dbfs:6.1f} dBFS SINR {rep.snr_db:5.1f} dB CFI {cfi} CCE {load:3.0f}% | "
         f"SI {counts.get('SI', 0)} RAR {counts.get('RAR', 0)} PCH {counts.get('PCH', 0)}")
    conf = rep.confirmed_rntis()
    if conf:
        s += f" C-RNTI {len(conf)}"
    return s


def scan_table(reports: list[CellReport]) -> str:
    if not reports:
        return "No LTE cells found."
    hdr = (f"{'Freq MHz':>10} {'EARFCN':>6} {'Band':>4} {'PCI':>4} {'Dup':>3} {'CP':>4} "
           f"{'BW':>5} {'Ant':>3} {'RSRP dBFS':>9} {'SNR dB':>6} {'CFO Hz':>7}  PLMN / TAC / CellID")
    L = [hdr, "-" * len(hdr)]
    for r in sorted(reports, key=lambda r: (r.fc or 0, r.sync.n_id)):
        m = r.mib
        bw = f"{BANDWIDTH_MHZ[m.n_rb]}M" if m else "?"
        ant = str(m.n_ports) if m else "?"
        sib = ""
        if r.sib1:
            sib = (f"{','.join(str(p) for p in r.sib1.plmns)} / {r.sib1.tac} / "
                   f"{r.sib1.enb_id}-{r.sib1.local_cell_id}")
        L.append(f"{(r.fc or 0) / 1e6:10.3f} {r.earfcn if r.earfcn is not None else '-':>6} "
                 f"{r.band if r.band else '-':>4} {r.sync.n_id:4d} {r.sync.duplex:>3} "
                 f"{'ext' if r.sync.cp_ext else 'norm':>4} {bw:>5} {ant:>3} {r.rsrp_dbfs:9.1f} "
                 f"{r.snr_db:6.1f} {r.sync.cfo:7.0f}  {sib}")
    return "\n".join(L)


def to_dict(rep: CellReport) -> dict:
    s = rep.sync
    d = {
        "freq_hz": rep.fc, "earfcn": rep.earfcn, "band": rep.band,
        "pci": s.n_id, "duplex": s.duplex, "cp": s.cp, "cfo_hz": s.cfo,
        "receiver_ppm": None if np.isnan(rep.ppm) else rep.ppm,
        "pss_metric": s.pss_metric, "sss_metric": s.sss_metric,
        "rsrp_dbfs": rep.rsrp_dbfs, "rsrq_db": rep.rsrq_db, "sinr_db": rep.snr_db,
        "mib": None, "sib1": None, "si": [], "cfi": dict(rep.cfi_hist),
        "subframes": rep.subframes, "paging_subframes": rep.paging, "rar": rep.rar,
        "cce_load": float(np.mean(rep.cce_load)) if rep.cce_load else None,
        "c_rnti": {f"0x{r:04x}": n for r, n in rep.confirmed_rntis().items()},
        "rar_responses": [{"rapid": e.rapid, "ta": e.ta, "distance_m": round(e.distance_m),
                           "temp_crnti": f"0x{e.temp_crnti:04x}"} for e in rep.rar_entries],
        "paging_messages": [p.__dict__ for p in rep.paging_msgs],
        "notes": rep.notes,
    }
    if rep.mib:
        m = rep.mib
        d["mib"] = {"n_rb": m.n_rb, "n_ports": m.n_ports, "phich_ext": m.phich_ext,
                    "phich_ng": PHICH_NG_NAME[m.phich_ng], "sfn": m.sfn}
    if rep.sib1:
        b = rep.sib1
        d["sib1"] = {
            "plmn": [str(p) for p in b.plmns], "tac": b.tac, "cell_identity": b.cell_identity,
            "enb_id": b.enb_id, "cell_id": b.local_cell_id, "band": b.band,
            "cell_barred": b.cell_barred, "q_rx_lev_min_dbm": b.q_rx_lev_min_dbm,
            "p_max": b.p_max, "si_window_ms": b.si_window_ms, "value_tag": b.value_tag,
            "scheduling": [{"periodicity_rf": p, "sibs": s} for p, s in b.scheduling],
            "tdd_config": b.tdd_config, "raw_hex": rep.sib1_raw.hex() if rep.sib1_raw else None,
        }
    for sfn, sf, si, raw in rep.si:
        d["si"].append({"sfn": sfn, "sf": sf, "sibs": si.sibs, "n_sibs": si.n_sibs,
                        "raw_hex": raw.hex()})
    return d


def save_json(path, reports: list[CellReport], extra: dict | None = None):
    out = {"cells": [to_dict(r) for r in reports]}
    out.update(extra or {})
    Path(path).write_text(json.dumps(out, indent=2, default=_conv), encoding="utf-8")


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# ----------------------------------------------------------------- plots
def plot_spectrum(path, spectra, carriers=(), reports=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    for f, p in spectra:
        ax.plot(f / 1e6, p, lw=0.6, color="C0")
    for c in carriers:
        ax.axvspan(c.lo / 1e6, c.hi / 1e6, color="C1", alpha=0.15)
    for r in reports:
        ax.axvline(r.fc / 1e6, color="C3", lw=0.8)
        ax.text(r.fc / 1e6, ax.get_ylim()[1], f"PCI {r.sync.n_id}", rotation=90,
                va="top", ha="right", fontsize=8)
    ax.set_xlabel("MHz")
    ax.set_ylabel("dB")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cell(path_prefix, grid: np.ndarray, pbch_syms: np.ndarray | None = None,
              pdsch_syms: np.ndarray | None = None, h: np.ndarray | None = None):
    """Resource grid magnitude, constellations, channel estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axs[0]
    ax.imshow(20 * np.log10(np.abs(grid.T) + 1e-6), aspect="auto", origin="lower",
              cmap="viridis", interpolation="nearest")
    ax.set_title("Resource grid (one frame, |Y| dB)")
    ax.set_xlabel("OFDM symbol")
    ax.set_ylabel("subcarrier")
    ax = axs[1]
    if pbch_syms is not None:
        ax.plot(pbch_syms.real, pbch_syms.imag, ".", ms=2, label="PBCH")
    if pdsch_syms is not None:
        ax.plot(pdsch_syms.real, pdsch_syms.imag, ".", ms=2, label="PDSCH (SIB1)")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Constellations after equalization")
    ax = axs[2]
    if h is not None:
        for p in range(h.shape[0]):
            ax.plot(20 * np.log10(np.abs(h[p, 0]) + 1e-9), lw=0.8, label=f"port {p}")
        ax.legend()
    ax.set_title("Channel estimate |H| (symbol 0)")
    ax.set_xlabel("subcarrier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_cell.png", dpi=120)
    plt.close(fig)
