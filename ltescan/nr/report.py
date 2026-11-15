"""Text and JSON reports of NR cells."""
from __future__ import annotations

from .search import NrCell


def cell_text(c: NrCell) -> str:
    L = []
    head = (f"NR cell PCI {c.n_id} (N_ID1={c.n_id // 3}, N_ID2={c.n_id % 3}) @ "
            f"{c.f_ref / 1e6:.3f} MHz, GSCN {c.gscn}, SSB ARFCN {c.arfcn}, band {c.band}")
    L.append(head)
    L.append("-" * len(head))
    b = c.best
    m = c.mib
    beams = ", ".join(f"#{i} {p:.1f}" for i, p in c.beams.items())
    L.append(f"SSB: SCS {c.scs // 1000} kHz"
             + (f", L_max {m.l_max}" if m else "")
             + f", blocks received: {len(c.obs)}, SS-RSRP per SSB index [dBFS]: {beams}")
    L.append(f"Sync: PSS metric {b.pss_metric:.2f}, SSS margin {b.sss_margin:.2f}, "
             f"DMRS margin {b.dmrs_margin:.2f}")
    L.append(f"CFO: {c.cfo:+.0f} Hz, receiver clock error ~ {c.ppm:+.2f} ppm")
    L.append(f"SS-RSRP: {c.rsrp_dbfs:.1f} dBFS/RE, SS-SINR: {c.sinr_db:.1f} dB")
    if m is None:
        L.append("MIB: not decoded (PBCH CRC failed) - cell not confirmed")
        return "\n".join(L)
    L.append(f"MIB: SFN {m.sfn}, half frame {m.half_frame}, SSB index {m.ssb_index}, "
             f"subCarrierSpacingCommon {m.scs_common // 1000} kHz, "
             f"dmrs-TypeA-Position pos{m.dmrs_type_a_pos}")
    L.append(f"     k_SSB {m.k_ssb}, controlResourceSetZero {m.coreset0}, "
             f"searchSpaceZero {m.search_space0}, "
             f"cellBarred: {'barred' if m.cell_barred else 'notBarred'}, "
             f"intraFreqReselection: {'allowed' if m.intra_freq_reselection else 'notAllowed'}")
    if m.sib1_present:
        L.append("SIB1: scheduled (CORESET#0 present) - standalone-capable cell")
    elif m.k_ssb == 31:
        L.append("SIB1: none (k_SSB = 31, no CORESET#0) - typical of a non-standalone (EN-DC) cell")
    elif m.k_ssb >= 24:
        L.append(f"SIB1: none on this SSB; k_SSB {m.k_ssb} points to another GSCN that carries "
                 f"one (TS 38.213 13)")
    else:
        L.append("SIB1: none (k_SSB reserved)")
    return "\n".join(L)


def scan_table(cells: list[NrCell]) -> str:
    L = [f"{'GSCN':>5} {'SSB MHz':>9} {'ARFCN':>7} {'Band':>4} {'PCI':>4} {'RSRP':>6} "
         f"{'SINR':>5} {'SFN':>4} {'kSSB':>4} {'SIB1':>4} {'Barred':>6} {'SSBs':>4}"]
    for c in cells:
        m = c.mib
        L.append(f"{c.gscn:>5} {c.f_ref / 1e6:>9.3f} {c.arfcn:>7} {c.band:>4} {c.n_id:>4} "
                 f"{c.rsrp_dbfs:>6.1f} {c.sinr_db:>5.1f} "
                 + (f"{m.sfn:>4} {m.k_ssb:>4} {'yes' if m.sib1_present else 'no':>4} "
                    f"{'yes' if m.cell_barred else 'no':>6} " if m else f"{'-':>4} " * 3 + f"{'-':>6} ")
                 + f"{len(c.beams):>4}")
    return "\n".join(L)


def to_dict(c: NrCell) -> dict:
    m = c.mib
    return {
        "rat": "NR", "pci": c.n_id, "gscn": c.gscn, "ssb_freq_mhz": round(c.f_ref / 1e6, 4),
        "ssb_arfcn": c.arfcn, "band": c.band, "scs_khz": c.scs // 1000,
        "confirmed": c.confirmed, "cfo_hz": round(c.cfo), "ppm": round(c.ppm, 2),
        "ss_rsrp_dbfs": round(c.rsrp_dbfs, 1), "ss_sinr_db": round(c.sinr_db, 1),
        "ssb_rsrp_dbfs": {str(i): round(p, 1) for i, p in c.beams.items()},
        "mib": None if m is None else {
            "sfn": m.sfn, "half_frame": m.half_frame, "ssb_index": m.ssb_index, "l_max": m.l_max,
            "scs_common_khz": m.scs_common // 1000, "k_ssb": m.k_ssb,
            "dmrs_type_a_position": m.dmrs_type_a_pos, "coreset0": m.coreset0,
            "search_space0": m.search_space0, "cell_barred": m.cell_barred,
            "intra_freq_reselection_allowed": m.intra_freq_reselection,
            "sib1_present": m.sib1_present},
    }
