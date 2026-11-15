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
    comb = max((o.combined for o in c.obs if o.mib), default=1)
    L.append(f"MIB: SFN {m.sfn}, half frame {m.half_frame}, SSB index {m.ssb_index}, "
             f"subCarrierSpacingCommon {m.scs_common // 1000} kHz, "
             f"dmrs-TypeA-Position pos{m.dmrs_type_a_pos}"
             + (f" (soft-combined from {comb} blocks)" if comb > 1 else ""))
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


def sib1_text(r) -> str:
    """Report of a Sib1Result."""
    c0 = r.coreset
    L = []
    if c0 is not None:
        L.append(f"CORESET#0: {c0.n_rb} RB x {c0.n_symb} symbol(s), {c0.scs // 1000} kHz, "
                 f"{c0.f_low / 1e6:.3f}-{(c0.f_low + c0.bandwidth) / 1e6:.3f} MHz; "
                 f"search space #0: O = {c0.o:g} ms, M = {c0.m:g}")
    if r.sib1 is None:
        msg = r.error or "not decoded"
        extra = (f" ({len(r.dcis)} DCI 1_0 with SI-RNTI in {r.slots_tried} slots)"
                 if r.slots_tried else "")
        L.insert(0, f"SIB1: {msg}{extra}")
        return "\n".join("     " + x if i else x for i, x in enumerate(L))
    a = r.alloc
    mod = {2: "QPSK", 4: "16QAM", 6: "64QAM"}[a.qm]
    lv = sorted({d.level for d in r.dcis})
    rv = ",".join(str(v) for v in r.rvs) if len(r.rvs) <= 4 else         "/".join(str(v) for v in sorted(set(r.rvs)))
    how = "" if r.combined == 1 else ", soft-combined"
    L.insert(0, f"SIB1: decoded from {r.combined} transmission(s){how} (RV {rv}), "
                f"PDSCH SINR {r.pdsch_sinr_db:.1f} dB")
    L.append(f"PDSCH: DCI 1_0 SI-RNTI x{len(r.dcis)} (AL {'/'.join(map(str, lv))}), "
             f"MCS {a.dci.mcs} {mod}, {len(a.prbs)} PRB, symbols {a.s}-{a.s + a.length - 1} "
             f"(mapping {a.mapping}), TBS {a.tbs} bits")
    s = r.sib1
    for pi in s.plmn_info:
        plmns = ", ".join(f"{mcc or '(same)'}-{mnc}" for mcc, mnc in pi.plmns)
        tac = f"{pi.tac} (0x{pi.tac:06X})" if pi.tac is not None else "-"
        L.append(f"PLMN {plmns}: TAC {tac}, NR cell identity {pi.cell_identity} "
                 f"(0x{pi.cell_identity:09X})"
                 + (", reserved for operator use" if pi.reserved_for_operator else ""))
    sel = []
    if s.q_rx_lev_min is not None:
        sel.append(f"q-RxLevMin {s.q_rx_lev_min} dBm")
    if s.q_qual_min is not None:
        sel.append(f"q-QualMin {s.q_qual_min} dB")
    if s.reserved_for_other_use:
        sel.append("cellReservedForOtherUse")
    if sel:
        L.append("Cell selection: " + ", ".join(sel))
    if s.si:
        si = "; ".join(f"{'+'.join(sibs)} every {per} frames" for per, sibs in s.si)
        L.append(f"SI scheduling: {si}; window {s.si_window_slots} slots")
    if s.bands or s.carriers:
        car = []
        for scs, n_rb, lo, ctr in r.carriers():
            bw = next((c.channel_mhz for c in s.carriers if c.scs_khz == scs and c.n_rb == n_rb),
                      None)
            car.append(f"{n_rb} PRB x {scs} kHz" + (f" ({bw} MHz channel)" if bw else "")
                       + f", centre {ctr / 1e6:.3f} MHz")
        pa = r.point_a
        L.append("Carrier: band " + "/".join(f"n{b}" for b in s.bands)
                 + (f", point A {pa / 1e6:.3f} MHz" if pa else "")
                 + ("; " + "; ".join(car) if car else ""))
    if not s.complete:
        L.append("(rest of SIB1 not parsed: si-RequestConfig present)")
    L += si_lines(r.si)
    return "\n".join("     " + x if i else x for i, x in enumerate(L))


def si_lines(msgs) -> list[str]:
    out = []
    for m in msgs:
        head = f"SI {m.index} ({'+'.join(m.sibs)}, every {m.period * 10} ms): "
        if m.payload is None:
            out.append(head + f"not decoded - {m.error}"
                       + (f" ({m.windows} window(s), {m.slots_tried} slots)" if m.windows else ""))
            continue
        a = m.alloc
        mod = {2: "QPSK", 4: "16QAM", 6: "64QAM"}[a.qm]
        rv = ",".join(map(str, m.rvs)) if len(m.rvs) <= 4 else \
            "/".join(map(str, sorted(set(m.rvs))))
        out.append(head + f"decoded from {m.combined} transmission(s) (RV {rv}), MCS {a.dci.mcs} "
                   f"{mod}, {len(a.prbs)} PRB, TBS {a.tbs} bits, PDSCH SINR {m.pdsch_sinr_db:.1f} dB")
        info = m.info
        if info is None:
            if m.error:
                out.append("  " + m.error)
            continue
        if info.first != (m.sibs[0] if m.sibs else info.first):
            out.append(f"  first element {info.first} (SIB1 maps {'+'.join(m.sibs)})")
        b = info.sib2
        if b is not None:
            out.append(f"  SIB2: q-Hyst {b.q_hyst_db} dB, cellReselectionPriority "
                       f"{b.cell_reselection_priority}, threshServingLowP "
                       f"{2 * b.thresh_serving_low_p} dB"
                       + (f", s-NonIntraSearchP {2 * b.s_non_intra_search_p} dB"
                          if b.s_non_intra_search_p is not None else "")
                       + f", q-RxLevMin {b.q_rx_lev_min} dBm, s-IntraSearchP "
                       f"{2 * b.s_intra_search_p} dB, t-ReselectionNR {b.t_reselection_nr} s")
    if msgs and any(m.payload for m in msgs):
        out.append("  (full SIB contents: --pcap and Wireshark)")
    return out


def si_dict(m) -> dict:
    d = {"index": m.index, "period_frames": m.period, "sibs": m.sibs, "decoded": m.payload is not None,
         "error": m.error, "windows": m.windows, "dci_count": len(m.dcis)}
    if m.alloc is not None:
        a = m.alloc
        d["pdsch"] = {"mcs": a.dci.mcs, "qm": a.qm, "n_prb": len(a.prbs), "tbs": a.tbs,
                      "transmissions": m.combined}
    if m.payload is not None:
        d["payload_hex"] = m.payload.hex()
    if m.info is not None:
        d["n_sibs"] = m.info.n_sibs
        d["first_sib"] = m.info.first
        b = m.info.sib2
        if b is not None:
            d["sib2"] = {"q_hyst_db": b.q_hyst_db, "cell_reselection_priority":
                         b.cell_reselection_priority, "thresh_serving_low_p_db":
                         2 * b.thresh_serving_low_p,
                         "s_non_intra_search_p_db": None if b.s_non_intra_search_p is None
                         else 2 * b.s_non_intra_search_p,
                         "q_rx_lev_min_dbm": b.q_rx_lev_min,
                         "s_intra_search_p_db": 2 * b.s_intra_search_p,
                         "t_reselection_nr_s": b.t_reselection_nr}
    return d


def sib1_dict(r) -> dict:
    c0 = r.coreset
    out = {"decoded": r.sib1 is not None, "error": r.error,
           "dci_count": len(r.dcis), "slots_tried": r.slots_tried}
    if c0 is not None:
        out["coreset0"] = {"n_rb": c0.n_rb, "n_symb": c0.n_symb, "scs_khz": c0.scs // 1000,
                           "f_low_mhz": round(c0.f_low / 1e6, 4), "offset_rb": c0.offset,
                           "ss0_o_ms": c0.o, "ss0_m": c0.m}
    if r.alloc is not None:
        a = r.alloc
        out["pdsch"] = {"mcs": a.dci.mcs, "qm": a.qm, "n_prb": len(a.prbs), "start_symbol": a.s,
                        "n_symbols": a.length, "tbs": a.tbs, "transmissions": r.combined,
                        "sinr_db": None if r.pdsch_sinr_db is None else round(r.pdsch_sinr_db, 1)}
    if r.payload is not None:
        out["payload_hex"] = r.payload.hex()
    if r.si:
        out["si_messages"] = [si_dict(m) for m in r.si]
    s = r.sib1
    if s is not None:
        out["plmn_info"] = [{"plmns": [f"{m}-{n}" if m else n for m, n in pi.plmns],
                             "tac": pi.tac, "ranac": pi.ranac, "nr_cell_identity": pi.cell_identity,
                             "reserved_for_operator_use": pi.reserved_for_operator}
                            for pi in s.plmn_info]
        out["q_rx_lev_min_dbm"] = s.q_rx_lev_min
        out["q_qual_min_db"] = s.q_qual_min
        out["si_scheduling"] = [{"period_frames": p, "sibs": sibs} for p, sibs in s.si]
        out["si_window_slots"] = s.si_window_slots
        out["bands"] = s.bands
        out["offset_to_point_a"] = s.offset_to_point_a
        out["point_a_mhz"] = None if r.point_a is None else round(r.point_a / 1e6, 4)
        out["carriers"] = [{"scs_khz": c.scs_khz, "n_rb": c.n_rb, "offset_to_carrier":
                            c.offset_to_carrier, "channel_mhz": c.channel_mhz}
                           for c in s.carriers]
    return out
