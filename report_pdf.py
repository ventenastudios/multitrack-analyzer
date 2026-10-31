"""Report PDF (reportlab + matplotlib) per l'Album Mastering Analyzer - Ventena Studios."""
import datetime as dt
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from album_analysis import THIRD_FC
from album_tables import (BAND_METRICS, DEFAULT_THRESH, GLOBAL_COLS, LABEL, album_mean, albums, band_value,
                          deviation_df, spectrum_norm, target_status)

AMBER = colors.HexColor("#e8a020")
DARK = colors.HexColor("#0a0a09")
GREY = colors.HexColor("#2a2a26")
LOGO_CANDIDATES = ["assets/logo.png", "logo.png", "assets/logo_ventena.png", "assets/ventenastudios.png",
                   "static/logo.png", "images/logo.png"]
PALETTE = list(plt.get_cmap("tab20").colors)


def find_logo(base_dirs):
    for d in base_dirs:
        for c in LOGO_CANDIDATES:
            p = os.path.join(d, c)
            if os.path.isfile(p):
                return p
    return None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _color(i, ref):
    return "#111111" if ref else PALETTE[i % len(PALETTE)]


def _style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--"); ax.tick_params(labelsize=7)


def fig_spectrum(tracks, mode, deviation=False):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    mean = album_mean(tracks, lambda t: spectrum_norm(t["g"]["third_db"], mode))
    for i, t in enumerate(tracks):
        y = spectrum_norm(t["g"]["third_db"], mode)
        if deviation:
            y = y - mean
        ax.semilogx(THIRD_FC, y, color=_color(i, t["ref"]), lw=1.1 if not t["ref"] else 1.6,
                    ls="--" if t["ref"] else "-", label=("REF " if t["ref"] else "") + t["name"][:28])
    if not deviation and mean is not None:
        ax.semilogx(THIRD_FC, mean, color="#e8a020", lw=2.6, label="Media album")
    else:
        ax.axhline(0, color="#e8a020", lw=2)
    _style_ax(ax, "Frequenza [Hz]", "Scostamento dalla media album [dB]" if deviation else "Livello 1/3 ottava [dB]")
    ax.set_xlim(20, 20000)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
    ax.legend(fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return fig


def fig_bands(tracks, weighting, metric="rel"):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    labels = tracks[0]["b"]["labels"]
    xs = np.arange(len(labels))
    for i, t in enumerate(tracks):
        y = [band_value(t["g"], bd, metric, weighting) for bd in t["b"]["bands"]]
        ax.plot(xs, y, marker="o", ms=3, color=_color(i, t["ref"]), ls="--" if t["ref"] else "-",
                lw=1.0 if not t["ref"] else 1.6, label=("REF " if t["ref"] else "") + t["name"][:28])
    m = album_mean(tracks, lambda t: [band_value(t["g"], bd, metric, weighting) for bd in t["b"]["bands"]])
    if m is not None:
        ax.plot(xs, m, color="#e8a020", lw=2.6, marker="s", label="Media album")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    _style_ax(ax, "Sottobanda", BAND_METRICS[metric])
    ax.legend(fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig


def fig_timeline(tracks):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, t in enumerate(tracks):
        tl = t["g"]["timeline"]
        ax.plot(tl["t"], tl["st_lufs"], color=_color(i, t["ref"]), lw=0.9, ls="--" if t["ref"] else "-",
                label=("REF " if t["ref"] else "") + t["name"][:28])
    _style_ax(ax, "Tempo [s]", "Loudness short-term [LUFS]")
    ax.legend(fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return fig


def fig_track_spectrum(tracks, t, mode):
    fig, ax = plt.subplots(figsize=(10, 3.0))
    mean = album_mean(tracks, lambda x: spectrum_norm(x["g"]["third_db"], mode))
    y = spectrum_norm(t["g"]["third_db"], mode)
    ax.semilogx(THIRD_FC, y, color="#1f77b4", lw=2, label=t["name"][:40])
    if mean is not None:
        ax.semilogx(THIRD_FC, mean, color="#e8a020", lw=2, label="Media album")
    for r in [x for x in tracks if x["ref"]]:
        ax.semilogx(THIRD_FC, spectrum_norm(r["g"]["third_db"], mode), color="#111111", ls="--", lw=1.4,
                    label="REF " + r["name"][:28])
    _style_ax(ax, "Frequenza [Hz]", "Livello 1/3 ottava [dB]")
    ax.set_xlim(20, 20000)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
    ax.legend(fontsize=7)
    return fig


def _img(buf, max_w, max_h):
    """Immagine che entra in max_w x max_h mantenendo le proporzioni."""
    from PIL import Image as PILImage
    w, h = PILImage.open(buf).size
    buf.seek(0)
    k = min(max_w / w, max_h / h)
    return Image(buf, width=w * k, height=h * k)


def _fmt(v, d=1):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def build_pdf(tracks, settings, album_title="Album", notes="", logo_path=None, per_track_pages=True):
    """tracks: lista di {'name','ref','g','b'}; settings: dict con weighting, spec_norm, target_lufs, tol_lu,
    target_tp, thresholds. Restituisce i bytes del PDF."""
    buf = io.BytesIO()
    page = landscape(A4)
    W, H = page
    mode, weighting = settings["spec_norm"], settings["weighting"]
    thresh = {**DEFAULT_THRESH, **settings.get("thresholds", {})}

    def deco(c, doc):
        c.saveState()
        c.setFillColor(DARK); c.rect(0, H - 16 * mm, W, 16 * mm, stroke=0, fill=1)
        x0 = 12 * mm
        if logo_path and os.path.isfile(logo_path):
            try:
                c.drawImage(logo_path, x0, H - 14 * mm, height=12 * mm, width=36 * mm, preserveAspectRatio=True, mask="auto")
                x0 += 40 * mm
            except Exception:
                pass
        c.setFillColor(AMBER); c.setFont("Helvetica-Bold", 14)
        c.drawString(x0, H - 10.5 * mm, "VENTENA STUDIOS")
        c.setFillColor(colors.HexColor("#e8e4d8")); c.setFont("Helvetica", 9)
        c.drawRightString(W - 12 * mm, H - 10.5 * mm, "Album Mastering Analyzer - Report tecnico")
        c.setFillColor(colors.grey); c.setFont("Helvetica", 7.5)
        c.drawString(12 * mm, 7 * mm, "Ventena Studios Toolkit - analisi 100% privata, nessun file viene conservato")
        c.drawRightString(W - 12 * mm, 7 * mm, f"Pagina {doc.page}")
        c.restoreState()

    doc = BaseDocTemplate(buf, pagesize=page, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=22 * mm,
                          bottomMargin=13 * mm, title=f"{album_title} - Mastering report", author="Ventena Studios")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(12 * mm, 13 * mm, W - 24 * mm, H - 35 * mm, id="f")],
                                       onPage=deco)])
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=18, textColor=colors.HexColor("#111111"), spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#b97a00"), spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.HexColor("#444444"))
    imgw = W - 24 * mm
    story = []

    # ---- pagina 1: sintesi
    story += [Paragraph(album_title, h1),
              Paragraph(dt.datetime.now().strftime("Report del %d/%m/%Y %H:%M"), small), Spacer(1, 3 * mm)]
    xo = tracks[0]["b"]["crossovers"]
    cfg = (f"Target: {settings['target_lufs']:g} LUFS (+/- {settings['tol_lu']:g} LU), true peak max {settings['target_tp']:g} dBTP. "
           f"Crossover sottobande: {', '.join(f'{v:g} Hz' for v in xo)} (Linkwitz-Riley 4° ordine). "
           f"Ponderazione bande: {'K-weighting (LUFS)' if weighting == 'k' else 'RMS lineare'}. "
           f"Normalizzazione spettro: {dict(total='livello totale', mid='media 200 Hz-5 kHz', abs='nessuna')[mode]}. "
           f"Brani analizzati: {len(albums(tracks))}" + (f", reference: {len([t for t in tracks if t['ref']])}." if any(t['ref'] for t in tracks) else "."))
    story += [Paragraph(cfg, body), Spacer(1, 3 * mm)]
    if notes.strip():
        story += [Paragraph("<b>Note:</b> " + notes.replace("\n", "<br/>"), body), Spacer(1, 3 * mm)]

    dev, flag = deviation_df(tracks, weighting, thresh)
    hdr = ["#", "Brano", "LUFS", "TP dBTP", "LRA", "Crest", "PLR", "Corr", "Width %", "Tilt", "Stato / note"]
    rows = [hdr]
    flagcells = []
    keymap = {2: "lufs_i", 4: "lra", 5: "crest_db", 6: "plr", 7: "corr", 8: "width_pct", 9: "tilt"}
    for r, t in enumerate(tracks, 1):
        g = t["g"]
        ok, nts = target_status(g, settings["target_lufs"], settings["tol_lu"], settings["target_tp"])
        rows.append(["REF" if t["ref"] else str(sum(1 for x in tracks[:r] if not x["ref"])), t["name"][:34],
                     _fmt(g["lufs_i"]), _fmt(g["tp_db"], 2), _fmt(g["lra"]), _fmt(g["crest_db"]), _fmt(g["plr"]),
                     _fmt(g["corr"], 2), _fmt(g["width_pct"]), _fmt(g["tilt"], 2),
                     "-" if t["ref"] else ("OK" if ok else "; ".join(nts))[:48]])
        for c, k in keymap.items():
            if flag[LABEL[k]].iloc[r - 1] and not t["ref"]:
                flagcells.append((c, r))
    m = {k: album_mean(tracks, lambda t, k=k: t["g"][k]) for k in ("lufs_i", "tp_db", "lra", "crest_db", "plr", "corr", "width_pct", "tilt")}
    rows.append(["", "MEDIA ALBUM", _fmt(m["lufs_i"]), _fmt(m["tp_db"], 2), _fmt(m["lra"]), _fmt(m["crest_db"]),
                 _fmt(m["plr"]), _fmt(m["corr"], 2), _fmt(m["width_pct"]), _fmt(m["tilt"], 2), ""])
    tbl = Table(rows, repeatRows=1, colWidths=[9 * mm, 62 * mm, 16 * mm, 17 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 17 * mm, 14 * mm, None])
    st = [("BACKGROUND", (0, 0), (-1, 0), GREY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"), ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f6e7c4")),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")), ("ALIGN", (2, 0), (9, -1), "CENTER"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f4f4f2")]),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    st += [("BACKGROUND", (c, r), (c, r), colors.HexColor("#ffd9a8")) for c, r in flagcells]
    tbl.setStyle(TableStyle(st))
    story += [tbl, Spacer(1, 2 * mm),
              Paragraph("Celle arancioni: scostamento dalla media album oltre la soglia (LUFS +/-{lufs_i:g} LU, LRA +/-{lra:g} LU, crest +/-{crest_db:g} dB, "
                        "PLR +/-{plr:g} dB, correlazione +/-{corr:g}, width +/-{width_pct:g} %, tilt +/-{tilt:g} dB/ott).".format(**thresh), small)]

    # ---- pagina 2-3: grafici album
    story += [PageBreak(), Paragraph("Spettro a terzi d'ottava", h2), _img(_png(fig_spectrum(tracks, mode)), imgw, 74 * mm),
              _img(_png(fig_spectrum(tracks, mode, deviation=True)), imgw, 74 * mm)]
    story += [PageBreak(), Paragraph("Bilanciamento sottobande (livello relativo al totale)", h2),
              _img(_png(fig_bands(tracks, weighting, "rel")), imgw, 72 * mm),
              Paragraph("Crest factor per sottobanda", h2),
              _img(_png(fig_bands(tracks, weighting, "crest_db")), imgw, 72 * mm)]
    story += [PageBreak(), Paragraph("Andamento della loudness (short-term, 3 s)", h2),
              _img(_png(fig_timeline(tracks)), imgw, 70 * mm)]

    # ---- scostamenti per banda (tabella)
    story += [Paragraph("Scostamento dalla media album (sottobande, livello relativo)", h2)]
    bcols = [c for c in dev.columns if c.startswith("Banda")]
    brow = [["Brano"] + [c.replace("Banda ", "").replace(" rel. [dB]", "") for c in bcols]]
    cells = []
    for r, (name, row) in enumerate(dev[bcols].iterrows(), 1):
        brow.append([name[:40]] + [f"{v:+.1f}" for v in row.values])
        cells += [(c + 1, r) for c, f in enumerate(flag[bcols].iloc[r - 1].values) if f]
    bt = Table(brow, repeatRows=1, colWidths=[90 * mm] + [32 * mm] * len(bcols))
    bt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                            ("ALIGN", (1, 0), (-1, -1), "CENTER")] +
                           [("BACKGROUND", (c, r), (c, r), colors.HexColor("#ffd9a8")) for c, r in cells]))
    story += [bt, Paragraph(f"Soglia di segnalazione: +/-{thresh['band_rel']:g} dB rispetto alla media album.", small)]

    # ---- pagine per brano
    if per_track_pages:
        for i, t in enumerate(tracks):
            g = t["g"]
            story += [PageBreak(), Paragraph(("REFERENCE - " if t["ref"] else f"{sum(1 for x in tracks[:i + 1] if not x['ref'])}. ") + t["name"], h2)]
            mrows = []
            for k, lab, u, d in GLOBAL_COLS:
                mrows.append((lab + (f" [{u}]" if u else ""), _fmt(g[k], d)))
            half = (len(mrows) + 1) // 2
            data = [[a[0], a[1], b[0] if b else "", b[1] if b else ""] for a, b in
                    zip(mrows[:half], mrows[half:] + [None] * (half - len(mrows[half:])))]
            mt = Table(data, colWidths=[62 * mm, 24 * mm, 62 * mm, 24 * mm])
            mt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.2), ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#cccccc")),
                                    ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("ALIGN", (3, 0), (3, -1), "RIGHT"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
            btab = [["Banda", "Rel. [dB]", "Livello [dB]", "Quota %", "Crest [dB]", "Dinamica [dB]", "Width %", "Corr"]]
            for bd in t["b"]["bands"]:
                btab.append([bd["label"], _fmt(band_value(g, bd, "rel", weighting)), _fmt(band_value(g, bd, "level", weighting)),
                             _fmt(bd["share_pct"]), _fmt(bd["crest_db"]), _fmt(band_value(g, bd, "dyn", weighting)),
                             _fmt(bd["width_pct"]), _fmt(bd["corr"], 2)])
            bt2 = Table(btab, colWidths=[34 * mm] + [24 * mm] * 7)
            bt2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                     ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                                     ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
            story += [mt, Spacer(1, 2 * mm), bt2, Spacer(1, 2 * mm)]
            if g["sections"]:
                story += [Paragraph("<b>Passaggi da riascoltare:</b> " + " | ".join(f"{lab} a {int(tt // 60)}:{int(tt % 60):02d} ({v})" for lab, tt, v in
                                                                                    (g["sections"] + t["b"]["sections"])[:6]), small), Spacer(1, 2 * mm)]
            story += [_img(_png(fig_track_spectrum(tracks, t, mode)), imgw * 0.8, 50 * mm)]

    doc.build(story)
    return buf.getvalue()
