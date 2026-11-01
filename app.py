import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "pages":  # funziona sia come pagina in pages/ sia come app.py principale
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import album_analysis as A
import album_tables as T
from report_pdf import build_pdf, find_logo

st.set_page_config(page_title="Album Mastering Analyzer", layout="wide")

st.markdown("""
    <style>
    :root { --bg: #ffffff; --bg2: #f6f4ee; --amber: #b97a00; --text: #1a1a17; --border: #ddd9cf; }
    .stApp { background-color: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
    h1 { color: #b97a00 !important; font-family: 'Bebas Neue', sans-serif !important; }
    h2, h3 { font-family: 'Bebas Neue', sans-serif !important; color: var(--text); }
    div[data-testid="stFileUploader"] label { color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
    [data-testid="stMetric"] { background-color: var(--bg2) !important; border: 1px solid var(--border) !important; padding: 8px 12px; border-radius: 8px; }
    [data-testid="stMetricLabel"] { color: var(--text) !important; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: var(--amber) !important; font-family: 'DM Mono', monospace; }
    [data-testid="stFileUploader"] button p { color: var(--amber) !important; }
    button[data-baseweb="tab"] p { font-size: 1.0rem; }
    </style>
    """, unsafe_allow_html=True)


def centered_title(text):
    try:
        st.title(text, width="stretch", text_alignment="center")
    except TypeError:  # versioni piu' vecchie di Streamlit
        st.markdown(f"<h1 style='text-align:center'>{text}</h1>", unsafe_allow_html=True)


centered_title("VENTENA STUDIOS TOOLKIT")
centered_title("🎼 Album Mastering Analyzer 🎼")
st.write("Controllo finale del mastering di un album: loudness, true peak, dinamica, stereo, spettro e sottobande, "
         "con confronto tra i brani, verso la media dell'album e (opzionalmente) verso una reference.")

S = st.session_state
MAX_TRACKS, MAX_REFS = 20, 3
AUDIO_TYPES = ["wav", "mp3", "flac", "aif", "aiff", "ogg", "m4a", "aac"]
MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4",
        "aac": "audio/aac", "aif": "audio/aiff", "aiff": "audio/aiff"}
PAL = [c for c in px.colors.qualitative.Dark24 if c not in ("#222A2A", "#0D2A63")]   # colori leggibili su sfondo bianco
AMBER = "#e08a00"
REF_COLOR = "#111111"

S.setdefault("xo", list(A.DEFAULT_CROSSOVERS))
S.setdefault("cache_g", {})
S.setdefault("cache_b", {})
S.setdefault("digests", {})
S.setdefault("run", False)
S.setdefault("sig", None)
S.setdefault("last_tracks", None)
S.setdefault("pdf", None)

status_slot = st.container()
tab_up, tab_set, tab_res, tab_ref, tab_det, tab_exp = st.tabs(
    ["📥 Caricamento", "🎚️ Bande & Target", "📊 Risultati album", "🎯 Vs Reference", "🔍 Dettaglio & Ascolto", "📄 Export"])


# =============================================================================== utilita'
def digest_of(f):
    key = getattr(f, "file_id", None) or (f.name, f.size)
    if key not in S.digests:
        S.digests[key] = A.file_digest(f.getvalue())
    return S.digests[key]


def style_fig(fig, height=430, ytitle=None, xtitle=None):
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#1a1a17", family="DM Sans"),
        height=height, margin=dict(t=30, b=40, l=10, r=10), hovermode="closest",
        legend=dict(bgcolor="#ffffff", bordercolor="#ddd9cf", borderwidth=1, itemclick="toggle",
                    itemdoubleclick="toggleothers", font=dict(size=11)))
    fig.update_xaxes(gridcolor="#e8e5dd", zerolinecolor="#c9c5ba", title=xtitle)
    fig.update_yaxes(gridcolor="#e8e5dd", zerolinecolor="#c9c5ba", title=ytitle)
    return fig


def mmss(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def line_kw(t, width=1.6):
    """Stile di linea: brani colorati, reference bianche tratteggiate."""
    if t["ref"]:
        return dict(color=REF_COLOR, width=2.4, dash="dot")
    return dict(color=t["color"], width=width)


def add_mean_line(fig, x, y, name="Media album"):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=AMBER, width=4, dash="dash")))


def dev_heatmap(dsub, refmask, thr, dec_by_col=None, height=None, top=None, top_label="MEDIA ALBUM (valori)"):
    """Heatmap degli scostamenti. Righe brani: blu(-)/bianco/arancio(+). Righe reference: verde(-)/bianco/viola(+) con etichetta viola in grassetto.
    Il colore e' proporzionale a scostamento/soglia (pieno = 2 volte la soglia). Se `top` (Series con i valori assoluti medi) e' dato,
    aggiunge in testa una riga grigia con quei valori."""
    cols = list(dsub.columns)
    dec = dec_by_col or {}
    refmask = np.asarray(refmask, dtype=bool)
    names = list(dsub.index)
    z = np.clip(dsub.values / thr, -2, 2)
    txt = [[f"{v:+.{dec.get(c, 1)}f}" for v, c in zip(row, cols)] for row in dsub.values]
    ys = ([top_label] if top is not None else []) + names
    off = 1 if top is not None else 0
    nr, nc = len(ys), len(cols)
    za, zr, zt = (np.full((nr, nc), np.nan) for _ in range(3))
    ta, tr_, tt = (np.full((nr, nc), "", dtype=object) for _ in range(3))
    for i in range(len(names)):
        (zr if refmask[i] else za)[i + off] = z[i]
        (tr_ if refmask[i] else ta)[i + off] = txt[i]
    if top is not None:
        zt[0] = 0.0
        tt[0] = [f"{top[c]:.{dec.get(c, 1)}f}" for c in cols]
    common = dict(x=[c.replace(" [", "<br>[") for c in cols], y=ys, texttemplate="%{text}", zmin=-2, zmax=2,
                  showscale=False, xgap=2, ygap=2, hoverongaps=False)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=za, text=ta, colorscale=[[0, "#3b7dd8"], [0.5, "#ffffff"], [1, "#e8890c"]], **common))
    if refmask.any():
        fig.add_trace(go.Heatmap(z=zr, text=tr_, colorscale=[[0, "#1b9e77"], [0.5, "#ffffff"], [1, "#8e44ad"]], **common))
    if top is not None:
        fig.add_trace(go.Heatmap(z=zt, text=tt, colorscale=[[0, "#e3e0d6"], [1, "#e3e0d6"]], **common))
    ticks = [f"<b><span style='color:#b97a00'>{n}</span></b>" if (top is not None and i == 0) else
             (f"<b><span style='color:#8e44ad'>{n}</span></b>" if refmask[i - off] else n) for i, n in enumerate(ys)]
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=ys, tickvals=ys, ticktext=ticks)
    return style_fig(fig, height or 120 + 30 * nr)


def add_slope_refs(fig, anchor, offset=None):
    """Aggiunge le rette di riferimento (rumore rosa -3 dB/ott e musica tipica -4.5 dB/ott), ancorate a 1 kHz su `anchor`."""
    for name, sl, dash in (("Rumore rosa (-3 dB/ott)", -3.0, "longdash"), ("Musica tipica (-4,5 dB/ott)", -4.5, "dashdot")):
        y = T.slope_line(sl, anchor)
        if offset is not None:
            y = y - offset
        fig.add_trace(go.Scatter(x=T.THIRD_FC, y=y, name=name, mode="lines", line=dict(color="#7a7a7a", width=2.5, dash=dash)))


# =============================================================================== TAB: BANDE & TARGET
with tab_set:
    st.subheader("Frequenze di divisione delle sottobande")
    st.caption("Crossover Linkwitz-Riley 4° ordine (somma delle bande piatta in ampiezza). "
               "Le modifiche si applicano con il pulsante: se l'analisi e' gia' stata eseguita, "
               "vengono ricalcolate solo le sottobande (le metriche globali restano in memoria).")
    with st.form("xo_form"):
        c1, c2, c3 = st.columns(3)
        x1 = c1.number_input("Crossover 1 [Hz]", 30.0, 2000.0, float(S.xo[0]), 10.0)
        x2 = c2.number_input("Crossover 2 [Hz]", 100.0, 10000.0, float(S.xo[1]), 50.0)
        x3 = c3.number_input("Crossover 3 [Hz]", 500.0, 18000.0, float(S.xo[2]), 100.0)
        b1, b2, _ = st.columns([1, 1, 3])
        apply_xo = b1.form_submit_button("Applica crossover", type="primary")
        reset_xo = b2.form_submit_button("Ripristina default")
    if reset_xo:
        S.xo = list(A.DEFAULT_CROSSOVERS)
        st.rerun()
    if apply_xo:
        try:
            S.xo = A.validate_crossovers([x1, x2, x3], fs=44100)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    st.info("Bande attive: " + "  |  ".join(A.band_labels(S.xo)))

    st.subheader("Visualizzazione")
    v1, v2 = st.columns(2)
    weighting_label = v1.radio(
        "Ponderazione dei livelli di banda", ["RMS lineare", "K-weighting (LUFS)"], horizontal=True,
        help="RMS lineare = energia pura. K-weighting = curva BS.1770 (shelf alle alte, taglio sotto ~40 Hz), piu' vicina alla percezione. "
             "Entrambe sono calcolate durante l'analisi: puoi cambiare in qualsiasi momento senza rielaborare.")
    weighting = "k" if weighting_label.startswith("K") else "rms"
    norm_label = v2.selectbox(
        "Normalizzazione dello spettro 1/3 ottava", ["Livello totale", "Media 200 Hz - 5 kHz", "Assoluto (nessuna)"],
        help="Normalizzare permette di confrontare la FORMA dello spettro tra brani con loudness diversa. "
             "La media 200 Hz-5 kHz e' piu' sensibile alle differenze su bassi e acuti.")
    spec_norm = {"Livello totale": "total", "Media 200 Hz - 5 kHz": "mid", "Assoluto (nessuna)": "abs"}[norm_label]

    st.subheader("Target di mastering")
    t1, t2, t3 = st.columns(3)
    target_lufs = t1.number_input("Target loudness integrata [LUFS]", -30.0, 0.0, -10.0, 0.5)
    tol_lu = t2.number_input("Tolleranza [± LU]", 0.0, 5.0, 1.0, 0.1)
    target_tp = t3.number_input("True peak massimo [dBTP]", -6.0, 3.0, -1.0, 0.1)
    with st.expander("Soglie di segnalazione outlier (scostamento dalla media dell'album)"):
        d = T.DEFAULT_THRESH
        o1, o2, o3, o4 = st.columns(4)
        thresholds = {
            "lufs_i": o1.number_input("LUFS [LU]", 0.1, 10.0, d["lufs_i"], 0.1),
            "lra": o2.number_input("LRA [LU]", 0.1, 10.0, d["lra"], 0.1),
            "crest_db": o3.number_input("Crest [dB]", 0.1, 10.0, d["crest_db"], 0.1),
            "plr": o4.number_input("PLR [dB]", 0.1, 10.0, d["plr"], 0.1),
            "corr": o1.number_input("Correlazione", 0.01, 1.0, d["corr"], 0.01),
            "width_pct": o2.number_input("Width [%]", 1.0, 50.0, d["width_pct"], 1.0),
            "tilt": o3.number_input("Tilt [dB/ott]", 0.05, 3.0, d["tilt"], 0.05),
            "band_rel": o4.number_input("Livello banda [dB]", 0.5, 10.0, d["band_rel"], 0.5),
        }

# =============================================================================== TAB: CARICAMENTO
with tab_up:
    up1, up2 = st.columns([3, 2])
    album_files = up1.file_uploader(f"Brani dell'album (max {MAX_TRACKS}) - WAV, MP3, FLAC, AIFF, OGG, M4A",
                                    type=AUDIO_TYPES, accept_multiple_files=True, key="up_album")
    ref_files = up2.file_uploader(f"Reference (opzionale, max {MAX_REFS})", type=AUDIO_TYPES,
                                  accept_multiple_files=True, key="up_ref",
                                  help="Brani commerciali o di riferimento: vengono analizzati allo stesso modo e mostrati "
                                       "come linee nere tratteggiate. Non sono incluse nella media dell'album.")
    if len(album_files) > MAX_TRACKS:
        st.error(f"Hai caricato {len(album_files)} brani: verranno analizzati solo i primi {MAX_TRACKS}.")
        album_files = album_files[:MAX_TRACKS]
    if len(ref_files) > MAX_REFS:
        ref_files = ref_files[:MAX_REFS]
    album_title = st.text_input("Titolo album / progetto (usato nel report PDF)", value="", key="album_title")
    if album_files or ref_files:
        st.dataframe(pd.DataFrame(
            [{"Ruolo": "Album", "File": f.name, "Dimensione [MB]": round(f.size / 1e6, 1)} for f in album_files] +
            [{"Ruolo": "Reference", "File": f.name, "Dimensione [MB]": round(f.size / 1e6, 1)} for f in ref_files]),
            hide_index=True, width="stretch")
    go_btn = st.button("▶️ Analizza", type="primary", disabled=not album_files)
    st.caption("Per ogni brano: loudness BS.1770 (integrata, short-term, momentary, LRA), true peak 4x, crest factor, PLR/PSR, "
               "stereo (correlazione, M/S, compatibilita' mono), spettro a terzi d'ottava, tilt, centroide e analisi per sottobande. "
               "Nessun file viene salvato su disco: tutto resta in memoria nella tua sessione.")

# =============================================================================== ANALISI
items = [(f, False) for f in album_files] + [(f, True) for f in ref_files]
sig = tuple((digest_of(f), ref) for f, ref in items)
if go_btn:
    S.run, S.sig = True, sig


def compute_tracks(items, xo):
    out, errors = [], []
    n = len(items)
    with status_slot:
        box = st.status("Analisi in corso...", expanded=True)
        bar = box.progress(0.0)
    for k, (f, ref) in enumerate(items):
        dg = digest_of(f)
        gk, bk = dg, (dg, tuple(xo))
        need_g, need_b = gk not in S.cache_g, bk not in S.cache_b
        if need_g or need_b:
            box.write(f"{'Reference' if ref else 'Brano'} {k + 1}/{n}: {f.name}"
                      + ("" if need_g else " (ricalcolo sottobande)"))
            try:
                res = A.analyze_bytes(f.getvalue(), f.name, xo, want_global=need_g, want_bands=need_b)
            except Exception as e:  # file corrotto / formato non decodificabile
                errors.append(f"{f.name}: {e}")
                bar.progress((k + 1) / n)
                continue
            if need_g:
                S.cache_g[gk] = {**res["global"], "fs": res["fs"], "duration": res["duration"], "info": res["info"]}
            if need_b:
                S.cache_b[bk] = res["bands"]
        out.append({"name": Path(f.name).stem, "file": f.name, "ref": ref, "digest": dg,
                    "g": S.cache_g[gk], "b": S.cache_b[bk]})
        bar.progress((k + 1) / n)
    box.update(label="Analisi completata", state="complete", expanded=False)
    for e in errors:
        st.error(e)
    return out


tracks = None
if S.run and items:
    if sig == S.sig:
        tracks = compute_tracks(items, S.xo)
    else:
        with tab_up:
            st.warning("I file caricati sono cambiati dall'ultima analisi: premi **Analizza** per aggiornare i risultati.")
        tracks = S.last_tracks
elif S.run:
    tracks = S.last_tracks

if tracks and not any(not t["ref"] for t in tracks):
    tracks = None
if tracks:
    ai = 0
    for i, t in enumerate(tracks):
        if t["ref"]:
            t["label"], t["color"] = f"REF · {t['name']}", REF_COLOR
        else:
            ai += 1
            t["label"], t["color"] = f"{ai:02d} · {t['name']}", PAL[(ai - 1) % len(PAL)]
    S.last_tracks = tracks
    S.pdf = None if S.get("pdf_sig") != (sig, tuple(S.xo)) else S.pdf

files_by_digest = {digest_of(f): f for f, _ in items}

# =============================================================================== TAB: RISULTATI
with tab_res:
    if not tracks:
        st.info("Carica i brani nella scheda **Caricamento** e premi **Analizza**.")
    else:
        album = T.albums(tracks)
        labels = [t["label"] for t in tracks]
        if S.get("vis_labels") != labels:
            S.vis, S.vis_labels = list(labels), labels
        for t in tracks:
            if t["g"]["info"]["was_mono"]:
                st.warning(f"{t['name']}: file mono, duplicato su due canali (correlazione e stereo non significativi).")

        dev, flag = T.deviation_df(tracks, weighting, thresholds)
        dev.index = flag.index = labels
        status = {t["label"]: T.target_status(t["g"], target_lufs, tol_lu, target_tp) for t in tracks}
        n_off = sum(1 for t in album if not status[t["label"]][0])
        n_out = int(flag.loc[[t["label"] for t in album]].any(axis=1).sum())
        lufs_all = [t["g"]["lufs_i"] for t in album]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Brani analizzati", len(album))
        k2.metric("LUFS medio album", f"{np.nanmean(lufs_all):.1f}", f"spread {np.nanmax(lufs_all) - np.nanmin(lufs_all):.1f} LU", delta_color="off")
        k3.metric("True peak max", f"{max(t['g']['tp_db'] for t in album):.2f} dBTP")
        k4.metric("Fuori target", f"{n_off} / {len(album)}")
        k5.metric("Brani con outlier", f"{n_out} / {len(album)}")

        # --- visibilita' brani
        st.markdown("#### Brani visibili nei grafici")
        s1, s2, s3 = st.columns([1, 1, 8])
        s1.button("Tutti", on_click=lambda: S.update(vis=list(labels)))
        s2.button("Nessuno", on_click=lambda: S.update(vis=[]))
        sel = s3.multiselect("Seleziona i brani (oppure clicca sulle voci della legenda del grafico: "
                             "click = accendi/spegni, doppio click = isola)", labels, key="vis", label_visibility="collapsed")
        vis = [t for t in tracks if t["label"] in sel]
        st.caption("La linea ambra tratteggiata e' la media dei soli brani dell'album (reference escluse), sempre calcolata su tutti i brani e non solo su quelli visibili. "
                   "Le reference sono nere tratteggiate ed escluse da ogni media.")

        # --- tabella riepilogo
        st.markdown("### 📋 Riepilogo")
        keys = [k for k, *_ in T.GLOBAL_COLS]
        rows = [{"Brano": t["label"], **{T.LABEL[k]: t["g"][k] for k in keys}} for t in tracks]
        alb_rows = [r for r, t in zip(rows, tracks) if not t["ref"]]
        mean_row = {"Brano": "MEDIA ALBUM", **{T.LABEL[k]: np.nanmean([r[T.LABEL[k]] for r in alb_rows]) for k in keys}}
        df = pd.concat([pd.DataFrame(rows), pd.DataFrame([mean_row])], ignore_index=True)
        n_t = len(tracks)
        alb_idx = [i for i, t in enumerate(tracks) if not t["ref"]]

        def shade(_):
            """Intensita' crescente con il valore assoluto del parametro: colore piu' tenue sul brano con |valore| minore della colonna,
            piu' forte su quello con |valore| maggiore (scala calcolata sui soli brani dell'album)."""
            sty = pd.DataFrame("", index=df.index, columns=df.columns)
            for c in df.columns[1:]:
                mags = df[c].abs().to_numpy(dtype=float)[:n_t]
                lo, hi = np.nanmin(mags[alb_idx]), np.nanmax(mags[alb_idx])
                if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                    continue
                for r in range(n_t):
                    if np.isfinite(mags[r]):
                        x = float(np.clip((mags[r] - lo) / (hi - lo), 0, 1))
                        sty.loc[r, c] = f"background-color: rgb({int(255 - 15 * x)},{int(255 - 105 * x)},{int(255 - 225 * x)}); color: #1a1a17"
            sty.loc[n_t, :] = "font-weight: bold; color: #7a5200; background-color: #f6f4ee"
            return sty

        fmt = {T.LABEL[k]: "{:." + str(d) + "f}" for k, _, _, d in T.GLOBAL_COLS}
        styled = df.style.apply(shade, axis=None).format(fmt, na_rep="-")
        try:
            st.dataframe(styled, hide_index=True, width="stretch", height=min(60 + 35 * len(df), 800),
                         column_config={"Brano": st.column_config.TextColumn("Brano", pinned=True)})
        except TypeError:  # versioni di Streamlit senza colonne bloccate
            st.dataframe(styled, hide_index=True, width="stretch", height=min(60 + 35 * len(df), 800))
        st.caption("Colore: in ogni colonna l'intensita' cresce con il valore assoluto del parametro, cioe' man mano che ci si allontana dallo zero "
                   "(tenue = valore piu' basso della colonna, pieno = valore piu' alto tra i brani dell'album). Nessuna soglia standard: "
                   "serve a vedere a colpo d'occhio dove i brani si differenziano. La riga MEDIA ALBUM esclude le reference. "
                   "Crest, PLR e PSR in dB; Tilt = pendenza dello spettro a terzi d'ottava tra 100 Hz e 10 kHz (0 = rumore rosa); "
                   "scorri la tabella a destra per vedere tutte le grandezze.")

        if not vis:
            st.info("Nessun brano selezionato nei grafici.")
        else:
            # --- metrica per brano
            st.markdown("### 📈 Una metrica, tutti i brani")
            mk = st.selectbox("Metrica", [k for k, *_ in T.GLOBAL_COLS], format_func=lambda k: T.LABEL[k], key="metric_sel")
            vals_m = np.array([t["g"][mk] for t in vis], dtype=float)
            dm = {k: d for k, _, _, d in T.GLOBAL_COLS}[mk]
            bar_cols = [t["color"] if not t["ref"] else "#555555" for t in vis]
            labs_m = [t["label"] for t in vis]
            txt_m = [f"{v:.{dm}f}" for v in vals_m]
            below = bool(np.nanmax(vals_m) <= 0 and np.nanmin(vals_m) < 0)   # grandezze negative (LUFS, dBTP...): barre che salgono dal basso
            if below:
                ext = list(vals_m) + [np.nanmean([t["g"][mk] for t in album])] + [t["g"][mk] for t in tracks if t["ref"]]
                if mk == "lufs_i":
                    ext += [target_lufs - tol_lu, target_lufs + tol_lu]
                if mk == "tp_db":
                    ext.append(target_tp)
                lo, hi = float(np.nanmin(ext)), float(np.nanmax(ext))
                span = max(hi - lo, 1.0)
                y0, y1 = lo - 0.30 * span, hi + 0.22 * span
                fig = go.Figure(go.Bar(x=labs_m, y=vals_m - y0, base=y0, customdata=vals_m, text=txt_m, textposition="outside", cliponaxis=False,
                                       hovertemplate="%{x}<br>%{customdata:.2f}<extra></extra>", marker_color=bar_cols, showlegend=False))
                fig.update_yaxes(range=[y0, y1])
            else:
                fig = go.Figure(go.Bar(x=labs_m, y=vals_m, text=txt_m, textposition="outside", cliponaxis=False,
                                       marker_color=bar_cols, showlegend=False))
            mean_v = np.nanmean([t["g"][mk] for t in album])
            fig.add_hline(y=mean_v, line=dict(color=AMBER, width=3, dash="dash"), annotation_text=f"media album {mean_v:.2f}",
                          annotation_font_color=AMBER)
            for t in tracks:
                if t["ref"]:
                    fig.add_hline(y=t["g"][mk], line=dict(color=REF_COLOR, width=2, dash="dot"),
                                  annotation_text=t["label"], annotation_position="bottom right", annotation_font_color=REF_COLOR)
            if mk == "lufs_i":
                fig.add_hrect(y0=target_lufs - tol_lu, y1=target_lufs + tol_lu, fillcolor="#00ff9f", opacity=0.10, line_width=0)
            if mk == "tp_db":
                fig.add_hline(y=target_tp, line=dict(color="#ff007c", width=2), annotation_text=f"max {target_tp:g} dBTP",
                              annotation_font_color="#ff007c")
            st.plotly_chart(style_fig(fig, 380, T.LABEL[mk]), width="stretch")

            # --- heatmap scostamenti
            st.markdown("### 🌡️ Mappa degli scostamenti dalla media album")
            vl = [t["label"] for t in vis]
            dsub = dev.loc[vl]
            rmask = np.array([t["ref"] for t in vis])
            dec = {c: 2 for c in dsub.columns if "Correlazione" in c}
            top_row = T.album_mean_row(tracks, weighting)
            st.plotly_chart(dev_heatmap(dsub, rmask, T.thr_for(dsub.columns, thresholds), dec, top=top_row), width="stretch")
            st.caption("La riga grigia in alto riporta i valori medi dell'album (reference escluse), nelle unita' di misura indicate in colonna. "
                       "Ogni altra cella = valore del brano meno quella media (le reference NON entrano nella media). "
                       "Brani: blu = sotto la media, arancio = sopra. Righe delle reference: verde = sotto, viola = sopra, con il nome in viola e grassetto. "
                       "Colore pieno = scostamento pari o superiore a 2 volte la soglia impostata in *Bande & Target*.")

            # --- spettro
            st.markdown("### 🔊 Spettro a terzi d'ottava")
            sv = st.radio("Vista", ["Spettro", "Scostamento dalla media album"], horizontal=True, key="spec_view")
            mean_spec = T.album_mean(tracks, lambda t: T.spectrum_norm(t["g"]["third_db"], spec_norm))
            fig = go.Figure()
            for t in vis:
                y = T.spectrum_norm(t["g"]["third_db"], spec_norm)
                y = y - mean_spec if sv.startswith("Scost") else y
                fig.add_trace(go.Scatter(x=T.THIRD_FC, y=y, name=t["label"], mode="lines", line=line_kw(t, 1.8)))
            if sv.startswith("Scost"):
                fig.add_hline(y=0, line=dict(color=AMBER, width=3, dash="dash"))
            else:
                add_mean_line(fig, T.THIRD_FC, mean_spec)
            show_std = st.checkbox("Mostra le rette di riferimento (rumore rosa -3 dB/ott e musica tipica -4,5 dB/ott)", value=True, key="spec_std")
            if show_std and np.isfinite(mean_spec[17]):
                add_slope_refs(fig, mean_spec[17], offset=mean_spec if sv.startswith("Scost") else None)
            fig.update_xaxes(type="log", tickvals=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                             ticktext=["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"], range=[np.log10(18), np.log10(21000)])
            st.plotly_chart(style_fig(fig, 480, "Scostamento [dB]" if sv.startswith("Scost") else "Livello 1/3 ottava [dB]", "Frequenza [Hz]"),
                            width="stretch")
            st.caption("Le due rette grigie sono pendenze di riferimento ancorate al valore medio dell'album a 1 kHz: servono a confrontare la FORMA, non il livello. "
                       "Il rumore rosa ha -3 dB/ottava come densita' spettrale (cosi' lo misurano gli analizzatori FFT) e proprio per questo in un grafico a terzi d'ottava "
                       "e' una retta orizzontale. La musica masterizzata tipica scende di piu': analizzatori come Voxengo SPAN usano -4,5 dB/ott come impostazione "
                       "predefinita per leggerla 'piatta', e i valori reali variano in genere tra -3 e -6 dB/ott a seconda di genere e arrangiamento. "
                       "In questo grafico -4,5 dB/ott corrisponde a circa -1,5 dB per ottava (la banda di terzo d'ottava raddoppia di larghezza a ogni ottava, +3 dB). "
                       "Non e' un target: e' un metro di confronto, il genere puo' discostarsene molto.")

            # --- sottobande
            st.markdown("### 🎛️ Sottobande")
            BAND_HELP = {
                "rel": "Quanto pesa la banda rispetto al livello totale del brano (equilibrio tonale). Se un brano e' a -2 dB rispetto alla media nella banda dei bassi, "
                       "ha meno basso degli altri brani dell'album, a prescindere da quanto e' forte.",
                "level": "Livello assoluto della banda: dipende anche dalla loudness generale del brano.",
                "share_pct": "Percentuale dell'energia complessiva che cade in quella banda.",
                "crest_db": "Differenza picco-RMS nella banda: piu' e' alto, piu' la banda e' dinamica/impulsiva; piu' e' basso, piu' e' compressa o limitata.",
                "dyn": "Quanto varia il livello della banda nel tempo (finestre da 3 s): basso = banda statica o molto compressa.",
                "width_pct": "Quota di energia nel canale Side: indica quanto la banda e' larga. Nei bassi dovrebbe essere molto bassa.",
                "corr": "Correlazione tra canale sinistro e destro nella banda: 1 = mono, valori bassi o negativi = problemi di fase.",
            }
            BAND_THR = {"rel": thresholds["band_rel"], "crest_db": thresholds["crest_db"], "width_pct": thresholds["width_pct"], "corr": thresholds["corr"]}
            bm1, bm2, bm3 = st.columns([3, 2, 2])
            bmet = bm1.selectbox("Metrica di banda", list(T.BAND_METRICS), format_func=lambda k: T.BAND_METRICS[k], key="band_metric")
            bdev = bm2.checkbox("Scostamento dalla media album", value=True, key="band_dev",
                                help="Attivo: ogni barra e' il valore del brano meno la media dell'album (reference escluse). 0 = nella media.")
            bview = bm3.radio("Vista", ["Barre per banda", "Mappa brani × bande"], key="band_view")
            st.caption(BAND_HELP[bmet])
            blabels = tracks[0]["b"]["labels"]
            nbd = len(blabels)
            bmean = T.album_mean(tracks, lambda t: [T.band_value(t["g"], bd, bmet, weighting) for bd in t["b"]["bands"]])
            vals = {t["label"]: np.array([T.band_value(t["g"], bd, bmet, weighting) for bd in t["b"]["bands"]]) for t in tracks}
            thr_b = BAND_THR.get(bmet)
            vis_alb = [t for t in vis if not t["ref"]]
            vis_ref = [t for t in vis if t["ref"]]
            short = lambda t: t["label"].split(" · ")[0]
            if bview == "Barre per banda":
                fig = make_subplots(rows=1, cols=nbd, subplot_titles=blabels, shared_yaxes=bdev, horizontal_spacing=0.015 if bdev else 0.05)
                for t in vis_alb:
                    for b in range(nbd):
                        y = vals[t["label"]][b] - (bmean[b] if bdev else 0)
                        fig.add_trace(go.Bar(x=[short(t)], y=[y], name=t["label"], legendgroup=t["label"], showlegend=(b == 0),
                                             marker_color=t["color"], hovertemplate=f"{t['label']}<br>%{{y:+.2f}}<extra></extra>"), row=1, col=b + 1)
                for b in range(nbd):
                    if bdev and thr_b:
                        fig.add_hrect(y0=-thr_b, y1=thr_b, fillcolor="#2ca02c", opacity=0.10, line_width=0, row=1, col=b + 1)
                    fig.add_hline(y=0 if bdev else bmean[b], line=dict(color=AMBER, width=2.5, dash="dash"), row=1, col=b + 1)
                    for r in vis_ref:
                        fig.add_hline(y=vals[r["label"]][b] - (bmean[b] if bdev else 0), line=dict(color=REF_COLOR, width=2.2, dash="dot"), row=1, col=b + 1)
                for r in vis_ref:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=r["label"], line=dict(color=REF_COLOR, width=2.2, dash="dot")))
                fig.update_layout(barmode="overlay")
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=[short(t) for t in vis_alb], tickfont=dict(size=9), tickangle=-90)
                style_fig(fig, 470, T.BAND_METRICS[bmet] + (" - scostamento dalla media" if bdev else ""))
                fig.update_yaxes(title=None)
                fig.update_yaxes(title=("Scostamento" if bdev else "Valore"), row=1, col=1)
                st.plotly_chart(fig, width="stretch")
                st.caption("Un pannello per banda, una barra per brano (numero del brano in ascissa, nome nel tooltip). "
                           + ("La fascia verde e' la tolleranza impostata: barre che escono dalla fascia indicano un brano diverso dagli altri in quella banda. " if bdev and thr_b else "")
                           + "Linea ambra = media album; linee nere tratteggiate = reference. Clicca sulla legenda per accendere/spegnere i brani.")
            else:
                rows_t = vis_alb + vis_ref
                if bdev:
                    dmat = pd.DataFrame([vals[t["label"]] - bmean for t in rows_t], index=[t["label"] for t in rows_t], columns=blabels)
                    th = thr_b or max(float(np.nanmax(np.abs(dmat.values))) / 2, 1e-6)
                    st.plotly_chart(dev_heatmap(dmat, np.array([t["ref"] for t in rows_t]), np.full(nbd, th),
                                                {c: 2 for c in blabels} if bmet in ("corr",) else None), width="stretch")
                    st.caption("Righe = brani, colonne = bande. Arancio = sopra la media dell'album, blu = sotto; righe verdi/viola = reference. Colore pieno = scostamento pari a 2 volte la soglia.")
                else:
                    amat = np.array([vals[t["label"]] for t in rows_t])
                    fig = go.Figure(go.Heatmap(z=amat, x=blabels, y=[t["label"] for t in rows_t], text=np.round(amat, 2), texttemplate="%{text}",
                                               colorscale=[[0, "#ffffff"], [1, "#e8890c"]], showscale=False, xgap=2, ygap=2))
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(style_fig(fig, 120 + 30 * len(rows_t)), width="stretch")
            with st.expander("Tabella sottobande (tutti i brani)"):
                st.dataframe(T.bands_df(tracks, weighting).style.format(precision=2), hide_index=True, width="stretch")

            # --- tempo
            st.markdown("### ⏱️ Andamento nel tempo")
            st.info("**A cosa serve.** Mostra come cambia il livello durante il brano (media mobile su 3 secondi): serve a vedere se la loudness e' stabile o "
                    "ha sbalzi, dove stanno i passaggi piu' forti e piu' deboli, e se due brani dell'album hanno una gestione della dinamica simile "
                    "(per esempio uno resta piatto mentre l'altro ha crescendo molto marcati). "
                    "Scegliendo una sottobanda vedi anche *dove nel tempo* quella banda sale o scende: un'asprezza che compare solo nel ritornello, "
                    "o un basso che sparisce nelle parti piu' piene. "
                    "Attiva *Centra sul livello medio* per sovrapporre le curve di brani con loudness diversa e confrontarne solo la forma.")
            tt1, tt2 = st.columns([3, 2])
            tsel = tt1.selectbox("Segnale", ["Full band"] + blabels, key="time_sel")
            tcenter = tt2.checkbox("Centra sul livello medio di ogni brano", key="time_center")
            fig = go.Figure()
            for t in vis:
                if tsel == "Full band":
                    tl = t["g"]["timeline"]
                    x, y = tl["t"], (tl["st_lufs"] if weighting == "k" else tl["st_rms"])
                else:
                    b = blabels.index(tsel)
                    tl = t["b"]["timeline"]
                    x, y = tl["t"], (tl["k"][b] if weighting == "k" else tl["rms"][b])
                y = np.asarray(y, dtype=float)
                if tcenter and y.size:
                    y = y - np.nanmean(y)
                fig.add_trace(go.Scatter(x=x, y=y, name=t["label"], mode="lines", line=line_kw(t, 1.4)))
            unit = "dB rispetto alla media del brano" if tcenter else ("LUFS" if weighting == "k" else "dBFS RMS")
            st.plotly_chart(style_fig(fig, 400, f"Livello short-term [{unit}]", "Tempo [s]"), width="stretch")

# =============================================================================== TAB: VS REFERENCE
with tab_ref:
    if not tracks:
        st.info("Esegui prima l'analisi.")
    elif not any(t["ref"] for t in tracks):
        st.info("Nessuna reference caricata. Aggiungine una (o piu') nella scheda **Caricamento** e premi di nuovo **Analizza**.")
    else:
        refs = [t for t in tracks if t["ref"]]
        rsel = st.selectbox("Reference di confronto", [r["label"] for r in refs], key="ref_sel")
        base = next(r for r in refs if r["label"] == rsel)
        rows_t = [t for t in tracks if t is not base]
        devr, flagr = T.deviation_df(tracks, weighting, thresholds, baseline=base)
        devr.index = [t["label"] for t in rows_t] + ["MEDIA ALBUM"]
        rmask = np.array([t["ref"] for t in rows_t] + [False])
        bg = base["g"]
        st.markdown(f"**{base['label']}** - {bg['lufs_i']:.1f} LUFS, {bg['tp_db']:.2f} dBTP, LRA {bg['lra']:.1f} LU, crest {bg['crest_db']:.1f} dB, tilt {bg['tilt']:+.2f} dB/ott")
        st.markdown("### 🎯 Mappa degli scostamenti dalla reference")
        dec = {c: 2 for c in devr.columns if "Correlazione" in c}
        st.plotly_chart(dev_heatmap(devr, rmask, T.thr_for(devr.columns, thresholds), dec), width="stretch")
        st.caption("Ogni cella = valore del brano meno il valore della reference selezionata: positivo = il brano ha PIU' della reference "
                   "(piu' loudness, piu' dinamica, piu' energia in quella banda...), negativo = ne ha meno. "
                   "L'ultima riga (MEDIA ALBUM) riassume la distanza dell'album intero dalla reference: la media e' calcolata solo sui brani dell'album. "
                   "Se hai caricato altre reference, compaiono in verde/viola. Le soglie dei colori sono le stesse impostate in *Bande & Target*.")
        st.markdown("### 🔊 Spettro: brani vs reference")
        figr = go.Figure()
        bs = T.spectrum_norm(base["g"]["third_db"], spec_norm)
        for t in rows_t:
            if t["ref"]:
                continue
            figr.add_trace(go.Scatter(x=T.THIRD_FC, y=T.spectrum_norm(t["g"]["third_db"], spec_norm) - bs, name=t["label"], mode="lines", line=line_kw(t, 1.8)))
        figr.add_hline(y=0, line=dict(color=REF_COLOR, width=3, dash="dot"), annotation_text=base["label"], annotation_font_color=REF_COLOR)
        figr.update_xaxes(type="log", tickvals=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                          ticktext=["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"], range=[np.log10(18), np.log10(21000)])
        st.plotly_chart(style_fig(figr, 430, "Scostamento dalla reference [dB]", "Frequenza [Hz]"), width="stretch")
        st.caption("Sopra lo zero il brano ha piu' energia della reference a quella frequenza, sotto lo zero ne ha meno. "
                   "Normalizzazione dello spettro scelta in *Bande & Target*.")

# =============================================================================== TAB: DETTAGLIO & ASCOLTO
with tab_det:
    if not tracks:
        st.info("Esegui prima l'analisi.")
    else:
        album = T.albums(tracks)
        labels = [t["label"] for t in tracks]
        tl_sel = st.selectbox("Brano", labels, key="det_track")
        t = tracks[labels.index(tl_sel)]
        g, b = t["g"], t["b"]
        mean = lambda k: np.nanmean([x["g"][k] for x in album])
        cols = st.columns(4)
        for i, (k, nm, fmt) in enumerate([("lufs_i", "LUFS integrato", "{:.1f}"), ("tp_db", "True peak [dBTP]", "{:.2f}"),
                                          ("lra", "LRA [LU]", "{:.1f}"), ("crest_db", "Crest [dB]", "{:.1f}"),
                                          ("plr", "PLR [dB]", "{:.1f}"), ("corr", "Correlazione", "{:.2f}"),
                                          ("width_pct", "Width [%]", "{:.1f}"), ("tilt", "Tilt [dB/ott]", "{:+.2f}")]):
            cols[i % 4].metric(nm, fmt.format(g[k]), f"{g[k] - mean(k):+.2f} vs media album", delta_color="off")
        ok, nts = T.target_status(g, target_lufs, tol_lu, target_tp)
        if not t["ref"]:
            (st.success if ok else st.warning)("Entro target." if ok else "Fuori target: " + "; ".join(nts))
        if abs(g["dc_l"]) > 0.001 or abs(g["dc_r"]) > 0.001:
            st.warning(f"DC offset rilevato (L {g['dc_l']:+.4f}, R {g['dc_r']:+.4f}).")

        # spettro brano vs media vs reference
        fig = go.Figure()
        ms = T.album_mean(tracks, lambda x: T.spectrum_norm(x["g"]["third_db"], spec_norm))
        fig.add_trace(go.Scatter(x=T.THIRD_FC, y=T.spectrum_norm(g["third_db"], spec_norm), name=t["label"], line=dict(color="#1f77b4", width=3)))
        add_mean_line(fig, T.THIRD_FC, ms)
        if np.isfinite(ms[17]):
            add_slope_refs(fig, ms[17])
        for r in [x for x in tracks if x["ref"] and x is not t]:
            fig.add_trace(go.Scatter(x=T.THIRD_FC, y=T.spectrum_norm(r["g"]["third_db"], spec_norm), name=r["label"], line=line_kw(r)))
        fig.update_xaxes(type="log", tickvals=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                         ticktext=["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"], range=[np.log10(18), np.log10(21000)])
        st.plotly_chart(style_fig(fig, 360, "Livello 1/3 ottava [dB]", "Frequenza [Hz]"), width="stretch")

        bd_rows = [{"Banda": bd["label"], "Livello rel. [dB]": T.band_value(g, bd, "rel", weighting),
                    "Livello [dB]": T.band_value(g, bd, "level", weighting), "Quota energia [%]": bd["share_pct"],
                    "Crest [dB]": bd["crest_db"], "Dinamica ST [dB]": T.band_value(g, bd, "dyn", weighting),
                    "Width [%]": bd["width_pct"], "Corr": bd["corr"]} for bd in b["bands"]]
        st.dataframe(pd.DataFrame(bd_rows).style.format(precision=2, subset=pd.IndexSlice[:, pd.DataFrame(bd_rows).columns[1:]]),
                     hide_index=True, width="stretch")

        # timeline brano: full + bande, con marker sezioni
        sections = list(g["sections"]) + list(b["sections"])
        fig = go.Figure()
        tl = g["timeline"]
        fig.add_trace(go.Scatter(x=tl["t"], y=tl["st_lufs"] if weighting == "k" else tl["st_rms"], name="Full band",
                                 line=dict(color=REF_COLOR, width=2.5)))
        for i, lab in enumerate(b["labels"]):
            fig.add_trace(go.Scatter(x=b["timeline"]["t"], y=b["timeline"]["k"][i] if weighting == "k" else b["timeline"]["rms"][i],
                                     name=lab, line=dict(color=PAL[(i * 5 + 1) % len(PAL)], width=1.6)))
        for lab, ts, val in sections:
            fig.add_vline(x=ts, line=dict(color="#ff007c", width=1, dash="dot"))
        st.plotly_chart(style_fig(fig, 380, "Livello short-term [" + ("LUFS" if weighting == "k" else "dBFS RMS") + "]", "Tempo [s]"),
                        width="stretch")
        st.caption("Il grafico mostra il livello del brano nel tempo (media mobile su 3 s): linea nera = full band, linee colorate = le 4 sottobande. "
                   "Le linee rosa verticali sono i passaggi che l'app segnala come potenzialmente critici (piu' forte, piu' debole, true peak massimo, "
                   "minima correlazione stereo, massimo di ogni banda rispetto al totale): selezionali nel player qui sotto per riascoltarli.")

        # ----- player
        st.markdown("### 🎧 Player")
        f = files_by_digest.get(t["digest"])
        if f is None:
            st.info("File audio non piu' caricato: ricaricalo per riascoltarlo.")
        else:
            dur = float(g["duration"])
            mode = st.radio("Cosa ascoltare", ["Intero brano", "Passaggio rilevato", "Selezione manuale"], horizontal=True, key="play_mode")
            start, end = 0, None
            if mode == "Passaggio rilevato":
                p1, p2, p3 = st.columns([4, 1, 1])
                opt = p1.selectbox("Passaggio", range(len(sections)),
                                   format_func=lambda i: f"{sections[i][0]} - {mmss(sections[i][1])} ({sections[i][2]})", key="sec_sel")
                pre = p2.number_input("Anticipo [s]", 0, 30, 4, key="sec_pre")
                length = p3.number_input("Durata [s]", 3, 120, 15, key="sec_len")
                start = int(max(0, sections[opt][1] - pre))
                end = int(min(dur, start + length))
            elif mode == "Selezione manuale":
                a, bb = st.slider("Intervallo [s]", 0.0, dur, (0.0, min(dur, 20.0)), 0.5, key="play_rng")
                start, end = int(a), int(np.ceil(bb))
            ext = f.name.rsplit(".", 1)[-1].lower()
            if end is not None:
                st.caption(f"Riproduzione da {mmss(start)} a {mmss(end)}")
            st.audio(f.getvalue(), format=MIME.get(ext, "audio/wav"), start_time=start, end_time=end)
            if ext in ("aif", "aiff"):
                st.caption("Nota: alcuni browser non riproducono AIFF.")

# =============================================================================== TAB: EXPORT
with tab_exp:
    if not tracks:
        st.info("Esegui prima l'analisi.")
    else:
        st.subheader("CSV")
        fmt_csv = st.radio("Formato", ["Excel italiano (; e virgola decimale)", "Standard (, e punto)"], horizontal=True)
        kw = dict(index=False, sep=";", decimal=",") if fmt_csv.startswith("Excel") else dict(index=False)
        enc = lambda d: d.to_csv(**kw).encode("utf-8-sig")
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ Riepilogo brani", enc(T.summary_df(tracks)), "album_riepilogo.csv", "text/csv")
        e2.download_button("⬇️ Sottobande", enc(T.bands_df(tracks, weighting)), "album_sottobande.csv", "text/csv")
        e3.download_button("⬇️ Spettro 1/3 ottava", enc(T.third_df(tracks, spec_norm)), "album_spettro_terzi.csv", "text/csv")

        st.subheader("Report PDF")
        ventena_logo = find_logo([str(ROOT), str(Path.cwd())])
        st.caption("Logo Ventena Studios (testata a sinistra): " + (f"{Path(ventena_logo).name} trovato in assets" if ventena_logo else
                   "assets/logo.png non trovato nel repository, uso l'intestazione testuale."))
        pc1, pc2 = st.columns([2, 3])
        header_name = pc1.text_input("Nome album in testata (opzionale)", key="pdf_header_name",
                                     help="Compare nella testata del report, a destra.")
        album_logo_up = pc2.file_uploader("Logo / copertina dell'album (opzionale, testata a destra)", type=["png", "jpg", "jpeg"], key="album_logo_up")
        album_logo_path = None
        if album_logo_up is not None:
            ext_l = album_logo_up.name.rsplit(".", 1)[-1].lower()
            tmp = Path(tempfile.gettempdir()) / f"ventena_album_logo_{getattr(album_logo_up, 'file_id', 'x')}.{ext_l}"
            tmp.write_bytes(album_logo_up.getvalue())
            album_logo_path = str(tmp)
        header_style = st.radio("Colore della testata", ["Scura", "Chiara"], horizontal=True, key="pdf_header_style",
                                help="Scegli quella su cui i due loghi si leggono meglio (es. loghi chiari su testata scura).")
        notes = st.text_area("Note da inserire nel report (opzionale)", key="pdf_notes")
        per_track = st.checkbox("Includi una pagina di dettaglio per ogni brano", value=True)
        if st.button("📄 Genera PDF", type="primary"):
            with st.spinner("Generazione del report..."):
                settings = dict(weighting=weighting, spec_norm=spec_norm, target_lufs=target_lufs, tol_lu=tol_lu,
                                target_tp=target_tp, thresholds=thresholds)
                S.pdf = build_pdf(tracks, settings, album_title or "Album", notes, ventena_logo, per_track,
                                  album_logo_path=album_logo_path, header_name=header_name.strip(), header_dark=(header_style == "Scura"))
                S.pdf_sig = (sig, tuple(S.xo))
        if S.pdf:
            st.download_button("⬇️ Scarica il report PDF", S.pdf, f"{(album_title or 'album').strip().replace(' ', '_')}_mastering_report.pdf",
                               "application/pdf", type="primary")

st.caption("Album Mastering Analyzer | Developed by VentenaStudios for Producers & Engineers | 100% Private Analysis")
