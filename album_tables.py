"""Tabelle derivate (pandas) condivise da dashboard, export CSV e report PDF."""
import warnings

import numpy as np
import pandas as pd

from album_analysis import THIRD_FC

# chiave, etichetta, unita', decimali
GLOBAL_COLS = [
    ("lufs_i", "LUFS integrato", "LUFS", 1), ("lufs_st_max", "Short-term max", "LUFS", 1),
    ("lufs_m_max", "Momentary max", "LUFS", 1), ("lra", "LRA", "LU", 1),
    ("tp_db", "True peak", "dBTP", 2), ("peak_db", "Sample peak", "dBFS", 2),
    ("rms_db", "RMS", "dBFS", 1), ("crest_db", "Crest factor", "dB", 1),
    ("plr", "PLR (TP - LUFS)", "dB", 1), ("psr", "PSR (TP - ST max)", "dB", 1),
    ("corr", "Correlazione L/R", "", 2), ("width_pct", "Larghezza stereo (side)", "%", 1),
    ("ms_ratio_db", "Rapporto Mid/Side", "dB", 1), ("mono_loss_db", "Perdita in mono", "dB", 1),
    ("centroid", "Centroide spettrale", "Hz", 0), ("rolloff85", "Rolloff 85%", "Hz", 0),
    ("tilt", "Tilt spettrale", "dB/ott", 2), ("clip_n", "Campioni a fondo scala", "", 0),
    ("tp_over0_n", "Campioni TP > 0 dBFS", "", 0), ("silence_head", "Silenzio in testa", "s", 1),
    ("silence_tail", "Silenzio in coda", "s", 1), ("duration", "Durata", "s", 1),
]
LABEL = {k: f"{lab} [{u}]" if u else lab for k, lab, u, _ in GLOBAL_COLS}

# metriche globali usate per outlier / heatmap e soglia di default (scostamento dalla media album)
DEV_METRICS = ["lufs_i", "lra", "crest_db", "plr", "corr", "width_pct", "tilt"]
DEFAULT_THRESH = {"lufs_i": 1.0, "lra": 2.0, "crest_db": 1.5, "plr": 1.5, "corr": 0.15, "width_pct": 8.0,
                  "tilt": 0.5, "band_rel": 2.0}

BAND_METRICS = {
    "rel": "Livello relativo al totale [dB]",
    "level": "Livello assoluto [dB]",
    "share_pct": "Quota di energia [%]",
    "crest_db": "Crest factor di banda [dB]",
    "dyn": "Dinamica short-term di banda (P95-P10) [dB]",
    "width_pct": "Larghezza stereo di banda [%]",
    "corr": "Correlazione L/R di banda",
}


def band_level(bd, weighting):
    return bd["k_db"] if weighting == "k" else bd["rms_db"]


def band_value(g, bd, metric, weighting):
    if metric == "rel":
        return band_level(bd, weighting) - (g["lufs_i"] if weighting == "k" else g["rms_db"])
    if metric == "level":
        return band_level(bd, weighting)
    if metric == "dyn":
        return bd["dyn_k"] if weighting == "k" else bd["dyn_rms"]
    return bd[metric]


def spectrum_norm(third_db, mode):
    """mode: 'total' (livello totale 20 Hz-20 kHz), 'mid' (media 200 Hz-5 kHz), 'abs' (nessuna)."""
    if mode == "abs":
        return third_db
    if mode == "mid":
        m = (THIRD_FC >= 200) & (THIRD_FC <= 5000)
        return third_db - np.nanmean(third_db[m])
    return third_db - 10 * np.log10(np.nansum(10 ** (third_db / 10)))


def albums(tracks):
    return [t for t in tracks if not t["ref"]]


def summary_df(tracks):
    rows = []
    for i, t in enumerate(tracks, 1):
        row = {"#": "REF" if t["ref"] else str(i), "Brano": t["name"]}
        for k, lab, u, d in GLOBAL_COLS:
            row[LABEL[k]] = t["g"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def bands_df(tracks, weighting):
    rows = []
    for i, t in enumerate(tracks, 1):
        for bd in t["b"]["bands"]:
            row = {"#": "REF" if t["ref"] else str(i), "Brano": t["name"], "Banda": bd["label"]}
            for m, lab in BAND_METRICS.items():
                row[lab] = band_value(t["g"], bd, m, weighting)
            rows.append(row)
    return pd.DataFrame(rows)


def third_df(tracks, mode):
    data = {"Freq [Hz]": np.round(THIRD_FC, 1)}
    for t in tracks:
        data[("REF - " if t["ref"] else "") + t["name"]] = np.round(spectrum_norm(t["g"]["third_db"], mode), 2)
    return pd.DataFrame(data)


def album_mean(tracks, getter):
    vals = [getter(t) for t in albums(tracks)]
    if not vals:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.array(vals, dtype=float), axis=0)


def thr_for(columns, thresh):
    """Soglia di segnalazione per ogni colonna di deviation_df."""
    rev = {v: k for k, v in LABEL.items()}
    return np.array([thresh["band_rel"] if c.startswith("Banda") else thresh[rev[c]] for c in columns])


def deviation_df(tracks, weighting, thresh, baseline=None):
    """Scostamenti per brano su metriche globali + livello relativo delle sottobande.

    baseline=None  -> scostamento dalla MEDIA ALBUM (calcolata solo sui brani dell'album, reference escluse);
                      le reference compaiono come righe e non entrano mai nella media.
    baseline=track -> scostamento da quella reference (brano - reference). Il brano di riferimento non e' tra le
                      righe; in fondo si aggiunge la riga 'MEDIA ALBUM' (media album - reference).
    Restituisce (dev, flag): DataFrame con indice = nomi dei brani.
    """
    rows = [t for t in tracks if t is not baseline]
    alb = albums(tracks)
    nb = len(tracks[0]["b"]["bands"])
    cols = {}
    for k in DEV_METRICS:
        cols[LABEL[k]] = (lambda t, k=k: t["g"][k], thresh[k])
    for b in range(nb):
        lab = f"Banda {tracks[0]['b']['bands'][b]['label']} rel. [dB]"
        cols[lab] = (lambda t, b=b: band_value(t["g"], t["b"]["bands"][b], "rel", weighting), thresh["band_rel"])
    dev, flag = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for lab, (fn, th) in cols.items():
            v = np.array([fn(t) for t in rows], dtype=float)
            alb_mean = np.nanmean([fn(t) for t in alb])
            if baseline is None:
                d = v - alb_mean
            else:
                d = np.r_[v - fn(baseline), alb_mean - fn(baseline)]
            dev[lab], flag[lab] = d, np.abs(d) > th
    names = [("REF - " if t["ref"] else "") + t["name"] for t in rows] + (["MEDIA ALBUM"] if baseline is not None else [])
    return pd.DataFrame(dev, index=names), pd.DataFrame(flag, index=names)


def target_status(g, target_lufs, tol_lu, target_tp):
    """(ok, [note]) rispetto a target LUFS +/- tolleranza e true peak massimo."""
    notes = []
    if np.isfinite(g["lufs_i"]) and abs(g["lufs_i"] - target_lufs) > tol_lu:
        notes.append(f"LUFS {g['lufs_i']:+.1f} ({g['lufs_i'] - target_lufs:+.1f} dal target)")
    if g["tp_db"] > target_tp:
        notes.append(f"TP {g['tp_db']:.2f} > {target_tp:g} dBTP")
    if g["clip_n"] > 0:
        notes.append(f"{g['clip_n']} campioni a fondo scala")
    return (len(notes) == 0), notes
