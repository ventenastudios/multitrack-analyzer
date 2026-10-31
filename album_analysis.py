"""
Ventena Studios Toolkit - Album Mastering Analyzer
Motore di analisi (nessuna dipendenza da Streamlit, testabile da solo).

Scelte tecniche
- Loudness conforme a ITU-R BS.1770 / EBU R128 (K-weighting + gating), calcolato a blocchi da 100 ms.
- True peak con oversampling 4x (come nell'app Audio Analyzer), elaborato a chunk per limitare la RAM.
- Sottobande: crossover Linkwitz-Riley 4° ordine (2 x Butterworth 2° ordine in cascata) ad albero,
  quindi la somma delle bande e' piatta in ampiezza. Filtri causali con stato continuo tra i chunk.
- Tutto viene elaborato a chunk di ~20 s: in memoria resta solo il brano decodificato (float32).
"""
import hashlib
import io
import os
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

BLOCK_S = 0.1            # blocco elementare (s)
M_BLOCKS = 4             # finestra momentary 400 ms
ST_BLOCKS = 30           # finestra short-term 3 s
TL_HOP = 5               # nei timeline si tiene 1 punto ogni 0.5 s
DEFAULT_CROSSOVERS = (150.0, 1500.0, 6000.0)
F_MIN, F_MAX = 20.0, 20000.0
THIRD_FC = 1000.0 * 10 ** (0.1 * np.arange(-17, 14))   # 19.95 Hz ... 19.95 kHz (31 bande)
EPS = 1e-20


# --------------------------------------------------------------------------- utilita'
def file_digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def db10(p):
    return 10.0 * np.log10(np.maximum(p, EPS))


def fmt_hz(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def band_labels(crossovers):
    edges = [F_MIN] + list(crossovers) + [F_MAX]
    return [f"{fmt_hz(a)}-{fmt_hz(b)} Hz" for a, b in zip(edges[:-1], edges[1:])]


def validate_crossovers(crossovers, fs=44100):
    xo = [float(v) for v in crossovers]
    if any(b <= a for a, b in zip(xo[:-1], xo[1:])):
        raise ValueError("Le frequenze di crossover devono essere strettamente crescenti.")
    if xo[0] < 30 or xo[-1] > 0.45 * fs:
        raise ValueError(f"I crossover devono stare tra 30 Hz e {0.45 * fs:.0f} Hz.")
    return xo


# --------------------------------------------------------------------------- decodifica
def load_audio(data: bytes, name: str = "audio.wav"):
    """Restituisce (x[n,2] float32, fs, info). Mono -> duplicato su 2 canali (info['was_mono'])."""
    try:
        x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        import librosa  # fallback (m4a/aac ecc., richiede ffmpeg sul sistema)
        suffix = os.path.splitext(name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            y, fs = librosa.load(path, sr=None, mono=False)
        finally:
            os.unlink(path)
        x = np.atleast_2d(y).T.astype(np.float32)
    ch = x.shape[1]
    was_mono = ch == 1
    if was_mono:
        x = np.repeat(x, 2, axis=1)
    elif ch > 2:
        x = x[:, :2]
    return np.ascontiguousarray(x), int(fs), {"was_mono": was_mono, "channels": ch}


# --------------------------------------------------------------------------- filtri
def k_weight_sos(fs):
    """K-weighting BS.1770 (high-shelf + high-pass) per qualsiasi fs."""
    G, f0, Q = 3.999843853973347, 1681.974450955533, 0.7071752369554196
    K = np.tan(np.pi * f0 / fs)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K ** 2
    sh = np.r_[[(Vh + Vb * K / Q + K ** 2) / a0, 2 * (K ** 2 - Vh) / a0, (Vh - Vb * K / Q + K ** 2) / a0],
               [1.0, 2 * (K ** 2 - 1) / a0, (1 - K / Q + K ** 2) / a0]]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)
    a0 = 1 + K / Q + K ** 2
    hp = np.r_[[1.0, -2.0, 1.0], [1.0, 2 * (K ** 2 - 1) / a0, (1 - K / Q + K ** 2) / a0]]
    return np.vstack([sh, hp])


class _KFilter:
    """K-weighting con stato (uso a chunk)."""

    def __init__(self, fs):
        self.sos = k_weight_sos(fs)
        self.zi = np.zeros((self.sos.shape[0], 2, 2))

    def __call__(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class BandSplitter:
    """Crossover LR4 ad albero: low_i = LP_i(rest), rest = HP_i(rest). Somma piatta."""

    def __init__(self, fs, crossovers):
        self.lp, self.hp, self.zlp, self.zhp = [], [], [], []
        for f in crossovers:
            lp = signal.butter(2, f, "low", fs=fs, output="sos")
            hp = signal.butter(2, f, "high", fs=fs, output="sos")
            self.lp.append(np.vstack([lp, lp]))
            self.hp.append(np.vstack([hp, hp]))
            self.zlp.append(np.zeros((2, 2, 2)))
            self.zhp.append(np.zeros((2, 2, 2)))

    def process(self, x):
        rest, bands = x, []
        for i in range(len(self.lp)):
            low, self.zlp[i] = signal.sosfilt(self.lp[i], rest, axis=0, zi=self.zlp[i])
            rest, self.zhp[i] = signal.sosfilt(self.hp[i], rest, axis=0, zi=self.zhp[i])
            bands.append(low)
        bands.append(rest)
        return bands


# --------------------------------------------------------------------------- loudness
def _slide(v, w):
    if len(v) < w:
        return np.array([])
    c = np.cumsum(np.r_[0.0, v])
    return (c[w:] - c[:-w]) / w


def lufs(power):
    return -0.691 + db10(power)


def gated_lufs(z_blocks):
    """LUFS integrato con doppio gate (-70 assoluto, -10 relativo) da blocchi da 100 ms (z = potenza K sommata sui canali)."""
    m = _slide(z_blocks, M_BLOCKS)
    if m.size == 0:
        return float("nan")
    lm = lufs(m)
    g1 = lm > -70
    if not g1.any():
        return float("nan")
    rel = lufs(m[g1].mean()) - 10
    g2 = g1 & (lm > rel)
    if not g2.any():
        return float("nan")
    return float(lufs(m[g2].mean()))


def spread_p95_p10(power, offset_db=0.0, abs_gate=-70.0, rel_gate=-20.0):
    """Stile LRA: P95-P10 delle finestre short-term dopo gate assoluto e relativo."""
    if power.size == 0:
        return float("nan")
    lv = offset_db + db10(power)
    g = lv > abs_gate
    if not g.any():
        return float("nan")
    rel = offset_db + db10(power[g].mean()) + rel_gate
    sel = lv[g & (lv > rel)]
    if sel.size < 2:
        return float("nan")
    return float(np.percentile(sel, 95) - np.percentile(sel, 10))


# --------------------------------------------------------------------------- true peak
def true_peak(x, oversample=4, step=1 << 19, pad=64):
    """(tp_dB, tempo_s_relativo_in_campioni, n_campioni_oversampled>0dBFS). Chunked con margini."""
    n = len(x)
    best, pos, over0 = 0.0, 0.0, 0
    for i in range(0, n, step):
        j = min(i + step, n)
        a, b = max(0, i - pad), min(n, j + pad)
        up = signal.resample_poly(x[a:b], oversample, 1, axis=0)
        lo = (i - a) * oversample
        core = np.abs(up[lo:lo + (j - i) * oversample])
        if core.size == 0:
            continue
        m = float(core.max())
        over0 += int(np.count_nonzero(core > 1.0))
        if m > best:
            best = m
            pos = i + float(np.argmax(core.max(axis=1))) / oversample
    return float(db10(best ** 2)), pos, over0


# --------------------------------------------------------------------------- spettro
def third_octave_levels(f, p, fs):
    df = f[1] - f[0]
    out = np.full(len(THIRD_FC), np.nan)
    for k, fc in enumerate(THIRD_FC):
        lo, hi = fc * 10 ** -0.05, fc * 10 ** 0.05
        if hi > fs / 2:
            continue
        m = (f >= lo) & (f < hi)
        pw = p[m].sum() * df if m.any() else np.interp(fc, f, p) * (hi - lo)
        out[k] = db10(pw)
    return out


def spectral_features(f, p, third_db):
    sel = f >= 20
    ff, pp = f[sel], p[sel]
    tot = pp.sum()
    centroid = float((ff * pp).sum() / tot) if tot > 0 else float("nan")
    cs = np.cumsum(pp) / tot if tot > 0 else np.zeros_like(pp)
    rolloff = float(ff[min(np.searchsorted(cs, 0.85), len(ff) - 1)])
    m = (THIRD_FC >= 100) & (THIRD_FC <= 10000) & np.isfinite(third_db)
    tilt = float(np.polyfit(np.log2(THIRD_FC[m]), third_db[m], 1)[0]) if m.sum() > 3 else float("nan")
    return centroid, rolloff, tilt


# --------------------------------------------------------------------------- analisi principale
def analyze(x, fs, crossovers=DEFAULT_CROSSOVERS, want_global=True, want_bands=True, chunk_s=20.0):
    bl = int(round(fs * BLOCK_S))
    nb = len(x) // bl
    if nb < M_BLOCKS + 1:
        raise ValueError("Brano troppo corto per l'analisi (servono almeno ~0.5 s).")
    n = nb * bl
    chunk = bl * max(1, int(chunk_s / BLOCK_S))
    xo = validate_crossovers(crossovers, fs) if want_bands else []
    nbd = len(xo) + 1

    kf = _KFilter(fs)
    splitter = BandSplitter(fs, xo) if want_bands else None
    kfb = [_KFilter(fs) for _ in range(nbd)] if want_bands else []

    ll, rr, lr, zk = [], [], [], []
    peak, peak_i, clip_n = 0.0, 0, 0
    sum_l = sum_r = 0.0
    psd_acc, psd_w, nper = None, 0, 16384
    b_ll = [[] for _ in range(nbd)]; b_rr = [[] for _ in range(nbd)]
    b_lr = [[] for _ in range(nbd)]; b_zk = [[] for _ in range(nbd)]
    b_peak = [0.0] * nbd

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        xc = x[i0:i1].astype(np.float64)
        k = (i1 - i0) // bl
        xb = xc.reshape(k, bl, 2)
        ll.append((xb[..., 0] ** 2).mean(1)); rr.append((xb[..., 1] ** 2).mean(1))
        lr.append((xb[..., 0] * xb[..., 1]).mean(1))
        zk.append((kf(xc) ** 2).reshape(k, bl, 2).mean(1).sum(1))
        ax = np.abs(xc)
        pm = float(ax.max())
        if pm > peak:
            peak, peak_i = pm, i0 + int(np.argmax(ax.max(axis=1)))
        clip_n += int(np.count_nonzero(ax >= 0.9999))
        sum_l += float(xc[:, 0].sum()); sum_r += float(xc[:, 1].sum())
        if want_global and len(xc) >= nper:
            f, p = signal.welch(xc, fs, nperseg=nper, noverlap=nper // 2, axis=0)
            p = p.mean(axis=1)
            psd_acc = p * len(xc) if psd_acc is None else psd_acc + p * len(xc)
            psd_w += len(xc)
        if want_bands:
            for b, yb in enumerate(splitter.process(xc)):
                ybb = yb.reshape(k, bl, 2)
                b_ll[b].append((ybb[..., 0] ** 2).mean(1)); b_rr[b].append((ybb[..., 1] ** 2).mean(1))
                b_lr[b].append((ybb[..., 0] * ybb[..., 1]).mean(1))
                b_zk[b].append((kfb[b](yb) ** 2).reshape(k, bl, 2).mean(1).sum(1))
                b_peak[b] = max(b_peak[b], float(np.abs(yb).max()))

    cat = np.concatenate
    ll, rr, lr, zk = cat(ll), cat(rr), cat(lr), cat(zk)
    ms_blk = (ll + rr) / 2
    st_k = _slide(zk, ST_BLOCKS)
    st_ms = _slide(ms_blk, ST_BLOCKS)
    t_st = (np.arange(len(st_k)) + ST_BLOCKS / 2) * bl / fs
    sel = slice(None, None, TL_HOP)
    full_rms_db = float(db10(ms_blk.mean()))

    out = {"fs": fs, "duration": n / fs, "full_rms_db": full_rms_db, "global": None, "bands": None}

    # ---------------- globale
    if want_global:
        peak_db = float(db10(peak ** 2))
        tp_db, tp_pos, over0 = true_peak(x[:n])
        lufs_i = gated_lufs(zk)
        m_pow = _slide(zk, M_BLOCKS)
        lufs_m_max = float(lufs(m_pow.max())) if m_pow.size else float("nan")
        lufs_st_max = float(lufs(st_k.max())) if st_k.size else float("nan")
        lra = spread_p95_p10(st_k, offset_db=-0.691)
        sll, srr, slr = ll.mean(), rr.mean(), lr.mean()
        corr = float(slr / np.sqrt(sll * srr + EPS))
        mid_p, side_p = (sll + srr + 2 * slr) / 4, (sll + srr - 2 * slr) / 4
        mono_loss = float(db10(mid_p) - db10((sll + srr) / 2))
        lvl_blk = db10(ms_blk)
        active = np.where(lvl_blk > -60)[0]
        head = float(active[0] * BLOCK_S) if active.size else float(n / fs)
        tail = float((nb - 1 - active[-1]) * BLOCK_S) if active.size else 0.0
        f = np.fft.rfftfreq(nper, 1 / fs)
        psd = psd_acc / psd_w if psd_w else np.zeros_like(f)
        third = third_octave_levels(f, psd, fs)
        centroid, rolloff, tilt = spectral_features(f, psd, third)

        corr_st = _slide(lr, ST_BLOCKS) / np.sqrt(_slide(ll, ST_BLOCKS) * _slide(rr, ST_BLOCKS) + EPS)
        lv_st = lufs(st_k) if st_k.size else np.array([])
        sections = []
        if lv_st.size:
            i = int(np.argmax(lv_st)); sections.append(("Passaggio piu' forte (short-term)", float(t_st[i]), f"{lv_st[i]:.1f} LUFS"))
            ok = np.where(lv_st > -70)[0]
            if ok.size:
                j = ok[int(np.argmin(lv_st[ok]))]
                sections.append(("Passaggio piu' debole (short-term)", float(t_st[j]), f"{lv_st[j]:.1f} LUFS"))
            ok = np.where(lv_st > -60)[0]
            if ok.size:
                j = ok[int(np.argmin(corr_st[ok]))]
                sections.append(("Minima correlazione stereo", float(t_st[j]), f"corr {corr_st[j]:.2f}"))
        sections.append(("True peak massimo", float(tp_pos / fs), f"{tp_db:.2f} dBTP"))

        out["global"] = {
            "lufs_i": lufs_i, "lufs_st_max": lufs_st_max, "lufs_m_max": lufs_m_max, "lra": lra,
            "tp_db": tp_db, "tp_time": float(tp_pos / fs), "peak_db": peak_db, "peak_time": float(peak_i / fs),
            "rms_db": full_rms_db, "crest_db": peak_db - full_rms_db,
            "plr": tp_db - lufs_i if np.isfinite(lufs_i) else float("nan"),
            "psr": tp_db - lufs_st_max if np.isfinite(lufs_st_max) else float("nan"),
            "corr": corr, "ms_ratio_db": float(db10(mid_p) - db10(side_p)),
            "width_pct": float(100 * side_p / (mid_p + side_p + EPS)), "mono_loss_db": mono_loss,
            "dc_l": float(sum_l / n), "dc_r": float(sum_r / n), "clip_n": clip_n, "tp_over0_n": over0,
            "silence_head": head, "silence_tail": tail,
            "centroid": centroid, "rolloff85": rolloff, "tilt": tilt, "third_db": third,
            "timeline": {"t": t_st[sel], "st_lufs": lufs(st_k)[sel] if st_k.size else np.array([]),
                         "st_rms": db10(st_ms)[sel] if st_ms.size else np.array([]), "corr": corr_st[sel]},
            "sections": sections,
        }

    # ---------------- sottobande
    if want_bands:
        edges = [F_MIN] + xo + [F_MAX]
        bands, tl_rms, tl_k, sections = [], [], [], []
        labels = band_labels(xo)
        full_st_db = db10(st_ms)
        for b in range(nbd):
            bll, brr, blr, bz = cat(b_ll[b]), cat(b_rr[b]), cat(b_lr[b]), cat(b_zk[b])
            bms = (bll + brr) / 2
            rms_db = float(db10(bms.mean()))
            pk_db = float(db10(b_peak[b] ** 2))
            sll, srr, slr = bll.mean(), brr.mean(), blr.mean()
            mid_p, side_p = (sll + srr + 2 * slr) / 4, (sll + srr - 2 * slr) / 4
            bst_k, bst_ms = _slide(bz, ST_BLOCKS), _slide(bms, ST_BLOCKS)
            k_db = gated_lufs(bz)
            bands.append({
                "label": labels[b], "lo": edges[b], "hi": edges[b + 1],
                "rms_db": rms_db, "k_db": k_db, "share_pct": float(bms.mean()),
                "peak_db": pk_db, "crest_db": pk_db - rms_db,
                "corr": float(slr / np.sqrt(sll * srr + EPS)),
                "width_pct": float(100 * side_p / (mid_p + side_p + EPS)),
                "dyn_rms": spread_p95_p10(bst_ms, abs_gate=-120.0),
                "dyn_k": spread_p95_p10(bst_k, offset_db=-0.691, abs_gate=-120.0),
            })
            tl_rms.append(db10(bst_ms)[sel]); tl_k.append(lufs(bst_k)[sel])
            if bst_ms.size:
                rel = db10(bst_ms) - full_st_db
                ok = np.where(full_st_db > -60)[0]
                if ok.size:
                    j = ok[int(np.argmax(rel[ok]))]
                    sections.append((f"Banda {labels[b]}: massimo rispetto al totale", float(t_st[j]), f"{rel[j]:+.1f} dB"))
        tot_p = sum(bd["share_pct"] for bd in bands) + EPS      # quote normalizzate: somma = 100%
        for bd in bands:
            bd["share_pct"] = float(100 * bd["share_pct"] / tot_p)
        out["bands"] = {
            "crossovers": xo, "labels": labels, "bands": bands,
            "timeline": {"t": t_st[sel], "full_rms": full_st_db[sel] if full_st_db.size else np.array([]),
                         "rms": tl_rms, "k": tl_k},
            "sections": sections,
        }
    return out


def analyze_bytes(data, name, crossovers=DEFAULT_CROSSOVERS, want_global=True, want_bands=True):
    x, fs, info = load_audio(data, name)
    res = analyze(x, fs, crossovers, want_global, want_bands)
    res["info"] = info
    return res
