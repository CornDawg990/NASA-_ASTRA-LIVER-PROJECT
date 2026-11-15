"""Streaming anomaly detection and artifact rejection for chip telemetry.

Design goal (from the proposal): "a spike is a real biological event, not a glitch".
A plain z-score cannot establish that, so detection is split into three stages:

Stage 1 - transient/artifact rejection (1-minute resolution)
    Causal rolling median / MAD z-score, floored by the instrument noise measured at
    calibration. Short outlier runs (<= ``max_artifact_len`` samples) are bubbles, packet
    glitches or electrode transients: they are masked before anything downstream sees them.

Stage 2 - sustained biological events (10-minute blocks)
    Cleaned data are block-averaged (noise drops by sqrt(10)) and compared with a *lagged
    baseline* (median of an 8 h window that ends 1 h ago). Averaging first is what makes hour-scale
    ramps (tau ~ 3 h) detectable; the 8 h window keeps the detector blind to slow chronic trends
    (about 1 sigma at typical chronic rates) while letting an event "end" so that a later
    washout is its own event. An event needs a persistent run in >= 2 independent channels;
    one channel alone is a sensor fault.

Stage 3 - paired flight-vs-ground CUSUM
    Chronic offsets that every rolling baseline adapts to are caught by differencing against the
    matched ground-control chip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .physiology import CHANNELS, SPH


@dataclass(frozen=True)
class DetectorConfig:
    # stage 1
    window: int = 360  # 6 h robust baseline window (samples)
    min_periods: int = 60
    art_z: float = 5.0
    max_artifact_len: int = 8  # samples
    gap_close: int = 2
    floor_frac: float = 0.7
    calib_samples: int = 720
    # stage 2
    block: int = 10  # samples per block (10 min)
    base_window_blocks: int = 48  # 8 h baseline window
    base_lag_blocks: int = 6  # ends 1 h ago, so a developing event is not in its own baseline
    base_min_blocks: int = 18  # 3 h
    event_z: float = 4.0
    min_run_blocks: int = 3  # 30 min persistence
    group_gap_blocks: int = 12  # 2 h: channels flagged within 2 h belong to one event


def robust_noise(x: np.ndarray, first: int = 720) -> float:
    """White-noise sigma from the MAD of first differences (insensitive to trends/events)."""
    seg = np.asarray(x[:first], float)
    seg = seg[np.isfinite(seg)]
    if len(seg) < 10:
        return float("nan")
    d = np.diff(seg)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def rolling_robust_z(x: np.ndarray, cfg: DetectorConfig, sigma_floor: float) -> np.ndarray:
    s = pd.Series(x)
    med = s.rolling(cfg.window, min_periods=cfg.min_periods).median().shift(1)
    dev = (s - med).abs()
    mad = 1.4826 * dev.rolling(cfg.window, min_periods=cfg.min_periods).median().shift(1)
    sigma = np.maximum(mad.to_numpy(), cfg.floor_frac * sigma_floor)
    with np.errstate(invalid="ignore"):
        return (s - med).to_numpy() / sigma


def _runs(mask: np.ndarray, close: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs of True, bridging gaps of up to ``close`` samples."""
    mask = np.asarray(mask, bool)
    if close > 0:
        pad = close + 1
        mask = ndimage.binary_closing(np.pad(mask, pad), structure=np.ones(close + 1, bool))[pad:-pad]
    lab, k = ndimage.label(mask)
    if k == 0:
        return []
    return [(sl[0].start, sl[0].stop - 1) for sl in ndimage.find_objects(lab)]


def _merge(intervals: list[tuple], gap: int) -> list[list[tuple]]:
    intervals = sorted(intervals)
    groups: list[list[tuple]] = []
    cur_end = -(10**9)
    for iv in intervals:
        if groups and iv[0] - cur_end <= gap:
            groups[-1].append(iv)
            cur_end = max(cur_end, iv[1])
        else:
            groups.append([iv])
            cur_end = iv[1]
    return groups


@dataclass
class DetectionResult:
    z: pd.DataFrame  # stage-1 rolling robust z (1 min)
    zblk: pd.DataFrame  # stage-2 block z vs lagged slow baseline (10 min)
    artifact_mask: pd.DataFrame
    clean: pd.DataFrame
    events: pd.DataFrame
    sigma_floor: dict
    block: int


def _block_means(x: np.ndarray, B: int, min_valid: int = 4) -> np.ndarray:
    nb = len(x) // B
    a = x[: nb * B].reshape(nb, B)
    valid = np.isfinite(a).sum(1)
    tot = np.where(np.isfinite(a), a, 0.0).sum(1)
    m = np.full(nb, np.nan)
    ok = valid >= min_valid
    m[ok] = tot[ok] / valid[ok]
    return m


def detect(obs: pd.DataFrame, cfg: DetectorConfig = DetectorConfig()) -> DetectionResult:
    n = len(obs)
    z = pd.DataFrame(index=obs.index)
    art = pd.DataFrame(False, index=obs.index, columns=CHANNELS)
    floors: dict[str, float] = {}
    art_runs: list[tuple[int, int, str]] = []

    # ---------------- stage 1: transients ----------------
    for c in CHANNELS:
        x = obs[c].to_numpy(float)
        xs = np.log(np.clip(x, 1e-6, None)) if c == "albumin" else x  # multiplicative noise
        floor = robust_noise(xs, cfg.calib_samples)
        floors[c] = floor
        zc = rolling_robust_z(xs, cfg, floor)
        z[c] = zc
        f = np.nan_to_num(np.abs(zc), nan=0.0) > cfg.art_z
        for a, b in _runs(f, cfg.gap_close):
            if (b - a + 1) <= cfg.max_artifact_len:
                art_runs.append((a, b, c))
                art.iloc[max(0, a - 1) : min(n, b + 2), art.columns.get_loc(c)] = True
    for c in CHANNELS:
        art[c] = art[c] | obs[c].isna()
    clean = obs.mask(art)

    rows = []
    for g in _merge(art_runs, 3):
        a, b = min(i[0] for i in g), max(i[1] for i in g)
        pk = max(np.nanmax(np.abs(z[c].to_numpy()[s0 : e0 + 1])) for s0, e0, c in g)
        rows.append(dict(kind="artifact", start=a, end=b, channels=sorted({i[2] for i in g}), confirmed=b, peak_z=float(pk)))

    # ---------------- stage 2: sustained events on cleaned blocks ----------------
    B = cfg.block
    nb = n // B
    zb = pd.DataFrame(index=np.arange(nb), columns=CHANNELS, dtype=float)
    chan_runs: list[tuple[int, int, str]] = []
    for c in CHANNELS:
        x = clean[c].to_numpy(float)
        xs = np.log(np.clip(x, 1e-6, None)) if c == "albumin" else x
        bm = _block_means(xs, B)
        s = pd.Series(bm)
        base = (
            s.rolling(cfg.base_window_blocks, min_periods=cfg.base_min_blocks).median().shift(cfg.base_lag_blocks)
        ).to_numpy()
        sig_emp = robust_noise(bm[np.isfinite(bm)], len(bm))
        sig = max(sig_emp if np.isfinite(sig_emp) else 0.0, cfg.floor_frac * floors[c] / np.sqrt(B))
        with np.errstate(invalid="ignore"):
            zz = (bm - base) / sig
        zb[c] = zz
        f = np.nan_to_num(np.abs(zz), nan=0.0) > cfg.event_z
        for a, b in _runs(f, 1):
            if (b - a + 1) >= cfg.min_run_blocks:
                chan_runs.append((a, b, c))

    for g in _merge(chan_runs, cfg.group_gap_blocks):
        chans = sorted({i[2] for i in g})
        a, b = min(i[0] for i in g), max(i[1] for i in g)
        pk = max(np.nanmax(np.abs(zb[c].to_numpy()[s0 : e0 + 1])) for s0, e0, c in g)
        first = {}
        for s0, _, c in g:
            first[c] = min(first.get(c, 10**9), s0)
        if len(chans) >= 2:
            second = sorted(first.values())[1]
            rows.append(
                dict(kind="biological", start=a * B, end=(b + 1) * B - 1, channels=chans,
                     confirmed=(second + cfg.min_run_blocks) * B, peak_z=float(pk))
            )
        else:
            rows.append(
                dict(kind="sensor_fault", start=a * B, end=(b + 1) * B - 1, channels=chans,
                     confirmed=(a + cfg.min_run_blocks) * B, peak_z=float(pk))
            )

    ev = pd.DataFrame(rows, columns=["kind", "start", "end", "channels", "confirmed", "peak_z"])
    if len(ev):
        ev = ev.sort_values("start").reset_index(drop=True)
    ev["start_h"] = ev["start"] / SPH
    ev["confirmed_h"] = ev["confirmed"] / SPH
    ev["duration_min"] = ev["end"] - ev["start"] + 1
    return DetectionResult(z=z, zblk=zb, artifact_mask=art, clean=clean, events=ev, sigma_floor=floors, block=B)


def evaluate_detection(truth_events: list[dict], det: DetectionResult, tol_h: float = 3.0) -> dict:
    """Score the detector against the events injected by the simulator."""
    ev = det.events
    bubbles = [e for e in truth_events if e["type"] == "bubble"]
    must = [e for e in truth_events if e["type"] in ("spe", "drug", "drug_end")]
    allow = must + [e for e in truth_events if e["type"] == "countermeasure"]
    bio = ev[ev.kind == "biological"] if len(ev) else ev
    tol = int(tol_h * SPH)

    hit = sum(
        int(det.artifact_mask.iloc[b["start"] : b["start"] + b["length"]][b["channels"]].to_numpy().any())
        for b in bubbles
    )
    bubble_recall = hit / len(bubbles) if bubbles else float("nan")

    detected, delays = 0, []
    for e in must:
        cand = bio[(bio.start >= e["start"] - tol) & (bio.start <= e["start"] + tol)] if len(bio) else bio
        if len(cand):
            detected += 1
            delays.append(float(cand.confirmed.min() - e["start"]))
    bio_recall = detected / len(must) if must else float("nan")

    def explained(r) -> bool:
        for e in allow:
            span = e["length"] if e["type"] in ("spe",) else 0
            if e["start"] - tol <= r.start <= e["start"] + span + tol + 24 * SPH:
                return True
        return False

    false_bio = int(sum(not explained(r) for r in bio.itertuples())) if len(bio) else 0
    bubble_alarms = 0  # unexplained biological alarms that begin within 1 h after a bubble
    if len(bio):
        for r in bio.itertuples():
            if explained(r):
                continue
            if any(0 <= r.start - b["start"] <= SPH for b in bubbles):
                bubble_alarms += 1
    return dict(
        n_bubbles=len(bubbles),
        bubble_recall=bubble_recall,
        n_bio_truth=len(must),
        bio_recall=bio_recall,
        median_delay_min=float(np.median(delays)) if delays else float("nan"),
        false_bio_alarms=false_bio,
        bubble_triggered_alarms=bubble_alarms,
        artifact_events=int((ev.kind == "artifact").sum()) if len(ev) else 0,
    )


# ---------------------------------------------------------------------------
# Stage 3: paired flight-vs-ground divergence (catches chronic offsets)
# ---------------------------------------------------------------------------
@dataclass
class DivergenceResult:
    hours: np.ndarray
    diff: pd.DataFrame
    pct: pd.DataFrame
    cusum: pd.DataFrame
    onset_h: dict
    confirmed_h: float | None


def paired_divergence(flight_clean: pd.DataFrame, ground_clean: pd.DataFrame, k: float = 0.5, h: float = 5.0) -> DivergenceResult:
    n = min(len(flight_clean), len(ground_clean))
    nh = n // SPH
    idx = np.repeat(np.arange(nh), SPH)
    fh = flight_clean.iloc[: nh * SPH].groupby(idx).mean()
    gh = ground_clean.iloc[: nh * SPH].groupby(idx).mean()
    diff = fh - gh
    pct = 100.0 * diff / gh.abs()
    cus = pd.DataFrame(0.0, index=diff.index, columns=CHANNELS)
    onset: dict = {}
    for c in CHANNELS:
        d = diff[c].interpolate(limit_direction="both").to_numpy()
        sig = robust_noise(d, len(d))
        if not np.isfinite(sig) or sig <= 0:
            sig = float(np.nanstd(d)) + 1e-9
        pos = neg = 0.0
        out = np.zeros(len(d))
        first = None
        for i, v in enumerate(d):
            zi = v / max(sig, 1e-9)
            pos = max(0.0, pos + zi - k)
            neg = max(0.0, neg - zi - k)
            out[i] = max(pos, neg)
            if first is None and out[i] > h:
                first = i
        cus[c] = out
        onset[c] = first
    hits = sorted(v for v in onset.values() if v is not None)
    return DivergenceResult(
        hours=np.arange(nh), diff=diff, pct=pct, cusum=cus, onset_h=onset,
        confirmed_h=float(hits[1]) if len(hits) >= 2 else None,
    )
