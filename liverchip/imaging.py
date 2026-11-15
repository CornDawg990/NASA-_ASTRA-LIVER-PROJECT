"""Optical mapping: synthetic hepatocyte fluorescence micrographs, segmentation, bleb classifier.

The proposal's second data stream is an automated fluorescence microscope that "segments cell
borders and flags early-stage toxicity when cells bleb or lose hexagonal shape".

What is real here
    * a classical segmentation pipeline (smoothing -> Otsu foreground -> nuclear markers ->
      marker-controlled watershed on the membrane image) that also runs on *your own* images
    * per-cell shape / texture features and a trained gradient-boosting classifier
What is simulated
    * the training and test micrographs. They are procedurally generated (hexagonal monolayer
      with jitter; damaged cells become rounded, bleb-ridden, with pyknotic nuclei). The
      classifier therefore reports performance on *synthetic* data only, and the app says so.
Swap-in path: replace ``segment`` with Cellpose/StarDist/MONAI and retrain ``fit_classifier``
on labelled real images.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import filters, measure, morphology, segmentation, feature

MODEL_PATH = os.path.join(os.path.dirname(__file__), "_bleb_model.joblib")
FEATURES = ["solidity", "eccentricity", "circularity", "hex6", "rough", "nc_ratio", "nuc_int", "cyto_int", "area_rel"]
# shape/texture-only features: usable when no nuclear channel is available (e.g. a user upload)
FEATURES_SHAPE = ["solidity", "eccentricity", "circularity", "hex6", "rough", "cyto_int", "area_rel"]


# ---------------------------------------------------------------------------
# Synthetic micrograph generator
# ---------------------------------------------------------------------------
@dataclass
class FieldImage:
    cyto: np.ndarray  # membrane / cytoplasm channel (green), float 0..1
    nuc: np.ndarray  # nuclear channel (blue), float 0..1
    truth_labels: np.ndarray  # per-pixel cell id (0 = background)
    damaged: np.ndarray  # bool per cell id (index 0 unused)
    p_damaged: float

    @property
    def rgb(self) -> np.ndarray:
        out = np.zeros(self.cyto.shape + (3,))
        out[..., 1] = self.cyto
        out[..., 2] = self.nuc
        out[..., 0] = 0.25 * self.cyto
        return np.clip(out, 0, 1)


def make_field(size: int = 384, pitch: float = 30.0, damaged_frac: float = 0.2, seed: int = 0) -> FieldImage:
    rng = np.random.default_rng(seed)
    # jittered hexagonal lattice of cell centres
    pts = []
    row_h = pitch * 0.866
    j = 0
    y = -pitch
    while y < size + 2 * pitch:
        x = -pitch + (pitch / 2 if j % 2 else 0)
        while x < size + 2 * pitch:
            pts.append((x + rng.normal(0, 0.10 * pitch), y + rng.normal(0, 0.10 * pitch)))
            x += pitch
        y += row_h
        j += 1
    pts = np.array(pts)
    # keep a padded ring of seeds outside the frame so border cells are not giant Voronoi wedges
    ring = pitch * 1.2
    keep = (pts[:, 0] > -ring) & (pts[:, 0] < size + ring) & (pts[:, 1] > -ring) & (pts[:, 1] < size + ring)
    pts = pts[keep]
    n = len(pts)
    damaged = np.zeros(n + 1, dtype=bool)
    k = int(round(damaged_frac * n))
    damaged[1 + rng.permutation(n)[:k]] = True

    yy, xx = np.mgrid[0:size, 0:size]
    tree = cKDTree(pts)
    d, idx = tree.query(np.c_[xx.ravel(), yy.ravel()], k=2)
    d1 = d[:, 0].reshape(size, size)
    d2 = d[:, 1].reshape(size, size)
    owner = idx[:, 0].reshape(size, size) + 1

    cyto = np.zeros((size, size))
    nuc = np.zeros((size, size))
    truth = np.zeros((size, size), dtype=int)
    margin = d2 - d1  # distance to the Voronoi edge (x2); small => membrane

    for cid in range(1, n + 1):
        cx, cy = pts[cid - 1]
        x0, x1 = int(max(cx - pitch, 0)), int(min(cx + pitch, size))
        y0, y1 = int(max(cy - pitch, 0)), int(min(cy + pitch, size))
        X, Y = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
        rho = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        own = owner[y0:y1, x0:x1] == cid
        if not damaged[cid]:
            body = own & (margin[y0:y1, x0:x1] > 1.8)  # polygon with a thin gap
            edge = own & (margin[y0:y1, x0:x1] <= 6.0) & body  # cortical actin ring
            val = np.where(body, 0.50 + 0.04 * rng.standard_normal(), 0.0)
            val = np.where(edge, 0.85, val)
            nr = 0.20 * pitch * rng.uniform(0.9, 1.1)
            nval = 0.55 * rng.uniform(0.9, 1.1) * np.exp(-0.5 * (rho / nr) ** 2) * (rho < 2.2 * nr)
            region = body
        else:
            r0 = 0.40 * pitch * rng.uniform(0.9, 1.05)
            rr = r0 * (
                1
                + 0.16 * np.sin(rng.integers(5, 9) * th + rng.uniform(0, 6.28))
                + 0.08 * np.sin(rng.integers(9, 14) * th + rng.uniform(0, 6.28))
            )
            body = rho < rr
            for _ in range(rng.integers(2, 5)):  # blebs: small bubbles budding off the membrane
                a = rng.uniform(0, 2 * np.pi)
                bx, by = 0.95 * r0 * np.cos(a), 0.95 * r0 * np.sin(a)
                body = body | (np.hypot(X - bx, Y - by) < rng.uniform(0.10, 0.17) * pitch)
            ring = body & ~ndimage.binary_erosion(body, iterations=2)
            val = np.where(body, 0.38 + 0.04 * rng.standard_normal(), 0.0)
            val = np.where(ring, 0.80, val)
            nr = 0.11 * pitch * rng.uniform(0.9, 1.1)  # pyknotic: small, dense, bright
            nval = 0.95 * np.exp(-0.5 * (rho / nr) ** 2) * (rho < 2.2 * nr)
            if rng.random() < 0.35:  # karyorrhexis: fragmented nucleus
                nval = nval + 0.8 * np.exp(-0.5 * (np.hypot(X - 0.45 * nr * 2, Y) / (0.7 * nr)) ** 2)
            region = body
        sub = cyto[y0:y1, x0:x1]
        sub[region] = np.maximum(sub[region], val[region])
        nsub = nuc[y0:y1, x0:x1]
        nsub[region] = np.maximum(nsub[region], nval[region])
        tsub = truth[y0:y1, x0:x1]
        tsub[region & (tsub == 0)] = cid

    cyto = ndimage.gaussian_filter(cyto, 1.1)
    nuc = ndimage.gaussian_filter(nuc, 1.4)
    photons = 220.0
    cyto = rng.poisson(np.clip(cyto, 0, None) * photons) / photons + rng.normal(0, 0.012, cyto.shape) + 0.03
    nuc = rng.poisson(np.clip(nuc, 0, None) * photons) / photons + rng.normal(0, 0.012, nuc.shape) + 0.02
    return FieldImage(np.clip(cyto, 0, 1), np.clip(nuc, 0, 1), truth, damaged, damaged_frac)


# ---------------------------------------------------------------------------
# Segmentation + features
# ---------------------------------------------------------------------------
def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, k = ndimage.label(mask)
    if k == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    holes = _drop_small(~mask, max_px + 1)  # background regions larger than max_px are real gaps
    return ~(~mask & holes) if False else (mask | (~mask & ~holes))


def segment(cyto: np.ndarray, nuc: np.ndarray | None = None) -> np.ndarray:
    """Marker-controlled watershed. Returns integer label image."""
    c = filters.gaussian(cyto, 1.0)
    thr = filters.threshold_otsu(c)
    fg = c > 0.75 * thr
    fg = _drop_small(fg, 40)
    fg = _fill_small_holes(fg, 30)  # only tiny holes: dark voids between cells are real
    if nuc is not None:
        n = filters.gaussian(nuc, 1.2)
        nmask = n > filters.threshold_otsu(n)
        nmask = ndimage.binary_dilation(nmask, structure=morphology.disk(3))  # merge fragmented nuclei
        markers = measure.label(nmask & fg)
    else:
        dist = ndimage.distance_transform_edt(fg)
        pk = feature.peak_local_max(dist, min_distance=7, labels=measure.label(fg), exclude_border=False)
        markers = np.zeros_like(fg, dtype=int)
        markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(c, markers, mask=fg)
    # drop tiny fragments
    for r in measure.regionprops(lab):
        if r.area < 60:
            lab[lab == r.label] = 0
    return lab


def _radial_signature(mask: np.ndarray, nb: int = 96) -> np.ndarray:
    er = ndimage.binary_erosion(mask)
    bd = mask & ~er
    ys, xs = np.nonzero(bd)
    cy, cx = ndimage.center_of_mass(mask)
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(ys - cy, xs - cx)
    bins = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    sig = np.full(nb, np.nan)
    for b in range(nb):
        m = bins == b
        if m.any():
            sig[b] = rad[m].max()
    ok = np.isfinite(sig)
    if ok.sum() < 8:
        return np.full(nb, np.nanmean(sig) if ok.any() else 1.0)
    xi = np.arange(nb)
    return np.interp(xi, xi[ok], sig[ok], period=nb)


def cell_features(labels: np.ndarray, cyto: np.ndarray, nuc: np.ndarray | None) -> pd.DataFrame:
    props = measure.regionprops(labels, intensity_image=cyto)
    areas = np.array([r.area for r in props]) if props else np.array([1.0])
    med_area = float(np.median(areas))
    nthr = None
    if nuc is not None:
        n = filters.gaussian(nuc, 1.2)
        nthr = filters.threshold_otsu(n)
    rows = []
    for r in props:
        sl = r.slice
        mask = np.pad(r.image, 1)
        sig = _radial_signature(mask)
        c0 = sig.mean()
        spec = np.abs(np.fft.rfft(sig - c0)) / (len(sig) * max(c0, 1e-6)) * 2
        hex6 = spec[6] if len(spec) > 6 else 0.0
        rough = float(np.sqrt(np.sum(spec[8:] ** 2)))
        circ = 4 * np.pi * r.area / max(r.perimeter, 1.0) ** 2
        if nuc is not None:
            nm = (filters.gaussian(nuc, 1.2)[sl] > nthr) & r.image
            nc = nm.sum() / r.area
            ni = float(nuc[sl][nm].mean()) if nm.any() else 0.0
        else:
            nc, ni = 0.15, 0.5
        rows.append(
            dict(
                label=r.label,
                cy=r.centroid[0],
                cx=r.centroid[1],
                area=r.area,
                solidity=r.solidity,
                eccentricity=r.eccentricity,
                circularity=min(circ, 1.2),
                hex6=hex6,
                rough=rough,
                nc_ratio=nc,
                nuc_int=ni,
                cyto_int=_mean_int(r),
                area_rel=r.area / med_area,
            )
        )
    return pd.DataFrame(rows)


def _mean_int(r) -> float:
    try:
        return float(r.intensity_mean)
    except AttributeError:  # older scikit-image
        return float(r.mean_intensity)


def _truth_for_cells(labels: np.ndarray, fld: FieldImage, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=int)
    for i, lab in enumerate(df["label"]):
        m = labels == lab
        t = fld.truth_labels[m]
        t = t[t > 0]
        out[i] = int(fld.damaged[np.bincount(t).argmax()]) if len(t) else 0
    return out


def build_dataset(n_fields: int, seed0: int, size: int = 384) -> tuple[pd.DataFrame, np.ndarray]:
    frames, ys = [], []
    rng = np.random.default_rng(seed0)
    for i in range(n_fields):
        frac = float(rng.choice([0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0]))
        f = make_field(size=size, damaged_frac=frac, seed=seed0 * 1000 + i)
        lab = segment(f.cyto, f.nuc)
        df = cell_features(lab, f.cyto, f.nuc)
        frames.append(df)
        ys.append(_truth_for_cells(lab, f, df))
    return pd.concat(frames, ignore_index=True), np.concatenate(ys)


def fit_classifier(features: list[str], data=None, n_train: int = 14, n_test: int = 6, seed: int = 7):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    if data is None:
        data = (build_dataset(n_train, seed), build_dataset(n_test, seed + 1))
    (Xtr, ytr), (Xte, yte) = data
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=0)
    clf.fit(Xtr[features], ytr)
    pte = clf.predict_proba(Xte[features])[:, 1]
    card = dict(
        features=list(features),
        n_train_cells=int(len(ytr)),
        n_test_cells=int(len(yte)),
        accuracy=float(accuracy_score(yte, pte > 0.5)),
        auc=float(roc_auc_score(yte, pte)),
        data="synthetic micrographs (procedurally generated) - NOT real hepatocytes",
        importance=dict(zip(features, map(float, clf.feature_importances_))),
    )
    return clf, card, data


def save_model(path: str = MODEL_PATH):
    import joblib

    clf_full, card_full, data = fit_classifier(FEATURES)
    clf_shape, card_shape, _ = fit_classifier(FEATURES_SHAPE, data=data)
    joblib.dump({"full": (clf_full, card_full), "shape": (clf_shape, card_shape)}, path)
    return {"full": card_full, "shape": card_shape}


def load_model(path: str = MODEL_PATH):
    """Returns {"full": (clf, card), "shape": (clf, card)}; trains on first use if absent."""
    import joblib

    if not os.path.exists(path):
        save_model(path)
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class ImageAnalysis:
    labels: np.ndarray
    cells: pd.DataFrame
    bleb_fraction: float
    overlay: np.ndarray
    n_cells: int


def analyze(cyto: np.ndarray, nuc: np.ndarray | None, models) -> ImageAnalysis:
    """``models`` is the dict from ``load_model``; the shape-only model is used without a nuclear channel."""
    labels = segment(cyto, nuc)
    df = cell_features(labels, cyto, nuc)
    if len(df) == 0:
        return ImageAnalysis(labels, df, float("nan"), np.zeros(cyto.shape + (3,)), 0)
    key = "full" if nuc is not None else "shape"
    clf, card = models[key]
    df["p_damaged"] = clf.predict_proba(df[card["features"]])[:, 1]
    frac = float((df["p_damaged"] > 0.5).mean())
    base = np.stack([0.25 * cyto, cyto, nuc if nuc is not None else 0.4 * cyto], -1)
    base = np.clip(base, 0, 1)
    bd = segmentation.find_boundaries(labels, mode="inner")
    overlay = base.copy()
    pd_map = np.zeros(labels.shape)
    pd_map[labels > 0] = df.set_index("label")["p_damaged"].reindex(labels[labels > 0]).to_numpy()
    overlay[bd & (pd_map > 0.5)] = (1.0, 0.25, 0.25)
    overlay[bd & (pd_map <= 0.5) & (labels > 0)] = (0.30, 1.0, 0.55)
    return ImageAnalysis(labels, df, frac, overlay, len(df))


def calibration_curve(models, fractions=None, reps: int = 2, seed: int = 99) -> pd.DataFrame:
    fractions = np.linspace(0, 1, 11) if fractions is None else fractions
    rows = []
    for i, p in enumerate(fractions):
        for r in range(reps):
            f = make_field(damaged_frac=float(p), seed=seed + 31 * i + r)
            a = analyze(f.cyto, f.nuc, models)
            true = float(f.damaged[1:].mean())
            rows.append(dict(true=true, measured=a.bleb_fraction))
    return pd.DataFrame(rows)
