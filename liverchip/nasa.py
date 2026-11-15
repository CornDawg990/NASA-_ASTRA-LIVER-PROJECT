"""NASA Open Science Data Repository (OSDR / GeneLab) connector and hepatic-stress scoring.

What this module does
    1. Queries the OSDR Biological Data API for a study's assay/sample metadata (live).
    2. Parses an OSDR differential-expression table (spaceflight vs ground, log2 fold change).
    3. Scores curated liver gene panels, bootstraps a confidence interval over genes, and maps the
       resulting hepatic-stress index (HSI) onto the simulator's ``severity`` multiplier.

What it does NOT do
    * It never invents NASA data. When the network is unavailable and no file is uploaded, the app
      can generate a clearly labelled SYNTHETIC table in the same format so the pipeline can be
      demonstrated end to end.
    * The HSI -> severity mapping is a modelling choice (``HSI_REF``), not a measured conversion.
      It exists so that real spaceflight liver signatures can modulate the demo in a transparent,
      auditable way.

Gene panels are curated by the author from the literature on spaceflight liver responses
(Nrf2-mediated antioxidant response, lipid dysregulation, DNA-damage / p53 stress, loss of
synthetic function); they are a starting point, not a validated clinical signature.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

OSDR_API = "https://visualization.osdr.nasa.gov/biodata/api/v2"
OSDR_STUDY_URL = "https://osdr.nasa.gov/bio/repo/data/studies/{acc}"

PANELS: dict[str, list[str]] = {
    "Nrf2 / oxidative stress": ["Nqo1", "Hmox1", "Gclc", "Gclm", "Txnrd1", "Srxn1", "Gsta1", "Gsta2", "Gstm1", "Sqstm1"],
    "Lipid metabolism": ["Cd36", "Fasn", "Scd1", "Acaca", "Srebf1", "Pparg", "Ppara", "Cpt1a", "Plin2"],
    "DNA damage / p53 stress": ["Cdkn1a", "Gadd45a", "Mdm2", "Bax", "Ddit3"],
    "Synthetic function (loss)": ["Alb", "Apoa1", "Ttr", "Serpina1a", "Fgb", "Tf"],
}
SIGN = {"Nrf2 / oxidative stress": 1.0, "Lipid metabolism": 1.0, "DNA damage / p53 stress": 1.0, "Synthetic function (loss)": -1.0}
HSI_REF = 0.35  # HSI (mean signed log2FC across panels) that maps to severity = 1.0  [assumption]
SEV_MIN, SEV_MAX = 0.25, 4.0


# ---------------------------------------------------------------------------
# Live OSDR access
# ---------------------------------------------------------------------------
def fetch_osdr_metadata(accession: str, timeout: float = 15.0):
    """GET assay/sample metadata for an OSD accession. Raises on network / HTTP errors."""
    import requests

    acc = accession.strip().upper()
    if not acc.startswith("OSD-"):
        acc = "OSD-" + acc.replace("OSD", "").replace("-", "")
    url = f"{OSDR_API}/dataset/{acc}/assay/*/sample/*/"
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json(), url


def summarize_metadata(js) -> dict:
    """Defensive summary of an OSDR metadata response (its schema is not guaranteed)."""
    text = json.dumps(js).lower()
    flat = pd.json_normalize(js, sep=".") if isinstance(js, (list, dict)) else pd.DataFrame()
    if len(flat) == 1:  # one wide row -> key/value table
        kv = flat.iloc[0].astype(str).str.slice(0, 140).rename("value").rename_axis("field").reset_index()
    else:
        kv = flat
    return dict(
        table=kv,
        n_rows=int(len(flat)),
        n_fields=int(flat.shape[1]),
        mentions={
            "spaceflight": text.count("space flight") + text.count("spaceflight"),
            "ground control": text.count("ground control") + text.count("ground"),
            "liver": text.count("liver") + text.count("hepat"),
            "mouse / Mus musculus": text.count("mus musculus") + text.count("mouse"),
        },
        size_kb=len(text) / 1024,
    )


# ---------------------------------------------------------------------------
# Differential-expression parsing and panel scoring
# ---------------------------------------------------------------------------
def parse_de_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an OSDR-style differential-expression table to columns gene, log2fc, p."""
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}

    gene_col = next((c for c in cols if low[c] in ("symbol", "gene", "gene_symbol", "genesymbol", "gene_name", "gene.symbol")), None)
    if gene_col is None:
        obj = [c for c in cols if df[c].dtype == object]
        gene_col = obj[0] if obj else None
    fc_col = next((c for c in cols if any(k in low[c] for k in ("log2fc", "logfc", "log2foldchange", "log2_fc", "log2 fold"))), None)
    if fc_col is None:
        raise ValueError("No log2 fold-change column found (looked for log2fc / logFC / log2FoldChange).")
    p_col = next((c for c in cols if "adj" in low[c] or "padj" in low[c] or "fdr" in low[c]), None) or next(
        (c for c in cols if any(k in low[c] for k in ("p.value", "pvalue", "p_value", "pval"))), None
    )
    out = pd.DataFrame(
        {
            "gene": (df[gene_col].astype(str) if gene_col is not None else pd.Series(df.index.astype(str), index=df.index)),
            "log2fc": pd.to_numeric(df[fc_col], errors="coerce"),
            "p": pd.to_numeric(df[p_col], errors="coerce") if p_col is not None else np.nan,
        }
    ).dropna(subset=["log2fc"])
    out["gene_u"] = out["gene"].str.upper()
    return out.drop_duplicates("gene_u").reset_index(drop=True)


def panel_scores(de: pd.DataFrame, n_boot: int = 800, seed: int = 0) -> dict:
    """Mean log2FC per panel (with gene-bootstrap CI) and the combined hepatic-stress index."""
    rng = np.random.default_rng(seed)
    lut = de.set_index("gene_u")["log2fc"]
    rows, vals = [], {}
    for name, genes in PANELS.items():
        present = [g for g in genes if g.upper() in lut.index]
        v = lut.loc[[g.upper() for g in present]].to_numpy(float) if present else np.array([])
        vals[name] = v
        rows.append(dict(panel=name, found=len(present), total=len(genes), genes=", ".join(present),
                         mean_log2fc=float(v.mean()) if len(v) else np.nan))
    tab = pd.DataFrame(rows)
    usable = [k for k, v in vals.items() if len(v) >= 2]
    if not usable:
        raise ValueError("Fewer than 2 genes from any liver panel were found in this table (check gene symbols: mouse, e.g. Nqo1).")

    def hsi_of(sample: dict) -> float:
        return float(np.mean([SIGN[k] * sample[k].mean() for k in usable]))

    hsi = hsi_of({k: vals[k] for k in usable})
    boots = np.array(
        [hsi_of({k: rng.choice(vals[k], size=len(vals[k]), replace=True) for k in usable}) for _ in range(n_boot)]
    )
    lo, hi = np.quantile(boots, [0.025, 0.975])
    lo_b = np.empty(len(tab))
    hi_b = np.empty(len(tab))
    for i, name in enumerate(tab.panel):
        v = vals[name]
        if len(v) >= 2:
            b = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
            lo_b[i], hi_b[i] = np.quantile(b, [0.025, 0.975])
        else:
            lo_b[i] = hi_b[i] = np.nan
    tab["ci_lo"], tab["ci_hi"] = lo_b, hi_b
    return dict(table=tab, hsi=hsi, hsi_ci=(float(lo), float(hi)), usable=usable, n_genes=len(de))


def severity_from_hsi(hsi: float) -> float:
    return float(np.clip(hsi / HSI_REF, SEV_MIN, SEV_MAX))


# ---------------------------------------------------------------------------
# Clearly-labelled synthetic fallback (same schema as an OSDR DE table)
# ---------------------------------------------------------------------------
def synthetic_de_table(seed: int = 3, n_genes: int = 1800, effect: float = 1.0) -> pd.DataFrame:
    """SYNTHETIC differential-expression table for demonstrating the pipeline offline.

    Panel genes are drawn around plausible directions of change (up for stress panels, down for
    synthetic function); the rest is noise. ``effect`` scales the panel shifts. Not NASA data.
    """
    rng = np.random.default_rng(seed)
    mu = {"Nrf2 / oxidative stress": 0.45, "Lipid metabolism": 0.30, "DNA damage / p53 stress": 0.35, "Synthetic function (loss)": -0.30}
    rows = []
    for panel, genes in PANELS.items():
        for g in genes:
            rows.append((g, rng.normal(mu[panel] * effect, 0.22)))
    for i in range(n_genes - len(rows)):
        rows.append((f"Gene{i:04d}", rng.normal(0, 0.30)))
    df = pd.DataFrame(rows, columns=["SYMBOL", "Log2fc_(Flight)v(Ground)"])
    se = 0.15
    from scipy.stats import norm

    z = df["Log2fc_(Flight)v(Ground)"] / se
    df["P.value"] = 2 * norm.sf(np.abs(z))
    order = np.argsort(df["P.value"].to_numpy())
    ranks = np.empty(len(df))
    ranks[order] = np.arange(1, len(df) + 1)
    df["Adj.p.value"] = np.minimum(1.0, df["P.value"] * len(df) / ranks)
    return df
