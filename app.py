"""Liver-Chip Mission Control · NASA Space Apps demo.

Run:  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import itertools
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from liverchip import comms, imaging, nasa
from liverchip.physiology import CHANNEL_LABELS, CHANNELS, PRESETS, SPH, bleb_probability
from liverchip.pipeline import CREW_THRESHOLD, benchmark, run_mission, summarize_benchmark
from liverchip.twin import LEVELS, forecast, index_at_hour

st.set_page_config(page_title="Liver-Chip Mission Control", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

LEVEL_COL = {0: "#34d399", 1: "#fbbf24", 2: "#fb923c", 3: "#f87171"}
C_CLEAN, C_ART, C_GROUND, C_TWIN, C_TRUTH, C_BAND = "#2dd4bf", "#f87171", "#7aa2f7", "#a78bfa", "#fbbf24", "rgba(167,139,250,0.22)"
_key = itertools.count()

st.markdown(
    """
<style>
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1500px}
.pill{display:inline-block;padding:.15rem .7rem;border-radius:999px;font-weight:700;letter-spacing:.06em;font-size:.85rem}
.sim-banner{border:1px solid #fbbf2466;background:#fbbf2410;border-radius:10px;padding:.55rem .9rem;font-size:.88rem;margin-bottom:.6rem}
div[data-testid="stMetric"]{background:#111a2e;border:1px solid #1e2b47;border-radius:12px;padding:.7rem .9rem}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Simulating mission and running the digital twin…")
def cached_run(scn, seed, mismatch, bubble_rate, crew_interval, shielding):
    return run_mission(scn, seed=seed, mismatch=mismatch, bubble_rate=bubble_rate, crew_interval_h=crew_interval, shielding=shielding)


@st.cache_resource(show_spinner="Loading cell-imaging models…")
def cached_models():
    return imaging.load_model()


@st.cache_resource(show_spinner="Rendering synthetic micrograph…")
def cached_field(frac: float, seed: int):
    return imaging.make_field(damaged_frac=float(frac), seed=int(seed))


@st.cache_resource(show_spinner="Measuring classifier calibration…")
def cached_calibration():
    return imaging.calibration_curve(cached_models(), reps=1)


def style(fig, h=None, sub=False):
    """Shared dark theme. ``sub=True`` reserves headroom so the legend clears subplot titles."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=110 if sub else 50, b=10),
        legend=dict(orientation="h", x=0, y=1.0 + (0.075 if sub else 0.0), yanchor="bottom"), font=dict(size=12),
        **({"height": h} if h else {}),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", key=f"pc{next(_key)}")


# --------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🛰️ Mission setup")
preset = st.sidebar.selectbox("Mission scenario", list(PRESETS), index=1)
base = PRESETS[preset]
k = preset
days = st.sidebar.slider("Duration (days)", 3.0, 21.0, float(base.days), 0.5, key=f"days_{k}")
dose = st.sidebar.slider("Chronic dose-equivalent rate (mSv/day)", 0.0, 3.0, float(base.dose_rate_msv_day), 0.05, key=f"dose_{k}",
                         help="Mars-transit cruise measured by MSL/RAD ≈ 1.8 mSv/day. The ISS value is an approximate assumption.")
sev_manual = st.sidebar.slider("Biological severity multiplier", 0.25, 4.0, float(base.severity), 0.05, key=f"sev_{k}")
nasa_sev = st.session_state.get("nasa_severity")
use_nasa = False
if nasa_sev is not None:
    use_nasa = st.sidebar.checkbox(f"Use NASA-derived severity (×{nasa_sev:.2f})", value=True,
                                   help=f"Source: {st.session_state.get('nasa_source', '?')}")
    if "SYNTHETIC" in st.session_state.get("nasa_source", ""):
        st.sidebar.caption("⚠️ Severity comes from a **synthetic** demo table, not real NASA data.")
severity = nasa_sev if use_nasa else sev_manual

add_spe = st.sidebar.checkbox("Inject an extra solar particle event", value=False, key=f"addspe_{k}")
extra = ()
if add_spe:
    spe_day = st.sidebar.slider("SPE onset (mission day)", 0.5, float(days) - 0.5, min(2.0, float(days) - 0.5), 0.25)
    spe_dose = st.sidebar.slider("SPE dose equivalent (mSv)", 2.0, 40.0, 15.0, 1.0)
    extra = ((spe_day, spe_dose, 6.0),)
scn = dataclasses.replace(
    base, days=float(days), dose_rate_msv_day=float(dose), severity=float(severity),
    spe_events=tuple(e for e in base.spe_events if e[0] < days) + extra,
)

with st.sidebar.expander("Instrument & crew realism"):
    seed = int(st.number_input("Random seed", 0, 9999, 1))
    mismatch = st.slider("Twin–hardware parameter mismatch (±)", 0.0, 0.25, 0.08, 0.01,
                         help="The simulated chip's true parameters differ from the twin's calibrated ones by up to this fraction.")
    bubble_rate = st.slider("Microgravity bubble rate (events/day)", 0.0, 10.0, 3.0, 0.5)
    crew_interval = st.select_slider("Crew blood-draw interval (h)", [6, 12, 24, 48], value=12)
    shielding = st.slider("Body shielding factor", 0.3, 1.0, 0.6, 0.05)
reveal = st.sidebar.toggle("Reveal simulator ground truth", value=False, help="Overlay the hidden true state on twin outputs to audit them.")
st.sidebar.caption("Everything in the tabs updates from these settings.")

run = cached_run(scn, seed, mismatch, bubble_rate, crew_interval, shielding)
fl, tw, det = run.flight, run.twin_f, run.det_f
total_h = scn.days * 24.0
M = run.metrics

# --------------------------------------------------------------------------- header
st.markdown("# 🛰️ Predictive Liver-Chip · Mission Control")
st.caption("Closed-loop AI digital twin for an orbital hepatic organ-chip · NASA Space Apps proof of concept")
st.markdown(
    '<div class="sim-banner"><b>Proof of concept on simulated chip data.</b> The physiology, instrument noise, microgravity artifacts and crew blood panel '
    "are generated by a transparent phenomenological model (see <i>Methods</i>). Real NASA OSDR liver data can be loaded in the <i>NASA data</i> tab to calibrate the simulated biological severity. "
    "No accuracy claim here is a clinical or flight result.</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers
def level_pill(level: int) -> str:
    c = LEVEL_COL[level]
    return f'<span class="pill" style="background:{c}22;color:{c};border:1px solid {c}88">{LEVELS[level]}</span>'


def mission_figure(t_h: float, show_ground=True, show_raw=True):
    n = max(3, int(t_h * SPH))
    stride = max(1, n // 2500)
    idx = np.arange(0, n, stride)
    t = fl.t_days * 24
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[1, 1, 1, 1, 1.3],
                        subplot_titles=[CHANNEL_LABELS[c] for c in CHANNELS] + ["Digital twin · hepatocyte viability (posterior mean ± 95 % CI)"])
    for r, c in enumerate(CHANNELS, start=1):
        o = fl.obs[c].to_numpy()
        cl = det.clean[c].to_numpy()
        if show_raw:
            fig.add_trace(go.Scattergl(x=t[idx], y=o[idx], mode="markers", marker=dict(size=2.5, color="rgba(160,170,185,.35)"),
                                       name="raw telemetry", legendgroup="raw", showlegend=(r == 1)), row=r, col=1)
        if show_ground:
            fig.add_trace(go.Scattergl(x=t[idx], y=run.det_g.clean[c].to_numpy()[idx], mode="lines", line=dict(width=1, color=C_GROUND, dash="dot"),
                                       name="ground-control chip", legendgroup="gnd", showlegend=(r == 1)), row=r, col=1)
        fig.add_trace(go.Scattergl(x=t[idx], y=cl[idx], mode="lines", line=dict(width=1.5, color=C_CLEAN),
                                   name="artifact-cleaned (flight)", legendgroup="clean", showlegend=(r == 1)), row=r, col=1)
        ai = np.flatnonzero(det.artifact_mask[c].to_numpy()[:n] & np.isfinite(o[:n]))
        ai = ai[:: max(1, len(ai) // 700)]
        fig.add_trace(go.Scattergl(x=t[ai], y=o[ai], mode="markers", marker=dict(symbol="x", size=6, color=C_ART),
                                   name="rejected artifact", legendgroup="art", showlegend=(r == 1)), row=r, col=1)
        lo, hi = np.nanquantile(det.clean[c].to_numpy(), [0.002, 0.998])
        pad = 0.6 * (hi - lo + 1e-9)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=1)

    ev = det.events[(det.events.kind != "artifact") & (det.events.start_h <= t_h)]
    for e in ev.itertuples():
        col = "#fbbf24" if e.kind == "biological" else "#a78bfa"
        fig.add_vrect(x0=e.start_h, x1=min(e.end / SPH, t_h), fillcolor=col, opacity=0.13, line_width=0, row="all", col=1)

    kk = index_at_hour(tw, t_h) + 1
    x = tw.t_h[:kk]
    V, sd = tw.V[:kk], tw.V_sd[:kk]
    fig.add_trace(go.Scatter(x=x, y=np.minimum(1.0, V + 1.96 * sd), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=V - 1.96 * sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C_BAND, showlegend=False, hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=V, mode="lines", line=dict(width=2.2, color=C_TWIN), name="twin viability"), row=5, col=1)
    lv = tw.level[:kk]
    mk = np.flatnonzero(lv > 0)
    mk = mk[:: max(1, len(mk) // 400)]
    if len(mk):
        fig.add_trace(go.Scatter(x=x[mk], y=V[mk], mode="markers", marker=dict(size=6, color=[LEVEL_COL[int(v)] for v in lv[mk]]),
                                 name="alert level", hovertext=[LEVELS[int(v)] for v in lv[mk]]), row=5, col=1)
    if reveal:
        i5 = (np.arange(len(tw.t_h)) + 1) * tw.cfg.step - 1
        fig.add_trace(go.Scatter(x=x, y=fl.V[i5][:kk], mode="lines", line=dict(width=1.5, color=C_TRUTH, dash="dash"), name="TRUE viability (hidden)"), row=5, col=1)
    for yv, nm in ((0.85, "WARNING 85 %"), (0.70, "CRITICAL 70 %")):
        fig.add_hline(y=yv, line_dash="dash", line_color="rgba(251,146,60,.5)" if yv == 0.85 else "rgba(248,113,113,.5)", row=5, col=1,
                      annotation_text=nm, annotation_position="bottom right", annotation_font_size=10)
    fig.update_yaxes(range=[min(0.55, float(np.min(tw.V)) - 0.05), 1.03], row=5, col=1)
    fig.update_xaxes(range=[0, total_h], title_text="Mission elapsed time (h)", row=5, col=1)
    return style(fig, 920, sub=True)


def render_frame(t_h: float, show_ground: bool, show_raw: bool):
    kk = index_at_hour(tw, t_h)
    lvl = int(tw.level[kk])
    i24 = index_at_hour(tw, max(t_h - 24, 0))
    n_art = int(((det.events.kind == "artifact") & (det.events.start_h <= t_h)).sum())
    n_bio = int(((det.events.kind == "biological") & (det.events.start_h <= t_h)).sum())
    c = st.columns(6)
    c[0].metric("Hepatocyte viability", f"{tw.V[kk] * 100:.1f} %", f"{(tw.V[kk] - tw.V[i24]) * 100:+.1f} pts / 24 h", delta_color="normal")
    c[1].metric("Metabolic stress", f"{tw.sh[kk]:.2f}", help="Hidden fast stress state (0 = none, 1 = saturated)")
    c[2].metric("Insult rate (inferred)", f"{tw.S[kk]:.2f}/d", help="Twin's estimate of the hidden environmental insult (radiation + microgravity + drug)")
    c[3].metric("Artifacts rejected", n_art, help="Bubbles, packet loss and electrode transients masked before the twin sees them")
    c[4].metric("Biological events", n_bio)
    c[5].markdown(f"**Alert level**<br>{level_pill(lvl)}", unsafe_allow_html=True)
    if lvl >= 2:
        st.error(f"**{LEVELS[lvl]}** at t = {t_h:.0f} h: twin estimates viability {tw.V[kk] * 100:.1f} % (95 % CI ±{1.96 * tw.V_sd[kk] * 100:.1f}). Insult rate ≈ {tw.S[kk]:.2f}/day.")
    elif lvl == 1:
        st.warning(f"**WATCH** at t = {t_h:.0f} h: sustained metabolic-stress signature, viability {tw.V[kk] * 100:.1f} %.")
    else:
        st.success(f"Nominal at t = {t_h:.0f} h. Viability {tw.V[kk] * 100:.1f} %.")
    show(mission_figure(t_h, show_ground, show_raw))


# --------------------------------------------------------------------------- tabs
T = st.tabs(["🛰️ Mission Control", "🧠 Digital Twin & Forecast", "🌍 Orbit vs Ground", "🔬 Cell Imaging", "🧬 NASA data", "📡 Digital Thread", "📊 Benchmark & Methods"])

# ===== 1. Mission control
with T[0]:
    with st.expander("🎬 90-second demo script", expanded=False):
        st.markdown(
            """
1. **Pick** *ISS · solar particle event (day 3)*. Show the grey raw telemetry: bubbles and packet loss everywhere, with red × marks where the detector masks them.
2. **Press ▶ Replay.** The flight chip (teal) and the dotted ground-control chip track each other in shape; the flight chip sits slightly offset because of the modelled chronic microgravity metabolic shift. At day 3 the O₂, glucose and pH curves move together and the ground chip does not.
3. **Point at the twin panel:** viability drifts down with its confidence band, and the alert escalates **WATCH → WARNING** within about an hour of onset.
4. Open **Orbit vs Ground** and show the simulated crew bloodwork flagging the same event *a day or more* later. That gap is the product.
5. Open **NASA data**, load a real OSDR liver table (or the labelled synthetic one), and show the severity calibrating from spaceflight gene signatures.
"""
        )
    c1, c2, c3 = st.columns([1, 1, 4])
    show_ground = c1.toggle("Ground chip", True)
    show_raw = c2.toggle("Raw points", True)
    t_sel = st.slider("Mission elapsed time (h)", 6.0, float(total_h), float(total_h), 1.0, key="t_sel")
    play = st.button("▶ Replay mission", type="primary")
    holder = st.empty()
    if play:
        for th in np.linspace(6, total_h, 36):
            with holder.container():
                render_frame(float(th), show_ground, show_raw)
            time.sleep(0.05)
    else:
        with holder.container():
            render_frame(float(t_sel), show_ground, show_raw)

    st.markdown("#### Event log")
    L, R = st.columns(2)
    ev = det.events[(det.events.kind != "artifact") & (det.events.start_h <= t_sel)].copy()
    with L:
        st.caption("Detector events (artifacts are filtered out; only sustained, multi-channel changes become biological events)")
        if len(ev):
            ev["channels"] = ev["channels"].apply(", ".join)
            st.dataframe(ev[["kind", "start_h", "confirmed_h", "duration_min", "channels", "peak_z"]].rename(
                columns={"start_h": "onset (h)", "confirmed_h": "flagged at (h)", "duration_min": "duration (min)", "peak_z": "peak |z|"}).round(2),
                width="stretch", hide_index=True)
        else:
            st.info("No sustained events yet.")
    with R:
        st.caption("Twin alert escalations")
        al = tw.alert_log[tw.alert_log.t_h <= t_sel]
        st.dataframe(al.rename(columns={"t_h": "time (h)"}).round(2), width="stretch", hide_index=True) if len(al) else st.info("No alerts yet.")

# ===== 2. Twin
with T[1]:
    st.markdown("### Physics-informed digital twin (extended Kalman filter)")
    st.caption("State = [viability, log insult rate, metabolic stress]. The twin never sees the simulator's hidden parameters; it infers the environmental insult itself from the telemetry and imaging stream.")
    default_anchor = M["alert_warning_h"] + 6 if M["alert_warning_h"] else 0.6 * total_h
    a, b, c_, d, e_ = st.columns(5)
    anchor = a.slider("Forecast from (h)", 12.0, float(total_h), float(min(default_anchor, total_h)), 1.0)
    horizon = b.slider("Horizon (h)", 12, 240, 96, 6)
    red = c_.slider("What-if: countermeasure cuts insult by", 0, 95, 60, 5, format="%d %%")
    delay = d.slider("…starting after (h)", 0, 48, 6, 1)
    hl_opt = e_.select_slider("Assumed insult half-life", ["persists", "3 h", "6 h", "12 h", "24 h", "72 h"], value="12 h",
                              help="'persists' = worst case: the inferred insult never subsides. Otherwise it relaxes exponentially toward the quiescent baseline (e.g. a solar particle event passing).")
    hl = None if hl_opt == "persists" else float(hl_opt.split()[0])
    ka = index_at_hour(tw, anchor)
    quiet = (tw.t_h > 6) & (tw.t_h <= anchor)
    s_floor = float(np.percentile(tw.S[quiet], 10)) if quiet.any() else 0.0
    f0 = forecast(tw, ka, horizon_h=float(horizon), s_mult=1.0, seed=1, half_life_h=hl, s_floor=s_floor)
    f1 = forecast(tw, ka, horizon_h=float(horizon), s_mult=1 - red / 100, mult_start_h=float(delay), seed=1, half_life_h=hl, s_floor=s_floor)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[1.6, 1, 0.8],
                        subplot_titles=["Viability: estimate, forecast and what-if", "Inferred environmental insult rate (log scale)", "Metabolic stress state"])
    fig.add_trace(go.Scatter(x=tw.t_h, y=np.minimum(1, tw.V + 1.96 * tw.V_sd), line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.V - 1.96 * tw.V_sd, line=dict(width=0), fill="tonexty", fillcolor=C_BAND, showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.V, line=dict(width=2.2, color=C_TWIN), name="twin estimate"), row=1, col=1)
    i5 = (np.arange(len(tw.t_h)) + 1) * tw.cfg.step - 1
    if reveal:
        fig.add_trace(go.Scatter(x=tw.t_h, y=fl.V[i5], line=dict(width=1.5, color=C_TRUTH, dash="dash"), name="TRUE viability"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tw.t_h, y=fl.S[i5], line=dict(width=1.5, color=C_TRUTH, dash="dash"), name="TRUE insult", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=tw.t_h, y=fl.sh[i5], line=dict(width=1.5, color=C_TRUTH, dash="dash"), name="TRUE stress", showlegend=False), row=2 + 1, col=1)
    for fc, col, nm in ((f0, "#f87171", "no countermeasure"), (f1, "#34d399", f"countermeasure −{red} %")):
        x = anchor + fc["t_h"]
        fig.add_trace(go.Scatter(x=x, y=fc["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=fc["q10"], line=dict(width=0), fill="tonexty", fillcolor=col.replace("#", "rgba(") if False else ("rgba(248,113,113,.18)" if col == "#f87171" else "rgba(52,211,153,.18)"), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=fc["q50"], line=dict(width=2, color=col, dash="dash"), name=f"forecast · {nm}"), row=1, col=1)
    fig.add_hline(y=0.70, line_dash="dash", line_color="rgba(248,113,113,.6)", row=1, col=1, annotation_text="CRITICAL 70 %", annotation_position="bottom right")
    fig.add_vline(x=anchor, line_dash="dot", line_color="rgba(255,255,255,.4)")
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.S_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.S_lo, line=dict(width=0), fill="tonexty", fillcolor=C_BAND, showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.S, line=dict(width=2, color=C_CLEAN), name="inferred insult", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.sh, line=dict(width=2, color="#fb923c"), name="metabolic stress", showlegend=False), row=3, col=1)
    fig.update_yaxes(type="log", row=2, col=1, title_text="1/day")
    fig.update_yaxes(range=[0.3, 1.03], row=1, col=1, title_text="viability")
    fig.update_xaxes(title_text="Mission elapsed time (h)", row=3, col=1)
    show(style(fig, 840, sub=True))

    m = st.columns(4)
    m[0].metric("P(viability < 70 %) · no countermeasure", f"{f0['p_cross'] * 100:.0f} %")
    m[1].metric("P(viability < 70 %) · countermeasure", f"{f1['p_cross'] * 100:.0f} %", f"{(f1['p_cross'] - f0['p_cross']) * 100:+.0f} pts", delta_color="inverse")
    m[2].metric("Median time to 70 %", "n/a" if not np.isfinite(f0["median_ttt"]) else f"{f0['median_ttt']:.0f} h")
    m[3].metric("Median viability at horizon", f"{f0['v_end_median'] * 100:.0f} % → {f1['v_end_median'] * 100:.0f} %")
    st.caption("Forecasts are Monte-Carlo draws from the twin's posterior, propagated through the same damage–repair dynamics. The insult half-life is a scenario assumption you choose (the twin cannot know future space weather); 'persists' is the worst case.")
    with st.expander("Twin audit (against the simulator's hidden truth)"):
        q = st.columns(4)
        q[0].metric("Viability RMSE", f"{M['v_rmse'] * 100:.2f} pts")
        q[1].metric("Bias", f"{M['v_bias'] * 100:+.2f} pts")
        q[2].metric("95 % CI coverage", f"{M['v_coverage95'] * 100:.0f} %", help="≈100 % means the bands are conservative (wider than necessary)")
        q[3].metric("Outlier-gated updates", int(tw.gated.to_numpy().sum()))

# ===== 3. Orbit vs ground
with T[2]:
    st.markdown("### Flight chip vs matched ground-control chip vs simulated crew bloodwork")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chip WARNING raised", "never" if M["alert_warning_h"] is None else f"t = {M['alert_warning_h']:.0f} h")
    k2.metric("Crew bloodwork flags", "never" if M["crew_detect_h"] is None else f"t = {M['crew_detect_h']:.0f} h",
              help=f"First 2 consecutive draws above {CREW_THRESHOLD:.2f}× pre-flight baseline (≈ +3 measurement CV).")
    lead = M["lead_warning_h"]
    k3.metric("Early-warning lead time", "n/a" if lead is None else f"{lead:.0f} h",
              help="Crew-bloodwork detection time minus chip WARNING time. 'n/a' = one of the two never fired in this run.")
    rv = M["r_twin_crew"]
    k4.metric("Twin ↔ crew correlation r", "n/a" if not np.isfinite(rv) else f"{rv:.2f}", help="Pearson r between twin damage and crew marker at blood-draw times (after day 1). The proposal's validation target is r > 0.85.")
    if M["crew_detect_h"] is None and M["alert_warning_h"] is not None:
        st.success("Bloodwork **never** crosses its detection threshold in this mission, yet the chip raised a WARNING. This is the failure mode the chip is designed to cover.")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[1, 1, 1.1],
                        subplot_titles=["Twin viability: flight vs ground", "Flight − ground divergence (%, hourly)", "Simulated crew marker (relative to pre-flight) vs chip alert"])
    fig.add_trace(go.Scatter(x=tw.t_h, y=tw.V, name="flight chip", line=dict(color=C_TWIN, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=run.twin_g.t_h, y=run.twin_g.V, name="ground chip", line=dict(color=C_GROUND, width=2, dash="dot")), row=1, col=1)
    palette = ["#2dd4bf", "#fbbf24", "#f472b6", "#60a5fa"]
    for c, col in zip(CHANNELS, palette):
        fig.add_trace(go.Scatter(x=run.div.hours, y=run.div.pct[c], name=CHANNEL_LABELS[c].split(" (")[0], line=dict(color=col, width=1.4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=run.crew.draw_idx / SPH, y=run.crew.values, mode="markers+lines", name="crew blood draw",
                             marker=dict(size=8, color="#e6edf7"), line=dict(width=1, color="rgba(230,237,247,.3)")), row=3, col=1)
    fig.add_hline(y=CREW_THRESHOLD, line_dash="dash", line_color="rgba(248,113,113,.7)", row=3, col=1, annotation_text="bloodwork detection threshold")
    if reveal:
        fig.add_trace(go.Scatter(x=np.arange(len(run.crew.truth)) / SPH, y=run.crew.truth, name="TRUE crew marker", line=dict(color=C_TRUTH, dash="dash")), row=3, col=1)
    for hh, lab, col in ((M["alert_warning_h"], "chip WARNING", "#fb923c"), (M["crew_detect_h"], "bloodwork flags", "#f87171")):
        if hh is not None:
            for rr in (1, 2, 3):
                fig.add_vline(x=hh, line_color=col, line_dash="dot", row=rr, col=1,
                              **(dict(annotation_text=lab, annotation_position="bottom left" if lab.startswith("chip") else "bottom right",
                                      annotation_font_size=11, annotation_font_color=col) if rr == 3 else {}))
    fig.update_xaxes(title_text="Mission elapsed time (h)", row=3, col=1)
    show(style(fig, 880, sub=True))

    L, R = st.columns(2)
    with L:
        cu = go.Figure()
        for c, col in zip(CHANNELS, palette):
            cu.add_trace(go.Scatter(x=run.div.hours, y=run.div.cusum[c], name=CHANNEL_LABELS[c].split(" (")[0], line=dict(color=col)))
        cu.add_hline(y=5, line_dash="dash", line_color="rgba(255,255,255,.5)", annotation_text="decision limit h = 5")
        cu.update_layout(title="Paired CUSUM: catches chronic offsets that rolling baselines adapt to")
        show(style(cu, 340))
        st.caption("Chronic-exposure onset (first CUSUM crossing, h): " + ", ".join(f"{c.split('_')[0]}: {v if v is not None else '–'}" for c, v in run.div.onset_h.items()))
    with R:
        di = run.crew.draw_idx[run.crew.draw_idx / SPH > 24]
        kk = np.clip(np.searchsorted(tw.t_h, di / SPH), 0, len(tw.t_h) - 1)
        xd, yd = 1 - tw.V[kk], run.crew.values[np.isin(run.crew.draw_idx, di)]
        sc = go.Figure(go.Scatter(x=xd, y=yd, mode="markers", marker=dict(size=9, color=di / SPH, colorscale="Viridis", colorbar=dict(title="t (h)")), name="draws"))
        if len(xd) > 3 and np.std(xd) > 0:
            sl, ic = np.polyfit(xd, yd, 1)
            xs = np.linspace(xd.min(), xd.max(), 20)
            sc.add_trace(go.Scatter(x=xs, y=sl * xs + ic, mode="lines", line=dict(color="#f87171", dash="dash"), name="linear fit"))
        sc.update_layout(title=f"Twin damage vs crew marker · r = {rv:.2f}", xaxis_title="twin-estimated damage (1 − viability)", yaxis_title="crew marker (× baseline)")
        show(style(sc, 340))

# ===== 4. Imaging
with T[3]:
    st.markdown("### Optical mapping: segmentation and bleb classification")
    st.caption("Classical segmentation (watershed) plus a gradient-boosted classifier on per-cell shape and texture features. **Training and demo images are procedurally generated**, so the accuracy below is for synthetic cells only.")
    models = cached_models()
    up = st.file_uploader("…or analyse your own micrograph (PNG/JPG/TIFF, grayscale membrane channel)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if up is not None:
        from PIL import Image

        im = Image.open(up).convert("L")
        im.thumbnail((768, 768))
        cy = np.asarray(im, float) / 255.0
        an = imaging.analyze(cy, None, models)
        raw_rgb = np.stack([0.25 * cy, cy, 0.4 * cy], -1)
        truth_txt = "uploaded image (shape-only model, no nuclear channel)"
        true_frac = None
    else:
        t_img = st.slider("Chip time to image (h)", 6.0, float(total_h), float(min(total_h, max(6.0, M["alert_warning_h"] or total_h))), 1.0)
        ii = int(np.clip(round(t_img * SPH / 30) - 1, 0, len(fl.bleb_meas) - 1))
        meas = float(fl.bleb_meas[ii])
        fld = cached_field(round(meas, 2), seed * 1000 + int(t_img))
        an = imaging.analyze(fld.cyto, fld.nuc, models)
        raw_rgb = fld.rgb
        true_frac = float(fld.damaged[1:].mean())
        truth_txt = f"synthetic field rendered at the chip's measured bleb fraction ({meas:.2f})"
    a1, a2 = st.columns(2)
    a1.image(raw_rgb, caption="Fluorescence (green = membrane/cytoplasm, blue = nuclei)", width="stretch")
    a2.image(an.overlay, caption="Segmentation: green = healthy hexagonal · red = blebbing / damaged", width="stretch")
    m = st.columns(4)
    m[0].metric("Cells segmented", an.n_cells)
    m[1].metric("Blebbing fraction (classifier)", f"{an.bleb_fraction * 100:.0f} %")
    if true_frac is not None:
        m[2].metric("Rendered ground truth", f"{true_frac * 100:.0f} %")
        m[3].metric("Absolute error", f"{abs(an.bleb_fraction - true_frac) * 100:.1f} pts")
    st.caption(truth_txt)
    if len(an.cells):
        L, R = st.columns([3, 2])
        with L:
            top = an.cells.sort_values("p_damaged", ascending=False).head(12)[["label", "p_damaged", "solidity", "circularity", "rough", "area_rel"]]
            st.markdown("**Most suspicious cells**")
            st.dataframe(top.round(3), width="stretch", hide_index=True)
        with R:
            h = go.Figure(go.Histogram(x=an.cells["p_damaged"], nbinsx=20, marker_color=C_TWIN))
            h.update_layout(title="Per-cell P(damaged)", xaxis_title="probability", yaxis_title="cells")
            show(style(h, 300))
    with st.expander("Model card & calibration"):
        cf, cs = models["full"][1], models["shape"][1]
        st.markdown(
            f"- **Full model** (shape + nuclear channel): accuracy {cf['accuracy'] * 100:.1f} %, AUC {cf['auc']:.3f} on {cf['n_test_cells']} held-out synthetic cells. "
            f"Importance is dominated by nuclear brightness (a cue built into the generator), so this number overstates real-world performance.\n"
            f"- **Shape-only model** (used for uploads): accuracy {cs['accuracy'] * 100:.1f} %, AUC {cs['auc']:.3f}.\n"
            "- To make this real: replace `segment` with Cellpose/StarDist and retrain `fit_classifier` on labelled hepatocyte images."
        )
        cal = cached_calibration()
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="ideal"))
        fg.add_trace(go.Scatter(x=cal.true, y=cal.measured, mode="markers+lines", marker=dict(size=8, color=C_CLEAN), name="classifier"))
        fg.update_layout(title="Fraction-damaged calibration on synthetic fields", xaxis_title="true fraction", yaxis_title="measured fraction")
        show(style(fg, 320))

# ===== 5. NASA data
with T[4]:
    st.markdown("### NASA Open Science Data Repository · spaceflight liver signatures")
    st.caption("Calibrates the simulator's biological *severity* from a real spaceflight-vs-ground liver differential-expression table. The metadata API can be queried live; expression values come from a study's processed files.")
    src = st.radio("Data source", ["Live NASA OSDR query", "Upload OSDR differential-expression file", "Synthetic demo table (offline)"], horizontal=True)
    de, label = None, ""
    if src.startswith("Live"):
        acc = st.text_input("OSDR accession", "OSD-48", help="Studies are listed at osdr.nasa.gov. Look for mouse liver (RNA-seq) from Rodent Research missions.")
        if st.button("Fetch metadata from NASA OSDR"):
            try:
                with st.spinner("Contacting visualization.osdr.nasa.gov…"):
                    js, url = nasa.fetch_osdr_metadata(acc)
                st.session_state["osdr_meta"] = (acc, nasa.summarize_metadata(js), url)
                st.session_state.pop("osdr_err", None)
            except Exception as e:  # noqa: BLE001
                st.session_state["osdr_err"] = f"{type(e).__name__}: {e}"
        if "osdr_err" in st.session_state:
            st.error(f"Could not reach the OSDR API from this environment ({st.session_state['osdr_err']}). "
                     "Run the app on a machine with internet access, or switch to *Upload* / *Synthetic demo*.")
        if "osdr_meta" in st.session_state:
            acc_, summ, url = st.session_state["osdr_meta"]
            st.success(f"Fetched {summ['n_fields']} metadata fields for {acc_} ({summ['size_kb']:.0f} kB).")
            st.json({"request": url, "keyword mentions in metadata": summ["mentions"]}, expanded=False)
            st.dataframe(summ["table"].head(200), width="stretch", hide_index=True)
        st.info(f"Next step: download the study's *differential_expression.csv* from {nasa.OSDR_STUDY_URL.format(acc=acc.strip().upper())} (Files tab) and upload it under **Upload OSDR differential-expression file** to score the liver panels.")
    elif src.startswith("Upload"):
        f = st.file_uploader("OSDR differential-expression CSV (needs a gene-symbol column and a log2 fold-change column)", type=["csv", "tsv", "txt"])
        if f is not None:
            try:
                raw = pd.read_csv(f, sep=None, engine="python")
                de = nasa.parse_de_table(raw)
                label = f"uploaded file '{f.name}'"
                st.success(f"Parsed {len(de):,} genes.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not parse that file: {e}")
    else:
        eff = st.slider("Synthetic effect size", 0.0, 2.5, 1.0, 0.1)
        de = nasa.parse_de_table(nasa.synthetic_de_table(effect=eff))
        label = "SYNTHETIC demo table (not NASA data)"
        st.warning("This table is **synthetic**: it only demonstrates the pipeline offline.")

    if de is not None:
        try:
            ps = nasa.panel_scores(de)
        except ValueError as e:
            st.error(str(e))
            ps = None
        if ps:
            tab = ps["table"]
            sev = nasa.severity_from_hsi(ps["hsi"])
            c = st.columns(4)
            c[0].metric("Hepatic-stress index (HSI)", f"{ps['hsi']:.2f}", help="Mean signed log2FC across liver panels (loss-of-function panel sign-flipped)")
            c[1].metric("95 % CI (gene bootstrap)", f"{ps['hsi_ci'][0]:.2f} … {ps['hsi_ci'][1]:.2f}")
            c[2].metric("Panels scored", f"{len(ps['usable'])}/4")
            c[3].metric("→ simulator severity", f"× {sev:.2f}", help=f"severity = HSI / {nasa.HSI_REF} (clipped {nasa.SEV_MIN}–{nasa.SEV_MAX}); a modelling assumption")
            if st.button("Apply to simulation", type="primary"):
                st.session_state["nasa_severity"] = sev
                st.session_state["nasa_source"] = label
                st.rerun()
            L, R = st.columns(2)
            with L:
                bar = go.Figure(go.Bar(x=tab.mean_log2fc, y=tab.panel, orientation="h", marker_color=[C_CLEAN if v > 0 else C_ART for v in tab.mean_log2fc.fillna(0)],
                                       error_x=dict(type="data", symmetric=False, array=(tab.ci_hi - tab.mean_log2fc).fillna(0), arrayminus=(tab.mean_log2fc - tab.ci_lo).fillna(0))))
                bar.update_layout(title="Panel mean log2 fold change (flight vs ground) · 95 % CI", xaxis_title="log2FC")
                show(style(bar, 340))
            with R:
                pc = "p" if de["p"].notna().any() else None
                vol = go.Figure()
                y = -np.log10(de["p"].clip(lower=1e-300)) if pc else np.zeros(len(de))
                vol.add_trace(go.Scattergl(x=de.log2fc, y=y, mode="markers", marker=dict(size=3, color="rgba(160,170,185,.4)"), name="all genes", hoverinfo="skip"))
                for (name, genes), col in zip(nasa.PANELS.items(), ["#2dd4bf", "#fbbf24", "#f472b6", "#60a5fa"]):
                    sub = de[de.gene_u.isin([g.upper() for g in genes])]
                    vol.add_trace(go.Scatter(x=sub.log2fc, y=(-np.log10(sub["p"].clip(lower=1e-300)) if pc else np.zeros(len(sub))), mode="markers+text", text=sub.gene,
                                             textposition="top center", textfont=dict(size=9), marker=dict(size=9, color=col), name=name))
                vol.update_layout(title="Volcano with liver panels highlighted", xaxis_title="log2FC", yaxis_title="−log10 p")
                show(style(vol, 340))
            st.dataframe(tab.round(3), width="stretch", hide_index=True)
            st.caption("Panels are curated from the spaceflight-liver literature (Nrf2 response, lipid dysregulation, DNA-damage stress, loss of synthetic function). The HSI→severity mapping is an explicit modelling assumption, not a measured conversion.")

# ===== 6. Digital thread
with T[5]:
    st.markdown("### Digital thread: why the twin has to live on board")
    link = st.radio("Link", [l.name for l in comms.LINKS], horizontal=True, index=3)
    ln = next(l for l in comms.LINKS if l.name == link)
    dd = M["det_median_delay_min"] if np.isfinite(M["det_median_delay_min"]) else 60.0
    think = st.slider("Ground analysis / decision time (min)", 5, 120, 20, 5, help="Assumption: time for ground teams to review and decide once data arrive.")
    c = st.columns(3)
    c[0].metric("One-way light time", f"{ln.one_way_s:.1f} s" if ln.one_way_s < 120 else f"{ln.one_way_s / 60:.1f} min", help=ln.note)
    c[1].metric("Detection delay (this run)", f"{dd:.0f} min", help="Median delay from true onset to confirmed detection in this simulated mission (60 min default if the scenario has no discrete event).")
    tl = comms.decision_timeline(dd, ln, ground_think_min=float(think))
    c[2].metric("Ground-in-loop penalty", f"+{tl.minutes.iloc[1] - tl.minutes.iloc[0]:.0f} min")
    bt = go.Figure(go.Bar(x=tl.minutes, y=tl["mode"], orientation="h", marker_color=[C_CLEAN, C_ART], text=[f"{m:.0f} min" for m in tl.minutes], textposition="outside"))
    bt.update_layout(title="Minutes from biological onset until an action reaches the crew", xaxis_title="minutes")
    show(style(bt, 260))
    st.caption(tl.breakdown.iloc[1])
    with st.expander("Data-volume assumptions", expanded=True):
        q = st.columns(4)
        w = q[0].select_slider("Image size (px)", [512, 1024, 2048, 4096], value=2048)
        ev_ = q[1].select_slider("Imaging cadence (min)", [10, 15, 30, 60], value=30)
        cells = q[2].slider("Cells per image", 50, 1000, 250, 25)
        ratio = q[3].slider("Lossless compression ratio", 0.2, 1.0, 0.5, 0.05)
        vol_ = comms.daily_volume(comms.DataBudget(img_w=w, img_h=w, img_every_min=ev_, cells_per_image=cells, lossless_ratio=ratio))
        fg = go.Figure(go.Bar(x=vol_.stream, y=vol_.MB_per_day, marker_color=[C_ART, C_ART, C_CLEAN, C_CLEAN], text=[f"{v:.3g} MB" for v in vol_.MB_per_day], textposition="outside"))
        fg.update_yaxes(type="log", title="MB per day (log)")
        fg.update_layout(title="Downlink volume per chip per day")
        show(style(fg, 320))
        red = vol_.MB_per_day.iloc[:2].sum() / vol_.MB_per_day.iloc[2:].sum()
        st.markdown(f"On-board segmentation and twin inference reduce downlink by about **{red:,.0f}×**, so only features, state packets and alerts need to cross the link. Image sizes and compression are assumptions; adjust them above.")

# ===== 7. Benchmark & methods
with T[6]:
    st.markdown("### Monte-Carlo benchmark")
    st.caption("Runs the full pipeline (simulate → detect → twin → crew comparison) over multiple seeds and scenarios, scoring against the simulator's hidden truth.")
    n_seeds = st.slider("Seeds per scenario", 2, 8, 3)
    if st.button("Run benchmark", type="primary"):
        pb = st.progress(0.0, text="Starting…")
        dfb = benchmark(seeds=range(n_seeds), progress=lambda p, t: pb.progress(p, text=t), mismatch=mismatch, bubble_rate=bubble_rate, crew_interval_h=crew_interval, shielding=shielding)
        pb.empty()
        st.session_state["bench"] = dfb
    if "bench" in st.session_state:
        dfb = st.session_state["bench"]
        sm = summarize_benchmark(dfb)
        st.dataframe(sm.round(2), width="stretch")
        st.download_button("Download per-run results (CSV)", dfb.to_csv(index=False), "liverchip_benchmark.csv", "text/csv")
        st.caption("'n/a' (blank) = not applicable, e.g. chronic scenarios have no discrete event to detect; the drug scenario doses the ground chip too, so ground alerts there are not false.")

    st.divider()
    st.markdown("### Methods, assumptions and honest limits")
    st.graphviz_chart(
        """digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box,style="rounded,filled",fillcolor="#111a2e",fontcolor="#e6edf7",color="#2dd4bf"]; edge [color="#7aa2f7"];
        chip[label="Organ-chip\\n(O2, pH, glucose, albumin\\n+ optical mapping)"]; art[label="Stage 1\\nartifact rejection\\n(robust z, MAD)"]; ev[label="Stage 2\\nsustained multi-channel\\nevents"]; twin[label="Digital twin\\n(EKF: viability,\\ninsult, stress)"]; pair[label="Stage 3\\nflight vs ground\\nCUSUM"]; out[label="Alerts +\\nforecast +\\nwhat-if"];
        chip->art->ev->twin->out; art->pair->out; }"""
    )
    st.markdown(
        r"""
**Latent dynamics (per chip).** Damage $D$ obeys $\dot D = a\,S(t)(1-D) - r\,D$ with viability $V=1-D$; a fast metabolic-stress state follows $\dot s_h = (g(S)-s_h)/\tau$, $g(S)=S/(S+S_{50})$.
The insult $S(t)$ combines chronic dose-equivalent rate, microgravity unloading, solar-particle events (exponential decay) and optional drug exposure, scaled by a severity multiplier.

**Observation model.** O₂ consumption $\propto V(1+c_{O_2}s_h)$, glucose uptake $\propto V(1+c_{glc}s_h)$, acidification $\propto V(1+c_{H^+}s_h)+k(1-V)$, albumin $\propto V^{1.5}$, bleb fraction $=0.03+0.9\,D^{1.3}$ (binomial counts from 250 imaged cells every 30 min).

**Estimation.** An extended Kalman filter over $[V,\ln S,s_h]$ with sequential scalar updates, innovation gating (re-admitted after 2 h so real regime changes are not rejected forever) and per-channel model-mismatch noise floors. The simulated chip uses parameters perturbed by up to ±8 % from the twin's calibration.

**What is real:** the signal-processing, filtering, segmentation, forecasting and calibration code; the OSDR connector and gene-panel scoring; the light-time arithmetic; the MSL/RAD cruise dose-rate figure (1.8 mSv/day).
**What is simulated:** all chip telemetry, imaging, crew bloodwork, and the biological response itself (coefficients are assumptions, not fitted to flight data).

**Known limits.**
(1) Because the twin's model family matches the simulator's, accuracy here is optimistic compared with a real chip.
(2) The 95 % intervals are conservative (near-100 % coverage in simulation).
(3) The twin–crew correlation and lead-time results depend on the assumed crew-marker kinetics (24 h lag, 10 % CV, 12 h draws).
(4) The image classifier is trained on procedural cells.
(5) The rolling-baseline detector is deliberately blind to slow chronic drift; the paired flight–ground CUSUM covers that case.
(6) Nothing here is a validated medical device or flight result.
"""
    )
    st.markdown(
        "**Sources:** [NASA OSDR Biological Data API](https://visualization.osdr.nasa.gov/biodata/api) · "
        "[Multi-omics analysis of multiple missions to space reveal a theme of lipid dysregulation in mouse liver (Sci Rep 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6915713) · "
        "[Nrf2 plays a critical role in the metabolic response during and after spaceflight (Commun Biol 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8660801) · "
        "[SwRI: MSL/RAD Mars-cruise radiation, 1.8 mSv/day](https://www.swri.org/press-release/swri-led-team-calculates-radiation-exposure-associated-trip-mars)"
    )
    st.download_button("Download twin state (CSV)", pd.DataFrame({"t_h": tw.t_h, "viability": tw.V, "viability_sd": tw.V_sd, "insult_per_day": tw.S, "metabolic_stress": tw.sh, "alert_level": tw.level}).to_csv(index=False),
                       "liverchip_twin_state.csv", "text/csv")
