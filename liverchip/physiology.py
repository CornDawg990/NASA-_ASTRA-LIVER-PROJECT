"""Hepatic organ-chip physiology simulator.

This module is the *ground-truth generator* for the demo. It is a phenomenological
model, not a validated biophysical one: every coefficient is an assumption that is
documented in ``ChipParams`` / ``Scenario`` and surfaced in the app's Methods tab.

Latent state (per chip)
    D(t)   cumulative hepatocyte damage, V = 1 - D is functional viability
    sh(t)  fast metabolic-stress response (hypermetabolic / glycolytic shift)

Dynamics (per day)
    dD/dt  = a * S(t) * (1 - D) - r * D          (insult drives damage, repair opposes it)
    dsh/dt = (g(S) - sh) / tau_sh,  g(S) = S / (S + S50)

S(t) is an abstract "oxidative insult rate" assembled from space radiation (chronic GCR +
solar particle events), microgravity, and optional drug exposure, scaled by a severity
multiplier that can be calibrated from NASA OSDR data (see ``nasa.py``).

Observed channels (sampled every 60 s, as in the proposal)
    O2 outlet (mmHg), pH outlet, glucose outlet (mM), albumin (ug/mL),
plus an optical-mapping stream (fraction of blebbing hepatocytes every 30 min).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.signal import lfilter

SPD = 1440  # samples per day (60 s cadence)
SPH = 60  # samples per hour
IMG_EVERY = 30  # optical mapping cadence in samples (30 min)
CHANNELS = ["o2_out", "ph_out", "glucose_out", "albumin"]
CHANNEL_LABELS = {
    "o2_out": "O₂ outlet (mmHg)",
    "ph_out": "pH outlet",
    "glucose_out": "Glucose outlet (mM)",
    "albumin": "Albumin (µg/mL)",
}


@dataclass(frozen=True)
class ChipParams:
    """Nominal (calibrated) chip parameters. The twin uses these; the simulator uses a
    hidden, perturbed copy so the twin is never handed the exact truth."""

    # inlet / calibration constants
    o2_in: float = 150.0  # mmHg, air-equilibrated medium
    our0: float = 55.0  # mmHg consumed across the chamber at full viability
    ph_in: float = 7.40
    acid0: float = 0.060  # pH units of acidification at full viability
    glc_in: float = 11.0  # mM
    gu0: float = 1.2  # mM glucose uptake at full viability
    alb0: float = 20.0  # ug/mL albumin at full viability
    # dynamics
    a_damage: float = 1.0  # damage per unit insult
    r_repair: float = 0.15  # 1/day
    s50: float = 0.05  # 1/day, half-saturation of metabolic response
    tau_sh_h: float = 3.0  # hours
    c_our: float = 0.6  # hypermetabolic gain on oxygen consumption
    c_glc: float = 1.0  # glycolytic gain on glucose uptake
    c_acid: float = 1.5  # acidification gain
    acid_dead: float = 0.05  # acidification from dying cells
    # sensor noise (1-sigma, per sample)
    sd_o2: float = 1.0
    sd_ph: float = 0.004
    sd_glc: float = 0.05
    sd_alb_rel: float = 0.04
    # optical mapping
    n_cells_imaged: int = 250


@dataclass(frozen=True)
class Scenario:
    name: str = "ISS · chronic LEO exposure"
    days: float = 7.0
    dose_rate_msv_day: float = 0.5  # chronic dose-equivalent rate
    alpha_dose: float = 0.02  # insult units per (mSv/day)  [model assumption]
    severity: float = 1.0  # multiplier; calibrate from NASA data
    microgravity_stress: float = 0.012  # insult units from unloading / fluid shifts
    spe_events: tuple = ()  # (t_day, total_dose_msv, decay_tau_h)
    drug_events: tuple = ()  # (t_start_day, t_end_day, potency)
    countermeasure_start_day: float | None = None
    countermeasure_efficacy: float = 0.0  # 0..1 reduction of space-derived insult


PRESETS: dict[str, Scenario] = {
    "ISS · chronic LEO exposure": Scenario(),
    "ISS · solar particle event (day 3)": Scenario(
        name="ISS · solar particle event (day 3)",
        days=7.0,
        spe_events=((3.0, 15.0, 6.0),),
    ),
    "Mars transit · chronic GCR": Scenario(
        name="Mars transit · chronic GCR",
        days=14.0,
        dose_rate_msv_day=1.8,
    ),
    "Mars transit · GCR + two SPEs": Scenario(
        name="Mars transit · GCR + two SPEs",
        days=14.0,
        dose_rate_msv_day=1.8,
        spe_events=((4.0, 12.0, 6.0), (9.5, 20.0, 8.0)),
    ),
    "Countermeasure trial (antioxidant from day 4)": Scenario(
        name="Countermeasure trial (antioxidant from day 4)",
        days=10.0,
        dose_rate_msv_day=1.8,
        countermeasure_start_day=4.0,
        countermeasure_efficacy=0.6,
    ),
    "Drug safety screen (hepatotoxic compound day 2-4)": Scenario(
        name="Drug safety screen (hepatotoxic compound day 2-4)",
        days=7.0,
        drug_events=((2.0, 4.0, 0.12),),
    ),
}


def ground_scenario(scn: Scenario) -> Scenario:
    """Matched ground control: identical drug schedule, no space-derived insult."""
    return replace(
        scn,
        name=scn.name + " [ground control]",
        dose_rate_msv_day=0.0,
        microgravity_stress=0.0,
        spe_events=(),
        countermeasure_start_day=None,
        countermeasure_efficacy=0.0,
    )


def stress_profile(scn: Scenario, n: int) -> np.ndarray:
    """Insult rate S(t) in 1/day for n samples at 60 s cadence."""
    t = np.arange(n) / SPD
    space = np.full(n, scn.severity * scn.alpha_dose * scn.dose_rate_msv_day + scn.severity * scn.microgravity_stress)
    for t0, dose, tau_h in scn.spe_events:
        tau_d = tau_h / 24.0
        peak_rate = dose / tau_d  # mSv/day at onset; integral over decay = dose
        env = np.where(t >= t0, np.exp(-(t - t0) / tau_d), 0.0)
        space = space + scn.severity * scn.alpha_dose * peak_rate * env
    if scn.countermeasure_start_day is not None and scn.countermeasure_efficacy > 0:
        factor = np.where(t >= scn.countermeasure_start_day, 1.0 - scn.countermeasure_efficacy, 1.0)
        space = space * factor
    drug = np.zeros(n)
    for ts, te, potency in scn.drug_events:
        drug = drug + np.where((t >= ts) & (t < te), potency, 0.0)
    return space + drug


def integrate_damage(S: np.ndarray, p: ChipParams) -> np.ndarray:
    """Euler integration of dD/dt = a S (1-D) - r D at 60 s steps."""
    dt = 1.0 / SPD
    a, r = p.a_damage, p.r_repair
    out = np.empty(len(S))
    d = 0.0
    for i, s in enumerate(S.tolist()):
        out[i] = d
        d += dt * (a * s * (1.0 - d) - r * d)
    return out


def metabolic_shift(S: np.ndarray, p: ChipParams) -> np.ndarray:
    """First-order lag of g(S); exact linear filter."""
    g = S / (S + p.s50)
    k = (1.0 / SPH) / p.tau_sh_h
    # sh[n+1] = (1-k) sh[n] + k g[n]; start in the unstressed steady state of the first sample
    zi = np.array([(1.0 - k) * g[0]])
    sh, _ = lfilter([k], [1.0, -(1.0 - k)], g, zi=zi)
    # lfilter gives y[n] = (1-k) y[n-1] + k g[n]; shift by one so sh[n] uses g[:n]
    return np.concatenate([[g[0]], sh[:-1]])


def perturb_params(p: ChipParams, rng: np.random.Generator, mismatch: float) -> ChipParams:
    """Hidden 'true' chip differs from the nominal calibration by +-mismatch."""
    if mismatch <= 0:
        return p
    f = lambda: 1.0 + rng.uniform(-mismatch, mismatch)  # noqa: E731
    return replace(
        p,
        r_repair=p.r_repair * f(),
        s50=p.s50 * f(),
        tau_sh_h=p.tau_sh_h * f(),
        c_our=p.c_our * f(),
        c_glc=p.c_glc * f(),
        c_acid=p.c_acid * f(),
        acid_dead=p.acid_dead * f(),
    )


@dataclass
class ChipRun:
    scenario: Scenario
    flight: bool
    t_days: np.ndarray
    S: np.ndarray
    D: np.ndarray
    sh: np.ndarray
    obs: pd.DataFrame  # noisy, artifact-corrupted telemetry (what the instrument reports)
    truth: pd.DataFrame  # noise-free channel values
    truth_events: list = field(default_factory=list)
    img_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    bleb_meas: np.ndarray = field(default_factory=lambda: np.array([]))
    bleb_true: np.ndarray = field(default_factory=lambda: np.array([]))
    params_true: ChipParams = field(default_factory=ChipParams)

    @property
    def n(self) -> int:
        return len(self.t_days)

    @property
    def V(self) -> np.ndarray:
        return 1.0 - self.D


def bleb_probability(D: np.ndarray | float) -> np.ndarray | float:
    """Fraction of hepatocytes showing blebbing / loss of polarity as a function of damage."""
    return 0.03 + 0.90 * np.power(np.clip(D, 0.0, 1.0), 1.3)


def simulate_chip(
    scn: Scenario,
    params: ChipParams = ChipParams(),
    *,
    flight: bool = True,
    seed: int = 0,
    bubble_rate_per_day: float | None = None,
    dropout_rate_per_day: float = 0.5,
    drift: bool = True,
    mismatch: float = 0.08,
) -> ChipRun:
    rng = np.random.default_rng(seed)
    n = int(round(scn.days * SPD))
    p = perturb_params(params, np.random.default_rng(seed + 9973), mismatch)

    S = stress_profile(scn, n)
    D = integrate_damage(S, p)
    sh = metabolic_shift(S, p)
    V = 1.0 - D
    t = np.arange(n) / SPD

    our = p.our0 * V * (1.0 + p.c_our * sh)
    pump = 0.5 * np.sin(2 * np.pi * np.arange(n) / 25.0)  # micropump pulsation, mmHg
    o2 = p.o2_in - our + pump
    ph = p.ph_in - (p.acid0 * V * (1.0 + p.c_acid * sh) + p.acid_dead * D)
    glc = p.glc_in - p.gu0 * V * (1.0 + p.c_glc * sh)
    alb = p.alb0 * np.power(V, 1.5)
    truth = pd.DataFrame({"o2_out": o2, "ph_out": ph, "glucose_out": glc, "albumin": alb})

    noise_mult = 1.3 if flight else 1.0  # microgravity: noisier optics / fluidics
    obs = truth.copy()
    obs["o2_out"] += rng.normal(0, p.sd_o2 * noise_mult, n)
    obs["ph_out"] += rng.normal(0, p.sd_ph * noise_mult, n)
    obs["glucose_out"] += rng.normal(0, p.sd_glc * noise_mult, n)
    obs["albumin"] *= np.exp(rng.normal(0, p.sd_alb_rel, n))
    if drift:  # slow sensor aging (identical on flight and ground hardware)
        obs["o2_out"] += 0.25 * t
        obs["ph_out"] -= 0.002 * t

    # ---- artifacts: microgravity bubbles linger (no buoyancy) => longer, more frequent
    events: list[dict] = []
    if bubble_rate_per_day is None:
        bubble_rate_per_day = 3.0 if flight else 0.3
    n_b = rng.poisson(bubble_rate_per_day * scn.days)
    amp = {"o2_out": (15, 40), "ph_out": (0.04, 0.12), "glucose_out": (0.4, 1.0)}
    for _ in range(n_b):
        start = int(rng.integers(45, max(46, n - 10)))
        length = int(rng.integers(1, 7 if flight else 4))
        chans = list(rng.choice(["o2_out", "ph_out", "glucose_out"], size=int(rng.integers(1, 4)), replace=False))
        for c in chans:
            lo, hi = amp[c]
            obs.iloc[start : start + length, obs.columns.get_loc(c)] += rng.choice([-1, 1]) * rng.uniform(lo, hi)
        events.append({"type": "bubble", "start": start, "length": length, "channels": chans})
    n_d = rng.poisson(dropout_rate_per_day * scn.days)
    for _ in range(n_d):  # telemetry packet loss
        start = int(rng.integers(45, max(46, n - 12)))
        length = int(rng.integers(1, 11))
        obs.iloc[start : start + length, :] = np.nan
        events.append({"type": "dropout", "start": start, "length": length, "channels": list(CHANNELS)})

    # ---- biological ground-truth events (used only for scoring the detector)
    for t0, dose, tau_h in scn.spe_events:
        events.append({"type": "spe", "start": int(t0 * SPD), "length": int(tau_h * SPH), "channels": []})
    for ts, te, potency in scn.drug_events:
        events.append({"type": "drug", "start": int(ts * SPD), "length": int((te - ts) * SPD), "channels": []})
        if int(te * SPD) < n - 12 * SPH:  # washout is itself a physiological change
            events.append({"type": "drug_end", "start": int(te * SPD), "length": 0, "channels": []})
    if scn.countermeasure_start_day is not None and scn.countermeasure_efficacy > 0:
        events.append({"type": "countermeasure", "start": int(scn.countermeasure_start_day * SPD), "length": 0, "channels": []})

    # ---- optical mapping stream: binomial count of blebbing cells in the field of view
    img_idx = np.arange(IMG_EVERY, n, IMG_EVERY)
    p_true = bleb_probability(D[img_idx])
    counts = rng.binomial(p.n_cells_imaged, p_true)
    bleb_meas = counts / p.n_cells_imaged

    return ChipRun(
        scenario=scn,
        flight=flight,
        t_days=t,
        S=S,
        D=D,
        sh=sh,
        obs=obs.reset_index(drop=True),
        truth=truth,
        truth_events=events,
        img_idx=img_idx,
        bleb_meas=bleb_meas,
        bleb_true=p_true,
        params_true=p,
    )


# ---------------------------------------------------------------------------
# Simulated crew blood panel (the "in-vivo" comparator in the proposal)
# ---------------------------------------------------------------------------
KAPPA_MARKER = 2.5  # relative marker rise per unit systemic damage (assumption)


def lowpass(x: np.ndarray, dt_h: float, tau_h: float) -> np.ndarray:
    k = dt_h / tau_h
    y, _ = lfilter([k], [1.0, -(1.0 - k)], np.asarray(x, float), zi=[(1.0 - k) * float(x[0])])
    return y


@dataclass
class CrewPanel:
    draw_idx: np.ndarray  # sample index of each blood draw
    values: np.ndarray  # measured marker (relative to pre-flight baseline = 1.0)
    truth: np.ndarray  # noise-free marker at each sample
    interval_h: float
    cv: float


def simulate_crew_panel(
    flight: ChipRun,
    params: ChipParams = ChipParams(),
    *,
    seed: int = 0,
    interval_h: float = 12.0,
    shielding: float = 0.6,
    tau_h: float = 24.0,
    cv: float = 0.10,
) -> CrewPanel:
    """Systemic marker (e.g. an oxidative-stress / injury marker) in a simulated astronaut.

    The astronaut is a *different* realisation: body shielding scales the insult, an
    individual susceptibility factor is drawn, and the marker is a slow, noisy, sparsely
    sampled low-pass of damage. This is what makes the chip an early-warning system.
    """
    rng = np.random.default_rng(seed + 424242)
    person = float(np.exp(rng.normal(0, 0.2)))
    S_crew = flight.S * shielding * person
    D_crew = integrate_damage(S_crew, params)
    marker = 1.0 + KAPPA_MARKER * lowpass(D_crew, 1.0 / SPH, tau_h)
    step = max(1, int(round(interval_h * SPH)))
    draw_idx = np.arange(step, flight.n, step)
    values = marker[draw_idx] * np.exp(rng.normal(0, cv, len(draw_idx)))
    return CrewPanel(draw_idx=draw_idx, values=values, truth=marker, interval_h=interval_h, cv=cv)
