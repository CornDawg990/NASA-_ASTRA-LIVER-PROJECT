"""Physics-informed digital twin: an extended Kalman filter over the chip's latent state.

State  x = [V, xi, sh]
    V   hepatocyte functional viability (1 = pristine)
    xi  ln(S): log of the *hidden* environmental insult rate (radiation + microgravity + drug)
    sh  fast metabolic-stress state

The process model is the same mechanistic ODE family as the simulator (damage vs repair,
saturating metabolic response) but uses the *nominal calibrated* parameters, while the
simulated chip uses a perturbed copy: the twin is never handed the truth.

Observations (ratios to healthy baseline, 5-minute block means of artifact-cleaned telemetry)
    O2 consumption, glucose uptake, acidification, albumin   (every block)
    fraction of blebbing cells from optical mapping            (every 30 min)

Robustness: sequential scalar updates with innovation gating. A channel that is persistently
gated for 2 h is re-admitted with inflated noise, so genuine regime changes are not rejected
forever while one-off glitches are.

Outputs: posterior over viability, inferred insult rate, metabolic stress, alert levels, and
Monte-Carlo forecasts (including what-if scenarios such as a countermeasure or shelter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .physiology import SPD, SPH, ChipParams, bleb_probability

XI_MIN, XI_MAX = np.log(1e-3), np.log(3.0)
LEVELS = {0: "NOMINAL", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}


@dataclass(frozen=True)
class TwinConfig:
    step: int = 5  # samples per filter step (5 min)
    q_xi: float = 0.9  # random-walk std of ln S per sqrt(hour)
    q_v: float = 0.10  # per sqrt(day)
    q_sh: float = 0.05  # per sqrt(hour)
    gate_sigma: float = 4.0
    gate_patience: int = 24  # steps (2 h) before a gated channel is re-admitted
    sensor_inflate: float = 1.3  # twin assumes microgravity-noisy sensors
    # model-mismatch noise floors (absolute, in ratio units) per observation channel: the twin
    # knows its calibrated coefficients are only approximately right
    me_our: float = 0.09
    me_glc: float = 0.09
    me_acid: float = 0.10
    me_alb: float = 0.03
    me_bleb: float = 0.03
    xi0: float = float(np.log(0.02))
    p0: tuple = (0.02, 1.5, 0.2)  # prior std of V, xi, sh
    persist_steps: int = 6  # 30 min persistence for alerts


@dataclass
class TwinResult:
    t_h: np.ndarray
    V: np.ndarray
    V_sd: np.ndarray
    S: np.ndarray
    S_lo: np.ndarray
    S_hi: np.ndarray
    sh: np.ndarray
    sh_sd: np.ndarray
    x_all: np.ndarray
    P_all: np.ndarray
    gated: pd.DataFrame
    level: np.ndarray
    alert_log: pd.DataFrame
    params: ChipParams
    cfg: TwinConfig


def _obs_model(name: str, x: np.ndarray, p: ChipParams):
    V, _, sh = x
    if name == "our":
        return V * (1 + p.c_our * sh), np.array([1 + p.c_our * sh, 0.0, p.c_our * V])
    if name == "glc":
        return V * (1 + p.c_glc * sh), np.array([1 + p.c_glc * sh, 0.0, p.c_glc * V])
    if name == "acid":
        k = p.acid_dead / p.acid0
        return V * (1 + p.c_acid * sh) + k * (1 - V), np.array([(1 + p.c_acid * sh) - k, 0.0, p.c_acid * V])
    if name == "alb":
        v = max(V, 1e-3)
        return v**1.5, np.array([1.5 * v**0.5, 0.0, 0.0])
    if name == "bleb":
        d = max(1.0 - V, 1e-4)
        return float(bleb_probability(d)), np.array([-0.90 * 1.3 * d**0.3, 0.0, 0.0])
    raise KeyError(name)


def _block_stats(x: np.ndarray, step: int):
    nb = len(x) // step
    a = x[: nb * step].reshape(nb, step)
    ok = np.isfinite(a)
    cnt = ok.sum(1)
    tot = np.where(ok, a, 0.0).sum(1)
    mean = np.full(nb, np.nan)
    good = cnt >= 2
    mean[good] = tot[good] / cnt[good]
    return mean, cnt


def run_twin(
    clean: pd.DataFrame,
    img_idx: np.ndarray,
    bleb_meas: np.ndarray,
    params: ChipParams = ChipParams(),
    cfg: TwinConfig = TwinConfig(),
    n_cells: int | None = None,
) -> TwinResult:
    n_cells = n_cells or params.n_cells_imaged
    st = cfg.step
    dt_h = st / SPH
    dt_d = st / SPD
    o2, cnt_o = _block_stats(clean["o2_out"].to_numpy(float), st)
    ph, cnt_p = _block_stats(clean["ph_out"].to_numpy(float), st)
    glc, cnt_g = _block_stats(clean["glucose_out"].to_numpy(float), st)
    alb, cnt_a = _block_stats(clean["albumin"].to_numpy(float), st)
    nb = len(o2)

    obs = {
        "our": ((params.o2_in - o2) / params.our0, params.sd_o2 * cfg.sensor_inflate / params.our0, cnt_o),
        "glc": ((params.glc_in - glc) / params.gu0, params.sd_glc * cfg.sensor_inflate / params.gu0, cnt_g),
        "acid": ((params.ph_in - ph) / params.acid0, params.sd_ph * cfg.sensor_inflate / params.acid0, cnt_p),
        "alb": (alb / params.alb0, params.sd_alb_rel * cfg.sensor_inflate, cnt_a),
    }
    bleb_at: dict[int, float] = {}
    for idx, m in zip(img_idx, bleb_meas):
        k = int(np.ceil(idx / st)) - 1
        if 0 <= k < nb:
            bleb_at[k] = float(m)

    x = np.array([1.0, cfg.xi0, float(params.s50 and 0.0)])
    g0 = np.exp(cfg.xi0) / (np.exp(cfg.xi0) + params.s50)
    x[2] = g0
    P = np.diag(np.array(cfg.p0) ** 2)
    Q = np.diag([cfg.q_v**2 * dt_d, cfg.q_xi**2 * dt_h, cfg.q_sh**2 * dt_h])
    a, r, tau = params.a_damage, params.r_repair, params.tau_sh_h

    X = np.empty((nb, 3))
    PA = np.empty((nb, 3, 3))
    names = list(obs.keys()) + ["bleb"]
    gated = {c: np.zeros(nb, dtype=np.int8) for c in names}
    gate_run = {c: 0 for c in names}
    I3 = np.eye(3)

    for k in range(nb):
        # ---- predict
        V, xi, sh = x
        S = np.exp(xi)
        g = S / (S + params.s50)
        dg = params.s50 * S / (S + params.s50) ** 2
        xn = np.array([V + dt_d * (-a * S * V + r * (1 - V)), xi, sh + (dt_h / tau) * (g - sh)])
        F = np.array(
            [
                [1 + dt_d * (-a * S - r), -dt_d * a * S * V, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, (dt_h / tau) * dg, 1 - dt_h / tau],
            ]
        )
        x = xn
        P = F @ P @ F.T + Q
        # ---- update (sequential scalar)
        for c in names:
            if c == "bleb":
                if k not in bleb_at:
                    continue
                y = bleb_at[k]
                hx, H = _obs_model("bleb", x, params)
                pc = min(max(hx, 0.02), 0.98)
                R = pc * (1 - pc) / n_cells + cfg.me_bleb**2
            else:
                ratio, sd, cnt = obs[c]
                y = ratio[k]
                if not np.isfinite(y):
                    continue
                hx, H = _obs_model(c, x, params)
                scale = hx if c == "alb" else 1.0
                me = {"our": cfg.me_our, "glc": cfg.me_glc, "acid": cfg.me_acid, "alb": cfg.me_alb}[c]
                R = (sd * scale) ** 2 / max(cnt[k], 1) + me**2
            nu = y - hx
            Sv = float(H @ P @ H) + R
            if nu * nu / Sv > cfg.gate_sigma**2:
                gate_run[c] += 1
                gated[c][k] = 1
                if gate_run[c] <= cfg.gate_patience:
                    continue
                R *= 9.0
                Sv = float(H @ P @ H) + R
            else:
                gate_run[c] = 0
            K = (P @ H) / Sv
            x = x + K * nu
            P = (I3 - np.outer(K, H)) @ P
            P = 0.5 * (P + P.T)
        x[0] = min(max(x[0], 0.02), 1.0)
        x[1] = min(max(x[1], XI_MIN), XI_MAX)
        x[2] = min(max(x[2], 0.0), 1.0)
        d = np.diag(P).copy()
        d[d < 1e-10] = 1e-10
        P[np.diag_indices(3)] = d
        X[k] = x
        PA[k] = P

    sd = np.sqrt(np.clip(PA[:, [0, 1, 2], [0, 1, 2]], 0, None))
    t_h = (np.arange(nb) + 1) * dt_h
    level, log = _alerts(t_h, X, sd, cfg)
    return TwinResult(
        t_h=t_h, V=X[:, 0], V_sd=sd[:, 0], S=np.exp(X[:, 1]),
        S_lo=np.exp(X[:, 1] - 1.96 * sd[:, 1]), S_hi=np.exp(X[:, 1] + 1.96 * sd[:, 1]),
        sh=X[:, 2], sh_sd=sd[:, 2], x_all=X, P_all=PA,
        gated=pd.DataFrame(gated), level=level, alert_log=log, params=params, cfg=cfg,
    )


def _alerts(t_h, X, sd, cfg: TwinConfig):
    V, sh = X[:, 0], X[:, 2]
    Vsd, shsd = sd[:, 0], sd[:, 2]
    cond = {
        1: (sh - 1.96 * shsd > 0.40) | (V + 1.64 * Vsd < 0.95),
        2: (V < 0.85) & (V + 1.64 * Vsd < 0.90),
        3: (V < 0.70),
    }
    m = cfg.persist_steps
    level = np.zeros(len(V), dtype=int)
    for lv in (1, 2, 3):
        c = pd.Series(cond[lv].astype(float)).rolling(m, min_periods=m).min().fillna(0).to_numpy() > 0
        level = np.where(c, lv, level)
    rows = []
    prev = 0
    reasons = {
        1: "Sustained metabolic-stress signature or early viability loss",
        2: "Functional compromise: viability below 85 %",
        3: "Critical: viability below 70 %",
        0: "Returned to nominal",
    }
    for i, lv in enumerate(level):
        if lv != prev:
            rows.append(dict(t_h=float(t_h[i]), level=int(lv), label=LEVELS[int(lv)], reason=reasons[int(lv)]))
            prev = lv
    return level, pd.DataFrame(rows, columns=["t_h", "level", "label", "reason"])


def first_alert_time(res: TwinResult, level: int) -> float | None:
    hit = np.where(res.level >= level)[0]
    return float(res.t_h[hit[0]]) if len(hit) else None


# ---------------------------------------------------------------------------
# Forecasting / what-if
# ---------------------------------------------------------------------------
def forecast(
    res: TwinResult,
    step_index: int,
    *,
    horizon_h: float = 72.0,
    n: int = 500,
    s_mult: float = 1.0,
    mult_start_h: float = 0.0,
    threshold: float = 0.70,
    seed: int = 0,
    dt_h: float = 0.25,
    half_life_h: float | None = None,
    s_floor: float = 0.0,
) -> dict:
    """Monte-Carlo forecast of viability from the posterior at ``step_index``.

    By default exposure is assumed to persist at the inferred level (worst case). With
    ``half_life_h`` the insult relaxes exponentially toward ``s_floor`` (e.g. a solar particle
    event passing). ``s_mult`` scales it from ``mult_start_h`` onwards (countermeasure / shelter).
    """
    p = res.params
    rng = np.random.default_rng(seed)
    x = res.x_all[step_index]
    P = res.P_all[step_index] + 1e-12 * np.eye(3)
    L = np.linalg.cholesky(P)
    s = x + rng.standard_normal((n, 3)) @ L.T
    V = np.clip(s[:, 0], 0.02, 1.0)
    S0 = np.exp(np.clip(s[:, 1], XI_MIN, XI_MAX))
    steps = int(round(horizon_h / dt_h))
    dt_d = dt_h / 24.0
    traj = np.empty((steps + 1, n))
    traj[0] = V
    ttt = np.full(n, np.inf)
    for k in range(steps):
        t = k * dt_h
        S_t = S0 if half_life_h is None else s_floor + np.maximum(S0 - s_floor, 0.0) * 0.5 ** (t / half_life_h)
        S = S_t * (s_mult if t >= mult_start_h else 1.0)
        V = V + dt_d * (-p.a_damage * S * V + p.r_repair * (1 - V))
        V = np.clip(V, 0.0, 1.0)
        traj[k + 1] = V
        newly = (V < threshold) & ~np.isfinite(ttt)
        ttt[newly] = (k + 1) * dt_h
    q = np.quantile(traj, [0.10, 0.50, 0.90], axis=1)
    return dict(
        t_h=np.arange(steps + 1) * dt_h,
        q10=q[0], q50=q[1], q90=q[2],
        ttt=ttt,
        p_cross=float(np.isfinite(ttt).mean()),
        median_ttt=float(np.median(ttt[np.isfinite(ttt)])) if np.isfinite(ttt).any() else float("inf"),
        v_end_median=float(q[1][-1]),
    )


def index_at_hour(res: TwinResult, hour: float) -> int:
    return int(np.clip(np.searchsorted(res.t_h, hour), 0, len(res.t_h) - 1))
