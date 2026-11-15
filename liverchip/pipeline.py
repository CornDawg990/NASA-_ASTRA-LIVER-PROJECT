"""End-to-end mission run + Monte-Carlo benchmark. Used by the app, tests and CLI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .detect import DetectionResult, DivergenceResult, detect, evaluate_detection, paired_divergence
from .physiology import (
    PRESETS, SPH, ChipRun, CrewPanel, Scenario, ground_scenario, simulate_chip, simulate_crew_panel,
)
from .twin import TwinResult, first_alert_time, run_twin

CREW_THRESHOLD = 1.30  # relative marker level counted as "bloodwork shows it" (~ +3 CV)
CREW_CONSECUTIVE = 2


@dataclass
class MissionRun:
    scenario: Scenario
    seed: int
    flight: ChipRun
    ground: ChipRun
    det_f: DetectionResult
    det_g: DetectionResult
    twin_f: TwinResult
    twin_g: TwinResult
    div: DivergenceResult
    crew: CrewPanel
    metrics: dict


def crew_detection_hour(crew: CrewPanel) -> float | None:
    above = crew.values > CREW_THRESHOLD
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= CREW_CONSECUTIVE:
            return float(crew.draw_idx[i] / SPH)
    return None


def run_mission(
    scn: Scenario,
    seed: int = 0,
    *,
    mismatch: float = 0.08,
    bubble_rate: float | None = None,
    crew_interval_h: float = 12.0,
    shielding: float = 0.6,
) -> MissionRun:
    f = simulate_chip(scn, seed=seed, mismatch=mismatch, bubble_rate_per_day=bubble_rate)
    g = simulate_chip(ground_scenario(scn), flight=False, seed=seed + 50, mismatch=mismatch)
    det_f, det_g = detect(f.obs), detect(g.obs)
    tw_f = run_twin(det_f.clean, f.img_idx, f.bleb_meas)
    tw_g = run_twin(det_g.clean, g.img_idx, g.bleb_meas)
    div = paired_divergence(det_f.clean, det_g.clean)
    crew = simulate_crew_panel(f, seed=seed, interval_h=crew_interval_h, shielding=shielding)

    idx = (np.arange(len(tw_f.t_h)) + 1) * tw_f.cfg.step - 1
    err = tw_f.V - f.V[idx]
    m = tw_f.t_h > 6
    a1, a2, a3 = (first_alert_time(tw_f, k) for k in (1, 2, 3))
    crew_h = crew_detection_hour(crew)

    # twin-vs-crew agreement at blood-draw times (after the first day)
    di = crew.draw_idx[crew.draw_idx / SPH > 24]
    r_val = float("nan")
    if len(di) >= 4:
        k = np.clip(np.searchsorted(tw_f.t_h, di / SPH), 0, len(tw_f.t_h) - 1)
        damage = 1.0 - tw_f.V[k]
        vals = crew.values[np.isin(crew.draw_idx, di)]
        if np.std(damage) > 1e-9 and np.std(vals) > 1e-9:
            r_val = float(np.corrcoef(damage, vals)[0, 1])

    det = evaluate_detection(f.truth_events, det_f)
    det_g_m = evaluate_detection(g.truth_events, det_g)
    metrics = dict(
        v_rmse=float(np.sqrt(np.mean(err[m] ** 2))),
        v_bias=float(err[m].mean()),
        v_coverage95=float(np.mean(np.abs(err[m]) < 1.96 * tw_f.V_sd[m])),
        v_end_true=float(f.V[-1]),
        v_end_est=float(tw_f.V[-1]),
        alert_watch_h=a1, alert_warning_h=a2, alert_critical_h=a3,
        crew_detect_h=crew_h,
        lead_watch_h=(crew_h - a1) if (crew_h is not None and a1 is not None) else None,
        lead_warning_h=(crew_h - a2) if (crew_h is not None and a2 is not None) else None,
        r_twin_crew=r_val,
        # the matched ground chip only counts as a false alarm if it received no drug exposure
        ground_false_alerts=(float(int((tw_g.level >= 2).any())) if not scn.drug_events else float("nan")),
        ground_false_bio=det_g_m["false_bio_alarms"],
        **{f"det_{k}": v for k, v in det.items()},
    )
    return MissionRun(scn, seed, f, g, det_f, det_g, tw_f, tw_g, div, crew, metrics)


def benchmark(names: list[str] | None = None, seeds: range | list[int] = range(5), progress=None, **kw) -> pd.DataFrame:
    names = names or list(PRESETS)
    rows = []
    total = len(names) * len(list(seeds))
    k = 0
    for n in names:
        for s in seeds:
            r = run_mission(PRESETS[n], seed=s, **kw)
            rows.append(dict(scenario=n, seed=s, **r.metrics))
            k += 1
            if progress:
                progress(k / total, f"{n} · seed {s}")
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("scenario", sort=False)
    out = pd.DataFrame(
        {
            "runs": g.size(),
            "bubble recall": g["det_bubble_recall"].mean(),
            "event recall": g["det_bio_recall"].mean(),
            "median detect delay (min)": g["det_median_delay_min"].median(),
            "false bio alarms / run": g["det_false_bio_alarms"].mean(),
            "ground false alerts": g["ground_false_alerts"].apply(lambda s: float(s.sum()) if s.notna().any() else float("nan")),
            "viability RMSE": g["v_rmse"].mean(),
            "95% CI coverage": g["v_coverage95"].mean(),
            "chip lead vs bloodwork (h, median)": g["lead_warning_h"].median(),
            "bloodwork never detected (runs)": g["crew_detect_h"].apply(lambda s: int(s.isna().sum())),
        }
    )
    return out
