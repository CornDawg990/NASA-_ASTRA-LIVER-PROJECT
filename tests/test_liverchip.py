"""Test-suite for Liver-Chip Mission Control. Run:  pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liverchip import comms, imaging, nasa
from liverchip.detect import detect, evaluate_detection, paired_divergence
from liverchip.physiology import (
    CHANNELS, PRESETS, SPD, Scenario, ChipParams, ground_scenario, integrate_damage, simulate_chip, stress_profile,
)
from liverchip.pipeline import run_mission
from liverchip.twin import forecast, run_twin

SPE = PRESETS["ISS · solar particle event (day 3)"]


# ------------------------------------------------------------------ physiology
def test_deterministic_given_seed():
    a, b = simulate_chip(SPE, seed=3), simulate_chip(SPE, seed=3)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    assert not a.obs.equals(simulate_chip(SPE, seed=4).obs)


def test_spe_integral_equals_dose():
    scn = Scenario(days=6, dose_rate_msv_day=0.0, microgravity_stress=0.0, spe_events=((1.0, 10.0, 6.0),), alpha_dose=0.02)
    S = stress_profile(scn, int(6 * SPD))
    assert np.sum(S) / SPD == pytest.approx(0.02 * 10.0, rel=0.02)


def test_viability_bounded_and_monotone_without_repair():
    p = ChipParams(r_repair=0.0)
    D = integrate_damage(np.full(5 * SPD, 0.3), p)
    assert np.all(np.diff(D) >= 0) and D.min() >= 0 and D.max() < 1


def test_ground_scenario_has_no_space_stress():
    g = ground_scenario(SPE)
    assert np.allclose(stress_profile(g, SPD), 0.0)


def test_artifacts_corrupt_observation_not_truth():
    f = simulate_chip(SPE, seed=1)
    assert f.truth.notna().all().all()
    assert f.obs.isna().any().any()  # packet dropouts


# ------------------------------------------------------------------ detection
def test_bubbles_are_masked_and_not_called_biological():
    recalls, false_bio = [], 0
    for seed in range(4):
        f = simulate_chip(PRESETS["ISS · chronic LEO exposure"], seed=seed)
        m = evaluate_detection(f.truth_events, detect(f.obs))
        recalls.append(m["bubble_recall"])
        false_bio += m["false_bio_alarms"]
    assert np.mean(recalls) > 0.9
    assert false_bio == 0


def test_solar_event_detected_quickly():
    for seed in range(3):
        f = simulate_chip(SPE, seed=seed)
        m = evaluate_detection(f.truth_events, detect(f.obs))
        assert m["bio_recall"] == 1.0
        assert m["median_delay_min"] <= 120


def test_paired_divergence_sees_flight_effect_not_ground_noise():
    f = simulate_chip(SPE, seed=1)
    g = simulate_chip(ground_scenario(SPE), flight=False, seed=51)
    d = paired_divergence(detect(f.obs).clean, detect(g.obs).clean)
    assert d.confirmed_h is not None
    g2 = simulate_chip(ground_scenario(SPE), flight=False, seed=52)
    d0 = paired_divergence(detect(g.obs).clean, detect(g2.obs).clean)
    assert d0.confirmed_h is None or d0.confirmed_h > d.confirmed_h


# ------------------------------------------------------------------ twin
@pytest.fixture(scope="module")
def spe_run():
    return run_mission(SPE, seed=1)


def test_twin_accuracy_and_coverage(spe_run):
    m = spe_run.metrics
    assert m["v_rmse"] < 0.03
    assert m["v_coverage95"] > 0.9


def test_alert_escalates_after_event_not_before(spe_run):
    w = spe_run.metrics["alert_warning_h"]
    assert w is not None and 72 <= w <= 100
    assert spe_run.metrics["ground_false_alerts"] == 0


def test_chip_beats_bloodwork(spe_run):
    m = spe_run.metrics
    assert m["crew_detect_h"] is None or m["lead_warning_h"] > 0


def test_forecast_ordering(spe_run):
    tw = spe_run.twin_f
    k = len(tw.t_h) // 2
    worst = forecast(tw, k, horizon_h=48, seed=0)
    decays = forecast(tw, k, horizon_h=48, seed=0, half_life_h=6.0, s_floor=0.02)
    cm = forecast(tw, k, horizon_h=48, seed=0, s_mult=0.3)
    assert worst["v_end_median"] <= decays["v_end_median"]
    assert worst["v_end_median"] < cm["v_end_median"]


# ------------------------------------------------------------------ imaging
@pytest.fixture(scope="module")
def models():
    return imaging.load_model()


def test_segmentation_finds_most_cells():
    f = imaging.make_field(damaged_frac=0.2, seed=3)
    lab = imaging.segment(f.cyto, f.nuc)
    visible = len(np.unique(f.truth_labels)) - 1
    assert 0.75 * visible <= lab.max() <= 1.1 * visible


def test_bleb_fraction_calibrated(models):
    for p in (0.0, 0.5, 1.0):
        f = imaging.make_field(damaged_frac=p, seed=11)
        a = imaging.analyze(f.cyto, f.nuc, models)
        assert abs(a.bleb_fraction - p) < 0.1


def test_shape_only_model_runs_without_nuclei(models):
    f = imaging.make_field(damaged_frac=0.5, seed=12)
    a = imaging.analyze(f.cyto, None, models)
    assert 0.3 < a.bleb_fraction < 0.8


# ------------------------------------------------------------------ NASA
def test_parse_de_table_autodetects_columns():
    raw = pd.DataFrame({"SYMBOL": ["Nqo1", "Alb"], "Log2fc_(Flight)v(Ground)": [1.0, -0.5], "Adj.p.value": [0.01, 0.2]})
    de = nasa.parse_de_table(raw)
    assert list(de.columns[:3]) == ["gene", "log2fc", "p"] and len(de) == 2


def test_parse_de_table_rejects_missing_fc():
    with pytest.raises(ValueError):
        nasa.parse_de_table(pd.DataFrame({"SYMBOL": ["a"], "count": [3]}))


def test_panel_scores_direction_and_severity():
    hi = nasa.panel_scores(nasa.parse_de_table(nasa.synthetic_de_table(effect=1.5)))
    lo = nasa.panel_scores(nasa.parse_de_table(nasa.synthetic_de_table(effect=0.0)))
    assert hi["hsi"] > 0.3 > abs(lo["hsi"]) - 0.1
    assert hi["hsi_ci"][0] < hi["hsi"] < hi["hsi_ci"][1]
    assert nasa.severity_from_hsi(10) == nasa.SEV_MAX and nasa.severity_from_hsi(-1) == nasa.SEV_MIN


# ------------------------------------------------------------------ comms
def test_light_time_physics():
    near = next(l for l in comms.LINKS if "closest" in l.name)
    far = next(l for l in comms.LINKS if "farthest" in l.name)
    assert near.one_way_s == pytest.approx(259.5, abs=1.0)
    assert far.one_way_s == pytest.approx(1332.3, abs=2.0)
    iss = comms.LINKS[0]
    assert iss.one_way_s < 1.0


def test_onboard_always_faster_and_edge_reduces_volume():
    for l in comms.LINKS:
        t = comms.decision_timeline(60, l)
        assert t.minutes.iloc[0] < t.minutes.iloc[1]
    v = comms.daily_volume()
    assert v.MB_per_day.iloc[:2].sum() / v.MB_per_day.iloc[2:].sum() > 100


# ------------------------------------------------------------------ app
def test_app_all_presets_render():
    import pathlib

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=240).run()
    assert not at.exception
    for name in PRESETS:
        at.sidebar.selectbox[0].select(name).run()
        assert not at.exception, name
