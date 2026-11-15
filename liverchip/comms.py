"""Digital-thread model: link latency, data volume, and ground-in-the-loop vs on-board decisions.

All physical constants are standard; the instrument data sizes are explicit assumptions that the
app exposes as sliders. Light-time is computed, not looked up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

C_KM_S = 299_792.458
AU_KM = 149_597_870.7
GEO_ALT_KM = 35_786.0


@dataclass(frozen=True)
class Link:
    name: str
    one_way_s: float
    note: str


def _ls(km: float) -> float:
    return km / C_KM_S


LINKS: list[Link] = [
    Link("ISS · relay via GEO satellite", _ls(2 * (GEO_ALT_KM + 400)) + 0.15, "≈ 2 x GEO hop + processing (assumption: 0.15 s)"),
    Link("Cislunar / Moon (384,400 km)", _ls(384_400) + 0.5, "straight-line light time + 0.5 s processing"),
    Link("Mars · closest approach (0.52 AU)", _ls(0.52 * AU_KM), "light time only"),
    Link("Mars · farthest (2.67 AU)", _ls(2.67 * AU_KM), "light time only (excludes solar conjunction blackout)"),
]


@dataclass(frozen=True)
class DataBudget:
    n_channels: int = 4
    bytes_per_sample: int = 4
    sample_s: int = 60
    img_w: int = 2048
    img_h: int = 2048
    img_channels: int = 2
    img_bytes: int = 2
    img_every_min: int = 30
    lossless_ratio: float = 0.5  # assumed lossless compression of raw frames
    cells_per_image: int = 250
    features_per_cell: int = 9
    feature_bytes: int = 4
    twin_packet_bytes: int = 256
    twin_packets_per_hour: int = 12


def daily_volume(b: DataBudget = DataBudget()) -> pd.DataFrame:
    tele = b.n_channels * b.bytes_per_sample * 86400 / b.sample_s
    imgs_per_day = 24 * 60 / b.img_every_min
    raw_img = imgs_per_day * b.img_w * b.img_h * b.img_channels * b.img_bytes * b.lossless_ratio
    feat = imgs_per_day * b.cells_per_image * b.features_per_cell * b.feature_bytes
    twin = b.twin_packet_bytes * b.twin_packets_per_hour * 24
    rows = [
        ("Raw telemetry", tele, "downlink all"),
        ("Raw micrographs (lossless)", raw_img, "downlink all"),
        ("Per-cell features (on-board segmentation)", feat, "edge-processed"),
        ("Twin state packets", twin, "edge-processed"),
    ]
    df = pd.DataFrame(rows, columns=["stream", "bytes_per_day", "mode"])
    df["MB_per_day"] = df.bytes_per_day / 1e6
    return df


def decision_timeline(detect_delay_min: float, link: Link, ground_think_min: float = 20.0) -> pd.DataFrame:
    """Minutes from onset until an action reaches the crew, for ground-in-loop vs on-board twin."""
    rt = 2 * link.one_way_s / 60.0
    onboard = detect_delay_min
    ground = detect_delay_min + link.one_way_s / 60.0 + ground_think_min + link.one_way_s / 60.0
    return pd.DataFrame(
        [
            dict(mode="On-board twin (autonomous alert)", minutes=onboard, breakdown=f"detect {detect_delay_min:.0f} min"),
            dict(
                mode="Ground-in-the-loop",
                minutes=ground,
                breakdown=f"detect {detect_delay_min:.0f} + downlink {link.one_way_s/60:.1f} + analysis {ground_think_min:.0f} + uplink {link.one_way_s/60:.1f} min (RTT {rt:.1f} min)",
            ),
        ]
    )
