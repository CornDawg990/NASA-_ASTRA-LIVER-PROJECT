# Predictive Liver-Chip · Mission Control

NASA Space Apps proof of concept: a closed-loop **AI digital twin for an orbital hepatic organ-chip**.
The app simulates chip telemetry from a spaceflight mission, rejects microgravity artifacts, infers the hidden
environmental insult with a physics-informed Kalman filter, raises tiered alerts, forecasts what-if scenarios,
and compares the chip's early warning against simulated crew bloodwork. Real NASA OSDR liver data can calibrate it.

## Run

```bash
./run.sh                       # installs requirements and launches
# or
pip install -r requirements.txt && streamlit run app.py
pytest -q                      # 21 tests
```

First launch trains/loads the cell-imaging model (`liverchip/_bleb_model.joblib`, shipped pre-trained).

## Tabs

| Tab | What it shows |
|---|---|
| Mission Control | live replay of telemetry, rejected artifacts, ground-control overlay, twin viability, alert level, event log |
| Digital Twin & Forecast | EKF posterior, inferred insult rate, Monte-Carlo forecast with a countermeasure what-if |
| Orbit vs Ground | flight-vs-ground divergence (CUSUM), crew-bloodwork comparison, lead time, twin↔crew correlation |
| Cell Imaging | procedural micrographs, watershed segmentation, bleb classifier; upload your own image |
| NASA data | OSDR metadata query, differential-expression upload, liver-panel scoring → simulator severity |
| Digital Thread | light-time, on-board vs ground-in-loop decisions, downlink volume |
| Benchmark & Methods | Monte-Carlo benchmark over all scenarios, equations, assumptions, limits |


## What is real vs simulated

Real: artifact rejection, event detection, Kalman twin, forecasting, segmentation, OSDR connector,
gene-panel scoring with bootstrap CI, light-time maths, test-suite. MSL/RAD Mars-cruise dose rate (1.8 mSv/day).

Simulated: all chip telemetry and images, the crew blood panel, and the biological response itself.
The twin's model family matches the simulator's, so reported accuracy is optimistic versus a real chip.
The image classifier is trained on procedurally generated cells. The HSI→severity mapping is an explicit assumption.

## Layout

```
app.py                 Streamlit UI
liverchip/physiology.py   chip simulator + crew panel
liverchip/detect.py       3-stage detection (artifacts, events, paired CUSUM)
liverchip/twin.py         extended Kalman filter, alerts, Monte-Carlo forecast
liverchip/imaging.py      micrograph generator, segmentation, classifier
liverchip/nasa.py         OSDR connector + gene panels
liverchip/comms.py        digital-thread latency and data budget
liverchip/pipeline.py     end-to-end run + benchmark
tests/                    pytest suite
```
