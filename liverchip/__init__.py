"""Liver-Chip Mission Control: a predictive digital twin for an orbital hepatic organ-chip.

Modules
-------
physiology : hepatic organ-chip simulator (ground truth generator) + crew blood panel
detect     : robust rolling z-score, artifact rejection, event classification, paired CUSUM
twin       : extended Kalman filter digital twin, alert logic, Monte-Carlo forecasting
imaging    : synthetic fluorescence micrographs, segmentation, bleb classifier
nasa       : NASA OSDR connector, gene-panel scoring, calibration from real data
comms      : digital-thread latency / bandwidth budget
pipeline   : end-to-end run used by the app, tests and benchmark
"""

__version__ = "1.0.0"
