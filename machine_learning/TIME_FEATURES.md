# Time features adapted from the acoustic-waveguide paper

Source: Yufan Ye et al., *Multipoint Simultaneous Tactile Force and Location
Sensing Through Soft Acoustic Waveguides*, IEEE Transactions on Instrumentation
and Measurement, volume 75, 2026, article 9521712.
[DOI: 10.1109/TIM.2026.3693449](https://doi.org/10.1109/TIM.2026.3693449).
The supplied PDF calls the algorithm **WITS**, not SFST, and uses **SVR** for
regression rather than SVM classification.

## What the paper establishes

Section III-A and Figure 2 (PDF page 3) describe received intensity, echo
occurrence times and echo amplitudes. The figures separate emission noise,
the directly received packet and reflected packets. Contact changes packet
amplitudes and can introduce earlier reflections. Figure 4 (page 4) combines
time features with wavelet-packet energies, dimension reduction and SVR.

Section IV-C (page 6) specifies 10 MHz sampling, 8,192 samples/channel,
nine-level db4 wavelet packets, PCA initially retaining 20 components,
supervised projection refinement (PB), and PSO-tuned RBF SVR. The apparatus has
two transducers and two contact points. The text does **not** provide a complete
peak detector, intensity definition, time-gate boundaries or thresholds that
can simply be copied into TASS.

This implementation adapts the feature ideas to four receivers and the existing
single-contact force/location labels. It does not reproduce WPD, PB, PSO,
the paper's multioutput coupling, its ultrasonic filtering or its physical setup.
Its 500 kHz–5 MHz filter would not apply to the low-rate example recordings.

## Features implemented

All existing FFT/state features are preserved. Set `time_features_enabled=false`
to recover the original extracted feature set. Amplitudes are not normalized
independently per recording, which would discard useful force information.

For each receiver `rx1`…`rx4`, `time_features.py` adds:

| Suffix | Definition | Interpretation |
| --- | --- | --- |
| `time_energy_V2_s` | `sum((V - mean(V))**2) / sample_rate` | Integrated squared voltage; intensity proxy, not calibrated acoustic energy |
| `time_peak_envelope_V` | Maximum smoothed magnitude of the Hilbert analytic signal | Largest wave-packet envelope amplitude |
| `time_energy_centroid_fraction` | `sum((k/(N-1))*V_centered[k]**2) / sum(V_centered**2)` | Energy's location within the acquisition window, from 0 to 1 |
| `time_segment1_rms_V` … `time_segment4_rms_V` | AC RMS in four consecutive equal-count portions of the window | Retains coarse information about where signal energy occurs |

Mean removal is over the whole window, not independently within each segment.
Segment count and envelope smoothing are configurable. The Hilbert envelope
uses a 9-sample moving average by default. Zero signals produce zero descriptors.
The energy centroid and segment indices are **window-relative descriptors**, not
physical propagation times. Continuous sine recordings may gain little from them;
the energy feature also overlaps mathematically with existing RMS information.

Seven features per receiver add **28 predictors**: the baseline 60 becomes 88.
Time descriptors are applied to receivers only. All eight channels retain their
original FFT features, including the four measured actuator channels.

## Optional trigger-relative packet features

`echo_features_enabled` defaults to **false**. Enable it only for isolated,
synchronized burst/pulse captures whose timing and ringdown have been measured.
Use [`templates/triggered_samples.csv`](templates/triggered_samples.csv), adding
`excitation_trigger_time_s` in the same clock/units as `time_s`, constant for
the entire capture/trial. A new excitation capture needs a new trial ID.

Supply `echo_gate_s = [start, end]`, in **seconds after that trigger**. The gate
must exclude transmitter ringdown and fit inside an accepted window. The code
does not infer a trigger from a sinusoidal crossing or absolute recording time.
It does not distinguish echoes from several simultaneous sources; use separate
source captures for interpretable timing. Gate peaks can include transmitted
packets or other paths: the detector alone cannot establish their physical origin.

For the first two prominent envelope peaks, sorted by arrival rather than
amplitude, it records `echo_peakN_present`, `echo_peakN_time_s` and
`echo_peakN_amplitude_V`. Detection uses configurable absolute height,
relative prominence and minimum separation. Peaks at gate endpoints are not
accepted by the interior-peak detector. Missing peaks have zero-valued fields
**and a presence flag of zero**; that is a missing-packet encoding, not a claim
that a physical echo arrived at zero delay. No propagation speed or distance
is inferred from these delays automatically.

Two packets × three fields × four receivers add 24 further features, giving
112 predictors with generic time and echo features enabled.

[`config_echo_example.json`](config_echo_example.json) is only a **synthetic
fixture example** with an 8–105 ms gate, not a measured TASS or paper setting.
The defaults for smoothing/thresholds require instrument-specific validation.
Do not tune them against final test labels.

## Matched comparison

```bash
# Existing steady-sine toy signals: 60 vs 88 predictors.
python -m machine_learning compare-demo --signal-mode steady \
  --output machine_learning/outputs/time_steady_02

# Pulse-echo positive control: 60 vs 88 vs 112 predictors.
python -m machine_learning compare-demo --signal-mode echo \
  --output machine_learning/outputs/time_echo_02

# Your recordings: generic time descriptors are enabled by default.
python -m machine_learning compare --input path/to/samples.csv \
  --output machine_learning/outputs/time_hardware_01

# Triggered hardware captures: first replace the example gate/thresholds.
python -m machine_learning compare --input path/to/triggered_samples.csv \
  --config path/to/measured_echo_settings.json \
  --output machine_learning/outputs/echo_hardware_01
```

`comparison.py` runs two **continuous** regressors for each target: OLS and RBF
SVR. It compares `fft_baseline`, `fft_plus_time`, and optionally
`fft_plus_time_echo`, using exactly the same eligible windows, held-out sessions
and development folds. Each arm keeps the same servo/load-cell/frequency inputs.
The echo arm cannot leak its arrival-time columns into the generic-time arm.

Every training fold fits its own constant-column removal, predictor
standardization, PCA (up to 20 components) and target standardization. Using the
same PCA procedure for every arm makes this a feature-ablation benchmark, not
a comparison against the earlier best model from 237 different specifications.
It estimates the effect of adding a feature block to this entire pipeline,
including any resulting change to the PCA projection.

SVR searches `C = [1,10,100]`, `gamma = ['scale',0.01,0.1]`, and
`epsilon = [0.05,0.1]` through grouped development CV, independently per
target/feature set. These are application defaults, not the paper's PSO ranges.
C and epsilon operate on standardized targets; reported errors are converted
back to N and m. See the [official SVR documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
for their meanings. Force is not fed into the location model.

The best development-CV candidate from each arm is evaluated once on identical
holdout sessions. Outputs include `cv_search.csv`, `feature_sets.json`, split/CV
manifests, `holdout_metrics.csv`, `holdout_predictions.csv`, saved models, and
`paired_mae_changes.csv`. Negative MAE change means the added features improved
the point estimate. The paired 95% interval resamples entire holdout groups;
it is descriptive, not corrected for multiple comparisons, and excludes
training/tuning variability. An interval spanning zero is inconclusive.

SVR does not supply conventional coefficient p-values. The original
`run`/`train` workflow still performs the independent OLS/logistic significance
search, and it can use the additional features. It now permits 300 specifications
so the default 265-specification time-feature search is not truncated. Interactions
and p-value correction remain in that workflow rather than being assigned to
RBF kernel features.

## Interpretation of synthetic runs

No hardware training dataset was supplied. The steady-sine demonstration is the
previous toy distribution. The new pulse fixture is a deliberate positive control:
force changes packet amplitude and location changes its arrival time, with noise
and session gain variation. Its delays are invented; it is not a digital-twin
simulation, a recreation of the paper's apparatus, or evidence of real performance.
Tests additionally use known Gaussian-modulated packets and pure time shifts to
verify the actual feature calculations independent of any fitted regressor.

The steady example reuses the existing synthetic seed/split, so treat it as a
development comparison. A real benefit must be confirmed using independently
collected, triggered hardware sessions and an excitation protocol suitable for
separating arrivals. More time descriptors are not guaranteed to improve force
prediction, and nothing here yet estimates multiple simultaneous contacts.

Completed local checks (21 tests passed) and the synthetic RBF-SVR comparisons:

| Synthetic dataset / target | FFT baseline MAE | + generic time MAE | + time and triggered packets MAE |
| --- | --- | --- | --- |
| Steady sine / force | 0.9101 N | 0.9111 N | Not applicable |
| Steady sine / location | 69.90 mm | 61.46 mm | Not applicable |
| Pulse fixture / force | 0.2433 N | 0.2614 N | 0.2522 N |
| Pulse fixture / location | 90.92 mm | 22.62 mm | 14.94 mm |

The paired group-bootstrap intervals for the SVR location MAE changes were
below zero in these runs. The force-change intervals spanned zero. These are
exploratory synthetic findings, with no demonstrated hardware force improvement.

Generated reports (outputs are intentionally ignored by Git):
[steady-sine comparison](outputs/time_comparison_steady/results/REPORT.md) and
[pulse-echo comparison](outputs/time_comparison_echo/results/REPORT.md).
