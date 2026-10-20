# Segment-1 force and location validation

For the same-feature classifier and MLP experiment, see the
[model comparison](ml_model_comparison.md). The existing Ridge pair remains the
baseline and its artifacts are preserved.

The recommended trained pair is
`outputs/ml_segment1_compliant/robust/model/model_bundle.pkl`.
Use the [robust recipe and blind results](#robust-recipe-band-peaks-and-broader-development-training)
below; the initial experiment is retained to document its stress-test failure.

The original fixed-node contact changes the vibration boundary when location
changes, but has no force-dependent response at a fixed proximal location.
The proximal force has zero distal-joint lever arm, so 1 N and 5 N also give
the same held-pose tendon tension. Paired simulations with identical random
seed give identical voltage waveforms for the two forces. A balanced force
regressor therefore tends toward 3 N, with an uninformative RMSE of 2 N.
More complex estimators cannot recover information absent from these signals.

This experiment adds an optional contact response to the existing TASS
generator and trains **two independent single-output Ridge regressors**.
The unchanged hard-clamp protocol remains available as the negative control.
The new results are conditional on an **assumed, uncalibrated contact law**;
they do not establish real-hardware accuracy.

## Contact mechanics

The `contact_response` block in
`config/experiments/dataset_segment1_compliant.json` selects `mode: compliant`.
Instead of eliminating the contacted node, it adds a local incremental reaction
around the preloaded equilibrium:

```text
dynamic reaction = -k(F) * displacement - c(F) * velocity
k(F) = 200 N/m   * (F / 1 N)^(1/3)
c(F) = 0.004 N s/m * (F / 1 N)^(1/3)
```

The prefactors, reference force, both exponents and coefficient jitter are
editable. The power laws are illustrative empirical assumptions, not a fitted
law for the finger or a claim that the contact follows a particular material
model. The applied preload affects only the forward response; neither force,
contact coefficients, true tension nor trial identifiers become ML inputs.
Tendon tension and geometry remain unchanged in the segment-1 intervention.
No indentation geometry, contact separation or friction is modeled.

The point spring/damper formulation follows the elastic and viscous boundary
conditions described in the [COMSOL spring–damper documentation](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_solid.07.095.html).
Local damping couples string modes, so the solver retains the full damping
matrix. It integrates the state-space poles under zero-order-held inputs.
An independent full-matrix [`scipy.signal.lsim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html)
calculation validates its sampled response. The zero-stiffness/damping limit
matches an uncontacted string; finite compliance allows transmission across
contact. Default `hard_clamp` reproduces the previous solver.

`contact_response_force_dependent` records structural force dependence.
For compliant contact, `contact_force_identifiable` is `null`: force dependence
alone does not prove that the measured receivers identify force uniquely.
Trial metadata and summary store assumed/measured parameter provenance, actual
coefficients, and the full configuration. Dataset schema is 2.4.

## Collection and training recipe

All three collections retain the existing TASS geometry, two active/saved pairs
1 and 2, 30%/70% of **150 mm** (45/105 mm), and exact 1/5-N labels.
Each trial records five seconds at 75 kHz with a simultaneous 20–2000-Hz,
nominal ±3.3-V chirp that stops at 0.3 s. Simulated ADC is disabled. The original
four-pair routing is retained; this is not a separate bench generator.

| Collection | Trials | Drive/gain variation | Contact-coefficient variation | Voltage noise |
| --- | --- | --- | --- | --- |
| Training/validation/test | 160: 112 / 24 / 24 | ±3% / ±3% | ±10%, independently for k and c | 10 µV RMS |
| Independent fresh test | 80, all test | ±3% / ±3% | ±10% | 10 µV RMS |
| Stress test | 80, all test | ±10% / ±10% | ±20% | 30 µV RMS |

All four conditions are balanced in every collection and split. The independent
collections use different random seeds; the runner checks that no trial seed is
reused across collections. These variation ranges are illustrative, not measured
hardware distributions. The stress test changes nuisance variation, not the
assumed force law; it does not test an unknown material law or ADC/front-end drift.

The default feature window is the first 1.5 s of each full five-second recording.
It covers the excitation and early decay; the original recording is retained.
Use `--window-s` on the validation runner to select a different interval length.
The two models share 21 sensor/timing feature names, but each has its own scaler,
coefficients and separately validation-selected Ridge alpha. Only training
trials fit the scalers and coefficients. Validation chooses from alpha 0.01,
0.1, 1, 10 and 100; independent test results do not select or refit models.

The two-span analytical feature equations assume a hard clamp and correctly
identified opposite-span modes. They are not used with the compliant response.
Physical force dependence enters the simulator through its contact law; the ML
features are calculated from receiver voltage. This also avoids assuming that
the projected 30% clamp lies between receivers 1 and 2.

## Reproduce

From `Zach/` with the environment active, use new empty output directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_segment1_compliant.json \
  --output outputs/dataset_segment1_compliant

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_segment1_compliant_test.json \
  --output outputs/dataset_segment1_compliant_test --split test

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_segment1_compliant_stress.json \
  --output outputs/dataset_segment1_compliant_stress --split test

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.segment1_validation \
  --training-data outputs/dataset_segment1_compliant \
  --fresh-test outputs/dataset_segment1_compliant_test \
  --stress-test outputs/dataset_segment1_compliant_stress \
  --output outputs/ml_segment1_compliant/validated
```

The runner saves `report.json`, `validation_summary.png/.svg`, extracted
features and a `model/model_bundle.pkl` holding both independent regressors.
`model/training_loss.png/.svg` show validation/training MSE across Ridge
candidates, not training epochs. `held_out`, `fresh_test` and `stress_test`
each have JSON metrics, prediction CSVs, preprocessing metadata, continuous
prediction plots, and an additional `.discrete.csv` with nearest allowed labels.
The original continuous outputs are preserved. The report includes RMSE, R²,
per-condition errors, nearest-label accuracy and 95% Wilson intervals, as well
as a same-seed force-sensitivity audit. The model file's hash is recorded before
independent evaluation and checked afterward to confirm it was not refitted.

For ordinary CLI use, equivalent extraction/training is:

```bash
python -m tass.ml_cli extract \
  --input outputs/dataset_segment1_compliant \
  --output outputs/my_ridge/features \
  --sensor-id 1 --sensor-id 2 --sensor-representation voltage \
  --remove-sensor-mean --window-s 1.5 --stride-s 5 \
  --target contact_location_m --target contact_force_N

python -m tass.ml_cli train \
  --features outputs/my_ridge/features --output outputs/my_ridge/model \
  --task regression --regressor ridge
```

## Inference and practical limits

The bundle restores both separate Ridge models, feature order, window and DC
removal automatically. It uses simulation voltage directly. Real raw counts
are calibrated to voltage before the same preprocessing; original ADC files
remain unchanged. This sensor-only recipe does not require reference captures.

```bash
python -m tass.ml_cli predict \
  --model outputs/ml_segment1_compliant/robust/model/model_bundle.pkl \
  --input captures/segment1_capture.csv \
  --adc-full-scale-v 3.3 --adc-resolution-bits 12 \
  --output outputs/ml_segment1_compliant/real_predictions.csv
```

Replace nominal ADC calibration with measured slope/offset as available.
To transfer force prediction to hardware, measure repeated receiver responses
at both positions and both known loads, fit/verify the contact coefficients
against resonances and decay, and evaluate on untouched real capture sessions.
The assumed actuator N/V conversion, sensor V/m gains, receiver front end and
measured sample clock also need to match the recordings; ADC count-to-voltage
calibration alone does not calibrate this mechanical/electrical transfer path.
If hardware responses do not vary with force above noise, a model cannot
identify force from these two receivers and this excitation alone.

Two tested locations and two tested forces support a four-condition task.
Low error on these endpoints does not establish continuous localization or
continuous force estimation, nor does nearest-label accuracy replace the
reported continuous regression errors.

## Robust recipe: band peaks and broader development training

The first 21-feature Ridge pair passed the nominal fresh-seed collection:
location RMSE 0.421 mm, force RMSE 0.248 N, both with 80/80 nearest labels correct.
It failed to remain a useful continuous regressor under the broader stress
variation: location RMSE 84.946 mm, force RMSE 1.307 N and force-class accuracy
75/80 (93.75%). Location-class accuracy alone was 100% despite severe location
overshoot, so class accuracy by itself would hide this failure.

The cause was a dominant-peak identity switch. In five 105-mm/5-N stress trials,
sensor 2's global peak moved from the low-frequency band near 105 Hz to about
427 Hz. Treating that global maximum as a continuous location cue produced
roughly 445-mm location estimates. The original reports remain in
`outputs/ml_segment1_compliant/validated/` and are not replaced.

The revised recipe uses `--spectral-mode band_peaks`. Each sensor contributes
mean, standard deviation, RMS, peak-to-peak, three absolute band powers,
three peak frequencies and three normalized power fractions for 20–200,
200–800 and 800–2000 Hz. There are 27 inputs including sample rate.
Global dominant frequency, global peak amplitude and global spectral SNR are
excluded from this sensor-only recipe. Fixed-band peaks retain low-frequency
information when a higher-frequency mode becomes strongest. Fractional powers
are invariant to a uniform multiplicative voltage gain. This is not a claim
that every band peak identifies the same physical mode across trials.

The exposed 80-trial stress collection becomes **development data**. Its repeats
are assigned 70% training / 15% validation / 15% development holdout within each
condition and combined with the original 160-trial collection. Complete trials
remain together and identifiers are prefixed by collection. The combined
development splits are **168 training / 36 validation / 36 holdout**.
Each target still has a separate Ridge model and validation-selected alpha;
there is no multi-output regressor, clipping or endpoint rounding inside the
model. Both feature extraction and its saved mode are reused during inference.

Because those stress failures informed the revised features, that collection
is no longer presented as an independent test. A **new 80-trial blind collection**
uses seed 2026101903 and the broader ±10% gain/drive, ±20% contact coefficients,
and 30-µV noise settings. It has disjoint trial seeds and is evaluated only
after model selection is complete. A saved hash verifies that the model file
does not change during blind evaluation. Development holdout metrics are
reported separately and do not replace this blind test.

After generating the training and stress collections above, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_segment1_compliant_blind.json \
  --output outputs/dataset_segment1_compliant_blind --split test

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.segment1_validation \
  --training-data outputs/dataset_segment1_compliant \
  --development-data outputs/dataset_segment1_compliant_stress \
  --blind-test outputs/dataset_segment1_compliant_blind \
  --output outputs/ml_segment1_compliant/robust
```

The final report, blind predictions, loss/inference plots and two-model bundle
are in `outputs/ml_segment1_compliant/robust/`. Use this model path for the
`predict` command above. The generic extraction CLI also accepts
`--spectral-mode band_peaks`; the default `global` feature mode remains compatible
with earlier models. Real ADC calibration and preservation apply to both modes.
This robustness check still uses the same assumed preload laws; validating
unknown hardware stiffness/damping, gain, noise and electrical transfer remains
a separate measurement task.

### Locked-model blind results

The two independent models selected alpha **0.01 for location** and **0.1 for
force**. Both use 27 voltage/timing features. The new blind collection contains
20 trials per condition and was not used for feature choice, scaling, fitting
or alpha selection. All four force/location conditions were selected correctly.

| Evaluation | Location RMSE | Force RMSE | Correct location labels | Correct force labels |
| --- | --- | --- | --- | --- |
| Development holdout (exposed collection) | 0.086 mm | 0.430 N | 36/36 | 36/36 |
| New independent blind collection | **2.009 mm** | **0.654 N** | **80/80** | **80/80** |

Blind R² is 0.9955 for location and 0.8932 for force. For each 80/80 accuracy,
the 95% Wilson interval is about 95.4%–100%; it is not proof of universally
perfect prediction. The continuous estimates have **not** been clipped or
snapped to make the RMSE smaller: some force estimates are negative and one
location estimate overshoots to about 122 mm. These are limitations of an
unconstrained Ridge model. The known-condition decision uses the nearest
allowed values, with midpoints 3 N and 75 mm, and is recorded separately in
`blind_test.predictions.discrete.csv`. Continuous regression errors remain
visible in the CSV, metrics and plots.

Artifacts in `outputs/ml_segment1_compliant/robust/`:

- `report.json`: disjoint-seed audit, frozen model hash, per-condition metrics,
  intervals, original development disclosure and same-seed physics interventions.
- `model/model_bundle.pkl`: separate location and force Ridge pipelines.
- `model/training_loss.png/.svg`: training/validation MSE by alpha.
- `blind_test.predictions.png/.svg`: unmodified continuous estimates.
- `validation_summary.png/.svg`: RMSE and known-condition selection accuracy.
- `force_sensitivity.png/.svg`: identical 1/5-N traces under hard or
  force-independent compliant contact; differing traces under force-dependent
  compliance, with drive, noise, gain and tendon tension matched.

All 44 relevant solver, collection, ADC, feature, CLI and model tests pass.
