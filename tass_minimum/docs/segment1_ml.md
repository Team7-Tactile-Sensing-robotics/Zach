# TASS segment-1 training

For the optional force-dependent contact experiment and independent-seed
validation of **two separate Ridge regressors**, see
[segment-1 force/location validation](segment1_force_validation.md).
The historical hard-clamp experiment below remains the negative control.

Use the existing `tass.dataset` simulator and `config/default.json` geometry,
spring, tendon, held-pose force balance and string response. There is no separate
bench generator. The full TASS finger remains modeled; only actuators 1/2 drive
it, and only actuator/sensor pairs 1/2 are exported and used for training.

The focused protocol is `config/experiments/dataset_segment1_30_70.json`:

| Setting | Value |
| --- | --- |
| Location reference | `segment1_from_base` |
| Locations | 30%, 70% of segment 1: 45 mm and 105 mm from the finger base |
| Forces | 1 N and 5 N, without label jitter |
| Active actuators / saved pairs | `[1, 2]` / `[1, 2]` |
| Trials | 4 conditions x 20 repeats = 80 |
| Split | 56 training / 12 validation / 12 test, by complete trial |
| Recording | 5 s at requested 75,000 samples/s |
| Excitation | Simultaneous 20..2000 Hz chirps for 0.3 s, nominal peak 3.3 V |
| Simulation input to ML | Saved sensor voltage directly; simulated ADC observation disabled |

The nominal actuator force-per-volt and sensor gains remain uncalibrated TASS
assumptions. One transmitter-off capture had a mean of 1887 counts and range
1729..2065, but this is not a quiet-baseline calibration or a fitted noise
distribution. DC removal subtracts each analysis window's mean rather than 1887.

`recorded_pair_ids` selects exported one-dimensional channels and the columns
of compact matrices such as `piezo_V` and `actuator_force_N`. Matrix column order
matches `recorded_pair_ids`, saved in protocol, trial metadata and summary.
Every active actuator must be recorded. Legacy protocols still export all four
pairs unless this setting is supplied. Other piezo locations are retained in
the forward model but are passive and are not saved in this focused collection.

## Run generation and training

From `Zach/` with the virtual environment active, use a new empty dataset folder:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_segment1_30_70.json \
  --output outputs/dataset_segment1_30_70_1_5

python -m tass.ml_cli extract \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_voltage/features \
  --sensor-id 1 --sensor-id 2 --sensor-representation voltage \
  --remove-sensor-mean \
  --window-s 5 --stride-s 5 \
  --target contact_location_m --target contact_force_N

python -m tass.ml_cli train \
  --features outputs/ml_segment1_voltage/features \
  --output outputs/ml_segment1_voltage/model

python -m tass.ml_cli evaluate \
  --model outputs/ml_segment1_voltage/model/model_bundle.pkl \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_voltage/model/simulation_evaluation.json
```

Extraction streams trials one at a time. In the default `voltage` mode,
simulation reads `piezo_sensor_N_V` directly, removes the DC mean and computes
voltage statistics, spectra and band powers. It does not quantize or decode
simulated ADC, even if old datasets contain ADC channels. New datasets disable
ADC observation; existing datasets can be reused without regeneration.
Model inputs are sensor-derived
features and sample rate; contact labels, tension, servo states and trial IDs
are excluded. `train` fits separate Linear/Ridge regressors per target, choosing
by validation RMSE. The held-out report includes MAE, RMSE, R² and a constant
training-mean baseline. Labels are continuous; regression outputs are not
restricted to the two allowed force/location values.

Each target is fitted independently with its own scaler, fitted coefficients
and validation-selected regularization. `model_bundle.pkl` is one container
holding two separate single-output models, not one joint multi-output regressor.
The default `--regressor auto` compares ordinary Linear and Ridge candidates
for each target independently. To require **two Ridge linear regressors**, use:

```bash
python -m tass.ml_cli train \
  --features outputs/ml_segment1_physics/features \
  --output outputs/ml_segment1_ridge_physics/model \
  --task regression --regressor ridge

python -m tass.ml_cli evaluate \
  --model outputs/ml_segment1_ridge_physics/model/model_bundle.pkl \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_ridge_physics/model/simulation_evaluation.json
```

This example reuses the analytical features extracted in the section below.
Use `outputs/ml_segment1_voltage/features` instead for sensor-only training.
Ridge alpha is selected separately for location and force from 0.01, 0.1, 1,
10 and 100 using validation RMSE. `--regressor linear` instead requires two
ordinary least-squares regressors. Both choices apply only to regression;
inference automatically loads both independent models and their preprocessing.

`train` saves `training_loss.png` and `.svg`: training/validation MSE against
model candidate, with the selected model marked. Linear/Ridge use direct
solvers, so this is a loss comparison rather than an epoch history.
`evaluate` saves `simulation_evaluation.predictions.csv`, `.png`, and `.svg`:
per-trial actual/predicted labels and parity plots with RMSE. Location is
displayed in mm; force is displayed in N. Unlabeled `predict` runs graph only
the predictions and do not invent true labels.

`plot_trial.py 0 --dataset outputs/dataset_segment1_30_70_1_5` works with this
two-pair export and still draws the recorded full TASS pose. Use `--start-s 0
--end-s 0.6` to inspect excitation and early decay.

## Physical limit on force prediction

In the current TASS model, contact on the fixed proximal segment has zero joint
lever arm. Consequently, 1 N and 5 N give the same tendon tension at a fixed
position. Contact is a hard clamp whose acoustic effect depends on position,
not force magnitude. Holding random seed and location fixed produces identical
sensor waveforms for 1 N and 5 N. The dataset records
`contact_force_identifiable: false` to make this limitation visible.

Both targets are trained as requested, but use force results as a negative
control for the pipeline. For balanced 1/5-N data, the constant 3-N predictor has
RMSE 2 N. Location can be learned; meaningful force inference needs validated
force-dependent contact physics or independently labeled real force data.
The generator does not add artificial force dependence to manufacture accuracy.

## Try discrete classification before changing contact mechanics

For this experiment, both targets have only two exact values: location 45/105 mm
and force 1/5 N. Use `train --task classification` on the same extracted features
to train separate classifiers for both targets. There is no need to regenerate
data or re-extract features. Regression remains the default.

```bash
python -m tass.ml_cli train \
  --features outputs/ml_segment1_voltage/features \
  --output outputs/ml_segment1_classifier_voltage/model \
  --task classification

python -m tass.ml_cli evaluate \
  --model outputs/ml_segment1_classifier_voltage/model/model_bundle.pkl \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_classifier_voltage/model/simulation_evaluation.json
```

The candidates are standardized logistic regression with C = 0.01, 0.1, 1, 10,
100 and random forests with 200 trees, depth 2 or unrestricted depth, and minimum
leaf size 2. Random seed is 42. Scaling and fitting use only training trials;
selection uses validation **log loss**. The original 56/12/12 complete-trial
splits are retained. Classification learns the exact numeric class values from
training labels, maps them to integer class indices internally and restores
physical values in predictions. Use discrete labels without jitter; unseen
evaluation classes are rejected instead of rounded to a known class.

`metrics.json` and evaluation report accuracy, balanced accuracy, macro F1,
log loss and confusion matrices, with training-class-prior baselines. Confusion
matrix rows are actual classes and columns are predicted classes, in the saved
`classes` order. `training_loss.png/.svg` compare training/validation log loss
across candidates; these are not epoch curves. Inference graphs show actual and
predicted classes plus confusion matrices, or class probabilities for unlabeled
captures. Classification can only choose a trained class; it cannot estimate an
intermediate force or demonstrate continuous localization between the two sites.

For the current unchanged 80-trial simulation dataset:

| Inputs | Selected force classifier | Location accuracy | Force accuracy | Regressor rounded to 1/5 N |
| --- | --- | --- | --- | --- |
| Sensor voltage | Logistic, C=100 | 12/12 (100%) | 8/12 (66.7%) | 8/12 (66.7%) |
| Sensor + reference changes | Logistic, C=100 | 12/12 (100%) | 8/12 (66.7%) | 8/12 (66.7%) |
| Sensor + reference changes + analytical terms | Logistic, C=100 | 12/12 (100%) | 8/12 (66.7%) | 8/12 (66.7%) |
| Training-prior baseline | — | 6/12 (50%) | 6/12 (50%) | — |

The force confusion matrix is `[[3,3],[1,5]]`, ordered `[1 N, 5 N]`, for all
three selected classifiers. Force log loss is about 0.683, close to the
uninformative 50/50 baseline of 0.693. Classification provides no accuracy gain
over the rounded regressor. Eight successes out of twelve do not establish a
force signal: the 95% Wilson interval is approximately 39%–86%, and the simulator
is exactly invariant to force at a fixed location and random seed. Repeat-to-repeat
noise/gain variations can create accidental associations in this small sample.
The held-out comparison is saved in `outputs/ml_segment1_classifier/comparison.json`
and `comparison.png/.svg`; each classifier has its own loss and inference plots.
Existing regressors and datasets are retained.

To repeat the reference/analytical controls, replace `voltage` with `reference`
or `physics` in the feature and classifier output paths above. The saved model
automatically selects classification and recomputes its saved features during
inference; do not repeat `--task` on `predict` or `evaluate`.

```bash
python -m tass.ml_cli predict \
  --model outputs/ml_segment1_classifier_voltage/model/model_bundle.pkl \
  --input captures/segment1_capture.csv \
  --adc-full-scale-v 3.3 --adc-resolution-bits 12 \
  --output outputs/ml_segment1_classifier_voltage/real_predictions.csv
```

Prediction preserves the original recording and writes physical class values
plus `probability_<target>_class_0`, `..._class_1`, etc. The preprocessing JSON
sidecar stores `class_values`, which maps these probability columns to physical
values, along with effective calibration and feature settings. Analytical
classifiers still require the matched reference signals described below.

### Next mechanics experiment

Location moves a hard acoustic boundary and changes resonances, whereas changing
proximal force does not change that boundary or tendon tension in this model.
Switching estimator families cannot create the missing force dependence.
The next useful simulator change is a compliant contact at the existing contact
node, with incremental reaction `-k_contact(F)*u - c_contact(F)*du/dt` around its
preloaded equilibrium, replacing the fixed-node constraint. Stiffness and damping
must have units N/m and N·s/m and should be fitted against measured preload,
frequency response and decay at 1 N and 5 N, with positive coefficients. This
could make transmission, resonances and decay force-dependent if the hardware
has that behavior. Indentation-induced geometry/tension change is a further
option if measured; force on segment 1 does not gain a distal joint lever arm.
No contact mechanics were changed for the classification comparison.

## Enable analytical features in training and inference

Use `--reference-features --physics-features` to add 15 analytical terms to the
sensor and reference-change features. The saved model recomputes all of these
at inference automatically; no physics flags need to be repeated then.
Simulation still uses sensor/reference voltage directly and bypasses ADC.

For the current saved dataset, the full acoustic length is 0.22302775637731992 m
(spool-to-P4 routed path plus 25-mm extension), not the 0.15-m first-segment
length. Linear density is 0.002 kg/m. These values are taken from the saved
`model_config.json` and held pose, not from contact-force/tension labels.

```bash
python -m tass.ml_cli extract \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_physics/features \
  --sensor-id 1 --sensor-id 2 --sensor-representation voltage \
  --remove-sensor-mean --window-s 5 --stride-s 5 \
  --target contact_location_m --target contact_force_N \
  --reference-features --physics-features --physics-sensor-pair 1 2 \
  --string-length-m 0.22302775637731992 \
  --linear-density-kg-m 0.002 --mode-number 1

python -m tass.ml_cli train \
  --features outputs/ml_segment1_physics/features \
  --output outputs/ml_segment1_physics/model

python -m tass.ml_cli evaluate \
  --model outputs/ml_segment1_physics/model/model_bundle.pkl \
  --input outputs/dataset_segment1_30_70_1_5 \
  --output outputs/ml_segment1_physics/model/simulation_evaluation.json
```

The 42 inputs comprise 21 original sensor/timing features, 6 reference-change
features, and 15 analytical features. These include acoustic contact-coordinate
estimate `x_physics_m`, reference/contact tension estimates, estimated tension
change, a heuristic confidence score and confidence-weighted terms. For matching
mode frequencies, the estimator uses `x = L*f_right/(f_left+f_right)` and
`T_hat = mu*(2*x*f_left/n)^2`. These are acoustic estimates derived from the
waveforms; simulated true tension and contact force are not input features.
The ML location target remains distance along the physical finger from its base.

For a reference-only control, use the same extraction command with
`--reference-features` and omit `--physics-features`, writing to
`outputs/ml_segment1_reference/features`. Train/evaluate in a separate reference
model folder. Keep identical trial splits to compare with the sensor-only model.
This separates any effect of a reference waveform from the analytical terms.

For the current 80-trial dataset, the matched 12-trial held-out comparison is:

| Inputs | Feature count | Force RMSE [N] |
| --- | --- | --- |
| Sensor only | 21 | 1.9783 |
| Sensor + reference changes | 27 | 1.9772 |
| Sensor + reference changes + analytical terms | 42 | 1.9766 |
| Constant 3-N prediction | — | 2.0000 |

The approximately 0.0017-N change from sensor-only is negligible. All three
models distinguish the two location labels, but the force result is still
essentially the mean baseline. Reports/plots are saved in
`outputs/ml_segment1_physics/comparison.json`, `comparison.png`, and
`comparison.svg`. The analytical model's own loss and inference graphs are in
`outputs/ml_segment1_physics/model/`. Historical sensor-only models are retained.

### Assumption audit for the current segment-1 dataset

The equations require sensors on opposite sides of the **projected tendon
clamp**, with peaks representing the same identified mode. At the 30% physical
contact, nearest-route projection chooses the slanted spool-P1 section: the
clamp snaps to acoustic coordinate 0.008364 m, while P1/P2 are at 0.018028 and
0.098028 m. Both sensors are on the same side, so the two-span equations do not
apply to those 40 trials. The 70% clamp is at 0.083635 m and is bracketed by
P1/P2. Mode identity is not verified by simply selecting the dominant FFT peak.
`physics_confidence` is a heuristic, not proof that these assumptions hold.

The geometry audit is recorded separately at
`outputs/ml_segment1_physics/physics_assumptions.json`. It uses ground-truth
geometry for diagnostics only; it is not fed to the model or used as a
label-dependent confidence gate. The comparison leaves the simulator unchanged.

The simulation reference is a no-clamp counterfactual at matched tension and
excitation. It is not a simulation of contact loading from a measured pre-contact
state. Analytical tension change is therefore not a calibrated contact force.
Even valid analytical features cannot recover a force dependence absent from
the forward model's proximal-segment response.

### Real inference with the analytical model

The model requires matched reference signals for both selected receivers in
addition to the contacted signals. Use
`data_collection/templates/segment1_waveforms_reference.csv`. For ADC recordings,
the columns are `piezo_sensor_1_raw_adc`, `piezo_sensor_2_raw_adc`,
`piezo_sensor_1_reference_raw_adc`, and `piezo_sensor_2_reference_raw_adc`.
Align references to the same excitation timeline and acquisition settings.
For already calibrated voltage input, use the corresponding `_voltage_v`
and `_reference_voltage_v` names with no ADC columns.

```bash
python -m tass.ml_cli predict \
  --model outputs/ml_segment1_physics/model/model_bundle.pkl \
  --input captures/segment1_with_reference.csv \
  --adc-full-scale-v 3.3 --adc-resolution-bits 12 \
  --output outputs/ml_segment1_physics/real_predictions.csv
```

Replace nominal calibration with measured slope/offset when available. Both
contacted and reference counts are converted before the shared DC removal and
feature extraction, and original recordings remain unchanged. Missing reference
columns are rejected rather than replaced by zeros. Prediction's preprocessing
sidecar records that physics was enabled and the constants used.

## Real capture representation and inference

Keep one row per synchronized sample, using either template:

- `data_collection/templates/segment1_waveforms.csv`: actual `time_s` plus raw
  sensor counts and optional synchronized actuator commands.
- `data_collection/templates/segment1_waveforms_indexed.csv`: `sample_index`
  plus a per-segment `measured_sample_rate_hz`; elapsed time is reconstructed
  from indices divided by this measured rate.

The dashboard requests 75 kHz, but the user reported a run near 74,970 Hz. Use
each capture's measured clock, not the requested value, for FFT frequency axes.
To match simulation timing to a particular run, copy the protocol and change
`sample_rate_Hz` to that run's measured rate before generation. The importer
checks supplied clock metadata against timestamp spacing and rejects irregular
sampling until explicitly resampled. A total recording with dropped samples
may be shorter than the configured five-second ML window.

The two data paths meet at the voltage waveform:

```text
simulation: saved sensor voltage ------------------------+
                                                        +-> shared DC removal -> features -> model
real: original ADC counts -> configured affine voltage --+
```

The default `voltage` mode always selects simulation voltage (it errors if
missing), and selects raw ADC for a real receiver when counts are present.
Already converted real voltage can be supplied when raw counts are absent.
Real raw counts require an explicit positive calibration slope, supplied at
extraction and saved in the bundle or supplied to `predict`/`evaluate`.
The real conversion is:

```python
adc_pin_voltage_v = raw_adc * volts_per_count + voltage_offset_v
ac_voltage_v = adc_pin_voltage_v - adc_pin_voltage_v.mean()
```

For nominal 12-bit 0..3.3 V decoding, use `--adc-full-scale-v 3.3
--adc-resolution-bits 12` (slope `3.3/4095`); this is an assumption, not a measured
calibration. For measured affine calibration, supply `--adc-volts-per-count SLOPE
--adc-voltage-offset-v OFFSET`. The same flags work on `extract`, `predict`, and
`evaluate`. Real calibration is never applied to simulation voltage and does
not require a simulated ADC block. The converted voltage must represent the
same sensor/front-end point as the simulated signal. See [ADC calibration](esp32_adc.md).

```bash
python -m tass.ml_cli predict \
  --model outputs/ml_segment1_voltage/model/model_bundle.pkl \
  --input captures/segment1_capture.csv \
  --adc-full-scale-v 3.3 --adc-resolution-bits 12 \
  --output outputs/ml_segment1_voltage/real_predictions.csv
```

Prediction reuses selected sensors, window duration, DC removal and feature
order. Calibration supplied at inference overrides only slope/offset for that
run; it does not edit model weights or the saved bundle. If calibration was
saved at extraction, the CLI flags can be omitted to reuse it. A
`*.preprocessing.json` sidecar records effective configuration and feature
provenance. Input recordings are read only, and output paths that would
overwrite a capture are rejected. Optional diagnostic voltage columns cannot
override real raw counts in this mode. The real ADC files are preserved.
For labeled, held-out real trials, replace `predict` with `evaluate` and use
a `.json` output; simulation test results do not establish hardware accuracy.

`auto` (prefers counts), `raw_adc`, and `adc_voltage` are legacy explicit
ADC-study modes. New training uses `voltage`; re-extract and retrain into a new
model directory to replace an older ADC-trained bundle. Historical model files
and datasets are not overwritten by this migration.

Actuator commands are retained as normalized waveforms derived from the
specified drive amplitude. Command 0 is off, and ±1 corresponds to the selected
chirp's peak voltage (±3.3 V for this protocol). These are command-derived, not
measured drive voltages. `--actuator-features --actuator-drive-peak-v 3.3` can add
their features, but the basic model above is sensor-only and commands must be
available with the same encoding at inference if this option is enabled.
