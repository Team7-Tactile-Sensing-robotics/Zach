# TASS machine learning framework

This folder compares **linear and logistic models independently for both contact
force and contact location**. No predicted force is fed to the location model,
and there is no contact-detection stage gating either model.

The [time-feature extension](TIME_FEATURES.md) adds received-wave intensity,
within-window energy timing, optional triggered packet peaks, and a matched
OLS/RBF-SVR comparison. Run `python -m machine_learning compare-demo --signal-mode steady --output machine_learning/outputs/time_demo`
to compare the original features with the additional time descriptors.

| Target | Linear model | Logistic model |
| --- | --- | --- |
| Contact force (N) | Ordinary least squares, continuous force | Multinomial logistic regression, force ranges |
| Contact location (m) | Ordinary least squares, continuous location | Multinomial logistic regression, location regions |

Logistic regression is a classifier, not a continuous regression estimator.
For direct numerical comparison, its class probabilities are multiplied by
**training-set mean target values within each bin** and summed. This produces
an approximate continuous prediction, bounded by the training-bin means.
Both families receive identical eligible rows and the same train/test groups.
OLS predictions are not clipped to physical limits.

## Run from the repository root

```bash
python -m pip install -r machine_learning/requirements.txt

# Complete runnable example; creates TOY signals, FFT features, and comparisons.
python -m machine_learning demo --output machine_learning/outputs/demo

# Process your synchronized, labeled measurements and run the complete comparison.
python -m machine_learning run --input path/to/samples.csv --output machine_learning/outputs/hardware_run_01

# Or run the two stages independently.
python -m machine_learning extract --input path/to/samples.csv --output machine_learning/outputs/features_01
python -m machine_learning train --input machine_learning/outputs/features_01/features.csv --output machine_learning/outputs/models_01

python -m unittest discover -s machine_learning/tests -v
```

Choose a **new output directory** for every run. To change settings, edit
[`config.json`](config.json), or pass `--config path/to/overrides.json` to any
extract/train/run/demo command. Override files can contain only changed keys.
No Streamlit app or hardware acquisition code is modified by this framework.

## Recording format

Copy [`templates/samples.csv`](templates/samples.csv). One row is one synchronous
piezo sample, on a common clock. The table includes every requested measurement:

| Your measurement | CSV field | Role |
| --- | --- | --- |
| Time | `time_s` | Sample timing, not a predictor |
| Contact force | `contact_force_N` | Independently measured target |
| Contact location | `contact_location_m` | Independently measured target |
| Four sensor voltages | `piezo_sensor_1_V` … `piezo_sensor_4_V` | Receiver FFT inputs |
| Four actuator voltages | `piezo_actuator_1_V` … `piezo_actuator_4_V` | Drive FFT inputs |
| Servo angle | `servo_angle_rad` | Actual measured state predictor |
| Servo torque | `servo_torque_Nm` | Actual measured/calibrated state predictor |
| Load-cell reading | `load_cell_N` | Tendon tension predictor; not contact-force ground truth |
| Current excitation frequency | `excitation_frequency_Hz` | Recorded excitation setting/state |

Also include `session_id`, globally unique `trial_id`, `data_source` (`hardware`
or `synthetic`), and optionally `quality_flag` (`ok`, `unassessed`, `clipped`,
`dropped`, `suspect`). Independent acquisition sessions should have different
session IDs. Each approach/contact/release recording has its own trial ID.
IDs keep related FFT windows together during validation; they are not predictors.

**Units and reference:** convert degrees to radians, mm to metres and ADC counts
to calibrated volts before ingestion. For this tendon study, location is arc
length from the spool feed point along the current tendon. If collecting finger
surface contacts instead, define one consistent surface coordinate in your
protocol; do not pool segment-local or tendon coordinates without conversion.
Leave location blank when there is no contact or the location is unknown.
Preserve measured zero-force noise; do not fill unknown force/torque with zero.
Use zero actuator voltage only when the channel is actually off/known zero.

All numeric columns must be present. Whole-window missing labels remain missing,
and that window is excluded only from the corresponding target's analysis.
Waveform channels need complete finite samples. State columns can be blank if
unavailable: predictors wholly missing in development are excluded and reported;
partially missing predictors cause complete-case exclusions for consistent
comparison across candidates. Use `feature_columns` to deliberately select a
smaller predictor set when necessary. Missing values are never imputed from labels.

Human-readable headers can be renamed using an explicit `column_map`, e.g.
`{"Time": "time_s", "Contact Force": "contact_force_N"}`. This only renames
columns; it does not convert units or supply IDs.

The earlier [`data_collection`](../data_collection/README.md) tables remain the
raw acquisition format. This wide table is a **derived synchronized export**:
align slower force/state measurements to the piezo clock with a documented
validity interval and maximum gap. Do not silently hold stale labels across
contact transitions. The old single-drive waveform schema does not contain
all four actuator recordings, so it cannot supply them automatically.

## FFT feature extraction

Defaults: 1,024 samples per window, 1,024-sample hop (no overlap). Sample rate
comes from the median timestamp spacing per trial; frequency resolution is
`sample_rate / window_samples`. Uniform spacing is checked in every window.
Timestamps must remain in acquisition order. Windows never cross trial boundaries.

For **each of the eight voltage channels**, the extractor removes the mean,
applies a periodic Hann window and computes a real FFT. It produces:

- AC RMS voltage from the unwindowed, mean-removed signal.
- Dominant non-DC frequency and coherent-gain-corrected peak amplitude.
- Spectral centroid and power fractions in three configurable frequency bands.

The original baseline has 56 channel features plus four state means: **60
predictors**. Generic time features now add 28 receiver descriptors by default,
giving **88 candidate predictors**. Disable `time_features_enabled` to restore
the original feature set. Triggered echo features are off by default. See
[exact definitions and comparison commands](TIME_FEATURES.md).
Peak frequency uses maximum one-sided spectral power. Endpoint
bins are not doubled. Band edges must fit below Nyquist; the configured bands
are [0,250), [250,1000), [1000,4000) Hz, not universal hardware bandwidths.
Change them to match your DAQ and excitation. No phase, FRF, or causal
transmitter-to-receiver transfer estimate is claimed for simultaneous drives.

Force, location, angle, torque, tension and excitation frequency are summarized
as window means, rather than FFT-transformed into model inputs. Force/location
are kept completely separate from the predictors. Windows with invalid quality,
missing voltages, timestamp gaps/jitter, or excessive force/location/frequency
variation are rejected with reasons in `window_audit.csv`. Current stationarity
limits are demonstration settings; configure them for your experiment. Partial
recording tails are also reported. Chirps and fast contact transitions need
appropriate windows/limits or a different time-frequency feature design.

## Parallel model comparisons and standardization

The original FFT/state search generates **237 specifications** and **948
target/family attempts**. With generic time features enabled, the default search
has **265 specifications** and **1,060 attempts**, plus development CV fits.
Available feature columns and config changes can change these counts.
The search includes:

- Every individual FFT/state feature.
- Every nonempty subset of the four receiver RMS channels.
- Receiver-only, receiver + actuator, receiver + mechanical state, and full FFT sets.
- All two- and three-variable subsets of `core_features`, with and without
  pairwise products. Every interaction retains both parent main effects.

The defaults intentionally bound the search; it is not all 2^60 subsets.
`max_subset_size`, `core_features` and `max_experiments` control expansion.
The manifest reports all specifications and any cap. Three-variable models
include pairwise products, not three-way products or squared terms.

Base predictors are centered and scaled to population standard deviation one
**using the training fold only**. Pair products are formed from those standardized
parents, then every resulting term is again standardized using that training fold.
These transformations are reused unchanged on validation/holdout/prediction data.
Targets stay in N and m. This follows the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

By default logistic class boundaries use training-only thirds. Repeated target
values can reduce the number of nonempty bins; unsuitable/empty bins are reported
instead of fabricating classes. For physically meaningful comparisons, configure
fixed internal boundaries in `target_bin_edges`. For example (illustrative only):

```json
{
  "target_bin_edges": {
    "contact_force_N": [1.0, 3.0],
    "contact_location_m": [0.12, 0.24]
  }
}
```

This creates three ranges per target, with boundary values entering the upper
range. Both families use identical bins for classification metrics. With
training-quantile bins, boundaries differ across CV folds; final boundaries and
training-bin means are recorded in `run_summary.json` and the saved models.

## Validation and statistical significance

25% of **whole sessions** form a held-out test set before any model fitting.
The remaining sessions undergo three-fold `GroupKFold` development validation.
You can group by trial instead, but related trials from one session can then
cross folds, so session-level generalization is the safer default. Overlapping
windows remain in the same group. This evaluates new recording groups, not
future-time forecasting. Synthetic and hardware data cannot share a run.

Each family/target selects its own specification by mean CV **MAE**. Only that
winner is evaluated on the holdout. Reports give MAE, RMSE and R², class accuracy
and macro F1, plus logistic log loss. A training-mean/majority-class baseline is
evaluated on the same test rows. Do not repeatedly tune settings against the
same holdout; use a new independent final collection after development.

Unregularized OLS and [multinomial logistic maximum likelihood](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.fit.html)
provide the inference tables. For development-set fits, they include:

- Standardized coefficients, cluster-robust standard errors, raw p-values and
  pointwise confidence intervals; logistic odds ratios for each class vs class 0.
- Clustering at the configured recording-group level to account for dependence
  among windows; OLS uses a G−1 t reference, logistic uses asymptotic Wald tests.
- [BH and BY multiple-testing corrections](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
  across **all valid non-intercept coefficient tests in the entire run**, including
  both targets, families, specifications and class contrasts. The default
  significance flag uses the more conservative BY q-value < 0.05.
- Explicit failures for constant/collinear predictors, insufficient observations,
  poor conditioning, logistic nonconvergence and indicators of separation.

Inference is suppressed unless there are at least `min_inference_groups` (20 by
default) **and more independent groups than estimated parameters**, with finite
positive standard errors. These are safeguards, not a guarantee of calibrated
p-values. A million samples from a few recordings are still only a few groups.
The row-per-parameter threshold and fit diagnostics are also recorded.

**The significance sweep is exploratory.** Reusing data across many models and
choosing specifications after inspecting them limits interpretation, even with
q-value correction. Small p-values are conditional associations, not causal
importance or proof of generalization. Correlated variables can exchange apparent
significance; interactions change main-effect interpretation. The variable
summary's minimum p/q and number of significant appearances are not additional
independent tests. Confirm selected hypotheses with a prespecified model on
new experiments; the prediction holdout is not used to generate p-values.

Including servo torque or tendon tension assumes those measurements will be
available at deployment. Sensor-only specifications allow comparison with that
assumption removed. A force sensor used to create the contact-force target must
not also be relabeled as a predictor. Unbalanced protocols (e.g. each location
always assigned its own excitation frequency) can create shortcuts; vary force,
location, motor state and excitation independently across repeated sessions.

## Outputs and saved-model prediction

| File | Contents |
| --- | --- |
| `features.csv`, `window_audit.csv` | One row per FFT window; acceptance/rejection reasons |
| `results/specifications.json`, `config_used.json` | Exact experiment definitions and settings |
| `results/split_manifest.csv`, `cv_manifest.csv` | Traceable group/window assignments |
| `results/experiments.csv` | All candidate CV scores, fit diagnostics and failure reasons |
| `results/coefficients.csv` | Coefficients, confidence intervals, raw p-values and adjusted q-values |
| `results/variable_significance.csv` | Exploratory term-by-term significance summary |
| `results/holdout_metrics.csv`, `holdout_predictions.csv` | Final parallel comparison and baseline |
| `results/*_linear.joblib`, `*_logistic.joblib` | Winners, fitted scalers, bin boundaries and bin means |
| `results/run_summary.json`, `REPORT.md` | Label exclusions, selected models and readable results |

```bash
python -m machine_learning predict \
  --model machine_learning/outputs/hardware_run_01/results/contact_force_N_linear.joblib \
  --input path/to/new_features.csv \
  --output machine_learning/outputs/force_predictions.csv
```

Prediction requires complete finite features and the same units/FFT configuration.
Load only your own trusted joblib files. Outputs/data folders are ignored by Git;
the source templates and tests are versioned. The demo uses artificial statistical
signals to exercise the workflow, **not measured data or the calibrated digital
finger model**. Its p-values and error scores are not evidence about the prototype.
