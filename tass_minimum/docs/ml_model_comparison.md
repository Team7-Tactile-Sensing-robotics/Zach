# Ridge, classifiers and feature-vector MLP comparison

The same 27 voltage/timing features are used for every model. Each approach
trains **two independent single-output models**, one for location and one for
force. The MLP consumes a flat 27-value feature vector per trial; it is not a
raw-waveform model or a Conv1D network. Classification is a task, while MLP is
an architecture, so both MLP regression and MLP classification are included.

The simulator, compliant contact laws, actuator/sensor pairs 1/2, 0.3-s chirp,
5-s recordings and first-1.5-s feature window remain unchanged. The four
conditions are 1/5 N at 45/105 mm (30%/70% of the 150-mm first segment).

## Matched evaluation

All new models reuse `outputs/ml_segment1_compliant/robust/features`:
168 training trials, 36 validation trials and 36 development holdout trials.
Scalers fit only training rows. Architecture, alpha, regularization and MLP
checkpoint selection use validation rows, not the benchmark. Comparison checks
identical feature order, feature configuration, complete-trial splits,
benchmark trial IDs and target labels. Existing Ridge artifacts are preserved.

The benchmark is the same 80-trial collection previously evaluated with Ridge,
`outputs/dataset_segment1_compliant_blind`. This is a **matched comparison on an
existing benchmark**, not a newly collected blind test. It has 20 trials per
condition and the broader ±10% drive/gain, ±20% contact-coefficient variation,
and 30-µV noise settings. There is no new simulator tuning or data generation
for this comparison. Reports explicitly record that this benchmark was exposed
in the earlier experiment.

| Model pair | Location RMSE | Force RMSE | Location selection | Force selection |
| --- | --- | --- | --- | --- |
| Existing Ridge regressors | 2.009 mm | 0.654 N | 80/80 | 80/80 |
| Logistic classifiers | — | — | 80/80 | 80/80 |
| MLP regressors | **1.481 mm** | **0.356 N** | 80/80 | 80/80 |
| MLP classifiers | — | — | 80/80 | 80/80 |

Regressors are evaluated using their unmodified continuous predictions and
separately by the nearest allowed target. Classifiers choose a known endpoint.
All classifiers make zero endpoint mistakes here, but zero hard-class error
does not establish continuous force estimation or localization between sites.
The comparison graph therefore shows continuous RMSE only for the regressors.

The MLP regressor reduces force RMSE by about 45.5% and location RMSE by 26.3%
on this benchmark. Class-selection accuracy is tied across all four pairs.
Classification log losses are:

| Classifier | Location log loss | Force log loss |
| --- | --- | --- |
| Logistic | 0.000215 | 0.009120 |
| MLP | 0.0000227 | 0.000760 |

Lower log loss favors the MLP on these samples, but this is not a separate
probability-calibration study. The small benchmark and one fixed neural-network
initialization seed do not establish universal superiority.

For continuous numeric estimates, the MLP regressor is better on this matched
benchmark. For selecting only the two known forces and two sites, logistic
classification is a simpler baseline with the same observed accuracy.
All results remain conditional on the assumed, uncalibrated preload-dependent
contact law. They do not establish real-hardware accuracy.

## Architecture and optimization

The classical classifier compares logistic C = 0.01, 0.1, 1, 10, 100 and random
forests with depth 2/unrestricted, selecting by validation log loss. Logistic
C=100 is selected independently for both targets on the current features.

Each MLP target compares ReLU hidden layers `(16,)` and `(32,16)` with L2 alpha
0.0001 or 0.01. Adam uses learning rate 0.001, batch size 32 (or training count
if smaller), initialization seed 42, a maximum of 600 epochs and validation
patience 60. The code uses the existing scikit-learn dependency; no PyTorch or
TensorFlow installation is required. See the primary documentation for
[`MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
and [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

Regression targets are standardized on training rows with
[`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
and converted back to metres/newtons during prediction. Input scalers are also
trained on training rows only. Classification internally encodes the exact
numeric classes and restores physical values in its outputs.

Every `partial_fit` performs another epoch. Training/validation prediction MSE
(regression) or log loss (classification) is recorded each epoch, excluding the
regularization penalty. The best validation checkpoint is saved; the last
epoch is not automatically selected. Candidate choice uses validation RMSE
for regression or validation log loss for classification. The existing holdout
and benchmark never fit weights or select checkpoints.

| Network | Selected hidden layers | L2 alpha | Saved epoch | Epochs run |
| --- | --- | --- | --- | --- |
| MLP location regressor | 16 | 0.01 | 600 | 600 |
| MLP force regressor | 32, 16 | 0.01 | 508 | 568 |
| MLP location classifier | 32, 16 | 0.0001 | 600 | 600 |
| MLP force classifier | 32, 16 | 0.0001 | 600 | 600 |

Reaching epoch 600 is the configured cap, not a claim that optimization has
fully converged. The force regression model restores the epoch-508 checkpoint
after 60 epochs without a better validation loss.

## Reproduce the complete comparison

From `Zach/` with the environment active, reuse the generated data/features and
choose a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.ml_compare \
  --features outputs/ml_segment1_compliant/robust/features \
  --ridge-model outputs/ml_segment1_compliant/robust/model/model_bundle.pkl \
  --input outputs/dataset_segment1_compliant_blind \
  --output outputs/ml_segment1_model_comparison_repeat
```

The command fits all three new model pairs before running their shared
benchmark evaluation. It checks model hashes across evaluation and writes
`comparison.json`, `comparison.csv`, and `comparison.png/.svg`. Each model
subdirectory contains its bundle, split definitions, metrics and inference
graphs. For MLPs, `training_loss.png/.svg` are actual epoch curves and
`candidate_comparison.png/.svg` compare saved candidate checkpoints.
The fitted artifacts from this run are in `outputs/ml_segment1_model_comparison/`.

Individual training commands are also available:

```bash
# Classical classification: validation selects logistic or random forest.
python -m tass.ml_cli train \
  --features outputs/ml_segment1_compliant/robust/features \
  --output outputs/my_classifier --task classification

# Two separate feature-vector MLP regressors.
python -m tass.ml_cli train \
  --features outputs/ml_segment1_compliant/robust/features \
  --output outputs/my_mlp_regressors --task regression --regressor mlp

# Two separate feature-vector MLP classifiers.
python -m tass.ml_cli train \
  --features outputs/ml_segment1_compliant/robust/features \
  --output outputs/my_mlp_classifiers --task classification --classifier mlp
```

Use `--mlp-max-epochs`, `--mlp-patience` and `--mlp-seed` to configure training.
Changing these is a new model-selection experiment; do not tune against the
benchmark and then describe it as untouched validation.

## Inference

All model types use the same `predict` and `evaluate` commands. The bundle saves
the task, selected sensors, DC removal, voltage representation, band features,
input/target scalers and fitted independent models. Classification additionally
saves class values and outputs their probabilities.

```bash
python -m tass.ml_cli predict \
  --model outputs/ml_segment1_model_comparison/mlp_regression/model/model_bundle.pkl \
  --input captures/segment1_capture.csv \
  --adc-full-scale-v 3.3 --adc-resolution-bits 12 \
  --output outputs/ml_segment1_model_comparison/mlp_real_predictions.csv
```

Choose `classifier` or `mlp_classifier` in the model path for endpoint decisions.
Simulation uses sensor voltage directly; real raw ADC is calibrated to voltage
before common DC removal/feature extraction. Original recordings remain read
only. Nominal ADC decoding is not a measured calibration of the sensor/front
end or contact mechanics. See the [force/contact validation guide](segment1_force_validation.md)
for those assumptions and collection commands.
