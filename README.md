# Zach: tendon-driven finger and string vibration

Zach includes an interactive finger-mechanics app, a time-domain reference
simulation, and the routed modal vibration model adapted from
`string-finger-simulator`. It runs independently; the sibling repository is
not required. Parameters are demonstration values, not hardware calibration.

## Install

Use Python 3.10 or newer. From the directory containing `Zach/`:

```bash
cd Zach
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ./tass_minimum
```

The requirements file installs NumPy, SciPy, Matplotlib, PyYAML, pandas and
Streamlit. The editable install makes the `tass` package available from `Zach/`.
Alternatively, `python -m pip install -e './tass_minimum[app]'` installs both
the package and all runtime dependencies in one command.
All commands below start in `Zach/` with that environment activated.

## Spring–tendon actuation

The default preset now models the three stages in the sketch: relaxed at -10°,
ready at 0°, and normal flexion. A geometric extension spring supplies the
opposing torque. See [spring model, equations and run commands](tass_minimum/docs/spring_model.md).
The upright spool height is 30 mm (`[0.03, 0]` in internal coordinates).

## Complete editable configuration

Edit [`tass_minimum/config/default.json`](tass_minimum/config/default.json).
The source-tendon preset `tass_minimum/config/string_finger_tendon.json` also
contains the complete schema. `outputs/.../config.json` is a saved run snapshot,
not the default input file. Pass the editable file explicitly to the CLI:

```bash
python -m tass --config tass_minimum/config/default.json --output outputs/configured
python -m tass --config tass_minimum/config/default.json --modal-sweep --output outputs/configured_modal
```

Without `--config`, the CLI loads `tass_minimum/config/default.json` when available
in this checkout (otherwise it falls back to legacy Python defaults). Explicit CLI overrides win
for scenario, tension mode and sweep settings. The coupled demo selects simulated
tension unless `--tension-mode measured` is given. Each simulated run saves the
resolved configuration in JSON and YAML. YAML input is also supported.

| Section | Used by |
| --- | --- |
| `spring` | Extension-spring stiffness, attachments and relaxed angle (or legacy torsional mode) |
| `motor`, `joint`, `axial`, `tendon_routing` | CLI finger/tendon mechanics |
| `string`, `contact`, `piezo`, `numerics` | Time-domain string, clamp, receiver and numerical settings |
| `experiment` | CLI motor/load trajectories, measured tension, pulse/chirp excitation |
| `modal_sweep` | Held pose, tension override, frequency range, mode count and damping |
| `routed_pairs` | Four time-domain pairs, selected actuator, reference pose, anchor extension and sensor gains/noise |
| `modal_piezo` | Four modal excitation/sensing pairs |
| `analysis` | CLI FFT interval, window, peak count and FRF threshold |
| `app` | Streamlit mechanics defaults, servo response, timestep and equilibrium table |

The Streamlit app loads `default.json` automatically; its sidebar accepts another
JSON/YAML path. Its independent demonstration solver uses the `app` section for
physical settings, not the CLI's `joint`/`axial` sections. Sliders override the
loaded values for the current session and do not write back to disk.

### Are there four piezo sensors and four actuators?

**The modal sweep has four idealized actuator/sensor pairs.** The arrays in
`modal_piezo` follow `pair_ids` order. Each `guide_numbers` entry selects P1-P4;
positions come from `tendon_routing`. `actuator_amplitudes_au` and
`actuator_phases_rad` set each sequential excitation, while `sensor_gains` scales
each received response. A zero amplitude disables an actuator. The result is a
4-by-4 complex response matrix at every frequency. Units remain arbitrary:
there is no calibrated voltage-to-force conversion, piezo circuit or ADC model.
The pair assigned to P4 remains a fixed endpoint and has zero modeled response.

**The default time-domain presets now have four force actuators and four receivers.**
`routed_pairs.enabled=true` connects them to the actual P1-P4 tendon route:

| Pair | Guide | Segment | Position along segment |
| --- | --- | --- | --- |
| pair1 | P1 | 1 (fixed) | Base/spool end |
| pair2 | P2 | 1 (fixed) | Joint end |
| pair3 | P3 | 2 (rotating) | Joint end |
| pair4 | P4 | 2 (rotating) | Distal/tip end |

“Top/bottom” means the two ends along each segment, not opposite surfaces.
Reference coordinates are in `tendon_routing`; all four sit along the same
routed tendon. Segment 2's world coordinates rotate with the joint.

From `Zach/`, select an actuator and read all four sensors:

```bash
python -m tass --scenario coupled --actuator pair1 --output outputs/four_pairs
python -m tass --scenario baseline --actuator pair3 --output outputs/pair3
```

The scalar demo waveform in `experiment` drives `routed_pairs.active_actuator`
(default pair1); `--actuator` overrides it. For independent simultaneous inputs:

```python
from tass import ModelConfig, Inputs, simulate
from tass.signals import Pulse, Sine

config = ModelConfig.load("tass_minimum/config/default.json")
result = simulate(config, Inputs(actuator_forces_N={
    "pair1": Pulse(amplitude=0.002),
    "pair3": Sine(amplitude=0.001, frequency_Hz=300),
}))
print(result.measurement.piezo_V.shape)  # (samples, 4), pair1 through pair4
```

Forces are in newtons, not actuator volts. Omitted dictionary entries are off;
`{}` turns all off. `None` uses the selected demo actuator. CSV records
`actuator_pair1_force_N` through `actuator_pair4_force_N`, four piezo voltages,
and each pair's moving world coordinates. Metadata identifies channel order,
segment assignment, acoustic positions, and actual mesh excitation locations.
With several driven actuators, the scalar Y/U FRF is disabled rather than dividing
by their summed forces. Individual drive channels are preserved for later MIMO
analysis; voltage spectra remain available.

The acoustic solver uses the route at `routed_pairs.reference_angle_rad` as a
**frozen reference span**, including the spool-to-P1 feed. It follows tension
changes, but does not implement a moving mesh or intermediate guide constraints.
Pair world coordinates still follow actual mechanics; large pose changes need
a moving-boundary model for quantitative acoustic predictions.

P4 is a fixed endpoint when `anchor_extension_m=0` (the default). Its point-force
actuator cannot move that endpoint, and its displacement receiver is silent
except for configured noise. If the real tendon continues beyond P4 to a separate
anchor, set `anchor_extension_m` to that **measured** distance. Do not add a fictitious
length merely to obtain a signal. Reaction-force sensing or driven-boundary piezo
coupling needs a different physical model. The empirical modal sweep retains its
own P4 endpoint assumption; the extension setting applies to time-domain mode.

In routed mode, sensor gains/noise come from `routed_pairs`, while sensing mode
and random seed come from `piezo`. Derived acoustic length, actuator positions,
and receiver positions override the legacy `string.length_m`,
`string.exciter_position_m` and `piezo.positions_m` for the solver. The saved
configuration remains replayable; resolved acoustic geometry is in metadata.
Set `routed_pairs.enabled=false` to run the legacy independent span with two
receivers and one exciter. `ModelConfig()` retains that legacy default for Python
API compatibility; load the supplied JSON to select the four-pair model.

The JSON contains all exposed configuration fields for these workflows. Model
assumptions (such as fixed endpoints), UI layout, and optional Python callbacks
or initial-state arrays remain code/API choices rather than file parameters.

## Interactive finger mechanics

```bash
python -m streamlit run tass_finger_dynamics_app.py
```

Open the local URL printed by Streamlit. Adjust motor command, tendon routing,
stiffness, physical rest length and slack, contact force and joint limits. The app plots finger
geometry, joint motion and tension. It shares the tendon model with both
simulation workflows; vibration sweeps are run through the CLI below.

## Integrated tendon model

The shared `tass/models/tendon.py` now includes string-finger-simulator's
routed geometry, winding/free-length state, fixed reference-length offset,
linear density (`mu`), rotation helper, tension and joint-torque interface.
Both the Streamlit app and CLI use this class. Zach retains its analytical
path derivative, vectorized calculations, validation and slack.

To run the source project's tendon/spool parameters in Zach, from `Zach/`
with the environment above activated:

```bash
python -m tass --config tass_minimum/config/string_finger_tendon.json --scenario coupled --output outputs/source_tendon
python -m tass --config tass_minimum/config/string_finger_tendon.json --modal-sweep --motor-angle-rad 0.3 --output outputs/source_tendon_modal
```

This preset carries over the source's routing, 5 mm spool radius, 0.25 m rest
length, 500 N/m tendon stiffness, 0.1 N s/m damping and 0.0005 kg/m density.
It retains Zach's other model defaults; it does not import the source's servo,
gravity, electronics or complete joint dynamics. The app reads slider defaults from the same file's `app` section.

Zach represents stiffness as `E*A/L0`. The preset uses E=2e9 Pa and A=6.25e-8 m²
with L0=0.25 m to reproduce k=500 N/m. This factorization matches stiffness;
it is not an independently measured material/area calibration.

For direct use after installation:

```python
from tass.models.tendon import Tendon

tendon = Tendon(rest_length=0.25, k=500, c=0.1, mu=0.0005)
tension = tendon.compute(q=0.2, q_dot=0.0,
                         wound_length=0.005*0.3, winding_velocity=0.0)
print(tension, tendon.joint_torque(0.2))
print(tendon.free_length, tendon.required_length_current)
print(tendon.guide_path_coordinates(0.2))
```

Use keyword arguments when porting constructors: `mu` is keyword-only in Zach
to preserve its existing positional argument order. The tendon uses physical rest
length: `extension = path_length - (rest_length - winding) - slack`.
There is no `fixed_extra_length` or independent preload parameter. Initial
pretension comes from the chosen rest length and winding.
CSV mechanical outputs include `tendon_free_length_m`, `tendon_required_length_m`,
signed `tendon_extension_m`, extension rate and P1-P4 path coordinates.

### Physical rest length and migration

The legacy Python defaults use a 0.23031128874149276 m unstretched length.
The active extension-spring JSON preset instead uses 0.21043479108399377 m,
matching its raised-spool, 15-mm-guide route at the -10° relaxed pose. With zero winding and zero additional slack, initial tension is
zero at the configured relaxed pose. Material E and area A are retained, so
`EA/L0` changes when the physical rest length changes. This is a demonstration setup, not a
measurement of your prototype. Changing geometry does not automatically resize
the physical tendon; update rest length deliberately when appropriate.

The source-tendon preset retains its 0.25 m rest length and 500 N/m stiffness.
With its 0.23031 m route it has about 19.69 mm of slack at zero winding; small
motor commands can legitimately produce no tension. Choose measured rest length
and initial winding for your hardware rather than hiding this slack with offsets.

Old `axial.preload_N` or `app.preload_N` values of zero load with a migration
warning and are dropped. Nonzero values are rejected so pretension cannot be
silently lost. For fixed E and A, zero winding and zero extra slack, an intended
static pretension T at a held pose of route length L requires
`L0 = L / (1 + T/(E*A))`. The app instead specifies stiffness directly, so it
uses `L0 = L - T/k` under those same conditions. These formulas assume a held
pose; a freely moving joint must also satisfy torque balance.

Winding must remain between zero and rest length. The app exposes physical rest
length and no longer passes an unsupported preload argument. Archived output
configurations retain their original values; rerun experiments with the updated
input presets to get results for the new setup.

## Integrated routed-tendon modal sweep

Hold the finger at a known pose, calculate tension from spool winding, and
sweep the configured excitation at each of P1-P4 while reading all four locations
(unit amplitudes by default):

```bash
python -m tass --modal-sweep --joint-angle-deg 20 --motor-angle-rad 0.3 --output outputs/modal_elastic
```

Or prescribe tension directly to compare poses at the same load:

```bash
python -m tass --modal-sweep --joint-angle-deg 45 --tension-N 4 --start-Hz 20 --stop-Hz 2000 --step-Hz 2 --modes 12 --output outputs/modal_45deg
```

Add `--config tass_minimum/config/default.json` to use edited parameters.
`--modal-damping-ratio` defaults to 0.02. Frequencies include the start and
advance by the requested step without exceeding the stop.

Each modal output directory contains:

- `modal_sweep.png`: four plots, one per excitation location.
- `modal_sweep.csv`: actuator, frequency, sensor, real/imaginary response and magnitude.
- `modal_sweep.npz`: complex `response` shaped `(4, frequencies, 4)`, frequencies,
  natural frequencies, guide coordinates and pair IDs. Load with `numpy.load`.
- `metadata.json`: pose, winding angle, tension source, routed length and modal settings.
- `config.json`: physical configuration used for the run.

### How the integration works

`models/tendon.py` already implements the source project's route
`spool -> P1 -> P2 -> P3 -> P4`. The integration reuses it rather than introducing
a second tendon class. It retains Zach's analytical path derivative, validation,
slack. Zach now uses physical rest length, whereas the source project uses a
reference-length offset; the same numerical rest length need not give the same tension. Stiffness is `E*A/rest_length`;
when transferring the source's `k`, choose these parameters to give that value.
The optional `string_finger_tendon.json` preset transfers the tendon/spool defaults;
source YAML is not loaded automatically.

`models/modal_string.py` adapts the source's natural frequencies, sine mode
shapes and complex empirical response. `modal_sweep.py` connects it to the
current routed length and cumulative guide positions. Tension comes from the
shared tendon elastic law unless `--tension-N` is supplied. This explicit sweep
choice is independent of `axial.mode`, which controls the time-domain workflow.

The pose is held by an assumed fixture/controller; the sweep does not solve
joint equilibrium or evolving mechanics. Its response is in **arbitrary units**,
not metres, volts, ADC counts or a calibrated force response. It is a direct
frequency-domain calculation, so it does not use the output sampling clock.

The imported model treats the whole route as one fixed-fixed span; intermediate
guides impose no acoustic constraints. P4 is its fixed endpoint, so pair4's
excitation and displacement response are zero. This is retained explicitly.
There is no contact splitting in the modal sweep; contact settings do not activate
an acoustic clamp here. Use the reference contact simulation below for that.

## Time-domain reference simulation

```bash
python -m tass --scenario baseline --output outputs/baseline_new
python -m tass --scenario contact --output outputs/contact_new
python -m tass --scenario coupled --output outputs/coupled_new
```

- `baseline`: prescribed tension and pulse excitation.
- `contact`: prescribed tension with an ideal string contact clamp.
- `coupled`: motor ramp, elastic tendon tension, opposing contact force and chirp.

These commands retain the finite-difference string solver and existing piezo
voltage processing. With the supplied presets they use the routed reference span described above.
With `routed_pairs.enabled=false`, they use the independent `string.length_m`. Frequencies from the two
workflows are comparable only when span, tension, density and boundaries match.

Outputs include `overview.png`, `timeseries.csv`, `string_state.npz`, spectra,
features and configuration/metadata. Mechanical channels now also include
signed tendon extension, extension rate and cumulative P1-P4 path coordinates.
In measured-tension mode those geometric quantities do not determine tension.

For hardware/saved CSV processing:

```bash
python -m tass --process-csv outputs/baseline_new/timeseries.csv --output outputs/reprocessed
```

## Generate a straight-finger sweep dataset

Run these commands from `Zach/` after installation. The physical model is in
[`tass_minimum/config/default.json`](tass_minimum/config/default.json); the
collection protocol is in
[`dataset_straight_segments.json`](tass_minimum/config/experiments/dataset_straight_segments.json).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_straight_segments.json \
  --output outputs/dataset_straight_chirp
```

Choose a **new, empty output directory** for every collection, including after a
failed run. For example, replace the output with `outputs/dataset_straight_chirp_run2`
if the first directory already contains files. Existing datasets are not overwritten.

The protocol records 320 trials: four locations × four forces × 20 repeats.
Each trial is one second at 48 kHz, with the joint held straight at 0 radians.
Locations are 20%, 40%, 60%, and 80% of the combined physical segment lengths,
measured from the finger base (0.05, 0.10, 0.15, and 0.20 m for a 0.25-m finger).
Force classes are 1, 2, 5, and 10 N. All four actuators use a simultaneous linear
20–2,000 Hz chirp with nominal 2-V peak amplitude; select `chirp_schedule: sequential`
in the JSON protocol to excite one actuator per quarter-window instead.

Outputs include:

- `trials/trial_XXXX.npz`: synchronized sensor/actuator voltages, mechanical
  channels, timestamps, labels, and metadata.
- `manifest.csv`: trial IDs, contact conditions, seeds, checksums, and data splits.
- `example_trial.csv`: the full first trial as a table.
- `model_config.json` and `protocol.json`: the exact input snapshots.
- `summary.json`: completion status and collection statistics.

This is synthetic static-contact data. Servo angle is measured from zero spool
winding, so it need not be zero when the joint is straight. Actuator voltage
uses the protocol's demonstration force-per-volt gain. The current hard-contact
model cannot distinguish force magnitudes at fixed-segment or joint contacts
through tension changes alone. Model distances ending in `_m` are metres:
`anchor_extension_m: 0.03` means 30 mm, whereas `30` means 30 metres.
See [model assumptions and channel definitions](tass_minimum/docs/dataset_channels.md).

## Plot and export a trial

Use a zero-based trial number (0–319 for this protocol) and the same dataset
folder used during generation:

```bash
.venv/bin/python plot_trial.py 12 --dataset outputs/dataset_straight_chirp
```

For the existing `dataset_straight_chirp_fixed` collection, the shortcut is:

```bash
.venv/bin/python plot_trial.py 12
```

The script reads saved data; it does not rerun the simulation. Results are saved
under the selected dataset's `trial_exports/` directory:

| File | Contents |
| --- | --- |
| `trial_0012.csv` | Every timestamp and the requested voltage/contact/servo values |
| `trial_0012_piezo.png` | Four rows: actuator drive voltage on the left, corresponding sensor voltage on the right |
| `trial_0012_mechanics.png` | Contact force/location, servo angle/torque, and recorded finger pose |

The pose uses the saved model configuration, with +x pointing from base toward
the fingertip and +y upward on the tendon side. It marks the spool, joint,
P1–P4, spring attachments, and contact. A new export replaces the same trial's
CSV/PNGs but leaves the source dataset unchanged.

To choose a different export folder:

```bash
.venv/bin/python plot_trial.py 12 \
  --dataset outputs/dataset_straight_chirp \
  --output outputs/my_trial_plots
```

## First simulation dataset (legacy pulse protocol)

The [Version 1 collection protocol](tass_minimum/docs/dataset_v1.md) specifies
16 location/force conditions, 20 trials per condition, 1-second four-channel
windows at 48 kHz, two class labels, and complete-trial train/validation/test
splits. Parameters are in `tass_minimum/config/experiments/dataset_v1.json`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_v1.json \
  --output outputs/dataset_v1
```

Use a new output directory for a rerun. This is synthetic data with ideal static
force balance, not digital-force-gauge measurements. The protocol documents the
trial-count/location ambiguities, P4 boundary limitation and hardware procedure.

## Tests and code map

```bash
cd tass_minimum
python -m unittest discover -s tests -v
```

| File inside `tass_minimum/tass/` | Responsibility |
| --- | --- |
| `models/tendon.py` | Shared routed geometry, extension, tension and joint torque |
| `models/mechanics.py` | Spool winding and joint dynamics |
| `models/string.py`, `models/contact.py` | Reference transient string/contact solver |
| `models/modal_string.py` | Imported empirical modal vibration model |
| `modal_sweep.py` | Held-pose tendon/modal integration, four-location sweeps and export |
| `simulation.py`, `models/piezo.py` | Time-domain orchestration and synthetic voltages |
| `config.py`, `signals.py` | Parameters and time-varying inputs |
| `processing.py`, `io.py`, `cli.py` | Analysis, data files and command-line entry point |

For full reference equations and hardware measurement interfaces, see
[tass_minimum/README.md](tass_minimum/README.md).
