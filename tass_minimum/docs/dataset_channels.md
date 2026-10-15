# Synchronized simulation data

From `Zach`, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_channels_v2.json \
  --output outputs/dataset_channels_v2
```

Use a new empty output directory. This collects 320 trials (four locations ×
four forces × 20 repeats), each with 48,000 synchronized samples over one second.
The current ReturnSpring model supplies the static spring torque. There is no
servo startup or probe approach transient; each trial begins at a held mechanical
pose with vibration initially at rest.

Each `trials/*.npz` contains these one-dimensional, sample-aligned arrays:

| Columns | Meaning |
| --- | --- |
| `time_s` | Seconds since trial start |
| `contact_force_N` | Actual synthetic applied force, including trial jitter |
| `contact_location_m` | Grid-snapped distance along tendon from spool/feed endpoint |
| `contact_from_joint_reference_m` | Requested path distance from projected joint reference, before snapping |
| `piezo_sensor_1_V` … `piezo_sensor_4_V` | Simulated sensor voltages |
| `piezo_actuator_1_V` … `piezo_actuator_4_V` | Equivalent applied voltage pulses under linear actuator assumption |
| `piezo_actuator_1_force_N` … `piezo_actuator_4_force_N` | Forces actually used by vibration solver |
| `servo_angle_rad` | Static spool angle (ideal direct drive) |
| `servo_torque_Nm` | Ideal spool holding torque, tension × radius |
| `string_tension_N` | Static axial tendon tension |
| `joint_angle_rad` | Held finger joint angle |
| `spring_resisting_torque_Nm` | ReturnSpring resisting torque |

The sensor and actuator signals vary with time; mechanical quantities are
constant within each static trial. Servo torque excludes friction, gearing,
servo inertia and motor limits. It is a required torque, not a measured or
validated achievable motor output.

Actuator conversion is `force_N = gain_N_per_V * voltage_V`, configured by
`actuator_force_per_volt_N_V`. The initial demonstration gain is 0.001 N/V for
each channel, so a nominal 0.002-N pulse corresponds to 2 V. These are NOT
calibrated voltages or a piezo impedance/capacitance model. Replace the gains
with measured transfer data before comparing against hardware.

`example_trial.csv` is the entire first trial in a spreadsheet-ready table.
`manifest.csv` contains one row per trial with target labels, actual force,
nominal/requested/snapped contact locations, seed, split and checksum.
`summary.json` marks completion. Model and protocol snapshots record inputs.
The previous compact `piezo_V`, `actuator_force_N`, and class-label arrays
remain available; `actuator_voltage_V` adds a matching N×4 voltage matrix.

```python
import numpy as np
with np.load('outputs/dataset_channels_v2/trials/trial_0000.npz') as trial:
    sensors = trial['piezo_V']                 # (48000, 4)
    excitation = trial['actuator_voltage_V']   # (48000, 4)
    force_label = int(trial['force_class'])
    location_label = int(trial['location_class'])
```

For sensor-based prediction, use the sensor time series and optionally the
known excitation waveform. Keep contact labels, tension, servo torque, and
force-dependent servo angle out of that model's inputs: the simulated torque
balance makes them shortcuts to ground truth. Preserve trial-level splits.

Locations retain the v1 full-tendon convention: 20%, 40%, 60%, 80% from the
spool/feed endpoint. Mechanical force acts at a separately fixed 40-mm lever
arm. This is a surrogate experiment, not a geometrically exact probe at every
full-string point. See [the v1 protocol](dataset_v1.md) for the alternative
joint-referenced experiment and assumptions. Pair 4 is at the fixed boundary
and therefore has no displacement response beyond sensor noise. Its voltage
command is recorded even though its point force cannot move that boundary.

## Export a trial by number

From `Zach/`:

```bash
.venv/bin/python plot_trial.py 12 --dataset outputs/dataset_channels_v2
```

Trial numbers are zero-based (0–319 for the 320-trial collection). This reads
existing data and writes `outputs/dataset_channels_v2/trial_exports/trial_0012.csv`
plus `trial_0012_piezo.png`, `trial_0012_mechanics.png`, and
`trial_0012_finger_upright.png`. The upright figure contains just the recorded
finger diagram, rotated with the base at the bottom and fingertip at the top.
The piezo figure
has four rows with actuator drive on the left and the corresponding sensor on
the right. The mechanics figure contains contact force/location, servo angle/torque
and the static finger pose reconstructed from the saved model snapshot (+x along
the finger, +y upward on the tendon side). The CSV preserves all timestamps and
voltage values. Repeating the command replaces these exports, not source data.

Optional paths:

```bash
.venv/bin/python plot_trial.py 12 --dataset outputs/dataset_channels_v2 --output outputs/my_trial_plots
```

## Straight finger, locations measured along both physical segments

Use `config/experiments/dataset_straight_segments.json` to select
`location_reference="total_segment"`, `held_joint_angle_rad=0.0`, and zero
position jitter. With the current 0.15-m proximal and 0.10-m distal segments,
the four nominal positions from the finger base are 0.05, 0.10, 0.15 and 0.20 m.
The mapping assumes the base is (0,0), the joint lies on +x, and angle zero is
straight; unsupported coordinate conventions are rejected. Physical length is
joint x coordinate plus distal length. At bent poses, distance follows both
segment centerlines rather than a world x coordinate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_straight_segments.json \
  --output outputs/dataset_straight_segments
.venv/bin/python plot_trial.py 12 --dataset outputs/dataset_straight_segments
```

For this mode `contact_location_m` is the physical distance from the finger
base. `contact_tendon_location_m` separately records the projected, grid-snapped
vibration coordinate. The manifest records the coordinate convention and both
nominal and requested physical positions. Older datasets retain their original
location meaning. The plot script reads the convention from trial metadata.

The static generator assumes angle zero is already established by ideal servo
holding; it computes the winding needed for spring/contact torque balance.
It does not simulate a startup movement or feedback controller.

Contact is projected to the nearest point of the tendon route, then clamped to
a mesh node. The perpendicular resisting force has joint lever arm zero on
segment 1 and at the joint, and distance from the joint on segment 2. Thus
1/2/5/10 N at 20%, 40%, or exactly 60% do not change tension or the ideal hard
clamp response. This model cannot supply identifiable force classes there.
A validated local deformation/contact-compliance or tendon-deflection model
is needed for force estimation across the whole finger. Do not add artificial
joint torque on segment 1 to manufacture a signal. Pair 4's endpoint limitation
and the existing pulse excitation are unchanged by this location convention.

### Short frequency sweep followed by vibration decay

The straight-segment protocol now selects a linear chirp:

```json
"excitation_type": "chirp",
"chirp_schedule": "simultaneous",
"window_s": 1.5,
"chirp_duration_s": 0.3,
"chirp_start_Hz": 20.0,
"chirp_end_Hz": 2000.0,
"chirp_amplitude_V": 2.0,
"chirp_ramp_s": 0.005
```

All four channels sweep from 20 to 2000 Hz during the first 0.3 seconds,
with smooth 5-ms amplitude ramps at the start and end. Drive is then exactly
zero while sensors continue recording until 1.5 seconds (72,000 samples at
48 kHz). The vibration solver preserves the motion at shutoff and integrates
its subsequent decay; it does not reset sensor voltage. Sensor noise remains
present, and the time needed to decay depends on model damping. Amplitude is peak
bipolar voltage (nominal ±2 V), subject to the configured trial jitter.
Force = voltage × actuator_force_per_volt_N_V. The saved drive voltage traces
therefore describe the actual excitation supplied to the model.

Edit `chirp_duration_s` to change actuation duration and `window_s` to change
the total sensing duration in `config/experiments/dataset_straight_segments.json`.
For 1.5 seconds of sensing after shutoff, use `window_s: 1.8` with
`chirp_duration_s: 0.3`. These settings apply to the static dataset generator;
`numerics.duration_s` and `app.duration_s` control other simulations.

Set `chirp_schedule` to `sequential` to give each actuator a full frequency
sweep in its own quarter-window. Each starts at the beginning of its slot and
drives for `chirp_duration_s`, which must fit in `window_s / 4`. With the current
settings, starts are 0, 0.375, 0.75 and 1.125 seconds and each drive lasts
0.3 seconds. Omit `chirp_duration_s` or set it to `null` to fill the entire
slot (or entire recording for simultaneous excitation), preserving older protocols.
Simultaneous channels use the same phase and
frequency trajectory: their combined response does not independently identify
four actuator transfer functions. Use sequential sweeps for that experiment.
Set `excitation_type` to `pulse` to restore the old pulse protocol; its existing
pulse parameters are ignored when chirp mode is selected. Existing v2 outputs
are unchanged. Generate into a new directory, e.g. `outputs/dataset_straight_chirp`.

P4's fixed-end condition is controlled indirectly by model configuration:
`routed_pairs.anchor_extension_m = 0.0`. Acoustic length is P4's path coordinate
plus that extension; both endpoints are clamped. A positive measured extension
places the acoustic endpoint beyond P4, but this scalar does not add mechanical
load-cell geometry or model piezo reaction-force sensing. Do not assume a
continuous drive removes the fixed-boundary constraint.

The plot script currently defaults to `outputs/dataset_straight_chirp_fixed`.
Use `--dataset` to select a different saved collection. Servo angle is measured
from zero spool winding, not from the straight finger pose. Joint angle zero
and servo angle zero are independent references.

Choose the plotted time interval separately from recording duration:

```bash
.venv/bin/python plot_trial.py 12 --dataset outputs/dataset_straight_chirp_decay \
  --start-s 0.3 --end-s 1.5
```

This displays the decay and scales the vertical axes using visible samples.
Omit the bounds to plot the full recording; the CSV always exports the full
trial. Generate a new dataset into the selected folder before using this example.
Changing a protocol does not change previously saved trials.
