# TASS minimum analytical model

> For installation, the interactive app, and the integrated routed modal sweep, see the [project quickstart](../README.md). The equations below describe the legacy time-domain reference workflow (`routed_pairs.enabled=false`). The default JSON presets now enable four routed actuator/sensor pairs; see the project quickstart for their input and geometry conventions.

A runnable Python model of the V1 single-joint finger, based on the supplied
**TASS Minimum Analytical Model** notes and **Team 7 Project Proposal**.

Motor angle → tendon winding → joint motion and tension → string vibration →
synthetic piezo voltage → frequency features.

**All dimensions and material properties are editable demonstration values.**
They have not been calibrated against your prototype. This is a forward simulator,
not a trained contact-location or contact-force estimator.

## Run it

Requires Python 3.10 or later. Unzip the project and open a terminal in
`tass_minimum/`:

```bash
python -m pip install -r requirements.txt
python -m tass --scenario coupled --output outputs/my_run
python -m unittest discover -s tests -v
```

Optional: `python -m pip install -e .` installs the `tass` command and lets other
projects import the package.

Three examples are included. The committed data and plots in `outputs/` are
snapshots from the earlier constant-moment-arm model; rerun the commands below
to generate results with the routed tendon:

| Scenario | Command | What it demonstrates |
| --- | --- | --- |
| Baseline | `python -m tass --scenario baseline --output outputs/baseline` | Constant measured tension, pulse excitation, no contact |
| Contact | `python -m tass --scenario contact --output outputs/contact` | Same tension, rigid contact at 40% of string length |
| Coupled | `python -m tass --scenario coupled --output outputs/coupled` | Motor ramp, elastic tendon tension, opposing contact force, chirp excitation |

`baseline` and `contact` use the tension mode in the configuration, which defaults
to `measured`. `coupled` selects `simulated` unless `--tension-mode` is explicitly
supplied. Inputs for these demonstrations are defined in `tass/cli.py`; custom
experiments belong in a small script like `examples/custom_run.py`.

## Change the model

Start with `config/default.json`. Field names include units. Run your version with:

```bash
python -m tass --config config/default.json --scenario coupled --output outputs/experiment_01
```

The parameters most likely to change first are:

| JSON section / field | Meaning | Default |
| --- | --- | --- |
| `motor.spool_radius_m` | Constant spool radius | 0.015 m |
| `joint.origin_xy_m` | Joint center in world coordinates | [0.150, 0] m |
| `tendon_routing` | Spool feed and P1–P4 reference coordinates | See routing below |
| `joint.distal_length_m` | Finger segment length | 0.100 m |
| `joint.inertia_kg_m2` | Distal rotational inertia | 2e-5 kg m² |
| `joint.stiffness_Nm_rad` | Return spring stiffness | 0.25 N m/rad |
| `joint.damping_Nm_s_rad` | Viscous joint damping | 0.004 N m s/rad |
| `axial.mode` | Tension source | `measured` |
| `axial.young_modulus_Pa`, `area_m2`, `rest_length_m` | Define effective axial stiffness `EA/L0` | approximately 4341.95 N/m combined |
| `axial.slack_m` | Additional free-length allowance | 0 m |
| `axial.damping_Ns_m` | Axial damping while taut | 0.05 N s/m |
| `string.length_m` | Fixed vibrating span, independent of axial rest length | 0.120 m |
| `string.linear_density_kg_m` | Mass per length | 0.002 kg/m |
| `string.distributed_damping_Ns_m2` | Distributed transverse damping | 0.008 N s/m² |
| `string.nodes` | Spatial resolution, including endpoints | 81 |
| `string.exciter_position_m` | Point-force location along string | 0.024 m |
| `contact.finger_lever_arm_m` | Distance from joint for opposing torque | 0.040 m |
| `contact.tendon_position_m` | Distance along vibrating string for clamp | 0.048 m |
| `piezo.positions_m` | Receiver locations along string | [0.030, 0.090] m |
| `piezo.mode` | Local displacement or velocity sensing | `displacement` |
| `piezo.gains` | V/m or V s/m, depending on mode | [1000, 1000] |
| `piezo.noise_std_V` | Independent Gaussian noise per receiver | [0, 0] V |
| `numerics.sample_rate_Hz`, `duration_s` | Output sampling and record duration | 48,000 Hz; 0.25 s |

Each run saves the resolved configuration. Noise is reproducible using `piezo.seed`.

### Supply your own time-varying inputs

```python
from dataclasses import replace
from tass import ModelConfig, Inputs, simulate
from tass.signals import Constant, SmoothStep, Sine
from tass.processing import extract_frequency_features
from tass.io import save_run

config = ModelConfig.load("config/default.json")
config = replace(config, axial=replace(config.axial, mode="simulated"))

inputs = Inputs(
    motor_angle_rad=SmoothStep(0.12, 0.18, start_s=0.02, end_s=0.10),
    contact_force_N=SmoothStep(0, 0.4, start_s=0.14, end_s=0.16),
    excitation_N=Sine(amplitude=0.002, frequency_Hz=200),
)
result = simulate(config, inputs)
frequency = extract_frequency_features(result.measurement)
save_run(result, frequency, "outputs/custom_sine")
```

Any callable `f(time_s) -> scalar` works. Built-in signals include `Constant`,
`Sine`, `SmoothStep`, `Pulse`, `Chirp`, and `SampledSignal`. Signal values are
sampled on the output clock and interpolated during integration. Resolve the
fastest input on that clock; interpolation cannot recover undersampled inputs.

A sinusoidal drive demonstrates forced response; a pulse or free ringdown is
better for identifying natural frequencies. A dominant FFT peak may be a higher
mode, and some modes disappear when a sensor or exciter lies at their nodes.

## What the model solves

Positive motor rotation winds the tendon; positive joint angle is flexion;
positive external force opposes flexion. Angles are radians. The joint origin is
at `joint.origin_xy_m`. The straight reference pose at
`theta = joint.equilibrium_rad` points along positive x; flexion rotates the
distal segment counter-clockwise about that point.

```text
s = r_motor * phi_motor
L(theta) = |spool-P1| + |P1-P2| + |P2-P3(theta)| + |P3-P4|
shortening = L(theta_0) - L(theta)
k_axial = EA/L0
free_length = L0 - s
extension = L(theta) - free_length - slack
extension_rate = s_dot + L_prime(theta)*theta_dot

T = measured_load_cell_input
or
T = max(0, k_axial*extension + c_axial*extension_rate) if extension > 0 else 0

I * theta_ddot = -L_prime(theta)*T - k_joint*(theta-theta_0)
                - b_joint*theta_dot - r_contact*F_contact

mu*y_tt + c_string*y_t - T*y_xx = F_excitation(t)*delta(x-x_exciter)
V_i = gain_i * y(x_i)       [or gain_i * y_t(x_i)] + noise_i
```

The elastic law returns zero tension when the tendon is slack. In measured mode, the prescribed load-cell signal
drives joint dynamics directly: changing the motor command alone cannot change
the prescribed tension. Use simulated mode for the complete motor-to-tension
causal chain.

### Three-span tendon routing

`models/tendon.py` implements the shared `Tendon` model, used by both
`models/mechanics.py` and the root-level `tass_finger_dynamics_app.py`.
The route is `spool -> P1 -> P2 -> P3 -> P4`:

| Part | Default reference coordinates / length | Motion |
| --- | --- | --- |
| Spool feed | (0, 0) to P1 | Fixed feed span, about 40.31 mm |
| Span 1: P1–P2 | (40, 5) to (120, 5) mm | Fixed 80 mm along segment 1 |
| Span 2: P3–P4 | (175, 5) to (230, 5) mm | Fixed 55 mm along segment 2 |
| Span 3: P2–P3 | Across joint at (150, 0) mm | 55 mm straight; 32.02 mm at 90° flexion |

P1/P2 are fixed in the world frame. P3/P4 rotate about the joint by
`theta-theta_0`. P1–P3 are frictionless sliding guides, and P4 is the distal
attachment. The spool coordinate is an effective fixed feed point; changing
spool tangency, guide wrap, joint-surface contact, and guide friction are not
represented. The connecting span must remain straight and unobstructed.
A collapsed P2–P3 span is rejected because its force direction is undefined.

Only span 3 changes geometric length. The same analytical derivative `dL/dtheta`
is used for stretch rate and joint torque, so virtual work is consistent.
The effective flexion moment arm is **`-dL/dtheta`**: with the supplied geometry
it is 5 mm at 0°, 15.04 mm at 45°, and 22.65 mm at 90°. This variation replaces
the old constant `joint.moment_arm_m`. Older JSON files containing that field
load with a warning and discard it; update their joint origin and routing
coordinates explicitly before comparing results.

Rest length is the physical unstretched tendon length at zero winding. The
new default is 0.23031128874149276 m, matching the straight reference route.
Extension at zero winding is `L(theta_0) - rest_length_m - slack_m`. Pretension
comes from shorter rest length or initial winding, not an independent preload
parameter. Stiffness remains `EA/L0`, now about 4341.95 N/m with the default
material and area. The model has uniform axial tension and no axial inertia.

The app draws the same route with swapped display axes to show an upright finger.
From the repository root, run `streamlit run tass_finger_dynamics_app.py` (requires
Streamlit and pandas in addition to the package dependencies). Its servo response
and joint stops remain active, and its equilibrium table uses the routed torque.

Guide path coordinates are available for geometry inspection, but the existing
fixed-span vibration model and sensor coordinates have not been remapped to this
route. Mechanical contact distance and acoustic contact position remain separate.

Joint dynamics use adaptive DOP853 integration. The string uses centered finite
differences and RK4 with automatic substeps based on the maximum tension and
damping. Fixed endpoints have zero displacement and velocity. A hard contact
clamps the nearest interior node when the applied force exceeds
`contact.force_threshold_N` and `contact.enabled` is true. Setting that flag to
false disables the acoustic clamp; the supplied external force still acts on
joint dynamics.

Initial states default to the unloaded joint angle with zero joint velocity and
zero string displacement/velocity. A nonzero initial motor command in elastic
mode can therefore create an initial tension transient. Supply
`initial_joint_state=(theta_initial, omega_initial)` or the optional initial
string arrays to start from a different condition.

Contact distance along the finger and contact distance along the string are
**separate inputs**. Their physical mapping depends on tendon routing and should
be calibrated separately; the new axial routing does not impose that mapping.

### Physical details worth keeping explicit

- A point force is divided by **nodal mass `mu*dx`**. This preserves force when
  you refine the mesh. The PDE force density is `F/dx`, not `F`.
- The two hard-contact spans are `x_contact` and `L-x_contact`; they are equal
  only at the midpoint. Use the actual snapped contact coordinate in metadata
  when comparing frequencies on a coarse mesh.
- With one exciter and an ideal rigid clamp, energy cannot cross the clamp.
  In the contact example, the right-hand receiver remains silent because that
  span starts at rest. Excite each side in separate runs to observe both sets
  of resonances. The proposal's additional exciters can be added later.
- Clamp engagement projects local displacement and velocity to zero. Switching
  is resolved on output-sample boundaries and can dissipate energy and create
  transients. This is an ideal constraint, not a collision or compliant-contact
  calculation.
- At fixed measured tension, clamp resonances do not determine force magnitude.
  Force affects joint torque; in elastic mode it can also change tension.
  Force inference needs additional mechanics, calibrated compliance, or data.

## Outputs and hardware handoff

Every CLI simulation writes:

| File | Contents |
| --- | --- |
| `timeseries.csv` | Time, motor command, winding displacement/velocity/acceleration, each routed span length, total path length, shortening, moment arm, tendon torque, joint angle/velocity/acceleration, tension, contact force/location/state, contact and tip world coordinates, excitation, piezo voltages |
| `string_state.npz` | Complete `time_s`, `grid_m`, `displacement_m`, `velocity_m_s` arrays |
| `spectrum.csv`, `spectrum.npz` | Frequency, voltage amplitude, excitation amplitude, complex approximate FRF, valid-bin mask |
| `features.json` | Dominant frequency, peak amplitude, spectral peaks, spectral energy and channel amplitude ratio |
| `config.json`, `metadata.json` | Resolved parameters, mesh positions, timestep and schema version |
| `overview.png` | Mechanical motion, tension, voltage histories and spectrum |

No active acoustic contact is represented by `contact_active=0` and `NaN` contact
coordinates. `contact_force_N` remains the supplied mechanical load. Undefined
FRF bins are `NaN` and flagged by `frf_valid`; undefined scalar features are JSON
`null`. `string_state.npz` can be opened with `numpy.load` without pickle.

Processing consumes only `Measurement(time_s, piezo_V, excitation_N, channels)`.
It does not depend on the simulator. Receiver data are an array shaped
`(number_of_samples, number_of_receivers)`.

```python
from tass import Measurement
from tass.processing import extract_frequency_features

record = Measurement(
    time_s=uniform_timestamps,
    piezo_V=calibrated_receiver_voltages,
    excitation_N=calibrated_exciter_force,
    channels={"joint_angle_rad": encoder_angle, "tension_N": load_cell_tension},
)
features = extract_frequency_features(record)
```

For recorded tension, use `SampledSignal` as `Inputs.tension_N`. For measured
angles and piezos, construct `Measurement` directly; that bypasses simulated
mechanics and sensors. See `examples/hardware_adapter.py`.

To process a CSV with the shared schema:

```bash
python -m tass --process-csv outputs/baseline/timeseries.csv --output outputs/reprocessed
```

Real ESP32 data need timestamp synchronization, conversion to SI units and
uniform resampling before this interface. Drive voltage is not automatically
force: physical `V/N` frequency response requires actuator calibration. If force
is unavailable, use a zero excitation array to disable FRF while still computing
voltage spectra and features. This project does not implement serial acquisition.

### Spectral interpretation

FFT amplitudes use a coherent-gain-corrected one-sided windowed spectrum.
Spectral energy is the exact discrete Parseval energy of the demeaned,
unwindowed signal, in V² s. The channel ratio divides each channel's largest
spectral amplitude; it is not a ratio at a shared frequency.

`Y/U` is a finite-record approximation. Weakly excited bins are masked. Windowing,
transients, changing tension and changing contact can bias it. Use a stationary
interval and a suitable excitation to estimate a transfer function; future work
can add averaged cross spectra and coherence. A whole-record FFT of the coupled
example combines multiple operating states and must not be treated as one set
of fixed natural frequencies. Use `start_s` / `end_s` to select an interval.

## Extend fidelity in small steps

| Improvement | Component to extend | Existing downstream interface |
| --- | --- | --- |
| New motor command, pulse train or experimental trajectory | `signals.py` / `Inputs` | Sampled scalar inputs |
| Load-cell tension or a different elastic law | `models/mechanics.py`, or `tension_law=` | Nonnegative tension in N |
| Variable spool radius, guide wrap or a different tendon route | `motor_kinematics`, `models/tendon.py` | Mechanical channel dictionary |
| Encoder-driven replay | `mechanical_solver=` or a direct `Measurement` | Joint/tension channels |
| Compliant contact | `models/contact.py` and string stability bound | Boundary constraints plus contact acceleration |
| Multiple exciters, variable length, bending stiffness, FE/modal model | `vibration_solver=` | `StringResult` grid, displacement and velocity |
| Piezo dynamics, electronics, ADC filtering/noise | `sensor_model=` | Voltage array |
| Contact inference or ML | New processing module | `Measurement` and extracted features |

`simulate()` accepts a tension law, contact boundary, mechanical solver,
vibration solver and sensor model. The built-in function signatures document
their contracts. A new compliant contact needs its own stiffness included in
the timestep bound; adding contact forces alone does not guarantee stability.

The first physical upgrades to consider are measured geometry and damping,
joint travel limits, compliant contact, and actuator/receiver transfer functions.
The present model assumes a constant straight vibrating span, uniform tension,
small transverse displacement, one joint and no friction hysteresis. Axial
length and acoustic span are independent constants. Zero tension is handled
numerically, but a slack cable's real transverse motion is outside the taut-string
model. There is no dynamic motor or longitudinal-wave model and no vibration
feedback into axial extension.

## Verification

Run `python -m unittest discover -s tests -v`. The tests check the supplied
zero-input, static equilibrium, contact-force equilibrium, string-frequency,
tension-scaling and contact-location requirements, plus elastic coupling,
force normalization, fixed boundaries and the common data/processing interface.
Routing checks cover constant link spans, cross-joint shortening, the analytical
path derivative, virtual work, stretch rate, physical slack and initial tension, coordinate transforms,
and nonlinear static equilibrium.

For the saved baseline (`L=0.12 m`, `T=4 N`, `mu=0.002 kg/m`), the first three
analytical frequencies are **186.34, 372.68, 559.02 Hz**. The pulse simulation has
FFT peaks at **188, 372, 560 Hz** on a 4 Hz frequency grid. The strongest peak is
the second mode, demonstrating why the strongest peak is not always the fundamental.

At 40% hard contact, the excited left span predicts **465.85 Hz**; the simulated
peak is **464 Hz**. These checks validate the implementation against its own
assumptions, not against experimental TASS measurements. See `docs/verification.md`.
