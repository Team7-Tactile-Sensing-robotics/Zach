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

## Complete editable configuration

Edit [`tass_minimum/config/default.json`](tass_minimum/config/default.json).
The source-tendon preset `tass_minimum/config/string_finger_tendon.json` also
contains the complete schema. `outputs/.../config.json` is a saved run snapshot,
not the default input file. Pass the editable file explicitly to the CLI:

```bash
python -m tass --config tass_minimum/config/default.json --output outputs/configured
python -m tass --config tass_minimum/config/default.json --modal-sweep --output outputs/configured_modal
```

Without `--config`, the CLI uses Python defaults. Explicit CLI overrides win
for scenario, tension mode and sweep settings. The coupled demo selects simulated
tension unless `--tension-mode measured` is given. Each simulated run saves the
resolved configuration in JSON and YAML. YAML input is also supported.

| Section | Used by |
| --- | --- |
| `motor`, `joint`, `axial`, `tendon_routing` | CLI finger/tendon mechanics |
| `string`, `contact`, `piezo`, `numerics` | Time-domain string, clamp, receiver and numerical settings |
| `experiment` | CLI motor/load trajectories, measured tension, pulse/chirp excitation |
| `modal_sweep` | Held pose, tension override, frequency range, mode count and damping |
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

**The time-domain preset has two receivers and one point-force exciter.**
Its `piezo.positions_m`, `gains`, and `noise_std_V` arrays determine receiver
count; extend all three arrays together to add receivers. Its single excitation
location is `string.exciter_position_m`, with waveform in `experiment`.
`modal_piezo` settings do not add actuators to this separate solver.

The JSON contains all exposed configuration fields for these workflows. Model
assumptions (such as fixed endpoints), UI layout, and optional Python callbacks
or initial-state arrays remain code/API choices rather than file parameters.

## Interactive finger mechanics

```bash
python -m streamlit run tass_finger_dynamics_app.py
```

Open the local URL printed by Streamlit. Adjust motor command, tendon routing,
stiffness, slack/preload, contact force and joint limits. The app plots finger
geometry, joint motion and tension. It shares the tendon model with both
simulation workflows; vibration sweeps are run through the CLI below.

## Integrated tendon model

The shared `tass/models/tendon.py` now includes string-finger-simulator's
routed geometry, winding/free-length state, fixed reference-length offset,
linear density (`mu`), rotation helper, tension and joint-torque interface.
Both the Streamlit app and CLI use this class. Zach retains its analytical
path derivative, vectorized calculations, validation, slack and preload.

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
to preserve its existing positional argument order. `fixed_extra_length` is a
read-only reference offset, which may be negative for an effective compliant
length shorter than the drawn route. At zero slack/preload, extension is
`required_length_current - free_length`. Slack and preload add independent offsets.
CSV mechanical outputs include `tendon_free_length_m`, `tendon_required_length_m`,
signed `tendon_extension_m`, extension rate and P1-P4 path coordinates.

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
slack and preload. With slack/preload zero, its extension law is equivalent to
the source's fixed-extra-length convention. Stiffness is `E*A/rest_length`;
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
voltage processing. They use the independently configured `string.length_m`,
not the full routed length used by the modal sweep. Frequencies from the two
workflows are comparable only when span, tension, density and boundaries match.

Outputs include `overview.png`, `timeseries.csv`, `string_state.npz`, spectra,
features and configuration/metadata. Mechanical channels now also include
signed tendon extension, extension rate and cumulative P1-P4 path coordinates.
In measured-tension mode those geometric quantities do not determine tension.

For hardware/saved CSV processing:

```bash
python -m tass --process-csv outputs/baseline_new/timeseries.csv --output outputs/reprocessed
```

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
