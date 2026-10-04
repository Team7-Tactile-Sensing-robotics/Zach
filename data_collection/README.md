# Finger data collection CSV format

Use these tables to collect piezo signals with independently measured contact
force and location. Schema version: **1.0**. This defines the recording format;
it does not start hardware acquisition or change the physical simulator.

| File | One row represents | Entered by |
| --- | --- | --- |
| [templates/trials.csv](templates/trials.csv) | One trial and its fixed setup/targets | Experimenter or logger |
| [templates/waveforms.csv](templates/waveforms.csv) | One simultaneous receiver/drive sample | High-rate acquisition logger |
| [templates/states.csv](templates/states.csv) | One timestamped measurement/event from a mechanical sensor or label source | Sensor logger or annotation tool |

The templates contain headers only. **All numbers in [examples/](examples/) are
invented format examples**, not measurements, calibrated predictions, recommended
settings, or training data. They show two receivers, a baseline trial, and a
contact trial. Each waveform is only a 2 ms snippet, not a useful experiment.

## Start a session

1. Copy the three templates into a new session directory, retaining their names.
2. Add one row to `trials.csv` for each independent approach/contact/release trial.
   Use a unique `trial_id` across the dataset and a shared `session_id` for trials
   collected in the same session. Record `data_source=hardware` for real readings.
3. Save a copy of the actual setup and calibration records alongside that session.
   For the modeled geometry, copy `finger_parameters.py` and update measured
   values. Fill in `setup_file` and `calibration_file`; paths are relative to the
   directory containing `trials.csv`. Keep these snapshots unchanged afterward.
4. Append waveform samples and mechanical readings with the same `trial_id`.
   The waveform/state file paths can point to session-wide files or separate
   files per trial. Filter shared files by `trial_id`.
5. Record before contact, loading, held contact, unloading, and after release.
   Keep repeated trials independent and use `repeat_index` to track repetitions.

Setup records should identify segment lengths, guides, spring attachments,
load-cell attachment, spool, tendon, and transmitter/receiver mounting. Calibration
records should identify each device, ADC voltage conversion, amplifier gain and
filters, sensor sample rates, force scale/zero/direction, angle zero, contact
position reference/method, timing offsets/drift, and calibration date. If they are
not yet available, leave their paths blank instead of referring to nonexistent
files. Hardware data are ready for labeled analysis only once the required
calibrations and labels are available.

## Shared conventions

- UTF-8, comma separator, one header row, decimal point `.`. Quote text containing
  commas, quotes, or newlines using standard CSV quoting. No unit text in numbers.
- Numeric headers carry SI units: seconds, volts, newtons, metres, radians, hertz.
  Convert mm to m and degrees to radians before writing. IDs remain text.
- An **empty field means missing, unavailable, or not applicable**. `0` means a
  measured/known zero. Do not use `-999`, `NA`, or an assumed zero for missing data.
- All `time_s` values are elapsed seconds from the **same start of that trial**.
  Sensor clocks must be aligned before writing these times. Record the method in
  `sync_method`; retain original device logs when clocks require correction.
- Trial targets are commanded/intended conditions, never ground-truth labels.
  Actual force, contact location, joint angle, and motor angle belong in `states.csv`.
- The distal load cell measures **tendon tension**. An independent sensor measures
  **object contact force**. Do not substitute one for the other.
- This version labels a single contact with a normal force. It does not represent
  multiple simultaneous contact locations or a full force vector.

## `trials.csv`: setup and intended conditions

Keep settings constant within a trial. Start a new trial when transmitter routing,
gains, mounting, or excitation protocol changes.

| Column | Meaning |
| --- | --- |
| `schema_version` | `1.0` for this format. |
| `session_id` | Session identifier, e.g. `session_20261004_A`. |
| `trial_id` | Unique identifier, e.g. `session_20261004_A_trial_001`. |
| `data_source` | `hardware` or `synthetic`; required to distinguish real data. |
| `start_time_utc` | Trial start in UTC, e.g. `2026-10-04T15:00:00Z`. Blank if unknown. |
| `waveforms_file`, `states_file` | Paths to the corresponding sample files. |
| `setup_file`, `calibration_file` | Paths to saved setup and calibration records. |
| `piezo_sample_rate_Hz` | Nominal rate of the synchronous waveform samples. Actual timestamps still matter. |
| `transmitter_id` | Physical exciter ID, e.g. `P2`; use `none` for an excitation-off trial. |
| `piezo_1_location` ... `piezo_4_location` | Physical location/ID assigned to each receiver column, e.g. channel 1 at `P4`. Blank for unused channels. Channel numbers are acquisition channels, not automatically P1-P4. |
| `excitation_type` | `off`, `sine`, `chirp`, `pulse`, `noise`, or `custom`. Document custom protocols in setup/notes. |
| `excitation_frequency_start_Hz`, `excitation_frequency_end_Hz` | Sweep endpoints; equal for a sine. Blank when not applicable. |
| `excitation_drive_peak_V` | Intended drive peak amplitude, not peak-to-peak. Actual sampled voltage belongs in waveforms. |
| `contact_surface` | Planned contacted surface: `finger` or `tendon`. Record this for baseline trials too. |
| `object_id`, `object_material` | Object/probe identifier and material. |
| `contact_tip_radius_m` | Probe radius when meaningful; blank for an unspecified/flat geometry. |
| `target_contact_force_N` | Intended normal force. |
| `target_contact_segment` | Intended segment `1` or `2`; blank for baseline or tendon-only coordinates. |
| `target_contact_position_m` | Intended finger-surface location using the segment convention below. Blank for baseline or tendon-only contact. |
| `target_joint_angle_rad` | Intended joint angle; straight is zero and flexion positive. |
| `preload_N` | Measured initial tendon tension at the documented reference pose before the trial. Blank if unknown. |
| `repeat_index` | Integer starting at 1 for repeats of the same intended condition. |
| `sync_method` | Actual clock/trigger/alignment method and calibration reference; do not claim synchronization that was not performed. |
| `notes` | Protocol details, contact direction, failures, and departures from the intended setup. |

At minimum identify the version, session, trial, source, data file paths, receiver
mapping, sample rate, and timing method. Fill applicable protocol fields; leave
unmeasured optional quantities blank. Preserve the header even for unused columns.

## `waveforms.csv`: raw time-domain signals

| Column | Meaning |
| --- | --- |
| `trial_id` | Links to exactly one row in `trials.csv`. |
| `sample_index` | Integer index starting at 0, increasing within each trial. Preserve gaps when known samples are lost. |
| `time_s` | Actual sample time on the common trial clock; strictly increasing within a trial. |
| `excitation_drive_V` | Sampled drive voltage at the measurement point documented in calibration. Blank if not measured. |
| `excitation_N` | Actual calibrated excitation force, if available. **Never copy drive volts into this column.** Blank when unknown, even if drive voltage is recorded. |
| `piezo_1_V` ... `piezo_4_V` | Raw ADC-input voltages after count-to-volt conversion, before digital filtering/gain removal. Document analog gain/filtering. Leave unconnected channels blank. |
| `quality_flag` | `ok`, `clipped`, `dropped`, or `suspect`; blank if quality was not assessed. A flag applies to the row; explain channel-specific issues in trial notes. |

Every recorded receiver must be mapped in the trial log. All populated channels
in a waveform row share a sample instant. If the DAQ scans channels sequentially,
record timing skew in calibration and account for it before phase analysis. Use
a separate capture stream/protocol if channels have unrelated clocks or rates.
The four slots are a starting format, not a requirement for four receivers.

Do not replace the waveforms with FFT peaks, averages, or RMS values. Preserve
dropped samples and timestamp gaps. Resampling and filtering belong in derived
analysis files, not in the original capture.

## `states.csv`: measured states and reference labels

Record at each sensor's actual update rate. Populate only quantities measured or
annotated at that row's time; leave the rest blank. Different sources may produce
rows at the same time. Sort by time within a trial. Do not repeat an old force
reading at every piezo sample or silently interpolate it in the raw recording.

| Column | Meaning |
| --- | --- |
| `trial_id` | Links to the trial log. |
| `time_s` | Measurement/event time in seconds on the common trial clock, not computer receipt time. |
| `source_id` | Device or annotation source, e.g. `object_force_sensor`, `distal_load_cell`, `joint_encoder`, or `camera_labels`. Multiple values in a row must come from the same synchronized acquisition source. |
| `contact_active` | Verified contact: `1` present, `0` absent, blank unknown. Use observation/reference instrumentation, not the future piezo prediction. |
| `contact_force_N` | Measured normal object-on-finger force, positive in compression. Preserve zero-offset noise rather than clamping it to zero. Document the sensor axis and frame. |
| `contact_segment` | `1` for proximal fixed segment; `2` for distal moving segment. |
| `contact_position_m` | Actual location along the specified finger segment's longitudinal axis: segment 1 from base toward joint; segment 2 from joint toward tip. |
| `tendon_contact_position_m` | For direct tendon contact only: arc length along the current routed tendon from the spool feed point. Independent of finger-surface position; blank unless measured. |
| `tension_N` | Distal load-cell measurement of tendon tension, positive in tension. |
| `joint_angle_rad` | Actual measured joint angle, zero straight and positive flexion. |
| `motor_command_rad` | Commanded motor/spool angle, positive winding, from a documented zero. Record when issued. |
| `motor_angle_rad` | Actual measured motor/spool angle using the same reference. Leave blank without feedback. |
| `quality_flag` | `ok`, `clipped`, `dropped`, or `suspect`; blank if unassessed. |

For confirmed no-contact rows, leave segment and contact-position fields blank.
Keep any real force sensor reading, including its noise. For contact with an
unknown location, leave that location blank; do not invent a location of zero.
If a fixture establishes one constant location throughout a hold, record the
verified location and its validity interval in trial notes. Camera-derived
locations should retain their own timestamps. Later alignment must document any
holding/interpolation and reject intervals where labels are unavailable.

## Using these files with the existing code

These are **raw collection tables**, not direct inputs to
`python -m tass --process-csv`. The existing
[`Measurement`](../tass_minimum/tass/data.py) interface expects one numeric table
with uniformly sampled `time_s`, finite receiver voltages, and `excitation_N`.

For subsequent analysis, select one trial and valid interval, select connected
receiver columns, align state labels to the waveform clock with documented gap
limits, and convert into a separate processed table. Preserve missing labels as
missing and exclude them from the corresponding training targets. If excitation
force is unavailable, the existing spectrum processor accepts an explicitly
documented zero excitation array to disable V/N FRF calculation; this is a
processing convention, **not a measured zero to write into the raw CSV**.

Use complete trial/session IDs when separating training and test sets. Keep
the `synthetic` examples out of experimental calibration and evaluation.
