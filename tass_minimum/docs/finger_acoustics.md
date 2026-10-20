# Interactive finger acoustics

Run `python -m streamlit run finger_physical_app.py` from the repository root.
The acoustic section uses the physical app's current tendon tension and routing.
It does not read the separate CLI JSON configuration.

## Controls and outputs

1. Wind the motor until the tendon is loaded. For a stationary experiment, let
   the finger settle and pause the mechanical simulation.
2. Enable any combination of **Excite P1**, **P2**, **P3**, and **P4**. Set each
   sine-wave frequency and peak force independently. Force is a demonstration
   mechanical input, not an inferred piezo drive voltage.
3. Enable **Run acoustic model**. Read all four receivers or select which traces
   to show. Disabling a receiver trace only hides it; all receivers keep recording.
4. Turn transmitters off to watch the remaining vibration decay. Turning off
   **Run acoustic model** freezes the acoustic clock and stored state instead.
   **Reset acoustic response** clears vibration and history but keeps the controls.
   **Reset model** resets both systems and the acoustic controls.
5. Enable **Apply contact** and adjust **Contact position (% from spool)** and
   **Contact force (N)**. The position spans the entire routed string, including
   the tail past P4: 0% is the spool and 100% is the load-cell anchor. Its distance
   in millimeters and a colored arrow appear on the acoustic graphic. The default
   force range is 0–5 N; edit `acoustic_contact_max_force_N` to change it.
   Compare receiver waveforms, RMS values, spectra and model frequencies as you
   change force or location. Zero force or disabling contact removes the support.
   Contact at either already-fixed endpoint has no additional effect.

The graphic shows instantaneous transverse deflection along the unfolded tendon,
with a fixed magnification from `acoustic_deflection_magnification`. The graphic
explicitly flags clipped deflection; underlying signals are not clipped.
These are slow, stroboscopic shape snapshots: use the full-rate waveform traces
to inspect oscillation timing. The displayed length/pose/tension are the last
operating point used by the acoustic solver, including while paused.

Waveform charts show the most recent 20 ms at full sample resolution. Spectra
use the rolling history and the shared Hann-window FFT implementation. RMS is
computed over the latest block. Dominant spectral peaks can reflect excitation,
ringdown, or changes of pose, and are not automatically natural frequencies.
The theoretical fundamental and retained mode count are displayed separately.

## Physical approximation

The shared `Tendon` determines the full path and cumulative guide positions:

`spool -> P1 -> P2 -> P3 -> P4 -> load cell`

The spool and load-cell attachment impose zero transverse displacement. P4 is
inside the span when the configured load-cell tail is nonzero. Each guide has
an ideal collocated transverse point-force actuator and displacement receiver.
Intermediate guides add no clamps, reflections, masses, or friction in this first
approximation. Real guide impedance and piezo transfer functions require calibration.

With path length L, guide coordinates s_i, mass per length mu, tension T, and
modal damping ratio zeta, the displacement is

`y(s,t) = sum_n a_n(t) sin(n*pi*s/L)`

and each physical modal coordinate obeys

`a_n'' + 2*zeta*omega_n*a_n' + omega_n^2*a_n = 2/(mu*L) * sum_i F_i(t)*sin(n*pi*s_i/L)`

`omega_n = n*pi/L * sqrt(T/mu)`

The factor `mu*L/2` is modal mass. Receiver output is
`V_i(t) = acoustic_receiver_gains_V_m[i] * y(s_i,t)`.
These gains are placeholders. No piezo electronics, direct electrical crosstalk,
ADC saturation or measurement noise is included.

### Adjustable held contact

Contact is a local compliant support for small vibrations about a held, pressed
equilibrium. For prescribed normal pressing force F, it adds tangent stiffness
`k_c = acoustic_contact_stiffness_per_N * F` and damping
`c_c = acoustic_contact_damping_per_N * F`. These demonstration laws require
experimental calibration; force is separate from the axial tendon tension T.
Defaults and all contact constants live in `finger_parameters.py`.

At contact coordinate s_c, let `b_n = sin(n*pi*s_c/L)` and `m = mu*L/2`.
The coupled modal matrices are

`K = diag(omega_n^2) + (k_c/m) b b^T`

`C = diag(2*zeta*omega_n) + (c_c/m) b b^T`

The solver advances `a'' + C a' + K a = modal_drive` with an exact held-input
matrix exponential. The reported natural frequencies come from K; receiver
signals include contact damping and mode coupling. This allows continuous
force-dependent responses rather than a binary hard clamp. The contact tracks
a fraction of the current routed length as the finger bends. Moving it or
removing it preserves the modal state, except for the bandwidth projection
described below. Abrupt changes of support can produce transients.

This approximation does not calculate static indentation, impact, slip,
contact-induced axial stretch, or a joint reaction torque. The selected force
changes acoustic support only; it does not change the finger's mechanical pose
or load-cell tension. The graphic shows vibration about equilibrium, not the
static deflection caused by pressing. It is a forward simulation, not a force
estimator from measured receiver data.

`FingerAcoustics` in `finger_acoustics.py` retains modal displacement, velocity,
oscillator phase and recent samples. It reuses the exact matrix-exponential /
held-input integration method from `tass/static_vibration.py`. Sine forces are
sampled and held over each ADC-sized interval. State persists when switching
transmitters, adjusting frequencies/amplitudes, or advancing to another block.
This physical transient model is separate from the older empirical arbitrary-unit
frequency-domain `ModalString` sweep; their amplitudes are not interchangeable.

## Timing and validity limits

Defaults in `finger_parameters.py` are 24 kHz sampling, 16 modes, 10 ms of acoustic
time per screen refresh, and 200 ms of retained history. Refreshes occur about
every 500 ms (`acoustic_refresh_interval_s`), independently of the mechanical
display's 100 ms refresh. This makes acoustic playback slower than mechanics.
Widget interactions can also trigger a refresh/block. The displayed acoustic
clock reports integrated time; it is not a synchronized hardware timestamp.

Geometry and tension are frozen within each block and refreshed from the finger
before the next. Modal coordinates persist in the instantaneous sine basis.
Terms due to moving supports, axial transport, and time derivatives of that
basis are omitted. This is a slow-geometry preview, not a fully coupled
moving-string solver. For quantitative comparison, use a held, settled pose.
Acoustic forces do not feed back into the finger mechanics.

Modes at or above 40% of sampling frequency are excluded, with their state
discarded, to keep modeled resonances below Nyquist with a margin. Drives must
also stay below that limit. Modal truncation is not a hardware anti-alias filter.
With contact enabled, the cutoff is applied to the coupled eigenmodes and state
is projected into the retained basis at integration time.
When T is below `acoustic_min_tension_N`, the model clears modal state and emits
zero receiver response with an explicit unsupported/slack status. It does not
predict the motion of a slack string. Existing history remains visible until
it ages out, so a recent spectrum can include earlier loaded states.

Because several independently driven inputs may be active, the UI reports
receiver voltage spectra rather than a scalar `Y/U` force-response function.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The acoustic tests check the oscillator against an analytical step response,
continuity across blocks, independent-input superposition, P4 excitation with
the real anchor beyond it, fixed-end displacement, damping energy decay,
tension/length frequency scaling, receiver gain scaling, FFT peaks, history
timing, slack handling and bandwidth limits. Contact checks cover the analytical
single-mode point-spring frequency, position/force sensitivity, dissipative
ringdown, release continuity, and zero-force/fixed-end equivalence. Streamlit
AppTest exercises exciter/frequency and contact controls, receiver selection,
pause, ringdown and resets.
These checks establish implementation consistency, not experimental accuracy.
