# Verification record

Routed tendon update checked on 2026-09-24. Tests use Python's built-in
`unittest` runner: `python -m unittest discover -s tests -v`.

All 23 tests pass. A full CLI coupled run with `config/default.json` also
completed successfully. The Streamlit app was checked at 0°, 90°, and 180° motor
commands, including its guide-route schematic and operating-point table.
At the app's default settings, 90° motor rotation settles at 44.2° finger flexion
and 4.15 N tendon tension; 180° settles at 73.6° and 5.07 N.

The committed `test-results.txt` and output plots are historical records from
2026-09-23, before routed mechanics replaced the constant moment arm. The acoustic
comparisons below remain applicable to those saved constant-tension runs.

## Analytical comparisons

The saved baseline and contact examples use 81 spatial nodes, a 48 kHz output
clock, a 0.25 s record and a 0.002 N smooth pulse lasting 0.5 ms.

| Condition | Analytical frequency | Numerical FFT peak | FFT spacing |
| --- | ---: | ---: | ---: |
| No contact, mode 1 | 186.34 Hz | 188 Hz | 4 Hz |
| No contact, mode 2 | 372.68 Hz | 372 Hz | 4 Hz |
| No contact, mode 3 | 559.02 Hz | 560 Hz | 4 Hz |
| 40% contact, left-span mode 1 | 465.85 Hz | 464 Hz | 4 Hz |

The reference is `f_n = n/(2*span)*sqrt(T/mu)`. Discrete mesh dispersion and FFT
bin width explain the small differences. Frequency tolerance for free-ringdown
tests is 5 Hz for the full span and 7.5 Hz for the shorter contact spans. The
20% span is represented by only 16 intervals, so its spatial error is larger.

The baseline's strongest peak is near mode 2. Under the rigid-contact example,
receiver 2 is exactly silent: it is on the opposite span from the exciter, with
zero initial displacement and velocity. Its dominant frequency and channel ratio
are correctly undefined.

## Test coverage

- Equilibrium with zero tension and zero load.
- Convergence to the nonlinear balance `-T*L'(theta) = k_joint*(theta-theta_0)`
  with measured tension, using an independent closed-form bridge geometry.
- Convergence with opposing contact torque `r_contact*F_contact`.
- Coupled elastic-tendon static equilibrium and nonnegative tension.
- Slack tendon cannot exert compressive force.
- Fixed segment spans and shortening of the connecting span at 0°, 45°, and 90°.
- Analytical path derivative against finite differences and torque against
  the elastic energy gradient (virtual work).
- Stretch rate, slack engagement, preload, unloading, and coordinate invariance.
- Measured and custom tension laws both use the routed joint moment arm.
- Routing configuration roundtrip, legacy-field warning, invalid geometry,
  reference-pose world coordinates, and mechanical output channels.
- Full-span fundamental obtained from a numerical ringdown FFT.
- Frequency ratio when tension changes from 2 N to 8 N.
- Contact positions at 20% and 40% of length, including exact zero motion at
  the clamp and isolation of the unexcited span.
- The right-hand contact span's frequency under its own initial ringdown.
- Mesh-independent integrated impulse for a point force.
- No spontaneous motion in the full zero-state pipeline.
- CSV roundtrip into the same processor used by simulation.
- Known FFT amplitude, Parseval energy and transfer ratio for a sinusoid.
- Undefined FRF for unexcited bins and undefined frequency for silent sensors.
- Rejection of invalid physical parameters, negative measured tension and
  nonuniform FFT timestamps.

## Numerical and physical limits

The output clock and string integration timestep are distinct. The string solver
subdivides each sample as needed using a conservative RK4 stability bound. This
protects integration stability; it does not remove mesh dispersion or replace
an output antialias filter. If the highest mesh mode exceeds output Nyquist,
the code emits a warning. Refine the mesh and clock for your frequency band of
interest and compare results before drawing quantitative conclusions.

The coupled example is a changing operating condition, so its whole-record FFT
is illustrative. The placeholder piezo gains and material parameters have not
been fit to a real prototype. Force/location inference and experimental accuracy
are not validated by these tests.
