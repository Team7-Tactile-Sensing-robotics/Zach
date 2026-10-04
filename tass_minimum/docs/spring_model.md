# Three-stage spring–tendon actuation

The main JSON preset now uses a geometric extension spring. Two attachment
points are defined in the same straight-reference coordinates as the tendon:
`spring.base_anchor_m` is fixed on segment 1; `spring.distal_anchor_m` rotates
with segment 2. These are editable demonstration dimensions, not measurements
inferred from the sketch. The spring constant is 250 N/m in this example.

`joint.equilibrium_rad` remains the straight **geometry reference angle**.
It is no longer necessarily the spring's relaxed position. The new
`spring.relaxed_angle_rad` is -10° in the default preset. At that angle, the
spring's natural length is calculated from the anchor separation. The configured
tendon rest length is the routed length at the same relaxed angle, giving zero
winding and zero initial tension. Changing geometry later does not silently
resize the physical tendon: update rest length explicitly.

## Equations

```
spring_length(q) = norm(rotated_distal_anchor(q) - base_anchor)
spring_rest_length = spring_length(relaxed_angle)
spring_extension = max(0, spring_length(q) - spring_rest_length)
spring_force = spring_stiffness * spring_extension
spring_resisting_torque = spring_force * d(spring_length)/dq

tendon_extension = tendon_path(q) - (tendon_rest_length - spool_radius*phi) - slack
T = max(0, k_tendon*tendon_extension + c_tendon*extension_rate) while taut

J*q_ddot = (-d(tendon_path)/dq)*T - spring_resisting_torque
           - joint_damping*q_dot - contact_force*contact_lever_arm
```

The spring's force and tendon tension are not equal: they act with different,
pose-dependent moment arms. At static equilibrium without external load:

```
T = spring_resisting_torque / (-d(tendon_path)/dq)
```

A stiffer spring therefore requires more tension to hold the same pose, and the
servo must wind slightly more to stretch the elastic tendon. Greater flexion
increases spring stretch for the chosen anchors. Tension is not universally
monotonic with angle: both moment arms change; inspect the operating range.
The extension spring can pull but cannot push. No spring initial tension,
hysteresis or coil dynamics is assumed. Joint damping remains a separate term.

## Default demonstration

| Stage | Joint angle | Spring extension | Spring force | Tendon tension | Spool angle |
| --- | --- | --- | --- | --- | --- |
| Initial | -10° | 0 mm | 0 N | 0 N | 0 rad |
| Ready | 0° | 4.510 mm | 1.127 N | 1.879 N | 0.1868 rad |
| Actuation | 40° | 19.199 mm | 4.800 N | 3.418 N | 1.1112 rad |

These are static operating points for the CLI preset, with zero external
contact force. Dynamic loading has transients. The app retains its own tendon
stiffness and spool radius, so its required motor angles can differ.

From `Zach/`:

```bash
python -m tass.spring_stages --config tass_minimum/config/default.json --output outputs/spring_stages
python -m tass --scenario coupled --output outputs/spring_actuation
python -m streamlit run tass_finger_dynamics_app.py
```

`spring_stages` writes a three-panel `stages.png`, numerical `stages.json`, and
configuration snapshot. The coupled demo starts at the relaxed angle with zero
winding, then winds toward the 40° operating point. Its default external force
is now zero so the spring contribution is visible. Baseline excitation starts
at the ready equilibrium using `experiment.baseline_motor_angle_rad`.
Mechanical CSVs include spring length, extension, force and resisting torque.

## Spool coordinates

Zach's internal **x axis is along the straight finger**. The upright drawing
swaps x and y. Therefore a spool raised 30 mm vertically in that drawing uses
`spool_position_m: [0.03, 0.0]`. A physical (horizontal, height) coordinate of
(0, 0.03) maps to that stored value. Literal internal [0, 0.03] would move it
sideways in the upright drawing. The preset uses the raised interpretation;
edit the coordinate if a different origin/axis convention is intended.

The active preset also carries the 15-mm guide offsets from the current tendon
constructor. `Tendon.from_config` uses the JSON values; direct `Tendon()` calls
retain their independent constructor defaults. The motor coordinate represents
an effective tendon feed point, not a detailed moving spool tangency model.

## Compatibility and datasets

`spring.mode="torsional"` retains the previous equivalent torsional spring:
`torque = joint.stiffness_Nm_rad * (q - relaxed_angle)`. Omitted spring settings
use that mode, with relaxed angle falling back to the geometry reference.
The source-tendon comparison preset retains its earlier spring choice.
In extension mode, `joint.stiffness_Nm_rad` is not an additional spring torque.

The app, dynamic mechanics, and static dataset generator all use the shared
`models/spring.py`. The dataset's static balance now uses this spring torque,
not a separate hard-coded torsional equation. Existing `outputs/dataset_v1`
files were NOT overwritten: they represent the old model and retain their
original config snapshot. Generate a new version to study the extension spring.
The acoustic P4 endpoint limitation is unchanged by this mechanical revision.

The shared implementation is `ReturnSpring(config)`; there is no
`LinearReturnSpring` class. Configuration stores the `Spring` parameter
record in `ModelConfig.spring`, not a constructed mechanical model.
`spring.free_length_m = null` derives the natural length from the relaxed
pose, preserving the three-stage model. An explicit positive free length
supports measured springs and preload in the physical app. The physical
app converts its joint-relative bracket offsets to world coordinates before
constructing `ReturnSpring`; applied joint torque is the negative of
`resisting_torque()`. `attachment_positions()` supplies the drawing geometry.
