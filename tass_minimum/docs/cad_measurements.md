# Finger app dimensions from TASS_assy_1.STEP

The interactive `finger_physical_app.py` uses these dimensions through
[`finger_parameters.py`](../../finger_parameters.py). Editing that file updates
the mechanics, graphic, routed acoustic length and sensor positions together.
The independent legacy CLI/JSON presets have not been changed.

## Source and coordinate system

- Source: `TASS_assy_1.STEP`, supplied from Downloads; STEP AP214, SolidWorks 2026.
- File header timestamp: `2026-10-13T03:11:52` (no timezone specified).
- SHA-256: `d0a21bf2ba93dc079dad2ad955082e35063429c0a2f111021fbadb03c5b0bd58`.
- Measurements use millimetres and assembled component transforms, extracted
  with Open CASCADE. The app converts them to metres. CAD tools are not app dependencies.
- Global hinge axis: parallel to CAD Y, through X = 1035.9124177,
  Z = 1186.5658126 mm. The selected finger plane is Y = 957.5192539 mm.
- Base datum: the base top at Z = 999.5658126 mm, directly below the hinge.
  Model +x = CAD +Z; model +y = CAD -X (toward the tendon).
- The assembly is bent approximately 82.26158 degrees. Distal measurements are
  rotated back to the straight pose, not taken from its global bounding box.
  Its local axial unit vector is (-0.9908931, 0, 0.1346506); its tendon-side
  vector is (-0.1346506, 0, -0.9908931).

## Active geometry

All coordinates below are **(axial, tendon-side) in mm**, in the straight pose.
Values are nominal CAD geometry, not measurements of manufacturing tolerances.

| Parameter | Value | Reference / basis |
| --- | --- | --- |
| Segment 1 length | 187 | Base top to hinge axis; excludes hinge's 3 mm outer radius |
| Segment 2 length | 187 | Hinge to end plane of segment body; excludes load cell |
| P1 | (31, 15.53607) | From base; average of installed lower tube pair centres |
| P2 | (147, 15.53607) | From base; average of installed upper tube pair centres |
| P3 | (40, 15.53607) | From hinge; inferred paired station at distal mounting holes |
| P4 | (164, 15.53607) | From hinge; inferred paired station at distal mounting holes |
| Load-cell attachment reference | (188, 16.68574) | From hinge; user-confirmed outer free-end hole, referenced at finger-facing surface |
| Fixed spring mount | (-23, -29.30940) | From hinge; transverse hole axis |
| Moving spring mount | (23, -29.30940) | From hinge; transverse hole axis |
| Spool centre | (39.110, 65) | From base; winding cylinder axis |
| Spool winding radius | 30 | Drum cylinder; outer flange radius is 33 |

**Stations:** the file contains four individual piezo/tube brackets, all on the
fixed segment. The user confirmed that each close pair is one TX/RX station
and that the corresponding two distal stations should be inferred from the
empty mounting holes. Proximal bracket centres are 23/39 and 139/155 mm from
the base; distal bracket centres are 32/48 and 156/172 mm from the hinge.
Reusing the same paired brackets gives the station centres above.

Within each installed pair the tube centres are 14.50638 mm apart axially.
Tubes are 12 mm long, with 2.6 mm outer diameter, 1.8 mm bore and opposite
5-degree tilts. The existing model still represents each pair by **one ideal
guide with collocated excitation and reception**, not two separate finite tubes.
These additional CAD dimensions are recorded here, but are not active constants
in the point-guide acoustic model.

**Spool routing:** `spool_position_m` is the drum centre.
`spool_feed_position_m` is derived as a tangent from the drum to P1 and is the
point passed to the shared tendon model. `spool_feed_branch = -1` selects the
lower exit at approximately (12.48810, 51.16979) mm; +1 selects the other
tangent. The lower branch is an assumed winding direction because no string
is included in the CAD. The free spool-to-P1 span is 40.15535 mm. Radius is
constant; wrap buildup and string thickness are not included.

At zero bend, P1–P2 is 116 mm, P2–P3 is 80 mm, P3–P4 is 124 mm, and
P4–load cell is approximately 24.02752 mm. Total ideal free route length is
384.18287 mm. Only P2–P3 changes with joint angle.

## Assumptions and measurements still needed

- No tendon or load-cell eyelet/knot is present. The user confirmed the outer
  free-end hole as the tendon attachment. The inner
  free-end hole is at (188, 6.63583) mm. Actual attachment hardware may shift
  the endpoint from the hole's finger-facing surface.
- No spring is included. The mounting-hole axes are projected into the same
  finger plane; hook offsets and any out-of-plane routing are omitted.
  **46 mm free length is an assumed zero-preload setup**, equal to straight
  mount separation, not a measured spring free length. The 500 N/m stiffness
  remains a demonstration value.
- Tendon E, area, damping, preload and effective compliant material length
  remain unmeasured. In particular, `tendon_rest_length_m = 0.250` is the
  assumed length used in EA/L0, **not** the CAD free route length. The app
  separately initializes its routed tendon free length from the geometry
  and specified preload. Replace EA/L0 or set the measured stiffness override
  when the actual string/compliant length has been characterized.
- Joint inertia/damping, 100-degree travel limit, motor response/limits,
  acoustic density/damping/gains and contact constants remain uncalibrated.
  The CAD's assembled bend angle is not a measured mechanical travel limit.
- The two-segment planar approximation remains in use. CAD body surfaces,
  collisions, guide friction and load-cell compliance are not simulated.

## Verification

Root tests check the CAD-derived straight spans, tangent orthogonality and
radius, an independent closed-form tendon/spring equilibrium, release, stops,
and app/acoustic integration. Run `python -m unittest discover -s tests -v`.
These are analytical/software checks, not experimental validation.
