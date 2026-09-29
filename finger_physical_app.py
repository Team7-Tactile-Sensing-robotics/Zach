"""Interactive, single-input finger mechanics. No acoustic simulation.

Run from the repository root:
    streamlit run finger_physical_app.py

Input: commanded motor/spool angle (0..180 degrees).
States: actual spool angle, finger angle and finger angular velocity.
The fixed demonstration parameters are collected in FingerParameters. The
existing routed Tendon supplies stretch, tension and geometry-dependent torque.
Importing this file does not launch the UI; FingerModel can be used on its own.
"""
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from tass_minimum.tass.models.tendon import Tendon


@dataclass(frozen=True)
class FingerParameters:
    proximal_length_m: float = 0.15
    distal_length_m: float = 0.10
    spool_radius_m: float = 0.006
    joint_inertia_kg_m2: float = 2e-4
    joint_damping_Nm_s_rad: float = 0.002
    return_spring_Nm_rad: float = 0.080
    tendon_stiffness_N_m: float = 2500.0
    tendon_damping_Ns_m: float = 0.1
    tendon_slack_m: float = 0.0
    tendon_preload_N: float = 0.0
    servo_time_constant_s: float = 0.10
    servo_speed_limit_deg_s: float = 300.0
    motor_limit_deg: float = 180.0
    joint_limit_deg: float = 100.0
    max_step_s: float = 0.0005


@dataclass
class FingerState:
    time_s: float = 0.0
    motor_angle_rad: float = 0.0
    joint_angle_rad: float = 0.0
    joint_velocity_rad_s: float = 0.0


class FingerModel:
    """Stateful finger driven only by a motor position command.

    phi_dot = clip((u-phi)/tau, -speed_limit, speed_limit)
    J*q_ddot = -T*L'(q) - k_s*q - b*q_dot

    T follows the shared unilateral elastic/damped tendon law. Joint stops at
    0 and joint_limit_deg remove outward velocity and support outward torque.
    There is no imposed joint angle, external load, gravity or acoustic forcing.
    """

    def __init__(self, parameters=None):
        self.parameters = parameters or FingerParameters()
        p = self.parameters
        self.tendon = Tendon(
            k=p.tendon_stiffness_N_m, c=p.tendon_damping_Ns_m,
            slack=p.tendon_slack_m, preload=p.tendon_preload_N,
            joint_position=(p.proximal_length_m, 0.0),
            p1=(p.proximal_length_m*4/15, 0.005),
            p2=(p.proximal_length_m*0.8, 0.005),
            p3=(p.proximal_length_m + p.distal_length_m*0.25, 0.005),
            p4=(p.proximal_length_m + p.distal_length_m*0.8, 0.005),
        )
        self.state = FingerState()
        self.motor_target_rad = 0.0

    def set_motor_target(self, degrees):
        """Set the one model input; retain every physical state."""
        if not math.isfinite(degrees):
            raise ValueError("Motor command must be finite")
        degrees = min(self.parameters.motor_limit_deg, max(0.0, degrees))
        self.motor_target_rad = math.radians(degrees)

    def _forces(self):
        p, s = self.parameters, self.state
        limit = math.radians(p.servo_speed_limit_deg_s)
        motor_speed = min(limit, max(-limit, (self.motor_target_rad-s.motor_angle_rad)/p.servo_time_constant_s))
        extension, extension_rate, tension = self.tendon.elastic_state(
            s.joint_angle_rad, s.joint_velocity_rad_s,
            p.spool_radius_m*s.motor_angle_rad, p.spool_radius_m*motor_speed,
        )
        moment_arm = -float(self.tendon.path_length_derivative(s.joint_angle_rad))
        tendon_torque = moment_arm*float(tension)
        spring_torque = -p.return_spring_Nm_rad*s.joint_angle_rad
        damping_torque = -p.joint_damping_Nm_s_rad*s.joint_velocity_rad_s
        net_torque = tendon_torque + spring_torque + damping_torque
        at_lower_stop = s.joint_angle_rad <= 0.0 and s.joint_velocity_rad_s <= 0.0
        at_upper_stop = (s.joint_angle_rad >= math.radians(p.joint_limit_deg)
                         and s.joint_velocity_rad_s >= 0.0)
        reaction = -net_torque if ((at_lower_stop and net_torque < 0)
                                  or (at_upper_stop and net_torque > 0)) else 0.0
        return {
            "motor_velocity_rad_s": motor_speed,
            "extension_m": float(extension),
            "extension_velocity_m_s": float(extension_rate),
            "tension_N": float(tension),
            "moment_arm_m": moment_arm,
            "tendon_torque_Nm": tendon_torque,
            "spring_torque_Nm": spring_torque,
            "damping_torque_Nm": damping_torque,
            "stop_reaction_Nm": reaction,
            "joint_acceleration_rad_s2": (net_torque+reaction)/p.joint_inertia_kg_m2,
        }

    def advance(self, duration_s):
        """Advance from the current state using small semi-implicit Euler steps."""
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("Elapsed simulation time must be finite and nonnegative")
        if duration_s == 0:
            return
        p, s = self.parameters, self.state
        steps = math.ceil(duration_s/p.max_step_s)
        dt = duration_s/steps
        upper_stop = math.radians(p.joint_limit_deg)
        for _ in range(steps):
            forces = self._forces()
            s.motor_angle_rad += forces["motor_velocity_rad_s"]*dt
            s.joint_velocity_rad_s += forces["joint_acceleration_rad_s2"]*dt
            s.joint_angle_rad += s.joint_velocity_rad_s*dt
            if s.joint_angle_rad <= 0:
                s.joint_angle_rad = 0.0
                s.joint_velocity_rad_s = max(0.0, s.joint_velocity_rad_s)
            elif s.joint_angle_rad >= upper_stop:
                s.joint_angle_rad = upper_stop
                s.joint_velocity_rad_s = min(0.0, s.joint_velocity_rad_s)
        s.time_s += duration_s

    def readouts(self):
        """All displayed quantities evaluated at the same current state."""
        s, p = self.state, self.parameters
        values = self._forces()
        spans = self.tendon.path_segments(s.joint_angle_rad)
        tip = self.tendon.joint_position + p.distal_length_m*np.array([
            math.cos(s.joint_angle_rad), math.sin(s.joint_angle_rad),
        ])
        values.update({
            "time_s": s.time_s,
            "motor_target_deg": math.degrees(self.motor_target_rad),
            "motor_angle_deg": math.degrees(s.motor_angle_rad),
            "motor_velocity_deg_s": math.degrees(values["motor_velocity_rad_s"]),
            "joint_angle_deg": math.degrees(s.joint_angle_rad),
            "joint_velocity_deg_s": math.degrees(s.joint_velocity_rad_s),
            "joint_acceleration_deg_s2": math.degrees(values["joint_acceleration_rad_s2"]),
            "wound_length_m": p.spool_radius_m*s.motor_angle_rad,
            "path_length_m": float(sum(spans.values())),
            "shortening_m": float(self.tendon.joint_displacement(s.joint_angle_rad)),
            "span_1_m": float(spans["p1_p2"]),
            "span_2_m": float(spans["p3_p4"]),
            "span_3_m": float(spans["p2_p3"]),
            "tip_x_m": float(tip[0]),
            "tip_y_m": float(tip[1]),
        })
        return values


def finger_svg(model):
    """Draw the current mechanics in a fixed viewport, with the finger upright.

    Display axes swap the model's x/y coordinates. Guide positions, span lengths
    and the distal segment all come from the same geometry used in dynamics.
    """
    s, p, tendon = model.state, model.parameters, model.tendon
    r = model.readouts()
    scale, base_x, base_y = 1550.0, 235.0, 475.0

    def screen(point):
        x, y = point
        return base_x+scale*y, base_y-scale*x

    def line(a, b, color, width=3, extra=""):
        x1, y1 = screen(a)
        x2, y2 = screen(b)
        return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" {extra}/>'

    def text(x, y, label, color="#a7b5c8", size=12, extra=""):
        return f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size}" {extra}>{label}</text>'

    guides = tendon.guide_positions(s.joint_angle_rad)
    joint = tendon.joint_position
    tip = joint + p.distal_length_m*np.array([math.cos(s.joint_angle_rad), math.sin(s.joint_angle_rad)])
    jx, jy = screen(joint)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 560" role="img" aria-labelledby="finger-title finger-description" style="width:100%;height:auto;display:block">',
        '<title id="finger-title">Live tendon-driven finger</title>',
        '<desc id="finger-description">Two rigid finger segments, four routing points, a motor spool and three tendon spans. The span crossing the joint shortens as the finger bends.</desc>',
        '<defs><pattern id="grid" width="31" height="31" patternUnits="userSpaceOnUse"><path d="M31 0H0V31" fill="none" stroke="#1c2a3e" stroke-width="0.7"/></pattern></defs>',
        '<rect width="720" height="560" rx="18" fill="#0e1728"/>',
        '<rect x="16" y="16" width="688" height="528" rx="12" fill="url(#grid)"/>',
        text(35, 42, 'LIVE MECHANICAL MODEL', '#edf4ff', 14, 'font-weight="700" letter-spacing="1"'),
        text(35, 64, f'Simulation time {r["time_s"]:.2f} s'),
        # Straight-pose reference and mechanical links.
        line(joint, joint+np.array([p.distal_length_m, 0]), '#42516a', 2, 'stroke-dasharray="5 6"'),
        line(np.array([0, 0]), joint, '#3c516f', 29),
        line(joint, tip, '#738cab', 26),
        line(np.array([0, 0]), joint, '#8294af', 1, 'stroke-dasharray="3 8"'),
        f'<circle cx="{jx}" cy="{jy}" r="20" fill="#142136" stroke="#a6bedc" stroke-width="2"/>',
    ]
    # A small torsion-spring symbol at the joint, anchored to the proximal link.
    spring = []
    for angle in np.linspace(0, 4*math.pi, 90):
        radius = 4 + angle/(4*math.pi)*10
        spring.append(f'{jx+radius*math.cos(angle):.2f},{jy+radius*math.sin(angle):.2f}')
    parts.append(f'<polyline points="{" ".join(spring)}" fill="none" stroke="#f5cd78" stroke-width="1.7"/>')
    colors = ['#8191a9', '#4cd5ca', '#ffb667', '#b69aff']
    points = [tendon.spool_position, *guides]
    for i, color in enumerate(colors):
        extra = 'stroke-dasharray="4 5"' if r['tension_N'] < 1e-6 else ''
        parts.append(line(points[i], points[i+1], color, 3.5, extra))
    for i, guide in enumerate(guides, start=1):
        x, y = screen(guide)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#0e1728" stroke="#eff5ff" stroke-width="2"/>')
        parts.append(text(x+11, y+4, f'P{i}' + (' · anchor' if i == 4 else ''), '#eff5ff', 11))
    # Spool at its actual fixed feed point; the spoke rotates with actual phi.
    sx, sy = screen(tendon.spool_position)
    radius = scale*p.spool_radius_m
    parts.append(f'<rect x="{sx-31}" y="{sy-23}" width="62" height="46" rx="10" fill="#25354e" stroke="#60728c"/>')
    parts.append(f'<circle cx="{sx}" cy="{sy}" r="{radius}" fill="#15243a" stroke="#4cd5ca" stroke-width="2"/>')
    parts.append(f'<line x1="{sx}" y1="{sy}" x2="{sx+radius*math.sin(s.motor_angle_rad):.2f}" y2="{sy-radius*math.cos(s.motor_angle_rad):.2f}" stroke="#eff5ff" stroke-width="2"/>')
    parts.extend([
        text(35, 496, 'MOTOR / SPOOL', '#eff5ff', 12, 'font-weight="600"'),
        text(35, 515, f'Actual {r["motor_angle_deg"]:.1f}° · target {r["motor_target_deg"]:.0f}°'),
        text(65, 352, 'SEGMENT 1', '#eff5ff', 12, 'font-weight="600"'),
        text(65, 371, f'Fixed · {p.proximal_length_m*1000:.0f} mm'),
        text(390, 93, 'SEGMENT 2', '#eff5ff', 12, 'font-weight="600"'),
        text(390, 112, f'Rotates · {p.distal_length_m*1000:.0f} mm'),
        text(35, 198, 'FINGER ANGLE', '#a7b5c8', 11),
        text(35, 231, f'{r["joint_angle_deg"]:.1f}°', '#eff5ff', 29, 'font-weight="700"'),
        text(35, 253, 'Return spring at joint', '#f5cd78', 11),
    ])
    # Angle arc in the flexion direction, independent of the tendon line.
    if s.joint_angle_rad > 0.005:
        arc_radius = 49
        end_x = jx+arc_radius*math.sin(s.joint_angle_rad)
        end_y = jy-arc_radius*math.cos(s.joint_angle_rad)
        parts.append(f'<path d="M {jx} {jy-arc_radius} A {arc_radius} {arc_radius} 0 0 1 {end_x:.2f} {end_y:.2f}" fill="none" stroke="#eff5ff" stroke-width="1.5"/>')
    # Fixed-position key keeps readouts clear at every pose.
    for index, (label, length, color) in enumerate([
        ('Span 1 · along segment 1', r['span_1_m'], colors[1]),
        ('Span 2 · along segment 2', r['span_2_m'], colors[3]),
        ('Span 3 · across joint', r['span_3_m'], colors[2]),
    ]):
        y = 373+index*42
        parts.append(f'<rect x="438" y="{y-8}" width="20" height="3" rx="1" fill="{color}"/>')
        parts.append(text(470, y, label, color, 11))
        parts.append(text(470, y+16, f'{length*1000:.2f} mm', '#eff5ff', 13))
    parts.append(text(438, 513, 'Solid = loaded · dashed = unloaded', '#a7b5c8', 10))
    parts.append('</svg>')
    return ''.join(parts)


def run_app():
    import streamlit as st

    st.set_page_config(page_title="TASS · Finger mechanics", page_icon="🦾", layout="wide")
    st.title("Finger mechanics")
    st.caption("One motor input · one moving joint · three routed tendon spans")

    def reset_model():
        st.session_state.finger_model = FingerModel()
        st.session_state.motor_target_deg = 0.0
        st.session_state.finger_paused = False
        st.session_state.finger_last_tick = time.monotonic()
        st.session_state.finger_history = deque(maxlen=240)

    if "finger_model" not in st.session_state:
        reset_model()

    def apply_target():
        st.session_state.finger_model.set_motor_target(st.session_state.motor_target_deg)

    def change_target(delta=None, target=None, hold=False):
        model = st.session_state.finger_model
        if hold:
            target = math.degrees(model.state.motor_angle_rad)
        elif delta is not None:
            target = st.session_state.motor_target_deg+delta
        st.session_state.motor_target_deg = float(np.clip(target, 0, model.parameters.motor_limit_deg))
        apply_target()

    def toggle_pause():
        st.session_state.finger_paused = not st.session_state.finger_paused
        st.session_state.finger_last_tick = time.monotonic()

    @st.fragment(run_every=0.1)
    def live_panel():
        model = st.session_state.finger_model
        now = time.monotonic()
        elapsed = now-st.session_state.finger_last_tick
        st.session_state.finger_last_tick = now
        if not st.session_state.finger_paused:
            # Bound catch-up work after an inactive browser tab. The simulation
            # clock shows time actually integrated, never discarded wall time.
            model.advance(min(0.1, max(0.0, elapsed)))
        r = model.readouts()

        controls, diagram, states = st.columns([1.05, 2.15, 1.0], gap="large")
        with controls:
            st.subheader("Motor control")
            st.slider("Commanded motor angle (°)", 0.0, 180.0, step=1.0,
                      key="motor_target_deg", on_change=apply_target)
            a, b = st.columns(2)
            a.button("−10° Unwind", key="unwind", on_click=change_target,
                     kwargs={"delta": -10.0}, use_container_width=True)
            b.button("+10° Wind", key="wind", type="primary", on_click=change_target,
                     kwargs={"delta": 10.0}, use_container_width=True)
            st.button("Release to 0°", key="release", on_click=change_target,
                      kwargs={"target": 0.0}, use_container_width=True)
            st.button("Hold motor here", key="hold", on_click=change_target,
                      kwargs={"hold": True}, use_container_width=True)
            st.caption("The slider and buttons set the same input: motor position. The finger angle follows from tendon and spring forces.")
            st.divider()
            st.button("Resume simulation" if st.session_state.finger_paused else "Pause simulation",
                      key="pause", on_click=toggle_pause, use_container_width=True)
            st.button("Reset model", key="reset", on_click=reset_model, use_container_width=True)
            st.caption("Pause freezes simulation time. Hold keeps the current motor position while the finger settles.")

        with diagram:
            st.markdown(finger_svg(model), unsafe_allow_html=True)
            motor_settled = abs(r['motor_target_deg']-r['motor_angle_deg']) < 0.05
            finger_settled = abs(r['joint_velocity_deg_s']) < 0.1
            motion = 'Paused' if st.session_state.finger_paused else ('Holding' if motor_settled and finger_settled else 'Moving')
            tendon_state = 'loaded' if r['tension_N'] > 1e-6 else ('slack' if r['extension_m'] < -1e-8 else 'unloaded')
            st.caption(f"{motion} · tendon {tendon_state} · model time {r['time_s']:.2f} s")

        with states:
            st.subheader("Current state")
            st.metric("Motor angle", f"{r['motor_angle_deg']:.1f}°")
            st.caption(f"Command: {r['motor_target_deg']:.1f}° · speed: {r['motor_velocity_deg_s']:.1f}°/s")
            st.metric("Finger angle", f"{r['joint_angle_deg']:.1f}°")
            st.caption(f"Speed: {r['joint_velocity_deg_s']:.1f}°/s")
            st.metric("Tendon tension", f"{r['tension_N']:.2f} N")
            st.metric("Tendon stretch", f"{max(0, r['extension_m'])*1000:.2f} mm")
            st.metric("Tendon torque", f"{r['tendon_torque_Nm']:.3f} N·m")

        history = st.session_state.finger_history
        if not history or r['time_s'] > history[-1]['Time (s)']:
            history.append({'Time (s)': r['time_s'], 'Motor command (°)': r['motor_target_deg'],
                            'Motor angle (°)': r['motor_angle_deg'], 'Finger angle (°)': r['joint_angle_deg']})
        with st.expander("Motion history"):
            st.line_chart(list(history), x='Time (s)',
                          y=['Motor command (°)', 'Motor angle (°)', 'Finger angle (°)'], height=220)
        with st.expander("All state readouts"):
            st.table([
                {"Quantity": label, "Current value": f"{value:.4f}", "Unit": unit}
                for label, value, unit in [
                    ('Simulation time', r['time_s'], 's'),
                    ('Motor command', r['motor_target_deg'], 'deg'),
                    ('Actual motor angle', r['motor_angle_deg'], 'deg'),
                    ('Motor speed', r['motor_velocity_deg_s'], 'deg/s'),
                    ('Finger angle', r['joint_angle_deg'], 'deg'),
                    ('Finger speed', r['joint_velocity_deg_s'], 'deg/s'),
                    ('Finger acceleration', r['joint_acceleration_deg_s2'], 'deg/s²'),
                    ('Tendon tension', r['tension_N'], 'N'),
                    ('Wound tendon length', r['wound_length_m']*1000, 'mm'),
                    ('Signed tendon extension', r['extension_m']*1000, 'mm'),
                    ('Extension rate', r['extension_velocity_m_s']*1000, 'mm/s'),
                    ('Route shortening', r['shortening_m']*1000, 'mm'),
                    ('Total route length', r['path_length_m']*1000, 'mm'),
                    ('Span 1 length', r['span_1_m']*1000, 'mm'),
                    ('Span 2 length', r['span_2_m']*1000, 'mm'),
                    ('Span 3 length', r['span_3_m']*1000, 'mm'),
                    ('Effective moment arm', r['moment_arm_m']*1000, 'mm'),
                    ('Tendon torque', r['tendon_torque_Nm'], 'N·m'),
                    ('Return spring torque', r['spring_torque_Nm'], 'N·m'),
                    ('Joint damping torque', r['damping_torque_Nm'], 'N·m'),
                    ('Stop reaction torque', r['stop_reaction_Nm'], 'N·m'),
                    ('Tip x (model frame)', r['tip_x_m']*1000, 'mm'),
                    ('Tip y (model frame)', r['tip_y_m']*1000, 'mm'),
                ]
            ])

    live_panel()
    with st.expander("Fixed physical model"):
        st.write("A servo winds the tendon around a 6 mm spool. Two rigid finger segments are 150 mm and 100 mm long. P1–P3 guide the sliding tendon; P4 anchors it to the moving segment. The 80 mm and 55 mm link spans stay fixed, while the span across the joint changes length.")
        st.latex(r"\delta = r_s\phi + L(\theta)-L(0),\quad \tau_t=-T\,L'(\theta)")
        st.latex(r"J\ddot\theta=\tau_t-k_s\theta-b\dot\theta,\qquad u(t)=\phi_{\mathrm{command}}(t)")
        st.caption("Demonstration values: tendon stiffness 2500 N/m, tendon damping 0.1 N·s/m, return spring 0.080 N·m/rad, joint damping 0.002 N·m·s/rad, inertia 0.0002 kg·m². Joint travel 0–100°; motor travel 0–180°; servo time constant 0.10 s and speed limit 300°/s. Initial slack and preload are zero. Parameters are fixed in FingerParameters at the top of this file.")
        st.caption("Frictionless guides and a straight, unobstructed joint span are assumed. No external load or gravity is applied. Tendon tension cannot be negative. No acoustic model runs in this app.")


if __name__ == "__main__":
    run_app()
