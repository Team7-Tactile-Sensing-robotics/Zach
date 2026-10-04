
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import brentq
from tass_minimum.tass.models.tendon import Tendon
from tass_minimum.tass.models.spring import LinearReturnSpring

# ============================================================
# TASS Single-Joint Tendon-Driven Finger Dynamics Demonstrator
# ============================================================
#
# Model:
#   Joint dynamics:
#       J*theta_ddot + b*theta_dot = -L'(theta)*T + tau_s - tau_ext
#       tau_s = -k_s*max(spring_length - free_length, 0)*spring_length'(theta)
#
#   Tendon extension:
#       delta_t = r_s*phi + L(theta) - L(0) - slack + preload/k_t
#
#   Tendon tension:
#       T = max(k_t*delta_t + c_t*delta_dot, 0) if delta_t > 0 else 0
#
#   Servo motion:
#       phi_dot = clip((phi_cmd - phi)/tau_servo,
#                      -omega_servo_max, omega_servo_max)
#
#   External contact torque (optional, force normal to finger):
#       tau_ext = F_ext * l_contact
#
# Variables:
#   theta          finger joint angle [rad]
#   theta_dot      finger joint angular velocity [rad/s]
#   phi            actual servo/spool angle [rad]
#   phi_cmd        commanded servo/spool angle [rad]
#   T              tendon tension [N]
#   J              finger rotational inertia [kg*m^2]
#   b              finger viscous damping [N*m*s/rad]
#   k_s            linear return spring stiffness [N/m]
#   k_t            effective tendon axial stiffness [N/m]
#   -L'(theta)     geometry-dependent tendon moment arm [m]
#   r_s            servo spool radius [m]
#   F_ext          external normal contact force [N]
#   l_contact      distance from joint to contact point [m]
#
# Notes:
# - This is a reduced-order engineering model for concept exploration.
# - The tendon is assumed massless in the arm-dynamics model.
# - Tension is zero if the tendon would otherwise go slack.
# - A hard joint stop is included at theta = 0 and theta = theta_max.
# - The servo is modeled as a first-order position actuator with a max speed.
# ============================================================

st.set_page_config(page_title="TASS Finger Dynamics", layout="wide")
st.title("TASS Single-Joint Finger Dynamics")
st.caption(
    "Interactive reduced-order model of a tendon-driven finger with a return spring. "
    "Two tendon spans follow the finger segments; the connecting span shortens during flexion."
)

# -----------------------------
# Sidebar: user-adjustable model
# -----------------------------
st.sidebar.header("Motor command")
motor_angle_deg = st.sidebar.slider(
    "Motor / spool angle φ_cmd [deg]",
    min_value=0.0,
    max_value=180.0,
    value=90.0,
    step=1.0,
)

st.sidebar.header("Mechanical parameters")
L1 = st.sidebar.slider(
    "Segment 1 length [mm]", 50.0, 150.0, 150.0, 5.0
) / 1000.0
L2 = st.sidebar.slider(
    "Segment 2 length [mm]", 50.0, 150.0, 100.0, 5.0
) / 1000.0
J = st.sidebar.number_input(
    "Distal segment rotational inertia J [kg·m²]",
    min_value=1e-6,
    max_value=0.01,
    value=2.0e-4,
    step=1.0e-5,
    format="%.6f",
)
b = st.sidebar.number_input(
    "Joint damping b [N·m·s/rad]",
    min_value=0.0,
    max_value=0.1,
    value=0.0020,
    step=0.0005,
    format="%.4f",
)
k_s = st.sidebar.number_input(
    "Linear return spring stiffness k_s [N/m]",
    min_value=0.0,
    max_value=5000.0,
    value=500.0,
    step=10.0,
)
spring_free_length = st.sidebar.number_input("Spring free length [mm]", min_value=1.0, value=24.0, step=1.0)/1000
guide_offset = st.sidebar.slider(
    "Tendon guide offset from segment axis [mm]", 1.0, 20.0, 5.0, 0.5
) / 1000.0
p2_fraction = st.sidebar.slider("P2 position along Segment 1 [%]", 40, 95, 80, 1) / 100.0
p3_fraction = st.sidebar.slider("P3 position along Segment 2 [%]", 5, 70, 25, 1) / 100.0
r_s = st.sidebar.slider(
    "Spool radius r_s [mm]", 2.0, 20.0, 6.0, 0.5
) / 1000.0
k_t = st.sidebar.number_input(
    "Effective tendon stiffness k_t [N/m]",
    min_value=50.0,
    max_value=20000.0,
    value=2500.0,
    step=100.0,
)
slack_mm = st.sidebar.slider(
    "Initial tendon slack [mm]", 0.0, 5.0, 0.0, 0.1
)
slack = slack_mm / 1000.0
c_t = st.sidebar.number_input("Tendon damping c_t [N·s/m]", min_value=0.0, value=0.1, step=0.01)
preload = st.sidebar.number_input("Reference preload [N]", min_value=0.0, value=0.0, step=0.1)

# Reference geometry is along +x; the plot below swaps axes to show the finger upright.
tendon = Tendon(k=k_t, c=c_t, slack=slack, preload=preload,
                joint_position=(L1, 0.0),
                p1=(L1*4/15, guide_offset), p2=(L1*p2_fraction, guide_offset),
                p3=(L1 + L2*p3_fraction, guide_offset), p4=(L1 + L2*0.8, guide_offset),
                load_cell=(L1 + L2, guide_offset))
spring = LinearReturnSpring(stiffness_N_m=k_s, free_length_m=spring_free_length,
                            joint_position=(L1, 0.0))

st.sidebar.header("Servo dynamics")
tau_servo = st.sidebar.slider(
    "Servo position time constant τ_servo [s]",
    0.02, 0.50, 0.10, 0.01
)
servo_speed_deg_s = st.sidebar.slider(
    "Servo max speed [deg/s]",
    30.0, 1000.0, 300.0, 10.0
)
servo_speed_max = np.deg2rad(servo_speed_deg_s)

st.sidebar.header("External contact load")
F_ext = st.sidebar.slider(
    "External normal contact force F_ext [N]",
    0.0, 10.0, 0.0, 0.1
)
contact_fraction = st.sidebar.slider(
    "Contact position along Segment 2 [% from joint]",
    10, 100, 70, 5
) / 100.0
l_contact = contact_fraction * L2

st.sidebar.header("Joint limits")
theta_max_deg = st.sidebar.slider(
    "Maximum finger angle [deg]",
    30.0, 120.0, 100.0, 1.0
)
theta_max = np.deg2rad(theta_max_deg)

st.sidebar.header("Simulation")
t_end = st.sidebar.slider(
    "Simulation duration [s]", 0.5, 5.0, 2.0, 0.1
)
dt = 0.0005

# -----------------------------
# Model helper functions
# -----------------------------
def tendon_tension(phi, theta, omega=0.0, phi_dot=0.0):
    """
    phi   : spool angle [rad]
    theta : finger joint angle [rad]
    return: tendon tension [N]
    """
    return tendon.elastic_state(theta, omega, r_s*phi, r_s*phi_dot)[2]


def static_equilibrium(phi_cmd):
    """
    Solve the quasi-static equilibrium numerically:
        tau_tendon(theta, phi_cmd) + tau_spring(theta) - tau_ext = 0

    Returns:
        theta_eq [rad], T_eq [N]
    """
    tau_ext = F_ext * l_contact

    def net_torque(theta):
        return tendon.joint_torque(theta, tendon_tension(phi_cmd, theta)) + spring.joint_torque(theta) - tau_ext

    # Follow the first stable balance from the straight-finger stop.
    if net_torque(0.0) <= 0:
        return 0.0, float(tendon_tension(phi_cmd, 0.0))
    theta_grid = np.linspace(0.0, theta_max, 5001)
    torque = net_torque(theta_grid)
    crossings = np.flatnonzero((torque[:-1] > 0) & (torque[1:] <= 0))
    if len(crossings):
        idx = crossings[0]
        theta_eq = brentq(net_torque, theta_grid[idx], theta_grid[idx + 1])
    else:
        theta_eq = theta_max  # Flexion stop supports the remaining torque.
    T_eq = tendon_tension(phi_cmd, theta_eq)
    return theta_eq, float(T_eq)


def simulate(phi_cmd):
    """
    Semi-implicit Euler simulation.

    States:
        theta      joint angle [rad]
        omega      joint angular velocity [rad/s]
        phi        actual servo angle [rad]
    """
    n = int(t_end / dt) + 1
    time = np.linspace(0.0, t_end, n)

    theta = np.zeros(n)
    omega = np.zeros(n)
    phi = np.zeros(n)
    tension = np.zeros(n)

    # Start in the straight, unspooled condition.
    theta[0] = 0.0
    omega[0] = 0.0
    phi[0] = 0.0
    tension[0] = tendon_tension(phi[0], theta[0])

    tau_ext = F_ext * l_contact

    for i in range(n - 1):
        # First-order servo position response with velocity saturation.
        phi_dot = (phi_cmd - phi[i]) / tau_servo
        phi_dot = np.clip(phi_dot, -servo_speed_max, servo_speed_max)
        phi[i + 1] = phi[i] + phi_dot * dt

        # Unilateral tendon: it can pull but cannot push.
        T = tendon_tension(phi[i], theta[i], omega[i], phi_dot)
        tension[i] = T

        # Joint equation:
        # The same routed path determines extension and pulling torque.
        theta_ddot = (
            tendon.joint_torque(theta[i], T)
            - b * omega[i]
            + spring.joint_torque(theta[i])
            - tau_ext
        ) / J

        # Semi-implicit Euler tends to be more stable than explicit Euler.
        omega[i + 1] = omega[i] + theta_ddot * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt

        # Straight-finger endstop.
        if theta[i + 1] < 0.0:
            theta[i + 1] = 0.0
            if omega[i + 1] < 0.0:
                omega[i + 1] = 0.0

        # Flexion endstop.
        if theta[i + 1] > theta_max:
            theta[i + 1] = theta_max
            if omega[i + 1] > 0.0:
                omega[i + 1] = 0.0

    final_phi_dot = np.clip((phi_cmd - phi[-1]) / tau_servo, -servo_speed_max, servo_speed_max)
    tension[-1] = tendon_tension(phi[-1], theta[-1], omega[-1], final_phi_dot)

    return time, theta, omega, phi, tension


phi_cmd = np.deg2rad(motor_angle_deg)
time, theta, omega, phi, tension = simulate(phi_cmd)
theta_eq, T_eq = static_equilibrium(phi_cmd)

# -----------------------------
# Summary metrics
# -----------------------------
final_theta_deg = np.rad2deg(theta[-1])
eq_theta_deg = np.rad2deg(theta_eq)
tip_x = L2 * np.sin(theta[-1])
tip_y = L1 + L2 * np.cos(theta[-1])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Commanded motor angle", f"{motor_angle_deg:.1f}°")
m2.metric("Final finger angle", f"{final_theta_deg:.1f}°")
m3.metric("Final tendon tension", f"{tension[-1]:.2f} N")
m4.metric("Quasi-static angle", f"{eq_theta_deg:.1f}°")

# -----------------------------
# Visual schematic
# -----------------------------
left, right = st.columns([1, 1.35])

with left:
    st.subheader("Finger configuration")

    fig, ax = plt.subplots(figsize=(6, 7))

    base = np.array([0.0, 0.0])
    joint = np.array([0.0, L1])

    # theta = 0 means segment 2 is straight above segment 1.
    tip = joint + np.array([
        L2 * np.sin(theta[-1]),
        L2 * np.cos(theta[-1])
    ])

    # Segment 1 and segment 2
    ax.plot([base[0], joint[0]], [base[1], joint[1]], linewidth=7)
    ax.plot([joint[0], tip[0]], [joint[1], tip[1]], linewidth=7)

    # Joint
    ax.scatter([joint[0]], [joint[1]], s=140, zorder=5)

    # Spool
    spool_center = tendon.spool_position[::-1]
    spool = plt.Circle(
        spool_center, r_s, fill=False, linewidth=2
    )
    ax.add_patch(spool)
    ax.scatter([spool_center[0]], [spool_center[1]], s=20)

    # P4 remains a sliding guide; the route ends at the load cell on segment 2.
    guides = np.array(tendon.guide_positions(theta[-1]))[:, ::-1]
    load_cell = tendon.load_cell_position(theta[-1])[::-1]
    route = np.vstack((spool_center, guides, load_cell))
    labels = ["Spool feed", "Span 1", "Span 3 (joint)", "Span 2", "P4 to load cell"]
    for i, label in enumerate(labels):
        ax.plot(route[i:i+2, 0], route[i:i+2, 1], "--", linewidth=2, label=label)
    ax.scatter(guides[:, 0], guides[:, 1], s=30, zorder=6)
    for i, point in enumerate(guides, start=1):
        ax.annotate(f"P{i}", point,
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.scatter(*load_cell, marker="s", s=80, zorder=7)
    ax.annotate("Load cell / anchor", load_cell, xytext=(5, 5), textcoords="offset points", fontsize=8)
    fixed, moving = np.array(spring.attachment_positions(theta[-1]))[:, ::-1]
    normal = np.array([-(moving-fixed)[1], (moving-fixed)[0]])/np.linalg.norm(moving-fixed)
    coil = [fixed]
    for i, fraction in enumerate(np.linspace(0.15, 0.85, 13)):
        coil.append(fixed+fraction*(moving-fixed)+(0.002 if i % 2 else -0.002)*normal)
    coil.append(moving)
    coil = np.array(coil)
    ax.plot(coil[:, 0], coil[:, 1], color="goldenrod", label="Linear spring")
    ax.legend(fontsize=8, loc="upper left")

    # External force arrow, assumed normal to the distal segment.
    contact_point = joint + contact_fraction * (tip - joint)
    if F_ext > 0:
        tangent = (tip - joint) / max(np.linalg.norm(tip - joint), 1e-12)
        normal = np.array([tangent[1], -tangent[0]])
        arrow_len = 0.035 * (F_ext / 10.0 + 0.2)
        ax.arrow(
            contact_point[0] + normal[0] * arrow_len,
            contact_point[1] + normal[1] * arrow_len,
            -normal[0] * arrow_len,
            -normal[1] * arrow_len,
            width=0.0015,
            head_width=0.007,
            length_includes_head=True
        )
        ax.text(
            contact_point[0] + normal[0] * arrow_len * 1.15,
            contact_point[1] + normal[1] * arrow_len * 1.15,
            f"{F_ext:.1f} N"
        )

    # Angle annotation
    ax.text(
        joint[0] - 0.065,
        joint[1] + 0.012,
        f"θ = {final_theta_deg:.1f}°"
    )
    ax.text(
        spool_center[0] + 0.02,
        spool_center[1] + 0.025,
        f"φ_cmd = {motor_angle_deg:.0f}°"
    )
    ax.text(
        spool_center[0] + 0.02,
        spool_center[1] + 0.005,
        f"T = {tension[-1]:.2f} N"
    )

    lim = L1 + L2 + 0.06
    ax.set_xlim(-0.08, 0.16)
    ax.set_ylim(-0.03, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Single-joint tendon-driven finger")
    ax.grid(True, alpha=0.25)

    st.pyplot(fig, clear_figure=True)
    spans = tendon.path_segments(theta[-1])
    st.caption(
        f"Span 1: {spans['p1_p2']*1000:.1f} mm · "
        f"Span 2: {spans['p3_p4']*1000:.1f} mm · "
        f"Span 3: {spans['p2_p3']*1000:.1f} mm. "
        f"Shortening: {tendon.joint_displacement(theta[-1])*1000:.1f} mm; "
        f"moment arm: {-tendon.path_length_derivative(theta[-1])*1000:.1f} mm."
    )

with right:
    st.subheader("Dynamic response")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time, np.rad2deg(theta), label="Finger angle θ")
    ax2.plot(time, np.rad2deg(phi), label="Actual servo angle φ")
    ax2.axhline(
        motor_angle_deg,
        linestyle="--",
        linewidth=1,
        label="Motor command φ_cmd"
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle [deg]")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(time, tension)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Tendon tension [N]")
    ax3.set_title("Tendon tension response")
    ax3.grid(True, alpha=0.25)
    st.pyplot(fig3, clear_figure=True)

# -----------------------------
# Operating-point table
# -----------------------------
st.subheader("Illustrative motor-angle / finger-angle / tension operating points")

angles_to_list = np.arange(0.0, 181.0, 30.0)
rows = []

for motor_deg in angles_to_list:
    phi_test = np.deg2rad(motor_deg)
    theta_test, T_test = static_equilibrium(phi_test)
    rows.append({
        "Motor angle φ [deg]": motor_deg,
        "Finger angle θ [deg]": np.rad2deg(theta_test),
        "Tendon tension T [N]": T_test,
    })

df = pd.DataFrame(rows)
df["Finger angle θ [deg]"] = df["Finger angle θ [deg]"].round(1)
df["Tendon tension T [N]"] = df["Tendon tension T [N]"].round(2)
st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Model equations
# -----------------------------
st.subheader("Equations used by the app")

st.latex(
    r"J\ddot{\theta} + b\dot{\theta} "
    r"= -L'(\theta) T + \tau_s - \tau_{\mathrm{ext}}"
)
st.write(
    "**Variables:** "
    "θ = finger joint angle [rad]; "
    "J = joint rotational inertia [kg·m²]; "
    "b = joint viscous damping [N·m·s/rad]; "
    "τ_s = torque from the linear spring between the bracket attachments [N·m]; "
    "−L′(θ) = tendon moment arm from guide geometry [m]; "
    "T = tendon tension [N]; "
    "τ_ext = external contact torque [N·m]."
)
st.latex(r"F_s=k_s\max(0,\ell_s-\ell_{s0}),\qquad \tau_s=-F_s\ell_s'(\theta)")
st.caption("Spring stiffness is in N/m. Bracket offsets from the joint are (−12, −20) and (12, −20) mm in the straight pose; the second rotates with segment 2.")

st.latex(
    r"\delta_t = r_s\phi + L(\theta)-L(0) - \delta_{\mathrm{slack}} + T_0/k_t"
)
st.write(
    "**Variables:** "
    "δ_t = tendon elastic extension [m]; "
    "r_s = spool radius [m]; "
    "φ = actual spool angle [rad]; "
    "L(θ) = routed tendon path length [m]; "
    "θ = finger joint angle [rad]; "
    "δ_slack = initial tendon slack [m]; T₀ = reference preload [N]."
)

st.latex(
    r"\dot\delta_t = r_s\dot\phi + L'(\theta)\dot\theta,\qquad "
    r"T = \begin{cases}\max(0,k_t\delta_t+c_t\dot\delta_t),&\delta_t>0\\0,&\delta_t\leq0\end{cases}"
)
st.write(
    "**Variables:** "
    "T = tendon tension [N]; "
    "k_t = effective tendon axial stiffness [N/m]; "
    "c_t = axial tendon damping [N·s/m]; δ_t = signed tendon extension [m]. "
    "A slack tendon carries no force, and tension cannot become negative."
)

st.latex(
    r"\tau_{\mathrm{ext}} = F_{\mathrm{ext}}\,l_{\mathrm{contact}}"
)
st.write(
    "**Variables:** "
    "τ_ext = external joint torque [N·m]; "
    "F_ext = force applied normal to the distal finger segment [N]; "
    "l_contact = distance from the joint axis to the contact point [m]."
)

st.latex(
    r"\dot{\phi} = \mathrm{sat}\left("
    r"\frac{\phi_{\mathrm{cmd}}-\phi}{\tau_{\mathrm{servo}}}"
    r"\right)"
)
st.write(
    "**Variables:** "
    "φ = actual servo/spool angle [rad]; "
    "φ_cmd = commanded servo/spool angle [rad]; "
    "τ_servo = first-order servo time constant [s]; "
    "sat(·) = saturation at the selected maximum servo angular speed."
)

st.info(
    "Demonstration parameters: replace guide coordinates, spool radius, stiffness, "
    "damping, inertia, and joint limits with measured prototype values. Guides are "
    "frictionless, the tendon passes P4 and anchors at the distal load cell, and the joint span is straight and unobstructed. "
    "P1 is at 26.7% of Segment 1 and P4 at 80% of Segment 2. Spring dimensions are demonstration values."
)
