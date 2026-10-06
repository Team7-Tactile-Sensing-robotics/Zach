"""Export one saved simulation trial: python plot_trial.py 12."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET = Path(__file__).resolve().parent / "outputs/dataset_straight_chirp_fixed"


def plot_pose(ax, dataset, metadata):
    """Draw the saved pose in the model's +x finger / +y tendon coordinates."""
    snapshot = dataset / "model_config.json"
    if not snapshot.is_file() or "joint_angle_rad" not in metadata:
        ax.text(.5, .5, "Pose unavailable: saved model configuration or joint angle missing",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
        return
    from tass_minimum.tass.config import ModelConfig
    from tass_minimum.tass.models.tendon import Tendon
    from tass_minimum.tass.models.spring import ReturnSpring
    config = ModelConfig.load(snapshot)
    tendon = Tendon.from_config(config)
    q = metadata["joint_angle_rad"]
    joint = tendon.joint_position
    direction = np.array([np.cos(q-tendon.q0), np.sin(q-tendon.q0)])
    tip = joint+config.joint.distal_length_m*direction
    segments = np.array([[0., 0.], joint, tip])
    ax.plot(*segments.T, "o-", color="slategray", lw=8, label="Finger segments", zorder=2)
    ax.scatter(*joint, s=100, facecolors="white", edgecolors="black", zorder=4)
    ax.annotate("Joint", joint, xytext=(0, -18), textcoords="offset points", ha="center")
    guides = np.array(tendon.guide_positions(q))
    route = np.vstack((tendon.spool_position, guides))
    ax.plot(*route.T, "o-", color="royalblue", lw=2, label="Tendon through P1–P4", zorder=3)
    for i, point in enumerate(guides, 1):
        ax.annotate(f"P{i}", point, xytext=(0, 9), textcoords="offset points", ha="center")
    ax.add_patch(plt.Circle(tendon.spool_position, config.motor.spool_radius_m,
                           fill=False, color="royalblue"))
    a, b = ReturnSpring(config).attachment_positions(q)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="darkorange", lw=3, label="Spring attachments")
    physical = metadata.get("contact_segment_requested_m")
    if physical is not None:
        point = (np.array([physical, 0.]) if physical <= joint[0]
                 else joint+(physical-joint[0])*direction)
        ax.scatter(*point, marker="x", color="crimson", s=85, zorder=5, label="Physical contact")
    distance = metadata.get("contact_snapped_m")
    if distance is not None:
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        point = np.array([np.interp(distance, cumulative, route[:, axis]) for axis in (0, 1)])
        ax.scatter(*point, marker="s", facecolors="none", edgecolors="crimson", s=65,
                   zorder=5, label="Tendon clamp (projected)")
    ax.set(title=f"Recorded finger pose: joint angle {np.degrees(q):.2f}° (static trial)",
           xlabel="x [m] — base toward fingertip", ylabel="y [m] — tendon side up")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=.12, y=.55)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    extension = config.routed_pairs.anchor_extension_m
    if extension:
        ax.text(.99, .02, f"Acoustic extension beyond P4: {extension*1000:g} mm; physical anchor geometry unspecified",
                transform=ax.transAxes, ha="right", fontsize=8)
    ax.grid(alpha=.2)


def export_trial(trial_number, dataset=DEFAULT_DATASET, output=None):
    """Export full-resolution CSV, paired piezo figure, and mechanics/pose figure."""
    if trial_number < 0:
        raise ValueError("Trial number must be nonnegative (numbering starts at 0)")
    dataset = Path(dataset)
    trial_id = f"trial_{trial_number:04d}"
    source = dataset / "trials" / f"{trial_id}.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Trial file not found: {source}")
    names = (["time_s"]
             + [f"piezo_sensor_{i}_V" for i in range(1, 5)]
             + [f"piezo_actuator_{i}_V" for i in range(1, 5)]
             + ["contact_force_N", "contact_location_m", "servo_angle_rad", "servo_torque_Nm"])
    with np.load(source, allow_pickle=False) as trial:
        missing = set(names) - set(trial.files)
        if missing:
            raise ValueError(f"Trial lacks required channels: {', '.join(sorted(missing))}. "
                             "Use a dataset generated with the synchronized-channel exporter.")
        channels = {name: trial[name] for name in names}
        metadata = json.loads(trial['metadata_json'].item()) if 'metadata_json' in trial.files else {}
        location_title = ('Contact location along physical finger from base'
                          if metadata.get('contact_location_reference') == 'finger_base_along_segments'
                          else 'Contact location along tendon from spool/feed endpoint')
    time = channels["time_s"]
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("Trial timestamps must be a strictly increasing vector")
    if any(values.shape != time.shape or not np.isfinite(values).all()
           for values in channels.values()):
        raise ValueError("All channels must be finite vectors matching the timestamps")
    output = Path(output) if output is not None else dataset / "trial_exports"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{trial_id}.csv"
    piezo_path = output / f"{trial_id}_piezo.png"
    mechanics_path = output / f"{trial_id}_mechanics.png"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(channels)
        writer.writerows(zip(*channels.values()))

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
    for i in range(4):
        for col, kind, label in ((0, "actuator", "Actuator drive"), (1, "sensor", "Sensor")):
            ax = axes[i, col]
            ax.plot(time, channels[f"piezo_{kind}_{i+1}_V"], lw=.6,
                    color="royalblue" if col == 0 else "seagreen")
            ax.set(title=f"{label} {i+1}", ylabel="Voltage [V]")
            ax.grid(alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time since trial start [s]")
    fig.suptitle(f"{trial_id} — actuator drive voltages (left) / sensor voltages (right)")
    fig.savefig(piezo_path, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
    axes = [fig.add_subplot(grid[i//2, i%2]) for i in range(4)]
    for ax, name, title, unit in zip(axes,
            ["contact_force_N", "contact_location_m", "servo_angle_rad", "servo_torque_Nm"],
            ["Contact force", location_title, "Servo / spool angle (zero winding reference)",
             "Ideal servo holding torque"],
            ["Force [N]", "Distance [m]", "Angle [rad]", "Torque [N m]"]):
        ax.plot(time, channels[name], lw=1)
        ax.set(title=title, ylabel=unit, xlabel="Time since trial start [s]")
        ax.grid(alpha=.25)
    plot_pose(fig.add_subplot(grid[2, :]), dataset, metadata)
    fig.suptitle(f"{trial_id} — contact, servo and finger pose")
    fig.savefig(mechanics_path, dpi=150)
    plt.close(fig)
    return csv_path, piezo_path, mechanics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=int, help="Zero-based trial number, e.g. 12")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Dataset directory containing trials/ (default: dataset_straight_chirp_fixed)")
    parser.add_argument("--output", type=Path, help="Export directory (default: DATASET/trial_exports)")
    args = parser.parse_args()
    try:
        paths = export_trial(args.trial, args.dataset, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
