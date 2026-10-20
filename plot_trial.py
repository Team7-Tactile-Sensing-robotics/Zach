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


def plot_pose(ax, dataset, metadata, *, upright=False):
    """Draw the recorded finger pose, optionally rotated 90° upright."""

    def orient(points):
        points = np.asarray(points, dtype=float)
        if upright:
            # Rotate coordinates 90° counterclockwise.
            return np.stack((-points[..., 1], points[..., 0]), axis=-1)
        return points

    snapshot = Path(dataset) / "model_config.json"
    if not snapshot.is_file() or "joint_angle_rad" not in metadata:
        ax.text(
            .5, .5,
            "Pose unavailable: saved model configuration or joint angle missing",
            ha="center", va="center",
            transform=ax.transAxes, wrap=True,
        )
        ax.set_axis_off()
        return

    from tass_minimum.tass.config import ModelConfig
    from tass_minimum.tass.models.tendon import Tendon
    from tass_minimum.tass.models.spring import ReturnSpring

    config = ModelConfig.load(snapshot)
    tendon = Tendon.from_config(config)

    q = metadata["joint_angle_rad"]
    joint = np.asarray(tendon.joint_position, dtype=float)
    direction = np.array([
        np.cos(q - tendon.q0),
        np.sin(q - tendon.q0),
    ])
    tip = joint + config.joint.distal_length_m * direction

    # Finger: base → joint → fingertip.
    segments = orient(np.array([[0., 0.], joint, tip]))
    ax.plot(
        *segments.T, "o-", color="slategray", lw=8,
        label="Finger segments", zorder=2,
    )

    ax.scatter(
        *orient(joint), s=100,
        facecolors="white", edgecolors="black", zorder=4,
    )
    ax.annotate(
        "Joint", orient(joint),
        xytext=(12, 0) if upright else (0, -18),
        textcoords="offset points",
        ha="left" if upright else "center", va="center",
    )

    # Tendon route through guides P1–P4.
    guides = np.asarray(tendon.guide_positions(q), dtype=float)
    route = np.vstack((tendon.spool_position, guides))
    ax.plot(
        *orient(route).T, "o-", color="royalblue", lw=2,
        label="Tendon through P1–P4", zorder=3,
    )

    for i, point in enumerate(guides, 1):
        ax.annotate(
            f"P{i}", orient(point),
            xytext=(-10, 0) if upright else (0, 9),
            textcoords="offset points",
            ha="right" if upright else "center",
            va="center" if upright else "bottom",
        )

    # Motor spool.
    ax.add_patch(plt.Circle(
        orient(tendon.spool_position),
        config.motor.spool_radius_m,
        fill=False, color="royalblue",
    ))

    # Return spring.
    a, b = ReturnSpring(config).attachment_positions(q)
    ax.plot(
        *orient(np.array([a, b])).T,
        color="darkorange", lw=3,
        label="Spring attachments",
    )

    # Physical contact position along the finger.
    physical = metadata.get("contact_segment_requested_m")
    if physical is not None:
        point = (
            np.array([physical, 0.])
            if physical <= joint[0]
            else joint + (physical - joint[0]) * direction
        )
        ax.scatter(
            *orient(point), marker="x", color="crimson",
            s=85, zorder=5, label="Physical contact",
        )

    # Clamp position projected onto the tendon route.
    distance = metadata.get("contact_snapped_m")
    if distance is not None:
        cumulative = np.r_[
            0.,
            np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1)),
        ]
        point = np.array([
            np.interp(distance, cumulative, route[:, axis])
            for axis in (0, 1)
        ])
        ax.scatter(
            *orient(point), marker="s",
            facecolors="none", edgecolors="crimson",
            s=65, zorder=5, label="Tendon clamp (projected)",
        )

    ax.set(
        title=(
            f"Recorded finger pose: joint angle {np.degrees(q):.2f}°"
            " (static trial)"
        ),
        xlabel=(
            "Horizontal position [m]"
            if upright else "x [m] — base toward fingertip"
        ),
        ylabel=(
            "Vertical position [m]"
            if upright else "y [m] — tendon side up"
        ),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.margins(
        x=.55 if upright else .12,
        y=.12 if upright else .55,
    )
    ax.legend(
        loc="upper left", fontsize=8, ncol=1 if upright else 2,
    )

    extension = config.routed_pairs.anchor_extension_m
    if extension:
        ax.text(
            .99, .02,
            f"Acoustic extension beyond P4: {extension * 1000:g} mm;\n"
            "physical anchor geometry unspecified",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        )

    ax.grid(alpha=.2)


def export_trial(trial_number, dataset=DEFAULT_DATASET, output=None, *, start_s=None, end_s=None):
    """Export CSV, piezo/mechanics figures, and a separate upright finger pose."""
    if trial_number < 0:
        raise ValueError("Trial number must be nonnegative (numbering starts at 0)")
    dataset = Path(dataset)
    trial_id = f"trial_{trial_number:04d}"
    source = dataset / "trials" / f"{trial_id}.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Trial file not found: {source}")
    with np.load(source, allow_pickle=False) as trial:
        pair_ids = [i for i in range(1, 5) if f"piezo_sensor_{i}_V" in trial.files]
        if not pair_ids:
            raise ValueError("Trial contains no saved piezo sensor voltage channels")
        names = (["time_s"]
                 + [f"piezo_sensor_{i}_V" for i in pair_ids]
                 + [f"piezo_actuator_{i}_V" for i in pair_ids]
                 + ["contact_force_N", "contact_location_m", "servo_angle_rad", "servo_torque_Nm"])
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
    start = time[0] if start_s is None else start_s
    end = time[-1] if end_s is None else end_s
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end:
        raise ValueError("Plot interval requires finite times with 0 <= start < end")
    visible = (time >= start) & (time <= end)
    if np.count_nonzero(visible) < 2:
        raise ValueError("Plot interval must contain at least two recorded samples")
    plot_time = time[visible]
    output = Path(output) if output is not None else dataset / "trial_exports"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{trial_id}.csv"
    piezo_path = output / f"{trial_id}_piezo.png"
    mechanics_path = output / f"{trial_id}_mechanics.png"
    finger_path = output / f"{trial_id}_finger_upright.png"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(channels)
        writer.writerows(zip(*channels.values()))

    fig, axes = plt.subplots(len(pair_ids), 2, figsize=(14, max(5, 2.75*len(pair_ids))),
                             sharex=True, squeeze=False, constrained_layout=True)
    for row, number in enumerate(pair_ids):
        for col, kind, label in ((0, "actuator", "Actuator drive"), (1, "sensor", "Sensor")):
            ax = axes[row, col]
            ax.plot(plot_time, channels[f"piezo_{kind}_{number}_V"][visible], lw=.6,
                    color="royalblue" if col == 0 else "seagreen")
            ax.set(title=f"{label} {number}", ylabel="Voltage [V]")
            ax.set_xlim(start, end)
            # Set X-axis interval to 0.1 seconds
            ax.set_xticks(np.arange(start, end + 0.001, 0.1))
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
        ax.plot(plot_time, channels[name][visible], lw=1)
        ax.set(title=title, ylabel=unit, xlabel="Time since trial start [s]")
        ax.set_xlim(start, end)
        ax.grid(alpha=.25)
    plot_pose(fig.add_subplot(grid[2, :]), dataset, metadata, upright=False)
    fig.suptitle(f"{trial_id} — contact, servo and finger pose")
    fig.savefig(mechanics_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 10), constrained_layout=True)
    plot_pose(ax, dataset, metadata, upright=True)
    if ax.get_legend() is not None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle(f"{trial_id} — upright finger pose")
    fig.savefig(finger_path, dpi=150)
    plt.close(fig)
    return csv_path, piezo_path, mechanics_path, finger_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=int, help="Zero-based trial number, e.g. 12")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Dataset directory containing trials/ (default: dataset_straight_chirp_fixed)")
    parser.add_argument("--output", type=Path, help="Export directory (default: DATASET/trial_exports)")
    parser.add_argument("--start-s", type=float, help="Plot start time in seconds (default: first sample)")
    parser.add_argument("--end-s", type=float, help="Plot end time in seconds (default: last sample); CSV stays complete")
    args = parser.parse_args()
    try:
        paths = export_trial(args.trial, args.dataset, args.output, start_s=args.start_s, end_s=args.end_s)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
