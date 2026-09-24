"""Four-panel diagnostic figure; headless export is supported."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(result, frequency, path, title="TASS minimum model"):
    m = result.measurement
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    axes[0, 0].plot(m.time_s, m.channels["motor_angle_rad"], label="Motor command")
    axes[0, 0].plot(m.time_s, m.channels["joint_angle_rad"], label="Finger joint")
    axes[0, 0].set(title="Mechanical motion", xlabel="Time [s]", ylabel="Angle [rad]")
    axes[0, 1].plot(m.time_s, m.channels["tension_N"], color="#14857b", label="Tendon tension")
    axes[0, 1].set(title="Tendon loading", xlabel="Time [s]", ylabel="Tension [N]")
    for i in range(m.piezo_V.shape[1]):
        label = f"Piezo {i+1}"
        axes[1, 0].plot(m.time_s, m.piezo_V[:, i], label=label, linewidth=0.7)
        axes[1, 1].plot(frequency.frequency_Hz, frequency.amplitude_V[:, i], label=label)
    axes[1, 0].set(title="Synthetic receiver signals", xlabel="Time [s]", ylabel="Voltage [V]")
    axes[1, 1].set(title="FFT amplitude (whole record)", xlabel="Frequency [Hz]", ylabel="Amplitude [V]",
                   xlim=(0, min(2500, m.sample_rate_Hz/2)))
    active = m.channels["contact_active"] > 0
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.fill_between(m.time_s, 0, 1, where=active, transform=ax.get_xaxis_transform(),
                        color="#dfb450", alpha=0.12)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.text(0.5, -0.012, "Shading: hard contact active. Demonstration parameters; piezo gain is uncalibrated.",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
