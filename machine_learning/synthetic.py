"""Toy statistical fixtures only: neither hardware measurements nor digital-twin output."""
import numpy as np
import pandas as pd

from .data import RX, TX


def make_samples(cfg, sessions=48, trials_per_session=6):
    rng = np.random.default_rng(cfg["seed"])
    n, fs = cfg["window_samples"], 8192.
    frames = []
    for session in range(sessions):
        gain = rng.normal(1, 0.08)
        for trial in range(trials_per_session):
            time = np.arange(n)/fs
            force = rng.uniform(0.1, 5.) if rng.random() > 0.15 else 0.
            location = rng.uniform(0.01, 0.37) if force > 0 else np.nan
            position = location if np.isfinite(location) else 0.
            freq = rng.choice([128., 256., 512.])
            angle = rng.uniform(0, 1.5)
            row = dict(session_id=f"toy_session_{session:03d}", trial_id=f"toy_{session:03d}_{trial:03d}",
                       data_source="synthetic", time_s=time, contact_force_N=force,
                       contact_location_m=location, servo_angle_rad=angle,
                       servo_torque_Nm=0.05*angle+rng.normal(0, .03),
                       load_cell_N=1+0.4*force+angle+rng.normal(0, 1.),
                       excitation_frequency_Hz=freq, quality_flag="ok")
            for i, name in enumerate(RX):
                spatial = 0.7*np.sin((i+1)*np.pi*position/.4)
                amp = max(.05, gain*(1+.22*force+spatial+rng.normal(0, .45)))
                row[name] = amp*np.sin(2*np.pi*freq*time+i*.3)+.18*rng.normal(size=n)
            for i, name in enumerate(TX):
                amp = rng.uniform(.3, 1.5) if rng.random() > .2 else 0.
                row[name] = amp*np.sin(2*np.pi*freq*time+i*.2)+.005*rng.normal(size=n)
            frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def make_echo_samples(cfg, sessions=48, trials_per_session=6):
    """Positive-control fixture: force changes packet amplitude; location changes delay.

    Delays/amplitudes are invented for a signal-processing test, not derived
    from the TASS mechanics, wave speed or the paper's ultrasonic apparatus.
    """
    rng = np.random.default_rng(cfg["seed"])
    n, fs = cfg["window_samples"], 8192.
    t = np.arange(n)/fs-.010
    frames = []
    for session in range(sessions):
        gain = rng.normal(1, .06)
        for trial in range(trials_per_session):
            force, location = rng.uniform(.1, 5), rng.uniform(.01, .37)
            arrival = .020+.060*location/.374
            row = dict(session_id=f"echo_session_{session:03d}", trial_id=f"echo_{session:03d}_{trial:03d}",
                       data_source="synthetic", time_s=t, excitation_trigger_time_s=0.,
                       contact_force_N=force, contact_location_m=location,
                       servo_angle_rad=rng.uniform(0, 1.5), servo_torque_Nm=rng.normal(.05, .02),
                       load_cell_N=rng.uniform(1, 5), excitation_frequency_Hz=1024., quality_flag="ok")
            for i, name in enumerate(RX):
                delay = arrival+i*.0005+rng.normal(0, .00015)
                amplitude = max(.05, gain*(.4+.25*force)+rng.normal(0, .06))
                packet = amplitude*np.exp(-.5*((t-delay)/.0015)**2)*np.cos(2*np.pi*1024*(t-delay))
                terminal = .25*np.exp(-.5*((t-.095)/.0015)**2)*np.cos(2*np.pi*1024*(t-.095))
                ringdown = 2*np.exp(-.5*((t-.002)/.0006)**2)*np.cos(2*np.pi*1024*t)
                row[name] = packet+terminal+ringdown+.01*rng.normal(size=n)
            for i, name in enumerate(TX):
                # One isolated excitation; other measured channels are off.
                row[name] = np.exp(-.5*(t/.0005)**2)*np.cos(2*np.pi*1024*t) if i == 0 else np.zeros(n)
            frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)
