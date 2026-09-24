"""Examples for replaying load-cell data and processing ADC-derived voltages.

Hardware capture/clock synchronization is outside this simulator. Convert to SI
and resample onto a uniform clock before calling these functions.
"""
from dataclasses import replace
from tass import Inputs, simulate, Measurement, ModelConfig
from tass.signals import SampledSignal
from tass.processing import extract_frequency_features


def replay_load_cell(time_s, load_cell_N, motor_angle_rad):
    config = ModelConfig()
    config = replace(config, numerics=replace(config.numerics,
                     duration_s=len(time_s)*(time_s[1]-time_s[0]),
                     sample_rate_Hz=1/(time_s[1]-time_s[0])))
    relative_time = time_s-time_s[0]
    inputs = Inputs(tension_N=SampledSignal(relative_time, load_cell_N),
                    motor_angle_rad=SampledSignal(relative_time, motor_angle_rad))
    return simulate(config, inputs)


def process_hardware(time_s, piezo_volts, excitation_newtons, encoder_angle_rad, load_cell_N):
    measurement = Measurement(time_s, piezo_volts, excitation_newtons,
                              channels={"joint_angle_rad": encoder_angle_rad,
                                        "tension_N": load_cell_N})
    return extract_frequency_features(measurement)
