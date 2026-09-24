"""Run from the project root: python -m examples.custom_run"""
from dataclasses import replace
from tass import ModelConfig, Inputs, simulate
from tass.signals import SmoothStep, Chirp
from tass.processing import extract_frequency_features
from tass.io import save_run
from tass.plotting import plot_run

config = ModelConfig.load("config/default.json")
config = replace(config,
                 axial=replace(config.axial, mode="simulated"),
                 contact=replace(config.contact, tendon_position_m=0.4*config.string.length_m))
inputs = Inputs(motor_angle_rad=SmoothStep(0.12, 0.18, 0.02, 0.10),
                contact_force_N=SmoothStep(0, 0.4, 0.14, 0.16),
                excitation_N=Chirp(amplitude=0.002, start_Hz=80, end_Hz=1800, duration_s=0.25))
result = simulate(config, inputs)
features = extract_frequency_features(result.measurement)
save_run(result, features, "outputs/custom")
plot_run(result, features, "outputs/custom/overview.png")
print(features.features)
