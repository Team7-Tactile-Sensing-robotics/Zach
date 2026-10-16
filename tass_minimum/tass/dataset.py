"""Reproducible static-contact synthetic dataset. Run python -m tass.dataset."""
import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import ModelConfig
from .models.tendon import Tendon
from .models.spring import ReturnSpring
from .static_vibration import static_response
from .adc import ADCConfig, observe_adc


@dataclass(frozen=True)
class Protocol:
    location_fractions: tuple = (0.2, 0.4, 0.6, 0.8)
    force_levels_N: tuple = (1., 2., 5., 10.)
    trials_per_condition: int = 20
    location_reference: str = 'total_tendon'
    window_s: float = 1.
    sample_rate_Hz: int = 48000
    held_joint_angle_rad: float = 0.5
    fixed_force_lever_arm_m: float = 0.04
    excitation_type: str = 'pulse'
    chirp_schedule: str = 'simultaneous'  # Or sequential, one sweep per quarter-window.
    chirp_start_Hz: float = 20.0
    chirp_end_Hz: float = 2000.0
    chirp_amplitude_V: float = 2.0
    chirp_ramp_s: float = 0.005
    pulse_amplitude_N: float = 0.002
    pulse_width_s: float = 0.001
    pulse_delay_in_slot_s: float = 0.01
    force_jitter_fraction: float = 0.02
    position_jitter_m: float = 0.0005
    gain_jitter_fraction: float = 0.03
    excitation_jitter_fraction: float = 0.03
    noise_std_V: float = 1e-5
    actuator_force_per_volt_N_V: tuple = (0.001, 0.001, 0.001, 0.001)
    adc: dict = field(default_factory=lambda: asdict(ADCConfig()))
    seed: int = 20260929

    def validate(self):
        ADCConfig(**self.adc).validate(self.sample_rate_Hz)
        gains = np.asarray(self.actuator_force_per_volt_N_V)
        if gains.shape != (4,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0):
            raise ValueError('Four positive finite actuator force-per-volt gains required')
        if self.location_reference not in {'total_tendon', 'segment2_from_joint', 'total_segment'}:
            raise ValueError('Unknown location reference')
        if not self.location_fractions or any(not 0 < x < 1 for x in self.location_fractions):
            raise ValueError('Location fractions must lie in (0,1)')
        if not self.force_levels_N or any(not np.isfinite(x) or x <= 0 for x in self.force_levels_N):
            raise ValueError('Force magnitudes must be positive')
        if type(self.trials_per_condition) is not int or self.trials_per_condition < 5:
            raise ValueError('At least five trials per condition required for grouped splits')
        if not np.isfinite(self.window_s) or not np.isfinite(self.sample_rate_Hz) or min(self.window_s, self.sample_rate_Hz) <= 0:
            raise ValueError('Window and sample rate must be positive and finite')
        if self.excitation_type not in {'pulse', 'chirp'}:
            raise ValueError('excitation_type must be pulse or chirp')
        if self.excitation_type == 'pulse':
            if not np.isfinite(self.pulse_width_s) or self.pulse_width_s <= 0:
                raise ValueError('Pulse width must be positive and finite')
            if not 0 <= self.pulse_delay_in_slot_s < self.window_s/4-self.pulse_width_s:
                raise ValueError('Pulse must fit inside each actuator slot')
            if self.pulse_width_s*self.sample_rate_Hz < 8:
                raise ValueError('Pulse requires at least eight samples')
        else:
            if self.chirp_schedule not in {'simultaneous', 'sequential'}:
                raise ValueError('chirp_schedule must be simultaneous or sequential')
            if not 0 < self.chirp_start_Hz < self.chirp_end_Hz < self.sample_rate_Hz/2:
                raise ValueError('Chirp requires 0 < start < end < Nyquist frequency')
            duration = self.window_s if self.chirp_schedule == 'simultaneous' else self.window_s/4
            if not np.isfinite(self.chirp_amplitude_V) or self.chirp_amplitude_V <= 0:
                raise ValueError('Chirp amplitude must be positive and finite')
            if not 0 <= self.chirp_ramp_s <= duration/2:
                raise ValueError('Chirp ramp must fit within half the sweep duration')
        for name in ('force_jitter_fraction', 'gain_jitter_fraction', 'excitation_jitter_fraction'):
            if not 0 <= getattr(self, name) < 0.5:
                raise ValueError(f'Invalid {name}')
        if self.position_jitter_m < 0 or self.noise_std_V < 0:
            raise ValueError('Noise and position jitter must be nonnegative')
        if not np.isfinite(self.held_joint_angle_rad):
            raise ValueError('Held angle must be finite')
        return self


def actuator_drives(time, protocol, rng):
    """Return forces in N; chirp voltage is converted with per-channel N/V gains."""
    p = protocol
    drives = np.zeros((len(time), 4))
    for i in range(4):
        variation = 1+rng.uniform(-p.excitation_jitter_fraction, p.excitation_jitter_fraction)
        if p.excitation_type == 'pulse':
            phase = (time-(i*p.window_s/4+p.pulse_delay_in_slot_s))/p.pulse_width_s
            drives[:, i] = np.where((phase>=0)&(phase<=1),
                p.pulse_amplitude_N*variation*np.sin(np.pi*phase)**2, 0)
        else:
            duration = p.window_s if p.chirp_schedule == 'simultaneous' else p.window_s/4
            local = time if p.chirp_schedule == 'simultaneous' else time-i*duration
            envelope = ((local >= 0)&(local < duration)).astype(float)
            if p.chirp_ramp_s > 0:
                edge = np.clip(np.minimum(local, duration-local)/p.chirp_ramp_s, 0, 1)
                envelope *= np.sin(np.pi*edge/2)**2
            phase = 2*np.pi*(p.chirp_start_Hz*local
                             + 0.5*(p.chirp_end_Hz-p.chirp_start_Hz)/duration*local**2)
            drives[:, i] = (p.chirp_amplitude_V*variation*p.actuator_force_per_volt_N_V[i]
                            * envelope*np.sin(phase))
    return drives


def project_to_route(point, route):
    best = (float('inf'), 0.)
    distance = 0.
    for start, end in zip(route[:-1], route[1:]):
        span = end-start
        length = np.linalg.norm(span)
        u = np.clip(np.dot(point-start, span)/(length*length), 0, 1)
        candidate = (np.linalg.norm(point-(start+u*span)), distance+u*length)
        best = min(best, candidate)
        distance += length
    return float(best[1])


def make_trial(config, protocol, location, target_force, seed, *, include_reference=False):
    p = protocol.validate()
    config.validate()
    tendon = Tendon.from_config(config)
    q = p.held_joint_angle_rad
    guides = tendon.guide_positions(q)
    route = np.array([tendon.spool_position, *guides])
    positions = tendon.guide_path_coordinates(q)
    length = positions[-1]+config.routed_pairs.anchor_extension_m
    joint_coordinate = project_to_route(tendon.joint_position, route)
    physical_nominal = physical_position = None
    proximal = float(config.joint.origin_xy_m[0])
    if p.location_reference == 'total_segment':
        if proximal <= 0 or config.joint.origin_xy_m[1] != 0 or config.joint.equilibrium_rad != 0:
            raise ValueError('total_segment assumes base (0,0), joint on +x, and straight reference angle 0')
        physical_nominal = location*(proximal+config.joint.distal_length_m)
        nominal = 0.0  # Computed by projection below.
        lever = max(0.0, physical_nominal-proximal)
    elif p.location_reference == 'total_tendon':
        nominal = location*length
        lever = p.fixed_force_lever_arm_m
    else:
        lever = location*config.joint.distal_length_m
        angle = q-config.joint.equilibrium_rad
        point = tendon.joint_position+lever*np.array([np.cos(angle), np.sin(angle)])
        nominal = project_to_route(point, route)
    if not 0 <= lever <= config.joint.distal_length_m:
        raise ValueError('Force lever arm must lie along segment 2')
    rng = np.random.default_rng(seed)
    force = target_force*(1+rng.uniform(-p.force_jitter_fraction, p.force_jitter_fraction))
    jitter = rng.uniform(-p.position_jitter_m, p.position_jitter_m)
    if physical_nominal is not None:
        physical_position = physical_nominal+jitter
        if not 0 < physical_position < proximal+config.joint.distal_length_m:
            raise ValueError('Jitter moved contact outside physical finger')
        def physical_point(distance):
            if distance <= proximal:
                return np.array([distance, 0.0])
            return tendon.joint_position+(distance-proximal)*np.array([np.cos(q), np.sin(q)])
        nominal = project_to_route(physical_point(physical_nominal), route)
        position = project_to_route(physical_point(physical_position), route)
        lever = max(0.0, physical_position-proximal)
    else:
        position = nominal+jitter
    if not 0 < position < length:
        raise ValueError('Jitter moved contact outside string')
    arm = -float(tendon.path_length_derivative(q))
    if arm <= 0:
        raise ValueError('Held pose must have a positive tendon flexion moment arm')
    tension = (float(ReturnSpring(config).resisting_torque(q))+force*lever)/arm
    winding = tendon.rest_length+config.axial.slack_m+tension/tendon.k-tendon.path_length(q)
    if tension <= 0 or not 0 <= winding <= tendon.rest_length:
        raise ValueError('Requested static state is infeasible for this tendon')
    grid = np.linspace(0, length, config.string.nodes)
    contact_node = int(np.argmin(abs(grid-position)))
    if contact_node in (0, len(grid)-1):
        raise ValueError(
            f'Contact at tendon coordinate {position:.6g} m snapped to fixed endpoint '
            f'{grid[contact_node]:.6g} m. String length={length:.6g} m, '
            f'mesh spacing={grid[1]-grid[0]:.6g} m, nodes={len(grid)}. '
            'Check routed_pairs.anchor_extension_m: units are metres '
            '(30 mm = 0.03 m, not 30). If geometry is correct, refine string.nodes '
            'or choose a contact inside the free span.')
    highest = np.sqrt(tension/config.string.linear_density_kg_m)/(np.pi*(grid[1]-grid[0]))
    if highest >= p.sample_rate_Hz/2:
        raise ValueError('Sampling rate below highest mesh mode: increase sample_rate_Hz or reduce nodes')
    time = np.arange(round(p.window_s*p.sample_rate_Hz))/p.sample_rate_Hz
    drives = actuator_drives(time, p, rng)
    motion, actuator_nodes = static_response(grid,tension,config.string.linear_density_kg_m,
        config.string.distributed_damping_Ns_m2,contact_node,positions,positions,drives,p.sample_rate_Hz)
    gains = np.asarray(config.routed_pairs.sensor_gains)*(1+rng.uniform(-p.gain_jitter_fraction,p.gain_jitter_fraction,4))
    volts = motion*gains+rng.normal(0,p.noise_std_V,motion.shape)
    # No-clamp counterfactual at matched tension, excitation, gains and rest
    # state. This isolates the acoustic boundary change; it is not a simulated
    # probe-approach transient or a load-cell measurement.
    reference_volts = None
    if include_reference:
        reference_motion, _ = static_response(grid,tension,config.string.linear_density_kg_m,
            config.string.distributed_damping_Ns_m2,None,positions,positions,drives,p.sample_rate_Hz)
        reference_volts = reference_motion*gains+rng.normal(0,p.noise_std_V,reference_motion.shape)
    metadata = dict(location_fraction=location,force_label_N=target_force,
        actual_force_N=float(force),contact_nominal_m=float(nominal),contact_requested_m=float(position),
        contact_snapped_m=float(grid[contact_node]),contact_from_joint_reference_m=float(position-joint_coordinate),
        force_lever_arm_m=float(lever),tension_N=float(tension),motor_angle_rad=float(winding/config.motor.spool_radius_m),
        servo_torque_Nm=float(tension*config.motor.spool_radius_m),
        spring_resisting_torque_Nm=float(ReturnSpring(config).resisting_torque(q)),joint_angle_rad=q,string_length_m=float(length),seed=int(seed),
        constrained_actuators=[i+1 for i,node in enumerate(actuator_nodes) if node in (0,len(grid)-1,contact_node)])
    metadata.update(excitation_type=p.excitation_type,
        chirp_schedule=p.chirp_schedule if p.excitation_type == "chirp" else None,
        location_reference=p.location_reference,
        contact_segment_nominal_m=physical_nominal,
        contact_segment_requested_m=physical_position,
        contact_location_reference=('finger_base_along_segments' if physical_position is not None else 'tendon_feed'))
    result = (time,volts,drives,metadata)
    return (*result, reference_volts) if include_reference else result


def recorded_channels(time, volts, drives, metadata, protocol, reference_volts=None,
                      raw_adc=None, reference_raw_adc=None):
    """Synchronized channels; static equilibrium quantities repeat over time."""
    constant = lambda value: np.full(len(time), value, dtype=np.float64)
    channels = {"time_s": time,
        "contact_force_N": constant(metadata["actual_force_N"]),
        "contact_location_m": constant(metadata["contact_segment_requested_m"]
            if metadata.get("contact_segment_requested_m") is not None else metadata["contact_snapped_m"]),
        "contact_tendon_location_m": constant(metadata["contact_snapped_m"]),
        "contact_from_joint_reference_m": constant(metadata["contact_from_joint_reference_m"]),
        "servo_angle_rad": constant(metadata["motor_angle_rad"]),
        "servo_torque_Nm": constant(metadata["servo_torque_Nm"]),
        "string_tension_N": constant(metadata["tension_N"]),
        "load_cell_readout_N": constant(metadata["tension_N"]),
        "joint_angle_rad": constant(metadata["joint_angle_rad"]),
        "spring_resisting_torque_Nm": constant(metadata["spring_resisting_torque_Nm"])}
    voltage = drives/np.asarray(protocol.actuator_force_per_volt_N_V)
    for i in range(4):
        channels[f"piezo_sensor_{i+1}_V"] = volts[:, i]
        if raw_adc is not None:
            channels[f"piezo_sensor_{i+1}_raw_adc"] = raw_adc[:, i]
        if reference_volts is not None:
            channels[f"piezo_sensor_{i+1}_reference_V"] = reference_volts[:, i]
        if reference_raw_adc is not None:
            channels[f"piezo_sensor_{i+1}_reference_raw_adc"] = reference_raw_adc[:, i]
        channels[f"piezo_actuator_{i+1}_V"] = voltage[:, i]
        channels[f"piezo_actuator_{i+1}_force_N"] = drives[:, i]
    return channels


def generate(config, protocol, output):
    p=protocol.validate()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; existing datasets are never overwritten')
    (output/'trials').mkdir()
    config.save(output/'model_config.json')
    protocol_snapshot=asdict(p)
    protocol_snapshot['adc']=asdict(ADCConfig(**p.adc))
    (output/'protocol.json').write_text(json.dumps(protocol_snapshot,indent=2)+'\n')
    schedule=[(l,f,repeat) for l in p.location_fractions for f in p.force_levels_N for repeat in range(p.trials_per_condition)]
    rng=np.random.default_rng(p.seed)
    rng.shuffle(schedule)
    seeds=rng.integers(0,2**32-1,len(schedule))
    rows=[]
    for index,((location,force,repeat),seed) in enumerate(zip(schedule,seeds)):
        time,volts,drives,metadata,reference_volts=make_trial(
            config,p,location,force,int(seed),include_reference=True)
        trial_id=f'trial_{index:04d}'
        # Assign complete trials, balanced within every condition, never windows.
        split='train' if repeat < int(0.7*p.trials_per_condition) else ('validation' if repeat < int(0.85*p.trials_per_condition) else 'test')
        path=output/'trials'/f'{trial_id}.npz'
        adc_config = ADCConfig(**p.adc)
        raw_adc = reference_raw_adc = None
        if adc_config.enabled:
            adc_rng = np.random.default_rng(int(seed) ^ 0xADC0DE)
            raw_adc = observe_adc(volts, adc_config, p.sample_rate_Hz, adc_rng)
            reference_raw_adc = observe_adc(reference_volts, adc_config, p.sample_rate_Hz, adc_rng)
        channels = recorded_channels(time, volts, drives, metadata, p, reference_volts,
                                     raw_adc, reference_raw_adc)
        np.savez_compressed(path,**channels,piezo_V=volts.astype('float32'),
            reference_piezo_V=reference_volts.astype('float32'),
            **({"piezo_raw_adc": raw_adc, "reference_piezo_raw_adc": reference_raw_adc}
               if raw_adc is not None else {}),
            actuator_voltage_V=(drives/np.asarray(p.actuator_force_per_volt_N_V)).astype('float32'),
            actuator_force_N=drives.astype('float32'),
            location_class=np.int64(list(p.location_fractions).index(location)),force_class=np.int64(list(p.force_levels_N).index(force)),
            location_fraction=location,force_N=force,metadata_json=json.dumps(metadata))
        if index == 0:
            with (output/'example_trial.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(channels)
                writer.writerows(zip(*channels.values()))
        rows.append(dict(trial_id=trial_id,condition_id=f'L{int(location*100)}_F{force:g}',repeat=repeat,split=split,
            location_class=list(p.location_fractions).index(location),force_class=list(p.force_levels_N).index(force),
            file=str(path.relative_to(output)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metadata))
        print(f'{index+1}/{len(schedule)} {trial_id} location={location:.0%} force={force:g} N',flush=True)
    with (output/'manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary=dict(status='complete',synthetic=True,trials=len(rows),conditions=len(p.location_fractions)*len(p.force_levels_N),
        shape_per_trial=[len(time),4],sample_rate_Hz=p.sample_rate_Hz,window_s=p.window_s,
        split_counts={name:sum(row['split']==name for row in rows) for name in ('train','validation','test')},
        channel_names=list(channels),schema_version='2.2',adc_observation=asdict(ADCConfig(**p.adc)),
        actuator_force_per_volt_N_V=list(p.actuator_force_per_volt_N_V),
        labels=['location_class','force_class'],location_reference=p.location_reference,
        assumptions=['static held pose with motor winding adjusted for force torque balance','ideal hard contact on tendon',
        'servo torque is ideal spool holding torque T*r; excludes friction, gearing and inertia',
        'actuator voltage uses an uncalibrated linear force-per-volt conversion; no piezo electrical dynamics',
        'no handheld gauge dynamics','pair4 at fixed endpoint is silent unless anchor extension is measured',
        'synthetic jitter is illustrative, not experimentally calibrated',
        'reference piezo channels are a no-clamp counterfactual at matched tension and excitation, not a probe-approach recording',
        'total_segment projects physical contact onto tendon; segment 1 force has no joint torque',
        'hard contact has no force-dependent compliance; fixed-segment force magnitude is not identifiable from this model'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adc-config',type=Path,
                        help='JSON ADCConfig object; overrides protocol adc block')
    parser.add_argument('--adc-resolution-bits',type=int,
                        help='Enable ADC output with this explicit resolution')
    parser.add_argument('--adc-counts-per-volt',type=float,
                        help='Explicit affine voltage-to-code slope at ADC input')
    parser.add_argument('--adc-zero-code',type=float,
                        help='ADC code corresponding to zero conditioned input volts')
    parser.add_argument('--adc-attenuation-db',type=float,
                        help='Firmware attenuation metadata; does not set conversion slope')
    parser.add_argument('--adc-analog-gain',type=float,default=1.0)
    parser.add_argument('--adc-analog-offset-V',type=float,default=0.0)
    parser.add_argument('--adc-voltage-noise-std-V',type=float,default=0.0)
    parser.add_argument('--adc-code-noise-std',type=float,default=0.0)
    parser.add_argument('--adc-highpass-Hz',type=float)
    parser.add_argument('--adc-lowpass-Hz',type=float)
    args=parser.parse_args()
    protocol_data=json.loads(args.protocol.read_text())
    inline_adc = (args.adc_resolution_bits, args.adc_counts_per_volt, args.adc_zero_code)
    if args.adc_config and any(value is not None for value in inline_adc):
        parser.error('--adc-config cannot be combined with inline ADC mapping options')
    if any(value is not None for value in inline_adc) and not all(value is not None for value in inline_adc):
        parser.error('--adc-resolution-bits, --adc-counts-per-volt, and --adc-zero-code are required together')
    if args.adc_config:
        protocol_data['adc']=json.loads(args.adc_config.read_text())
    elif all(value is not None for value in inline_adc):
        protocol_data['adc']=asdict(ADCConfig(
            enabled=True, resolution_bits=args.adc_resolution_bits,
            attenuation_db=args.adc_attenuation_db,
            voltage_to_code_gain_counts_per_V=args.adc_counts_per_volt,
            zero_voltage_code=args.adc_zero_code,
            analog_gain_V_per_V=args.adc_analog_gain,
            analog_offset_V=args.adc_analog_offset_V,
            voltage_noise_std_V=args.adc_voltage_noise_std_V,
            code_noise_std_counts=args.adc_code_noise_std,
            highpass_cutoff_Hz=args.adc_highpass_Hz,
            lowpass_cutoff_Hz=args.adc_lowpass_Hz,
            parameter_source='assumed',
            notes='Explicit dataset CLI values; replace assumptions with measured calibration.'))
    summary=generate(ModelConfig.load(args.config),Protocol(**protocol_data),args.output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
