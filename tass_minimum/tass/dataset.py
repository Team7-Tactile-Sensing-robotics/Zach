"""Reproducible static-contact synthetic dataset. Run python -m tass.dataset."""
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import ModelConfig
from .models.tendon import Tendon
from .models.spring import ReturnSpring
from .static_vibration import static_response


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
    pulse_amplitude_N: float = 0.002
    pulse_width_s: float = 0.001
    pulse_delay_in_slot_s: float = 0.01
    force_jitter_fraction: float = 0.02
    position_jitter_m: float = 0.0005
    gain_jitter_fraction: float = 0.03
    excitation_jitter_fraction: float = 0.03
    noise_std_V: float = 1e-5
    seed: int = 20260929

    def validate(self):
        if self.location_reference not in {'total_tendon', 'segment2_from_joint'}:
            raise ValueError('Unknown location reference')
        if not self.location_fractions or any(not 0 < x < 1 for x in self.location_fractions):
            raise ValueError('Location fractions must lie in (0,1)')
        if not self.force_levels_N or any(not np.isfinite(x) or x <= 0 for x in self.force_levels_N):
            raise ValueError('Force magnitudes must be positive')
        if type(self.trials_per_condition) is not int or self.trials_per_condition < 5:
            raise ValueError('At least five trials per condition required for grouped splits')
        if self.window_s <= 0 or self.sample_rate_Hz <= 0 or self.pulse_width_s <= 0:
            raise ValueError('Window, sample rate and pulse width must be positive')
        if not 0 <= self.pulse_delay_in_slot_s < self.window_s/4-self.pulse_width_s:
            raise ValueError('Pulse must fit inside each actuator slot')
        if self.pulse_width_s*self.sample_rate_Hz < 8:
            raise ValueError('Pulse requires at least eight samples')
        for name in ('force_jitter_fraction', 'gain_jitter_fraction', 'excitation_jitter_fraction'):
            if not 0 <= getattr(self, name) < 0.5:
                raise ValueError(f'Invalid {name}')
        if self.position_jitter_m < 0 or self.noise_std_V < 0:
            raise ValueError('Noise and position jitter must be nonnegative')
        if not np.isfinite(self.held_joint_angle_rad):
            raise ValueError('Held angle must be finite')
        return self


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


def make_trial(config, protocol, location, target_force, seed):
    p = protocol.validate()
    config.validate()
    tendon = Tendon.from_config(config)
    q = p.held_joint_angle_rad
    guides = tendon.guide_positions(q)
    route = np.array([tendon.spool_position, *guides])
    positions = tendon.guide_path_coordinates(q)
    length = positions[-1]+config.routed_pairs.anchor_extension_m
    joint_coordinate = project_to_route(tendon.joint_position, route)
    if p.location_reference == 'total_tendon':
        nominal = location*length
        lever = p.fixed_force_lever_arm_m
    else:
        lever = location*config.joint.distal_length_m
        angle = q-config.joint.equilibrium_rad
        point = tendon.joint_position+lever*np.array([np.cos(angle), np.sin(angle)])
        nominal = project_to_route(point, route)
    if not 0 < lever <= config.joint.distal_length_m:
        raise ValueError('Force lever arm must lie along segment 2')
    rng = np.random.default_rng(seed)
    force = target_force*(1+rng.uniform(-p.force_jitter_fraction, p.force_jitter_fraction))
    position = nominal+rng.uniform(-p.position_jitter_m, p.position_jitter_m)
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
        raise ValueError('Contact snapped to endpoint')
    highest = np.sqrt(tension/config.string.linear_density_kg_m)/(np.pi*(grid[1]-grid[0]))
    if highest >= p.sample_rate_Hz/2:
        raise ValueError('Sampling rate below highest mesh mode: increase sample_rate_Hz or reduce nodes')
    time = np.arange(round(p.window_s*p.sample_rate_Hz))/p.sample_rate_Hz
    drives = np.zeros((len(time),4))
    for i in range(4):
        amplitude = p.pulse_amplitude_N*(1+rng.uniform(-p.excitation_jitter_fraction,p.excitation_jitter_fraction))
        phase = (time-(i*p.window_s/4+p.pulse_delay_in_slot_s))/p.pulse_width_s
        drives[:,i] = np.where((phase>=0)&(phase<=1), amplitude*np.sin(np.pi*phase)**2,0)
    motion, actuator_nodes = static_response(grid,tension,config.string.linear_density_kg_m,
        config.string.distributed_damping_Ns_m2,contact_node,positions,positions,drives,p.sample_rate_Hz)
    gains = np.asarray(config.routed_pairs.sensor_gains)*(1+rng.uniform(-p.gain_jitter_fraction,p.gain_jitter_fraction,4))
    volts = motion*gains+rng.normal(0,p.noise_std_V,motion.shape)
    metadata = dict(location_fraction=location,force_label_N=target_force,
        actual_force_N=float(force),contact_nominal_m=float(nominal),contact_requested_m=float(position),
        contact_snapped_m=float(grid[contact_node]),contact_from_joint_reference_m=float(position-joint_coordinate),
        force_lever_arm_m=float(lever),tension_N=float(tension),motor_angle_rad=float(winding/config.motor.spool_radius_m),
        spring_resisting_torque_Nm=float(ReturnSpring(config).resisting_torque(q)),joint_angle_rad=q,string_length_m=float(length),seed=int(seed),
        constrained_actuators=[i+1 for i,node in enumerate(actuator_nodes) if node in (0,len(grid)-1,contact_node)])
    return time,volts,drives,metadata


def generate(config, protocol, output):
    p=protocol.validate()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; existing datasets are never overwritten')
    (output/'trials').mkdir()
    config.save(output/'model_config.json')
    (output/'protocol.json').write_text(json.dumps(asdict(p),indent=2)+'\n')
    schedule=[(l,f,repeat) for l in p.location_fractions for f in p.force_levels_N for repeat in range(p.trials_per_condition)]
    rng=np.random.default_rng(p.seed)
    rng.shuffle(schedule)
    seeds=rng.integers(0,2**32-1,len(schedule))
    rows=[]
    for index,((location,force,repeat),seed) in enumerate(zip(schedule,seeds)):
        time,volts,drives,metadata=make_trial(config,p,location,force,int(seed))
        trial_id=f'trial_{index:04d}'
        # Assign complete trials, balanced within every condition, never windows.
        split='train' if repeat < int(0.7*p.trials_per_condition) else ('validation' if repeat < int(0.85*p.trials_per_condition) else 'test')
        path=output/'trials'/f'{trial_id}.npz'
        np.savez_compressed(path,time_s=time,piezo_V=volts.astype('float32'),actuator_force_N=drives.astype('float32'),
            location_class=np.int64(list(p.location_fractions).index(location)),force_class=np.int64(list(p.force_levels_N).index(force)),
            location_fraction=location,force_N=force,metadata_json=json.dumps(metadata))
        rows.append(dict(trial_id=trial_id,condition_id=f'L{int(location*100)}_F{force:g}',repeat=repeat,split=split,
            location_class=list(p.location_fractions).index(location),force_class=list(p.force_levels_N).index(force),
            file=str(path.relative_to(output)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metadata))
        print(f'{index+1}/{len(schedule)} {trial_id} location={location:.0%} force={force:g} N',flush=True)
    with (output/'manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary=dict(status='complete',synthetic=True,trials=len(rows),conditions=len(p.location_fractions)*len(p.force_levels_N),
        shape_per_trial=[len(time),4],sample_rate_Hz=p.sample_rate_Hz,window_s=p.window_s,
        split_counts={name:sum(row['split']==name for row in rows) for name in ('train','validation','test')},
        labels=['location_class','force_class'],location_reference=p.location_reference,
        assumptions=['static held pose with motor winding adjusted for force torque balance','ideal hard contact on tendon',
        'no handheld gauge dynamics','pair4 at fixed endpoint is silent unless anchor extension is measured',
        'synthetic jitter is illustrative, not experimentally calibrated'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=generate(ModelConfig.load(args.config),Protocol(**json.loads(args.protocol.read_text())),args.output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
