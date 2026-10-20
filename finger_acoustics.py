"""Stateful small-deflection acoustic preview using the physical finger's route.

The spool -> P1 -> P2 -> P3 -> P4 -> load-cell path is unfolded into one taut
fixed-fixed string. Guides impose no additional acoustic constraints. Each P
location has an exciter/receiver pair; receiver volts use demonstration gains.
Geometry/tension are frozen within a short block and updated between blocks.
Moving-boundary terms and acoustic feedback on the joint are omitted.
"""
import math
import numpy as np
from scipy.linalg import eigh, expm

from tass_minimum.tass.data import Measurement
from tass_minimum.tass.models.string import analytical_frequencies
from tass_minimum.tass.processing import extract_frequency_features
from tass_minimum.tass.static_vibration import oscillator_response


class FingerAcoustics:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mode_numbers = np.arange(1, parameters.acoustic_modes+1)
        self.state = np.zeros((parameters.acoustic_modes, 2))  # q [m], qdot [m/s]
        self.phase_rad = np.zeros(4)
        self.sample_count = 0
        self.history = np.empty((0, 9))  # time, four receiver voltages, four forces
        self.last_volts = np.empty((0, 4))
        self.length_m = 0.0
        self.guide_positions_m = np.zeros(4)
        self.tension_N = 0.0
        self.joint_angle_rad = 0.0
        self.frequencies_Hz = np.zeros(parameters.acoustic_modes)
        self.retained = np.zeros(parameters.acoustic_modes, dtype=bool)
        self.status = 'Wind the motor to tension the tendon.'
        self.contact_enabled = False
        self.contact_position_ratio = parameters.acoustic_contact_position_ratio
        self.contact_force_N = parameters.acoustic_contact_force_N
        self._contact_cache_key = None
        self._stiffness = np.zeros((parameters.acoustic_modes, parameters.acoustic_modes))

    @property
    def contact_position_m(self):
        return self.contact_position_ratio*self.length_m

    @property
    def contact_stiffness_N_m(self):
        return (self.contact_force_N*self.parameters.acoustic_contact_stiffness_per_N
                if self.contact_enabled else 0.0)

    @property
    def contact_damping_Ns_m(self):
        return (self.contact_force_N*self.parameters.acoustic_contact_damping_per_N
                if self.contact_enabled else 0.0)

    @property
    def contact_active(self):
        return self.contact_enabled and self.contact_force_N > 0

    def set_contact(self, enabled, position_ratio, force_N):
        """Prescribed held-contact preload; position follows a fraction of the route.

        k_c = alpha*F and c_c = beta*F are demonstration tangent contact laws.
        They act on small vibration about a held contact, not on joint mechanics.
        The constant pressing deflection and impact impulse are not modeled.
        """
        if not math.isfinite(position_ratio) or not 0 <= position_ratio <= 1:
            raise ValueError('Contact position must be a finite fraction between 0 and 1')
        if not math.isfinite(force_N) or force_N < 0:
            raise ValueError('Contact force must be finite and nonnegative')
        self.contact_enabled = bool(enabled)
        self.contact_position_ratio = float(position_ratio)
        self.contact_force_N = float(force_N)

    @property
    def time_s(self):
        return self.sample_count/self.parameters.acoustic_sample_rate_Hz

    def set_operating_point(self, tendon, joint_angle_rad, tension_N):
        if not math.isfinite(tension_N) or tension_N < 0 or not math.isfinite(joint_angle_rad):
            raise ValueError('Acoustic pose and nonnegative tension must be finite')
        self.length_m = float(tendon.path_length(joint_angle_rad))
        self.guide_positions_m = np.asarray(tendon.guide_path_coordinates(joint_angle_rad), float)
        self.tension_N = float(tension_N)
        self.joint_angle_rad = float(joint_angle_rad)
        p = self.parameters
        self.frequencies_Hz = analytical_frequencies(
            self.length_m, self.tension_N, p.acoustic_linear_density_kg_m, p.acoustic_modes)
        self._stiffness = np.diag((2*np.pi*self.frequencies_Hz)**2)
        self._use_contact = (self.contact_active and 0 < self.contact_position_ratio < 1
                             and self.tension_N >= p.acoustic_min_tension_N
                             and (self.contact_stiffness_N_m > 0 or self.contact_damping_Ns_m > 0))
        if self._use_contact:
            self._prepare_contact()
        self.retained = ((self.frequencies_Hz < 0.4*p.acoustic_sample_rate_Hz)
                         & (self.tension_N >= p.acoustic_min_tension_N))
        if not self._use_contact:
            self.state[~self.retained] = 0.0
        if self.tension_N < p.acoustic_min_tension_N:
            self.status = 'Tendon slack/unloaded: taut-string vibration is suspended. Wind the motor.'
        elif not np.any(self.retained):
            self.status = 'No modes within the sample bandwidth. Increase acoustic_sample_rate_Hz.'
        else:
            self.status = 'Taut string'

    def _prepare_contact(self):
        """Couple sine modes through local stiffness and damping; cache the ZOH map."""
        p = self.parameters
        key = (self.length_m, self.tension_N, self.contact_position_ratio,
               self.contact_stiffness_N_m, self.contact_damping_Ns_m)
        if key != self._contact_cache_key:
            mass = p.acoustic_linear_density_kg_m*self.length_m/2
            shape = self.mode_shapes([self.contact_position_m])[0]
            coupling = np.outer(shape, shape)/mass
            K = self._stiffness + self.contact_stiffness_N_m*coupling
            C = np.diag(2*p.acoustic_damping_ratio*2*np.pi*self.frequencies_Hz)
            C += self.contact_damping_Ns_m*coupling
            omega2, basis = eigh(K)
            frequencies = np.sqrt(np.maximum(omega2, 0))/(2*np.pi)
            keep = frequencies < 0.4*p.acoustic_sample_rate_Hz
            basis = basis[:, keep]
            size = basis.shape[1]
            # q_c'' + C_c q_c' + diag(omega_c²) q_c = U.T * modal_forcing.
            A = np.block([[np.zeros((size, size)), np.eye(size)],
                          [-np.diag(omega2[keep]), -(basis.T @ C @ basis)]])
            B = np.vstack((np.zeros((size, size)), np.eye(size)))
            augmented = np.zeros((3*size, 3*size))
            augmented[:2*size, :2*size] = A
            augmented[:2*size, 2*size:] = B
            transition = expm(augmented/p.acoustic_sample_rate_Hz)
            self._contact_transition = transition[:2*size, :2*size]
            self._contact_input = transition[:2*size, 2*size:]
            self._contact_basis = basis
            self._contact_frequencies = frequencies
            self._contact_stiffness = K
            self._contact_cache_key = key
        self._stiffness = self._contact_stiffness
        self.frequencies_Hz = self._contact_frequencies.copy()

    def _contact_response(self, forcing):
        # Keep state in the original sine basis between updates so moving or
        # removing contact does not reset displacement/velocity or swap modes.
        U = self._contact_basis
        size = U.shape[1]
        z = np.concatenate((U.T @ self.state[:, 0], U.T @ self.state[:, 1]))
        result = np.empty((len(forcing), self.parameters.acoustic_modes))
        inputs = (forcing @ U) @ self._contact_input.T
        for i, u in enumerate(inputs):
            result[i] = U @ z[:size]
            z = self._contact_transition @ z + u
        self.state[:, 0], self.state[:, 1] = U @ z[:size], U @ z[size:]
        return result

    def mode_shapes(self, positions_m):
        x = np.atleast_1d(positions_m)
        shapes = np.sin(np.pi*x[:, None]*self.mode_numbers[None, :]/self.length_m)
        shapes[(x == 0) | (x == self.length_m)] = 0.0
        return shapes

    def advance(self, tendon, joint_angle_rad, tension_N, enabled, frequencies_Hz, amplitudes_N):
        """Advance a block; disabling transmitters preserves the free ringdown."""
        p = self.parameters
        enabled = np.asarray(enabled, dtype=bool)
        frequencies = np.asarray(frequencies_Hz, dtype=float)
        amplitudes = np.asarray(amplitudes_N, dtype=float)
        if any(v.shape != (4,) for v in (enabled, frequencies, amplitudes)):
            raise ValueError('Provide four exciter flags, frequencies, and amplitudes')
        if (not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0)
                or np.any(frequencies >= 0.4*p.acoustic_sample_rate_Hz)):
            raise ValueError('Drive frequencies must be positive and below 0.4 times the sample rate')
        if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
            raise ValueError('Drive amplitudes must be finite and nonnegative')
        self.set_operating_point(tendon, joint_angle_rad, tension_N)
        count = round(p.acoustic_block_s*p.acoustic_sample_rate_Hz)
        offsets = np.arange(count)/p.acoustic_sample_rate_Hz
        drive = np.sin(self.phase_rad + 2*np.pi*offsets[:, None]*frequencies)
        drive *= amplitudes*enabled
        self.phase_rad = (self.phase_rad + 2*np.pi*frequencies*count/p.acoustic_sample_rate_Hz) % (2*np.pi)
        shapes = self.mode_shapes(self.guide_positions_m)
        # For sine modes, modal mass = mu*L/2; actuators apply transverse forces.
        forcing = drive @ shapes * (2/(p.acoustic_linear_density_kg_m*self.length_m))
        if self._use_contact:
            q = self._contact_response(forcing)
        else:
            q = np.zeros((count, p.acoustic_modes))
            for j in np.flatnonzero(self.retained):
                omega = 2*np.pi*self.frequencies_Hz[j]
                q[:, j], _, self.state[j] = oscillator_response(
                    omega**2, 2*p.acoustic_damping_ratio*omega, forcing[:, j],
                    p.acoustic_sample_rate_Hz, self.state[j])
        self.last_volts = (q @ shapes.T)*np.asarray(p.acoustic_receiver_gains_V_m)
        times = (self.sample_count+np.arange(count))/p.acoustic_sample_rate_Hz
        block = np.column_stack((times, self.last_volts, drive))
        if not np.all(np.isfinite(block)) or not np.all(np.isfinite(self.state)):
            raise ValueError('Acoustic response became nonfinite; check acoustic parameters')
        limit = round(p.acoustic_history_s*p.acoustic_sample_rate_Hz)
        self.history = np.concatenate((self.history, block))[-limit:]
        self.sample_count += count

    def spectrum(self):
        if len(self.history) < 4:
            return None
        # Multiple independent drives have no single scalar FRF. Reuse voltage
        # spectrum processing only, with its scalar FRF explicitly disabled.
        measurement = Measurement(self.history[:, 0], self.history[:, 1:5],
                                  np.zeros(len(self.history)))
        return extract_frequency_features(measurement)

    def displacement(self, positions_m):
        return self.mode_shapes(positions_m) @ self.state[:, 0]

    def energy_J(self):
        mass = self.parameters.acoustic_linear_density_kg_m*self.length_m/2
        return float(0.5*mass*(np.sum(self.state[:, 1]**2)
                              + self.state[:, 0] @ self._stiffness @ self.state[:, 0]))


def acoustic_svg(model, enabled):
    """Unfolded route with real guide coordinates and magnified deflection."""
    length = model.length_m
    positions = np.linspace(0, length, 401)
    displacement = model.displacement(positions)
    gain = model.parameters.acoustic_deflection_magnification
    scale = 820/length
    y = 142-displacement*scale*gain
    clipped = bool(np.any((y < 78) | (y > 210)))
    y = np.clip(y, 78, 210)
    points = ' '.join(f'{60+x*scale:.2f},{v:.2f}' for x, v in zip(positions, y))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 275" role="img" aria-label="Unfolded tendon acoustic displacement" style="width:100%;height:auto">',
        '<rect width="960" height="275" rx="14" fill="#0e1728"/>',
        '<g font-family="sans-serif" font-size="12" fill="#a7b5c8">',
        '<text x="28" y="30" fill="#edf4ff" font-size="15">TENDON VIBRATION · UNFOLDED ROUTE</text>',
        f'<text x="28" y="53">Acoustic time {model.time_s*1000:.1f} ms · deflection ×{gain:g} · peak {np.max(abs(displacement))*1e6:.2f} µm</text>',
        '<line x1="60" y1="142" x2="880" y2="142" stroke="#53657d" stroke-dasharray="5 5"/>',
        f'<polyline points="{points}" fill="none" stroke="#4cd5ca" stroke-width="2.5"/>',
        '<circle cx="60" cy="142" r="5" fill="#edf4ff"/><circle cx="880" cy="142" r="5" fill="#edf4ff"/>',
        '<text x="60" y="252" text-anchor="middle">Spool · fixed</text>',
        '<text x="880" y="252" text-anchor="middle">Load cell · fixed</text>',
    ]
    for i, (position, on) in enumerate(zip(model.guide_positions_m, enabled)):
        x = 60+position*scale
        color = '#ffb667' if on else '#8191a9'
        parts.extend([
            f'<line x1="{x:.2f}" y1="96" x2="{x:.2f}" y2="216" stroke="{color}" stroke-dasharray="3 5"/>',
            f'<text x="{x:.2f}" y="87" text-anchor="middle" fill="{color}">P{i+1} TX {"ON" if on else "off"}</text>',
            f'<text x="{x:.2f}" y="230" text-anchor="middle">RX{i+1} · {position*1000:.1f} mm</text>',
        ])
    if clipped:
        parts.append('<text x="610" y="30" fill="#ffb667">Display clipped; reduce magnification</text>')
    if model.contact_enabled:
        x = 60+model.contact_position_m*scale
        color = '#ff758f' if model.contact_active else '#8191a9'
        # Force marker remains at the exact selected coordinate, including anchors.
        parts.extend([
            f'<line x1="{x:.2f}" y1="101" x2="{x:.2f}" y2="131" stroke="{color}" stroke-width="3"/>',
            f'<path d="M {x-6:.2f} 124 L {x:.2f} 134 L {x+6:.2f} 124" fill="none" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{x:.2f}" cy="142" r="7" fill="none" stroke="{color}" stroke-width="2"/>',
            f'<text x="480" y="269" text-anchor="middle" fill="{color}">Contact · {model.contact_force_N:.2f} N · {model.contact_position_m*1000:.1f} mm from spool</text>',
        ])
    parts.append('</g></svg>')
    return ''.join(parts)


def render_acoustics_panel(parameters):
    """Render the acoustic section without coupling Streamlit into the solver."""
    import streamlit as st

    st.divider()
    st.header('Acoustic dynamics')
    st.caption('Four excitation/receiver pairs · current finger geometry and tension · simulated signals')

    @st.fragment(run_every=parameters.acoustic_refresh_interval_s)
    def panel():
        finger = st.session_state.finger_model

        def fresh_model():
            model = FingerAcoustics(parameters)
            model.set_operating_point(finger.tendon, finger.state.joint_angle_rad,
                                      finger.readouts()['tension_N'])
            st.session_state.finger_acoustics = model
            return model

        acoustic = st.session_state.get('finger_acoustics')
        if acoustic is None:
            acoustic = fresh_model()
        left, middle, right = st.columns([1, 1, 2])
        running = left.toggle('Run acoustic model', key='acoustic_running')
        reset = middle.button('Reset acoustic response', key='acoustic_reset')
        if reset:
            acoustic = fresh_model()
        right.caption(f'Slow playback: {parameters.acoustic_block_s*1000:g} ms every '
                      f'{parameters.acoustic_refresh_interval_s:g} s. '
                      'Pause the mechanical simulation above to inspect a held pose.')
        enabled, frequencies, amplitudes = [], [], []
        for i, column in enumerate(st.columns(4)):
            with column:
                enabled.append(st.toggle(f'Excite P{i+1}', key=f'acoustic_tx_{i+1}'))
                frequencies.append(st.number_input(f'P{i+1} frequency (Hz)', min_value=1.0,
                    max_value=float(np.nextafter(0.4*parameters.acoustic_sample_rate_Hz, 0)),
                    value=float(parameters.acoustic_drive_frequencies_Hz[i]), step=10.0,
                    key=f'acoustic_frequency_{i+1}'))
                amplitudes.append(st.number_input(f'P{i+1} peak force (mN)', min_value=0.0,
                    value=parameters.acoustic_drive_amplitudes_N[i]*1000, step=0.1,
                    key=f'acoustic_amplitude_{i+1}')/1000)
        st.subheader('Contact on the string')
        contact_switch, contact_location, contact_load = st.columns([1, 2, 2])
        contact_enabled = contact_switch.toggle('Apply contact', key='acoustic_contact_enabled')
        position_ratio = contact_location.slider('Contact position (% from spool)',
            min_value=0.0, max_value=100.0,
            value=parameters.acoustic_contact_position_ratio*100, step=0.1,
            key='acoustic_contact_position')/100
        contact_force = contact_load.slider('Contact force (N)', min_value=0.0,
            max_value=float(parameters.acoustic_contact_max_force_N),
            value=float(parameters.acoustic_contact_force_N), step=0.01,
            key='acoustic_contact_force')
        acoustic.set_contact(contact_enabled, position_ratio, contact_force)
        # Recompute the selected acoustic boundary at the last sampled pose, even
        # when paused; no time integration or vibration reset happens here.
        acoustic.set_operating_point(finger.tendon, acoustic.joint_angle_rad, acoustic.tension_N)
        if running and not reset:
            acoustic.advance(finger.tendon, finger.state.joint_angle_rad,
                             finger.readouts()['tension_N'], enabled, frequencies, amplitudes)
        st.caption(f'Contact location: {acoustic.contact_position_m*1000:.2f} mm of '
                   f'{acoustic.length_m*1000:.2f} mm, measured along the route from the spool. '
                   '0% = spool; 100% = load-cell anchor. The selected fraction follows the route as it bends.')
        if contact_enabled:
            st.caption(f'Contact stiffness: {acoustic.contact_stiffness_N_m:.1f} N/m · '
                       f'contact damping: {acoustic.contact_damping_Ns_m:.4f} N·s/m. '
                       'Pressing force sets a local vibration support using demonstration constants. '
                       'This does not calculate static indentation or change joint motion/tendon tension.')
            if position_ratio in (0., 1.):
                st.info('This point is already a fixed anchor. Contact here has no additional acoustic effect.')
            elif contact_force == 0:
                st.caption('Zero contact force: the added contact support is inactive.')
        if not np.any(acoustic.retained):
            st.info(acoustic.status)
        elif not any(enabled):
            st.caption('All transmitters off. Existing vibration rings down while the acoustic model runs.')
        st.markdown(acoustic_svg(acoustic, enabled), unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        a.metric('Acoustic tension', f'{acoustic.tension_N:.3f} N')
        b.metric('Routed acoustic length', f'{acoustic.length_m*1000:.2f} mm')
        c.metric('Fundamental (model)', f'{acoustic.frequencies_Hz[0]:.1f} Hz' if acoustic.retained[0] else '—')
        d.metric('Retained modes', f'{int(acoustic.retained.sum())} / {parameters.acoustic_modes}')
        st.caption(f'Last acoustic pose: {math.degrees(acoustic.joint_angle_rad):.2f}°. '
                   'Pausing acoustics freezes its state and plots. Turning off excitation allows decay.')
        spectrum = acoustic.spectrum()
        if spectrum is None:
            st.info('Wind the motor, switch on one or more exciters, then enable Run acoustic model.')
        else:
            history = acoustic.history
            names = [f'RX{i+1} (V)' for i in range(4)]
            st.subheader('Recording piezos')
            selected = st.multiselect('Receivers to plot', names, default=names, key='acoustic_receivers')
            indices = [names.index(name) for name in selected]
            # Keep full sample resolution; a subsampled waveform could alias.
            recent = history[history[:, 0] >= history[-1, 0]-0.02]
            if selected:
                st.caption('Receiver waveforms · voltage (V)')
                st.line_chart({'Acoustic time (ms)': recent[:, 0]*1000,
                               **{names[i]: recent[:, i+1] for i in indices}},
                              x='Acoustic time (ms)', y=selected, height=230)
                upper = min(0.5*parameters.acoustic_sample_rate_Hz,
                            max(1000., float(np.max(acoustic.frequencies_Hz[acoustic.retained], initial=0))*1.1,
                                max((f for f, on in zip(frequencies, enabled) if on), default=0)*1.2))
                bins = spectrum.frequency_Hz <= upper
                st.caption('Receiver spectra · amplitude (V)')
                st.line_chart({'Frequency (Hz)': spectrum.frequency_Hz[bins],
                               **{names[i]: spectrum.amplitude_V[bins, i] for i in indices}},
                              x='Frequency (Hz)', y=selected, height=230)
            st.caption(f'Receiver voltage and Hann-windowed amplitude spectrum · '
                       f'{len(history)/parameters.acoustic_sample_rate_Hz*1000:.1f} ms history · '
                       f'{spectrum.features["frequency_resolution_Hz"]:.1f} Hz bin spacing. '
                       'Peaks include drive frequencies and transients; they are not automatic resonance estimates.')
            for i, column in enumerate(st.columns(4)):
                rms = float(np.sqrt(np.mean(acoustic.last_volts[:, i]**2)))
                peak = spectrum.features['channels'][i]['dominant_frequency_Hz']
                column.metric(f'RX{i+1} RMS (last block)', f'{rms*1000:.3f} mV')
                column.caption(f'Spectral peak: {peak:.1f} Hz' if peak is not None else 'No signal')
            with st.expander('Excitation force traces'):
                force_names = [f'TX{i+1} (mN)' for i in range(4)]
                st.line_chart({'Acoustic time (ms)': recent[:, 0]*1000,
                               **{name: recent[:, i+5]*1000 for i, name in enumerate(force_names)}},
                              x='Acoustic time (ms)', y=force_names, height=180)
        with st.expander('Acoustic model assumptions'):
            st.write('The route is unfolded into one fixed-fixed taut string from spool to load cell. '
                     'P4 is inside the span. Intermediate guides add no reflection or clamping. '
                     'Each pair applies a transverse force and records local displacement through a gain.')
            st.latex(r'\ddot a_n+2\zeta\omega_n\dot a_n+\omega_n^2a_n='
                     r'\frac{2}{\mu L}\sum_i F_i(t)\sin(n\pi s_i/L),\quad '
                     r'\omega_n=\frac{n\pi}{L}\sqrt{T/\mu}')
            st.write('A held contact adds local stiffness k = alpha × contact force and damping '
                     'c = beta × contact force, coupling the string modes at the chosen point. '
                     'This is a linearized contact about a pressed equilibrium, not an impact or '
                     'static-deflection model. Contact-force calibration is still required.')
            st.write('Geometry and tension are refreshed each acoustic block. This slow-geometry '
                     'approximation has its own playback clock; moving-boundary terms and feedback '
                     'on the mechanical joint are omitted. Guide friction, piezo electronics, '
                     'and noise are not modeled yet. Receiver voltages and exciter forces need hardware calibration.')
            st.caption('Modes above 40% of the sample rate are omitted and their state is discarded. '
                       'Slack tendon motion is outside this model; modal state is cleared when tension is lost. '
                       'Settings and receiver gains live in finger_parameters.py. No scalar multi-input FRF is reported.')

    panel()
