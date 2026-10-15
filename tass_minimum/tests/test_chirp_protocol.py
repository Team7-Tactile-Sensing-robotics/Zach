import unittest
from dataclasses import replace
import numpy as np
from scipy.signal import chirp
from tass.dataset import Protocol, actuator_drives

class ChirpProtocolTests(unittest.TestCase):
    def test_continuous_matches_independent_chirp(self):
        p=replace(Protocol(),excitation_type='chirp',chirp_ramp_s=0,excitation_jitter_fraction=0)
        t=np.arange(48000)/48000
        f=actuator_drives(t,p.validate(),np.random.default_rng(0))
        reference=chirp(t,f0=20,f1=2000,t1=1,method='linear',phi=-90)*.002
        for i in range(4):
            np.testing.assert_allclose(f[:,i],reference,atol=1e-14)
        self.assertGreater(np.count_nonzero(abs(f[:,0])>1e-8),47000)

    def test_sequential_slots_and_reproducibility(self):
        p=replace(Protocol(),excitation_type='chirp',chirp_schedule='sequential')
        t=np.arange(48000)/48000
        a=actuator_drives(t,p.validate(),np.random.default_rng(42))
        np.testing.assert_array_equal(a,actuator_drives(t,p,np.random.default_rng(42)))
        for i in range(4):
            self.assertTrue(np.all(a[(t<i/4)|(t>=(i+1)/4),i]==0))
            self.assertGreater(np.count_nonzero(a[:,i]),11000)

    def test_short_chirp_finishes_sweep_then_stops_during_recording(self):
        p=replace(Protocol(),excitation_type='chirp',window_s=1.5,chirp_duration_s=.3,
                  chirp_ramp_s=0,excitation_jitter_fraction=0)
        t=np.arange(72000)/48000
        forces=actuator_drives(t,p.validate(),np.random.default_rng(0))
        active=t<.3
        reference=chirp(t[active],f0=20,f1=2000,t1=.3,method='linear',phi=-90)*.002
        for i in range(4):
            np.testing.assert_allclose(forces[active,i],reference,atol=1e-14)
        np.testing.assert_array_equal(forces[~active],0)

    def test_short_sequential_chirps_leave_decay_time_in_each_slot(self):
        p=replace(Protocol(),excitation_type='chirp',chirp_schedule='sequential',
                  window_s=1.5,chirp_duration_s=.3)
        t=np.arange(72000)/48000
        forces=actuator_drives(t,p.validate(),np.random.default_rng(0))
        for i in range(4):
            active=(t>=i*.375)&(t<i*.375+.3)
            np.testing.assert_array_equal(forces[~active,i],0)
            self.assertGreater(np.count_nonzero(forces[active,i]),14000)

    def test_validation(self):
        for kwargs in ({'chirp_end_Hz':24000},{'chirp_start_Hz':0},
                       {'chirp_ramp_s':.6},{'chirp_schedule':'bad'}, {'chirp_amplitude_V':float('nan')},
                       {'chirp_duration_s':0}, {'chirp_duration_s':-1},
                       {'chirp_duration_s':float('nan')}, {'chirp_duration_s':float('inf')},
                       {'chirp_duration_s':1.1},
                       {'chirp_duration_s':.3,'chirp_schedule':'sequential'},
                       {'chirp_duration_s':.005,'chirp_ramp_s':.005}):
            with self.assertRaises(ValueError):
                replace(Protocol(),excitation_type='chirp',**kwargs).validate()
