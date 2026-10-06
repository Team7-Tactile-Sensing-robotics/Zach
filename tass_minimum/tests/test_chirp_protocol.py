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

    def test_validation(self):
        for kwargs in ({'chirp_end_Hz':24000},{'chirp_start_Hz':0},
                       {'chirp_ramp_s':.6},{'chirp_schedule':'bad'}, {'chirp_amplitude_V':float('nan')}):
            with self.assertRaises(ValueError):
                replace(Protocol(),excitation_type='chirp',**kwargs).validate()
