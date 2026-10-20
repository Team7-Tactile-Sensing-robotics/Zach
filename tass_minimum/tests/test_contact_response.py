import contextlib
import csv
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import lsim

from tass.config import ModelConfig
from tass.contact_response import ContactResponseConfig
from tass.dataset import Protocol, generate, make_trial
from tass.static_vibration import static_response


class ContactResponseTests(unittest.TestCase):
    def test_local_damping_solver_matches_independent_full_state_integration(self):
        grid=np.linspace(0,.2,11);dx=grid[1]-grid[0]
        mu=.002;tension=2.;damping=.008;rate=48000
        node=5;k=200.;c=.004
        t=np.arange(1000)/rate
        forces=(.001*np.sin(2*np.pi*200*t))[:,None]
        actual,_=static_response(grid,tension,mu,damping,node,[grid[3]],[grid[2],grid[8]],
            forces,rate,contact_mode='compliant',contact_stiffness_N_m=k,contact_damping_Ns_m=c)
        # Independent state-space assembly + matrix-exponential integration in lsim.
        size=len(grid)-2
        K=(2*np.eye(size)-np.eye(size,k=1)-np.eye(size,k=-1))*tension/(mu*dx**2)
        C=np.eye(size)*damping/mu
        K[node-1,node-1]+=k/(mu*dx);C[node-1,node-1]+=c/(mu*dx)
        A=np.block([[np.zeros((size,size)),np.eye(size)],[-K,-C]])
        B=np.zeros((2*size,1));B[2+size,0]=1/(mu*dx)
        output=np.zeros((2,2*size));output[0,1]=1;output[1,7]=1
        _,expected,_=lsim((A,B,output,np.zeros((2,1))),forces,t,interp=False)
        np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=1e-12)
        np.testing.assert_array_equal(actual[0],0)

    def test_zero_compliance_equals_uncontacted_string_and_transmits_across_contact(self):
        grid=np.linspace(0,.2,21);rate=48000
        forces=np.zeros((4800,1));forces[:100]=.001
        args=(grid,2.,.002,.008)
        plain,_=static_response(*args,None,[.04],[.16],forces,rate)
        zero,_=static_response(*args,10,[.04],[.16],forces,rate,contact_mode='compliant')
        np.testing.assert_array_equal(plain,zero)
        compliant,_=static_response(*args,10,[.04],[.16],forces,rate,contact_mode='compliant',
                                    contact_stiffness_N_m=200.,contact_damping_Ns_m=.004)
        hard,_=static_response(*args,10,[.04],[.16],forces,rate)
        self.assertGreater(np.max(abs(compliant)),1e-9)
        np.testing.assert_array_equal(hard,0)

    def test_preload_sensitivity_and_force_independent_ablation(self):
        root=Path(__file__).resolve().parents[1]
        config=ModelConfig.load(root/'config/default.json')
        protocol=Protocol(**json.loads((root/'config/experiments/dataset_segment1_30_70.json').read_text()))
        protocol=replace(protocol,window_s=.08,chirp_duration_s=.03)
        contact=ContactResponseConfig(mode='compliant',coefficient_jitter_fraction=.1)
        for location in (.3,.7):
            p=replace(protocol,contact_response=asdict(contact))
            low=make_trial(config,p,location,1.,42,include_reference=True)
            high=make_trial(config,p,location,5.,42,include_reference=True)
            self.assertEqual(low[3]['tension_N'],high[3]['tension_N'])
            np.testing.assert_array_equal(low[2],high[2])
            np.testing.assert_array_equal(low[4],high[4])
            self.assertFalse(np.allclose(low[1],high[1]))
            self.assertTrue(low[3]['contact_response_force_dependent'])
            self.assertAlmostEqual(high[3]['contact_stiffness_N_m']/low[3]['contact_stiffness_N_m'],5**(1/3))
            constant=replace(contact,stiffness_force_exponent=0.,damping_force_exponent=0.)
            p=replace(p,contact_response=asdict(constant))
            low=make_trial(config,p,location,1.,42)
            high=make_trial(config,p,location,5.,42)
            np.testing.assert_array_equal(low[1],high[1])
            self.assertFalse(low[3]['contact_response_force_dependent'])

    def test_validation(self):
        for kwargs in ({'mode':'wrong'},{'reference_force_N':0},{'stiffness_at_reference_N_m':-1},
                       {'damping_at_reference_Ns_m':-1},{'coefficient_jitter_fraction':.5},
                       {'stiffness_force_exponent':float('nan')},{'parameter_source':'calibrated_by_ml'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                ContactResponseConfig(**kwargs).validate()

    def test_fresh_collection_can_be_all_test_and_snapshot_roundtrips(self):
        root=Path(__file__).resolve().parents[1]
        config=ModelConfig.load(root/'config/default.json')
        config=replace(config,string=replace(config.string,nodes=21))
        protocol=Protocol(**json.loads((root/'config/experiments/dataset_segment1_30_70.json').read_text()))
        protocol=replace(protocol,window_s=.04,chirp_duration_s=.01,trials_per_condition=5,
                         contact_response=asdict(ContactResponseConfig(mode='compliant')))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'test'
            with contextlib.redirect_stdout(io.StringIO()):
                summary=generate(config,protocol,output,split='test')
            self.assertEqual(summary['split_counts'],{'train':0,'validation':0,'test':20})
            Protocol(**json.loads((output/'protocol.json').read_text())).validate()
            self.assertIsNone(summary['contact_force_identifiable'])
            with (output/'manifest.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(all(row['split']=='test' for row in rows))
            self.assertEqual({row['condition_id'] for row in rows},{'L30_F1','L30_F5','L70_F1','L70_F5'})


if __name__=='__main__':
    unittest.main()
