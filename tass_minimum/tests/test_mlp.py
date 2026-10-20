import csv
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tass.ml import (FeatureConfig,Segment,extract_features,write_feature_set,train,
                     load_bundle,predict,_matrix,_metrics)
from tass.ml_cli import main


class MLPTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(1000)/1000
        segments=[]
        for split,count in [('train',2),('validation',1),('test',1)]:
            for location,frequency in [(.045,40),(.105,80)]:
                for force in (1.,5.):
                    for repeat in range(count):
                        signal=.01*force*np.sin(2*np.pi*frequency*t)
                        segments.append(Segment(f'{split}_{location}_{force}_{repeat}',t,{
                            'piezo_sensor_1_V':signal,'piezo_sensor_2_V':.8*signal,
                            'contact_force_N':np.full(len(t),force),
                            'contact_location_m':np.full(len(t),location)},split=split,source='simulation'))
        config=FeatureConfig(sensor_ids=(1,2),remove_sensor_mean=True,spectral_mode='band_peaks',
            target_columns=('contact_location_m','contact_force_N'))
        rows,names,provenance=extract_features(segments,config)
        return segments,config,rows,names,provenance

    def test_regression_checkpoint_scaling_and_test_labels_do_not_affect_fit(self):
        _,cfg,rows,names,provenance=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            report=train(root/'features',root/'model',regressor='mlp',mlp_max_epochs=12,mlp_patience=3)
            bundle=load_bundle(root/'model/model_bundle.pkl')
            self.assertIsNot(*bundle['models'].values())
            validation=[row for row in rows if row['split']=='validation']
            training=[row for row in rows if row['split']=='train']
            for target,model in bundle['models'].items():
                self.assertEqual(model.named_steps['scaler'].n_samples_seen_,8)
                wrapper=model.named_steps['regressor']
                self.assertEqual(wrapper.transformer_.n_samples_seen_,8)
                self.assertAlmostEqual(wrapper.transformer_.mean_[0],np.mean([row[target] for row in training]))
                result=report['targets'][target];selected=result['selected_model']
                history=result['epoch_histories'][selected]
                self.assertEqual(history['selected_epoch'],int(np.argmin(history['validation_loss']))+1)
                measured=_metrics(_matrix(validation,[target]).ravel(),model.predict(_matrix(validation,names)))
                self.assertAlmostEqual(measured['mse'],result['validation_candidates'][selected]['mse'])
            self.assertTrue((root/'model/training_loss.png').is_file())
            self.assertTrue((root/'model/candidate_comparison.png').is_file())
            for row in rows:
                if row['split']=='test':
                    row['contact_force_N']=9999.
                    row['contact_location_m']=999.
            write_feature_set(rows,names,provenance,cfg,root/'changed')
            changed=train(root/'changed',root/'changed_model',regressor='mlp',plots=False,
                          mlp_max_epochs=12,mlp_patience=3)
            second=load_bundle(root/'changed_model/model_bundle.pkl')
            for target in bundle['models']:
                self.assertEqual(report['targets'][target]['selected_model'],changed['targets'][target]['selected_model'])
                first=bundle['models'][target].named_steps['regressor'].regressor_
                other=second['models'][target].named_steps['regressor'].regressor_
                for a,b in zip(first.coefs_,other.coefs_):np.testing.assert_array_equal(a,b)

    def test_mlp_classifier_and_regressor_infer_calibrated_adc_without_modifying_capture(self):
        segments,cfg,rows,names,provenance=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            segment=segments[-1]
            capture=root/'capture.csv'
            with capture.open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['time_s','piezo_sensor_1_raw_adc','piezo_sensor_2_raw_adc'])
                writer.writerows(zip(segment.time_s,segment.columns['piezo_sensor_1_V']/.001+1887,
                                      segment.columns['piezo_sensor_2_V']/.001+1887))
            original=hashlib.sha256(capture.read_bytes()).hexdigest()
            for task in ('classification','regression'):
                options={'classifier':'mlp'} if task=='classification' else {'regressor':'mlp'}
                report=train(root/'features',root/task,task=task,mlp_max_epochs=5,mlp_patience=2,
                             plots=False,**options)
                bundle=load_bundle(root/task/'model_bundle.pkl')
                output=root/(task+'.csv')
                result=predict(root/task/'model_bundle.pkl',capture,output,adc_volts_per_count=.001,plots=False)
                self.assertEqual(len(result),1)
                for target in bundle['target_names']:
                    self.assertTrue(np.isfinite(result[0]['predicted_'+target]))
                    if task=='classification':
                        self.assertIn(result[0]['predicted_'+target],bundle['class_values'][target])
                        p=[result[0][f'probability_{target}_class_{i}'] for i in range(2)]
                        self.assertAlmostEqual(sum(p),1.)
                self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),original)
                metadata=json.loads(output.with_suffix('.preprocessing.json').read_text())
                self.assertEqual(metadata['feature_config']['spectral_mode'],'band_peaks')

    def test_cli_supports_both_mlp_tasks(self):
        _,cfg,rows,names,provenance=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            for task,option in [('regression','--regressor'),('classification','--classifier')]:
                arguments=['tass.ml_cli','train','--features',str(root/'features'),
                    '--output',str(root/task),'--task',task,option,'mlp',
                    '--mlp-max-epochs','3','--mlp-patience','2','--mlp-seed','7']
                with patch('sys.argv',arguments),contextlib.redirect_stdout(io.StringIO()) as output:
                    main()
                report=json.loads(output.getvalue())
                self.assertEqual(report['task'],task)
                self.assertEqual(len(load_bundle(root/task/'model_bundle.pkl')['models']),2)
                self.assertTrue((root/task/'training_loss.svg').exists())


if __name__=='__main__':
    unittest.main()
