"""CLI voltage training, plotted evaluation and non-destructive ADC inference."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tass.ml_cli import main
from tass.ml import load_bundle, read_feature_set, train


class MLCLITests(unittest.TestCase):
    def command(self,*arguments):
        with patch.object(sys,'argv',['tass.ml_cli',*map(str,arguments)]),contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        return json.loads(output.getvalue())

    def test_voltage_training_and_raw_adc_prediction_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'dataset';dataset.mkdir()
            time=np.arange(1000)/1000
            rng=np.random.default_rng(5)
            manifest=[]
            for split in ('train','validation','test'):
                for condition in (0,1):
                    trial=f'{split}_{condition}'
                    signal=(condition+1)*np.sin(2*np.pi*(50+30*condition)*time)+rng.normal(0,.005,len(time))
                    np.savez(dataset/f'{trial}.npz',time_s=time,piezo_sensor_1_V=signal,
                             piezo_sensor_1_raw_adc=np.zeros(len(time)),
                             contact_location_m=np.full(len(time),.045+.06*condition))
                    manifest.append({'trial_id':trial,'file':f'{trial}.npz','split':split})
            with (dataset/'manifest.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
            features=root/'features';model=root/'model'
            result=self.command('extract','--input',dataset,'--output',features,'--remove-sensor-mean',
                                '--spectral-mode','band_peaks')
            self.assertEqual(result['windows'],6)
            config=json.loads((features/'feature_config.json').read_text())
            self.assertEqual(config['sensor_representation'],'voltage')
            self.assertEqual(config['spectral_mode'],'band_peaks')
            provenance=json.loads((features/'feature_provenance.json').read_text())
            self.assertTrue(all(value.get('kind')!='adc_to_voltage_derived' for value in provenance.values()))
            trained=self.command('train','--features',features,'--output',model,'--regressor','ridge')
            self.assertEqual(trained['regressor_family'],'ridge')
            self.assertTrue(all(result['selected_model'].startswith('ridge_alpha_')
                                for result in trained['targets'].values()))
            report=self.command('evaluate','--model',model/'model_bundle.pkl','--input',dataset,'--output',root/'evaluation.json')
            self.assertEqual(report['domain'],'simulation')
            self.assertTrue((model/'training_loss.png').is_file())
            self.assertTrue((root/'evaluation.predictions.svg').is_file())
            # A synthetic ADC CSV exercises the real loader; no hardware accuracy claim.
            capture=root/'capture.csv'
            with np.load(dataset/'test_0.npz') as trial:
                codes=np.rint(1887+trial['piezo_sensor_1_V']*(4095/3.3))
            np.savetxt(capture,np.column_stack((time,codes)),delimiter=',',comments='',
                       header='time_s,piezo_sensor_1_raw_adc',fmt=['%.12g','%d'])
            digest=hashlib.sha256(capture.read_bytes()).hexdigest()
            result=self.command('predict','--model',model/'model_bundle.pkl','--input',capture,
                '--adc-full-scale-v','3.3','--adc-resolution-bits','12',
                '--adc-voltage-offset-v','-1.5','--output',root/'capture_predictions.csv')
            self.assertEqual(result['predictions'],1)
            self.assertEqual(hashlib.sha256(capture.read_bytes()).hexdigest(),digest)
            applied=json.loads((root/'capture_predictions.preprocessing.json').read_text())['feature_config']
            self.assertEqual(applied['adc_volts_per_count'],3.3/4095)
            self.assertEqual(applied['adc_voltage_offset_v'],-1.5)
            conflicting=root/'features.csv'
            conflicting.write_bytes(capture.read_bytes())
            with self.assertRaisesRegex(ValueError,'must not overwrite'):
                self.command('extract','--input',conflicting,'--output',root)
            self.assertEqual(hashlib.sha256(conflicting.read_bytes()).hexdigest(),digest)

    def test_classification_numeric_classes_probabilities_and_real_adc_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dataset=root/'dataset';dataset.mkdir()
            time=np.arange(1000)/1000
            manifest=[]
            for split in ('train','validation','test'):
                for condition in (0,1):
                    trial=f'{split}_{condition}'
                    signal=.1*(condition+1)*np.sin(2*np.pi*(50+30*condition)*time)
                    np.savez(dataset/f'{trial}.npz',time_s=time,piezo_sensor_1_V=signal,
                             piezo_sensor_1_raw_adc=np.zeros(len(time)),
                             contact_location_m=np.full(len(time),(.045,.105)[condition]),
                             contact_force_N=np.full(len(time),(1.,5.)[condition]))
                    manifest.append({'trial_id':trial,'file':f'{trial}.npz','split':split})
            with (dataset/'manifest.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
            features=root/'features';model=root/'model'
            self.command('extract','--input',dataset,'--output',features,'--remove-sensor-mean',
                         '--target','contact_location_m','--target','contact_force_N')
            report=self.command('train','--features',features,'--output',model,'--task','classification')
            self.assertEqual(report['task'],'classification')
            self.assertEqual(report['selection_metric'],'log_loss')
            bundle=load_bundle(model/'model_bundle.pkl')
            self.assertEqual(bundle['class_values']['contact_location_m'],[.045,.105])
            self.assertEqual(bundle['class_values']['contact_force_N'],[1.,5.])
            for target, result in report['targets'].items():
                self.assertEqual(result['test']['accuracy'],1.)
                selected=min(result['validation_candidates'],key=lambda name:result['validation_candidates'][name]['log_loss'])
                self.assertEqual(result['selected_model'],selected)
            evaluation=self.command('evaluate','--model',model/'model_bundle.pkl',
                '--input',dataset,'--output',root/'evaluation.json')
            self.assertEqual(evaluation['targets']['contact_force_N']['model']['confusion_matrix'],[[1,0],[0,1]])
            self.assertTrue((model/'training_loss.png').is_file())
            self.assertTrue((root/'evaluation.predictions.png').is_file())
            # Logistic scaling must use training rows only.
            feature_rows,names,_=read_feature_set(features)
            training=np.asarray([[float(row[name]) for name in names] for row in feature_rows if row['split']=='train'])
            for fitted in bundle['models'].values():
                if hasattr(fitted,'named_steps'):
                    scaler=fitted.named_steps['scaler']
                    self.assertEqual(scaler.n_samples_seen_,len(training))
                    np.testing.assert_allclose(scaler.mean_,training.mean(axis=0))
            capture=root/'capture.csv'
            with np.load(dataset/'test_1.npz') as trial:
                codes=1887+trial['piezo_sensor_1_V']/.001
            np.savetxt(capture,np.column_stack((time,codes)),delimiter=',',comments='',
                       header='time_s,piezo_sensor_1_raw_adc')
            original=capture.read_bytes()
            self.command('predict','--model',model/'model_bundle.pkl','--input',capture,
                         '--adc-volts-per-count','.001','--output',root/'predictions.csv')
            with (root/'predictions.csv').open(newline='') as stream:
                prediction=next(csv.DictReader(stream))
            self.assertEqual(float(prediction['predicted_contact_force_N']),5.)
            for target,classes in bundle['class_values'].items():
                probabilities=[float(prediction[f'probability_{target}_class_{i}']) for i in range(len(classes))]
                self.assertAlmostEqual(sum(probabilities),1.)
                self.assertEqual(float(prediction[f'predicted_{target}']),classes[int(np.argmax(probabilities))])
            applied=json.loads((root/'predictions.preprocessing.json').read_text())
            self.assertEqual(applied['class_values'],bundle['class_values'])
            self.assertEqual(capture.read_bytes(),original)
            # Held-out classes cannot silently alter the class vocabulary or get rounded.
            feature_rows[-1]['contact_force_N']='3.0'
            with (features/'features.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(feature_rows[0]));writer.writeheader();writer.writerows(feature_rows)
            with self.assertRaisesRegex(ValueError,'Unseen class 3.0'):
                train(features,root/'invalid',task='classification',plots=False)

    def test_classification_cannot_train_on_a_single_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            from tass.ml import FeatureConfig, write_feature_set
            rows=[dict(source='simulation',segment_id=split,split=split,signal=1.,contact_force_N=1.)
                  for split in ('train','validation','test')]
            write_feature_set(rows,['signal'],{},FeatureConfig(target_columns=('contact_force_N',)),root/'features')
            with self.assertRaisesRegex(ValueError,'at least two training classes'):
                train(root/'features',root/'model',task='classification',plots=False)


if __name__=='__main__':
    unittest.main()
