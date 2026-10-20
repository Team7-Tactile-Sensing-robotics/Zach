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

from tass.ml import FeatureConfig,Segment,extract_features,train,write_feature_set
from tass.ml_compare import summarize,main


class ComparisonTests(unittest.TestCase):
    def test_comparison_cli_runs_all_families_and_preserves_ridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);benchmark=root/'benchmark';benchmark.mkdir()
            t=np.arange(1000)/1000
            segments=[];manifest=[]
            for split in ('train','validation','test'):
                for label in (0,1):
                    name=f'{split}_{label}'
                    columns={'piezo_sensor_1_V':(.1+.1*label)*np.sin(2*np.pi*(40+40*label)*t),
                        'contact_location_m':np.full(len(t),(.045,.105)[label]),
                        'contact_force_N':np.full(len(t),(1.,5.)[label])}
                    segments.append(Segment(name,t,columns,split=split,source='simulation'))
                    np.savez(benchmark/(name+'.npz'),time_s=t,**columns)
                    manifest.append(dict(trial_id=name,split=split,file=name+'.npz'))
            with (benchmark/'manifest.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
            cfg=FeatureConfig(target_columns=('contact_location_m','contact_force_N'))
            rows,names,provenance=extract_features(segments,cfg)
            write_feature_set(rows,names,provenance,cfg,root/'features')
            train(root/'features',root/'ridge',regressor='ridge',plots=False)
            model=root/'ridge/model_bundle.pkl';original=hashlib.sha256(model.read_bytes()).hexdigest()
            arguments=['tass.ml_compare','--features',str(root/'features'),'--ridge-model',str(model),
                       '--input',str(benchmark),'--output',str(root/'comparison'),
                       '--mlp-max-epochs','2','--mlp-patience','1']
            with patch('sys.argv',arguments),contextlib.redirect_stdout(io.StringIO()):main()
            report=json.loads((root/'comparison/comparison.json').read_text())
            self.assertEqual(len(report['models']),4)
            self.assertEqual(report['split_counts'],{'train':2,'validation':2,'test':2})
            self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(),original)
            self.assertTrue((root/'comparison/mlp_classifier/model/training_loss.png').exists())

    def test_matched_comparison_and_rejection_of_different_splits_or_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for split in ('train','validation','test'):
                for condition in (0,1):
                    rows.append(dict(source='simulation',segment_id=f'{split}_{condition}',split=split,
                        signal=float(condition),contact_force_N=(1.,5.)[condition],
                        contact_location_m=(.045,.105)[condition]))
            cfg=FeatureConfig(target_columns=('contact_location_m','contact_force_N'))
            write_feature_set(rows,['signal'],{},cfg,root/'features')
            benchmark=root/'benchmark';benchmark.mkdir()
            with (benchmark/'manifest.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['trial_id','split']);writer.writeheader()
                writer.writerows({'trial_id':row['segment_id'],'split':'test'} for row in rows if row['split']=='test')
            specs=[]
            for key in ('first','second'):
                train(root/'features',root/key,regressor='ridge',plots=False)
                predictions=root/(key+'.csv')
                output=[dict(source='simulation',segment_id=row['segment_id'],
                    contact_force_N=row['contact_force_N'],contact_location_m=row['contact_location_m'],
                    predicted_contact_force_N=row['contact_force_N'],
                    predicted_contact_location_m=row['contact_location_m']) for row in rows if row['split']=='test']
                with predictions.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
                specs.append({'name':key,'bundle':root/key/'model_bundle.pkl','predictions':predictions})
            report=summarize(root/'features',benchmark,specs,root/'comparison')
            self.assertTrue(report['same_features_and_splits_verified'])
            self.assertTrue(report['benchmark_previously_evaluated'])
            self.assertTrue((root/'comparison/comparison.png').exists())
            split_path=root/'second/splits.json'
            split=split_path.read_text()
            edited=json.loads(split);edited['train']=['wrong_trial']
            split_path.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError,'identical complete-trial splits'):
                summarize(root/'features',benchmark,specs,root/'bad')
            split_path.write_text(split)
            predictions=root/'second.csv'
            predictions.write_text(predictions.read_text().replace('test_1','wrong_trial'))
            with self.assertRaisesRegex(ValueError,'same complete benchmark trial set'):
                summarize(root/'features',benchmark,specs,root/'bad')


if __name__=='__main__':
    unittest.main()
