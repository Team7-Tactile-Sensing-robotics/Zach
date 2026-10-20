import csv
from pathlib import Path
import tempfile
import unittest

from tass.segment1_validation import check_collections, discrete_scores


class Segment1ValidationTests(unittest.TestCase):
    def test_fresh_collection_rejects_overlapping_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths={name:Path(directory)/name for name in ('training','fresh_test','stress_test')}
            for index,(name,path) in enumerate(paths.items()):
                path.mkdir()
                rows=[]
                for j,(location,force) in enumerate(((.045,1),(.045,5),(.105,1),(.105,5))):
                    rows.append(dict(seed=index*10+j,actual_force_N=force,
                        contact_segment_requested_m=location,active_actuator_ids='[1, 2]',
                        recorded_pair_ids='[1, 2]',split='test'))
                with (path/'manifest.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            result=check_collections(paths)
            self.assertEqual(result['fresh_test']['trials'],4)
            (paths['stress_test']/'manifest.csv').write_bytes((paths['fresh_test']/'manifest.csv').read_bytes())
            with self.assertRaisesRegex(ValueError,'disjoint trial seeds'):
                check_collections(paths)

    def test_nearest_class_accuracy_does_not_hide_continuous_errors_or_modify_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.csv'
            rows=[dict(contact_location_m=location,contact_force_N=force,
                       predicted_contact_location_m=location+.001,
                       predicted_contact_force_N=force+.4)
                  for location,force in ((.045,1),(.045,5),(.105,1),(.105,5))]
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            original=path.read_bytes()
            result=discrete_scores(path)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(result['contact_force_N']['accuracy'],1.)
            self.assertLess(result['contact_force_N']['wilson_95_interval'][0],1.)
            for condition in result['conditions'].values():
                self.assertAlmostEqual(condition['force_rmse_N'],.4)
                self.assertAlmostEqual(condition['location_rmse_mm'],1.)
            self.assertTrue(path.with_suffix('.discrete.csv').exists())


if __name__=='__main__':
    unittest.main()
