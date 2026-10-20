"""Compare independent Ridge, classifier and feature-vector MLP models on shared trials."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .ml import (read_feature_set,load_bundle,train,evaluate,_metrics,_class_indices,
                 _classification_metrics)


def _read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(feature_dir,benchmark,specifications,output):
    rows,features,config=read_feature_set(feature_dir)
    splits={split:sorted({r['segment_id'] for r in rows if r['split']==split})
            for split in ('train','validation','test')}
    classes={target:np.unique([float(r[target]) for r in rows if r['split']=='train']).tolist()
             for target in config.target_columns if all(target in row for row in rows)}
    benchmark_rows=[r for r in _read_csv(Path(benchmark)/'manifest.csv') if r['split']=='test']
    expected_ids=sorted(r['trial_id'] for r in benchmark_rows)
    reference_labels=None;records=[]
    for spec in specifications:
        bundle=load_bundle(spec['bundle'])
        if bundle['feature_order']!=features or json.dumps(bundle['feature_config'],sort_keys=True)!=json.dumps(asdict(config),sort_keys=True):
            raise ValueError('All compared models must use the identical feature order and preprocessing')
        saved_splits=json.loads((Path(spec['bundle']).parent/'splits.json').read_text())
        if saved_splits!=splits:
            raise ValueError('All compared models must use identical complete-trial splits')
        predicted=_read_csv(spec['predictions'])
        if sorted(r['segment_id'] for r in predicted)!=expected_ids:
            raise ValueError('Each model must predict the same complete benchmark trial set')
        predicted=sorted(predicted,key=lambda r:r['segment_id'])
        labels=[(r['segment_id'],*(float(r[target]) for target in classes)) for r in predicted]
        if reference_labels is None:reference_labels=labels
        if labels!=reference_labels:
            raise ValueError('Benchmark labels/order differ between model predictions')
        record={'name':spec['name'],'task':bundle['task'],'model_bundle':str(spec['bundle']),
                'model_sha256':_sha(spec['bundle']),'predictions':str(spec['predictions']),'targets':{}}
        for target,values in classes.items():
            actual=np.array([float(r[target]) for r in predicted])
            estimates=np.array([float(r['predicted_'+target]) for r in predicted])
            vocab=np.asarray(values)
            actual_codes=_class_indices(actual,values,target)
            codes=np.argmin(abs(estimates[:,None]-vocab),axis=1)
            result={'classes':values,'correct':int(np.sum(codes==actual_codes)),
                    'trials':len(predicted),'accuracy':float(np.mean(codes==actual_codes))}
            if bundle['task']=='regression':
                result['continuous_regression']=_metrics(actual,estimates)
            else:
                if bundle['class_values'][target]!=values:
                    raise ValueError('Classifier class order differs from training labels')
                probabilities=np.array([[float(r[f'probability_{target}_class_{i}'])
                                         for i in range(len(values))] for r in predicted])
                result['classification']=_classification_metrics(actual_codes,probabilities)
                result['hard_class_rmse']=_metrics(actual,estimates)['rmse']
            record['targets'][target]=result
        records.append(record)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report={'simulation_only':True,'benchmark_previously_evaluated':True,
        'benchmark_note':f'Matched comparison on the existing {len(expected_ids)}-trial benchmark, previously evaluated with Ridge; not a newly collected blind test. No benchmark-based parameter or epoch selection.',
        'input_representation':f'Same {len(features)}-value feature vector for every model; MLP is a dense feedforward network, not a waveform network or Conv1D.',
        'training_feature_csv_sha256':_sha(Path(feature_dir)/'features.csv'),
        'benchmark_manifest_sha256':_sha(Path(benchmark)/'manifest.csv'),
        'same_features_and_splits_verified':True,'same_benchmark_trials_verified':True,
        'split_counts':{key:len(value) for key,value in splits.items()},
        'benchmark_trial_ids':expected_ids,'models':records,
        'interpretation':'Classification selects only trained endpoints; zero hard-class RMSE does not establish continuous estimation. Simulation remains conditional on assumed compliant preload laws.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    table=[]
    for model in records:
        for target,metrics in model['targets'].items():
            table.append({'model':model['name'],'task':model['task'],'target':target,
                'accuracy':metrics['accuracy'],'correct':metrics['correct'],'trials':metrics['trials'],
                'continuous_rmse':metrics.get('continuous_regression',{}).get('rmse',''),
                'classifier_log_loss':metrics.get('classification',{}).get('log_loss','')})
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plot_comparison(report,output/'comparison')
    return report


def plot_comparison(report,stem):
    from .ml_plotting import _save,_target_label
    import matplotlib.pyplot as plt
    targets=list(report['models'][0]['targets'])
    fig,axes=plt.subplots(len(targets),2,figsize=(14,4*len(targets)),squeeze=False,constrained_layout=True)
    labels=[m['name'] for m in report['models']]
    for row,target in enumerate(targets):
        title,scale,unit=_target_label(target)
        x=np.arange(len(labels))
        values=[m['targets'][target]['accuracy']*100 for m in report['models']]
        axes[row,0].bar(x,values,color=['#64748b','#0d9488','#2563eb','#7c3aed'])
        for position,value in zip(x,values):axes[row,0].text(position,value,f'{value:g}%',ha='center',va='bottom')
        axes[row,0].set(title=f'{title}: nearest allowed label / classifier output',
            ylabel='Accuracy [%]',ylim=(0,112),xticks=x,xticklabels=labels)
        regressors=[m for m in report['models'] if m['task']=='regression']
        error=[m['targets'][target]['continuous_regression']['rmse']*scale for m in regressors]
        axes[row,1].bar(np.arange(len(error)),error,color=['#64748b','#2563eb'])
        for position,value in enumerate(error):axes[row,1].text(position,value,f'{value:.4g}',ha='center',va='bottom')
        axes[row,1].set(title=f'{title}: continuous estimates (regressors only)',ylabel=f'RMSE [{unit}]',
            ylim=(0,max(max(error)*1.25,1e-12)),xticks=np.arange(len(error)),xticklabels=[m['name'] for m in regressors])
        for ax in axes[row]:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same features, trial splits and benchmark | Separate location/force models | Simulation only')
    _save(fig,stem)


def run(feature_dir,ridge_model,benchmark,output,*,max_epochs=600,patience=60,seed=42):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new comparison directory to preserve existing models')
    output.mkdir(parents=True,exist_ok=True)
    configurations=[('classifier','Classifier',dict(task='classification')),
        ('mlp_regression','MLP regressor',dict(task='regression',regressor='mlp')),
        ('mlp_classifier','MLP classifier',dict(task='classification',classifier='mlp'))]
    specs=[{'name':'Ridge','bundle':Path(ridge_model),'key':'ridge'}]
    for key,label,options in configurations:
        print(f'Training {label}: separate location and force models',flush=True)
        train(feature_dir,output/key/'model',mlp_max_epochs=max_epochs,mlp_patience=patience,
              mlp_seed=seed,**options)
        specs.append({'name':label,'bundle':output/key/'model/model_bundle.pkl','key':key})
    # All three new pairs are fitted before evaluating the shared benchmark.
    hashes={spec['key']:_sha(spec['bundle']) for spec in specs}
    for spec in specs:
        print(f'Evaluating {spec["name"]} on the shared benchmark',flush=True)
        destination=output/spec['key']/'benchmark_evaluation.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        evaluate(spec['bundle'],benchmark,destination)
        spec['predictions']=destination.with_suffix('.predictions.csv')
        if _sha(spec['bundle'])!=hashes[spec['key']]:
            raise ValueError('A model changed during benchmark evaluation')
    return summarize(feature_dir,benchmark,specs,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--ridge-model',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mlp-max-epochs',type=int,default=600)
    parser.add_argument('--mlp-patience',type=int,default=60)
    parser.add_argument('--mlp-seed',type=int,default=42)
    args=parser.parse_args()
    run(args.features,args.ridge_model,args.input,args.output,max_epochs=args.mlp_max_epochs,
        patience=args.mlp_patience,seed=args.mlp_seed)
    print(f'Comparison: {args.output / "comparison.json"}',flush=True)


if __name__=='__main__':
    main()
