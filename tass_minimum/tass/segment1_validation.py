"""Train two Ridge models and audit fresh segment-1 simulation collections."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .config import ModelConfig
from .contact_response import ContactResponseConfig
from .dataset import Protocol, make_trial
from .ml import (FeatureConfig, evaluate, extract_features, load_bundle, load_segments,
                 train, write_feature_set, read_feature_set, _matrix)


CLASSES = {'contact_location_m': [.045, .105], 'contact_force_N': [1., 5.]}


def _manifest(path):
    with (Path(path)/'manifest.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_collections(collections):
    """Check the physical test setting and independent trial seeds before fitting."""
    seen = set()
    result = {}
    for name, path in collections.items():
        rows = _manifest(path)
        seeds = {int(row['seed']) for row in rows}
        if len(seeds) != len(rows) or seen.intersection(seeds):
            raise ValueError('Validation collections must have unique, disjoint trial seeds')
        seen.update(seeds)
        conditions = {}
        for row in rows:
            force = float(row['actual_force_N']); location = float(row['contact_segment_requested_m'])
            if force not in CLASSES['contact_force_N'] or not any(np.isclose(location,x,atol=1e-12) for x in CLASSES['contact_location_m']):
                raise ValueError('Expected only 1/5 N at 45/105 mm on the 150-mm first segment')
            if json.loads(row['active_actuator_ids']) != [1,2] or json.loads(row['recorded_pair_ids']) != [1,2]:
                raise ValueError('Expected only actuator/sensor pairs 1 and 2')
            key=f'{location*1000:g}mm_{force:g}N'
            conditions[key]=conditions.get(key,0)+1
        if len(conditions) != 4 or len(set(conditions.values())) != 1:
            raise ValueError('Each collection must balance all four force/location conditions')
        if name != 'training' and any(row['split'] != 'test' for row in rows):
            raise ValueError('Independent evaluation collections must contain test trials only')
        result[name]={'path':str(path),'trials':len(rows),'condition_counts':conditions,
                      'manifest_sha256':hashlib.sha256((Path(path)/'manifest.csv').read_bytes()).hexdigest()}
    return result


def sensitivity_audit(config, protocol, stem=None):
    """Paired same-seed interventions isolate force from excitation, gains and noise."""
    protocol=replace(protocol,window_s=1.5)
    response=ContactResponseConfig(**protocol.contact_response)
    cases={
        'hard_clamp':replace(response,mode='hard_clamp'),
        'compliant_constant_coefficients':replace(response,stiffness_force_exponent=0.,damping_force_exponent=0.),
        'compliant_force_dependent':response}
    result={}
    if stem is not None:
        from .ml_plotting import _save
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
    for case_index,(name, contact) in enumerate(cases.items()):
        rows=[]
        for location_index,location in enumerate((.3,.7)):
            p=replace(protocol,contact_response=asdict(contact))
            low=make_trial(config,p,location,1.,42)
            high=make_trial(config,p,location,5.,42)
            assert np.array_equal(low[2],high[2])
            assert low[3]['tension_N'] == high[3]['tension_N']
            selected=np.asarray(protocol.recorded_pair_ids)-1
            a,b=low[1][:,selected],high[1][:,selected]
            if stem is not None:
                ax=axes[case_index,location_index]
                visible=(low[0]>=.3)&(low[0]<.38)
                ax.plot(low[0][visible],a[visible,0]*1000,label='1 N',color='#2563eb',lw=1.1)
                ax.plot(high[0][visible],b[visible,0]*1000,label='5 N',color='#ea580c',ls='--',lw=1.1)
                ax.set(title=f'{name.replace("_"," ")} — {location:.0%} / {location*150:g} mm',
                       xlabel='Time after recording start [s]',ylabel='Sensor 1 [mV]')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            rows.append({'location_fraction':location,'location_m':location*.15,
                'force_1_and_5_signals_identical':bool(np.array_equal(a,b)),
                'max_abs_voltage_difference_V':float(np.max(abs(a-b))),
                'relative_voltage_difference_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                'tension_1N_N':low[3]['tension_N'],'tension_5N_N':high[3]['tension_N']})
        result[name]=rows
    if stem is not None:
        figure.suptitle('Same-seed force interventions | Matched drive, gain, noise and tendon tension')
        _save(figure,stem)
    return result


def discrete_scores(predictions):
    """Evaluate regression against the four known conditions; never alter model outputs."""
    with Path(predictions).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    report={}
    for target,classes in CLASSES.items():
        classes=np.asarray(classes)
        actual=np.asarray([float(row[target]) for row in rows])
        prediction=np.asarray([float(row['predicted_'+target]) for row in rows])
        nearest=classes[np.argmin(abs(prediction[:,None]-classes),axis=1)]
        for row,label in zip(rows,nearest):
            row['nearest_'+target]=float(label)
        correct=int(np.sum(np.isclose(actual,nearest)))
        interval=binomtest(correct,len(rows)).proportion_ci(confidence_level=.95,method='wilson')
        report[target]={'classes':classes.tolist(),'correct':correct,
            'total':len(rows),'accuracy':float(np.mean(np.isclose(actual,nearest))),
            'wilson_95_interval':[float(interval.low),float(interval.high)]}
    report['conditions']={}
    for location in CLASSES['contact_location_m']:
        for force in CLASSES['contact_force_N']:
            part=[row for row in rows if np.isclose(float(row['contact_location_m']),location)
                  and float(row['contact_force_N']) == force]
            report['conditions'][f'{location*1000:g}mm_{force:g}N']={
                'trials':len(part),
                'force_rmse_N':float(np.sqrt(np.mean([(float(row['predicted_contact_force_N'])-force)**2 for row in part]))),
                'force_class_accuracy':float(np.mean([row['nearest_contact_force_N']==force for row in part])),
                'location_rmse_mm':float(1000*np.sqrt(np.mean([(float(row['predicted_contact_location_m'])-location)**2 for row in part]))),
                'location_class_accuracy':float(np.mean([np.isclose(row['nearest_contact_location_m'],location) for row in part]))}
    output=Path(predictions).with_suffix('.discrete.csv')
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def plot_summary(report, stem):
    from .ml_plotting import _save
    import matplotlib.pyplot as plt

    names=list(report['evaluations']); x=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    labels=[name.replace('_',' ').title() for name in names]
    for i,(target,title,scale,unit) in enumerate((
        ('contact_location_m','Location',1000.,'mm'),('contact_force_N','Force magnitude',1.,'N'))):
        values=[report['evaluations'][name]['regression']['targets'][target]['model']['rmse']*scale for name in names]
        axes[i].bar(x,values,color=['#64748b','#2563eb','#0d9488'])
        axes[i].set(xticks=x,xticklabels=labels,ylabel=f'RMSE [{unit}]',title=title,ylim=(0,max(values)*1.25))
        for position,value in zip(x,values):
            axes[i].text(position,value,f'{value:.3g}',ha='center',va='bottom')
        axes[i].grid(axis='y',alpha=.2)
    for delta,target,color,label in ((-.18,'contact_location_m','#2563eb','Location class'),
                                     (.18,'contact_force_N','#0d9488','Force class')):
        values=[report['evaluations'][name]['discrete'][target]['accuracy']*100 for name in names]
        axes[2].bar(x+delta,values,width=.34,color=color,label=label)
        for position,value in zip(x+delta,values):
            axes[2].text(position,value,f'{value:.1f}',ha='center',va='bottom',fontsize=8)
    axes[2].axhline(50,color='#64748b',ls=':',label='Constant-class baseline')
    axes[2].set(xticks=x,xticklabels=labels,ylabel='Accuracy [%]',ylim=(0,112),title='Nearest allowed class')
    axes[2].legend(fontsize=8);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('Two independent Ridge regressors | Assumed compliant contact | Simulation only')
    _save(fig,stem)


def run(training_data, fresh_test, stress_test, output, *, window_s=1.5):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory to preserve prior models and reports')
    collections={'training':Path(training_data),'fresh_test':Path(fresh_test),'stress_test':Path(stress_test)}
    checked=check_collections(collections)
    output.mkdir(parents=True,exist_ok=True)
    config=ModelConfig.load(Path(training_data)/'model_config.json')
    protocol=Protocol(**json.loads((Path(training_data)/'protocol.json').read_text()))
    features=FeatureConfig(window_s=window_s,stride_s=protocol.window_s,sensor_ids=(1,2),remove_sensor_mean=True,
                           target_columns=('contact_location_m','contact_force_N'))
    print(f'Extracting training features (sensor voltage, first {window_s:g} s of each {protocol.window_s:g}-s recording)',flush=True)
    rows,names,provenance=extract_features(load_segments(training_data,stream=True),features)
    write_feature_set(rows,names,provenance,features,output/'features')
    fitted=train(output/'features',output/'model',regressor='ridge')
    model=output/'model/model_bundle.pkl'
    bundle=load_bundle(model)
    assert len({id(m) for m in bundle['models'].values()}) == 2
    report={'synthetic_only':True,'contact_parameters_calibrated':False,'collections':checked,
        'independent_trial_seeds_verified':True,'feature_count':len(names),'feature_config':asdict(features),
        'contact_response':asdict(ContactResponseConfig(**protocol.contact_response)),
        'training':fitted,'model_sha256_before_fresh_evaluation':hashlib.sha256(model.read_bytes()).hexdigest(),
        'evaluations':{},'interpretation':'Validation is conditional on the explicit assumed preload-dependent contact law. No hardware accuracy or continuous-force/location accuracy is established.'}
    for name,path in [('held_out',training_data),('fresh_test',fresh_test),('stress_test',stress_test)]:
        print(f'Evaluating locked Ridge models on {name}',flush=True)
        destination=output/f'{name}.json'
        regression=evaluate(model,path,destination)
        report['evaluations'][name]={'regression':regression,
            'discrete':discrete_scores(destination.with_suffix('.predictions.csv'))}
    assert hashlib.sha256(model.read_bytes()).hexdigest() == report['model_sha256_before_fresh_evaluation']
    print('Auditing same-seed force interventions',flush=True)
    report['sensitivity_audit']=sensitivity_audit(config,protocol,output/'force_sensitivity')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_summary(report,output/'validation_summary')
    return report


def prepare_robust(training_data, development_data, output, *, window_s=1.5):
    """Use the exposed stress collection as development data; never call it blind."""
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory to preserve earlier experiments')
    checked=check_collections({'training':Path(training_data),'stress_development':Path(development_data)})
    output.mkdir(parents=True,exist_ok=True)
    protocol=Protocol(**json.loads((Path(training_data)/'protocol.json').read_text()))
    config=FeatureConfig(window_s=window_s,stride_s=protocol.window_s,sensor_ids=(1,2),
        remove_sensor_mean=True,spectral_mode='band_peaks',
        target_columns=('contact_location_m','contact_force_N'))
    rows=[]
    for collection,path in [('original',training_data),('stress_development',development_data)]:
        print(f'Extracting band-peak features: {collection}',flush=True)
        extracted,names,provenance=extract_features(load_segments(path,stream=True),config)
        repeats={row['trial_id']:int(row['repeat']) for row in _manifest(path)}
        count=Protocol(**json.loads((Path(path)/'protocol.json').read_text())).trials_per_condition
        for row in extracted:
            if collection == 'stress_development':
                repeat=repeats[row['segment_id']]
                row['split']='train' if repeat<int(.7*count) else ('validation' if repeat<int(.85*count) else 'test')
            row['segment_id']=collection+'/'+row['segment_id']
        rows.extend(extracted)
    write_feature_set(rows,names,provenance,config,output/'features')
    fitted=train(output/'features',output/'model',regressor='ridge')
    model=output/'model/model_bundle.pkl'
    report={'synthetic_only':True,'contact_parameters_calibrated':False,
        'collections':checked,'feature_count':len(names),'feature_config':asdict(config),
        'training':fitted,'development_split_counts':{
            split:sum(row['split']==split for row in rows) for split in ('train','validation','test')},
        'model_sha256_before_blind_evaluation':hashlib.sha256(model.read_bytes()).hexdigest(),
        'development_disclosure':'Original stress failures informed the band-peak recipe. That exposed collection is now development data, not an independent test. A new collection with disjoint seeds is reserved for blind evaluation.',
        'interpretation':'Conditional simulation validation under assumed compliant preload laws. No hardware or continuous-range accuracy is established.',
        'evaluations':{}}
    (output/'training_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Two Ridge models locked; waiting for the new blind collection',flush=True)
    return report


def finish_robust(training_data, development_data, blind_test, output):
    output=Path(output)
    report=json.loads((output/'training_plan.json').read_text())
    checked=check_collections({'training':Path(training_data),'stress_development':Path(development_data),
                               'blind_test':Path(blind_test)})
    report['collections']=checked
    report['independent_trial_seeds_verified']=True
    model=output/'model/model_bundle.pkl'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == report['model_sha256_before_blind_evaluation']
    bundle=load_bundle(model)
    assert len({id(m) for m in bundle['models'].values()}) == 2
    rows,names,_=read_feature_set(output/'features')
    held_out=[row for row in rows if row['split']=='test']
    X=_matrix(held_out,names)
    for target,fitted in bundle['models'].items():
        predictions=fitted.predict(X)
        for row,prediction in zip(held_out,predictions):
            row['predicted_'+target]=float(prediction)
    destination=output/'development_held_out.predictions.csv'
    with destination.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(held_out[0]));writer.writeheader();writer.writerows(held_out)
    from .ml_plotting import plot_predictions
    plot_predictions(held_out,bundle['target_names'],destination.with_suffix(''))
    report['evaluations']['development_held_out']={
        'regression':{'targets':{t:{'model':v['test']} for t,v in report['training']['targets'].items()}},
        'discrete':discrete_scores(destination)}
    print('Evaluating frozen models on new blind trials',flush=True)
    regression=evaluate(model,blind_test,output/'blind_test.json')
    report['evaluations']['blind_test']={'regression':regression,
        'discrete':discrete_scores(output/'blind_test.predictions.csv')}
    assert hashlib.sha256(model.read_bytes()).hexdigest() == report['model_sha256_before_blind_evaluation']
    source=ModelConfig.load(Path(training_data)/'model_config.json')
    protocol=Protocol(**json.loads((Path(training_data)/'protocol.json').read_text()))
    report['sensitivity_audit']=sensitivity_audit(source,protocol,output/'force_sensitivity')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_summary(report,output/'validation_summary')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-data',type=Path,required=True)
    parser.add_argument('--fresh-test',type=Path)
    parser.add_argument('--stress-test',type=Path)
    parser.add_argument('--development-data',type=Path,help='Exposed stress collection for band-peak development training')
    parser.add_argument('--blind-test',type=Path,help='New independent test collection reserved for the robust recipe')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-s',type=float,default=1.5,help='Feature window starting at recording time zero')
    args=parser.parse_args()
    if args.development_data is not None or args.blind_test is not None:
        if args.development_data is None or args.blind_test is None or args.fresh_test or args.stress_test:
            parser.error('Use --development-data and --blind-test together, without --fresh-test/--stress-test')
        prepare_robust(args.training_data,args.development_data,args.output,window_s=args.window_s)
        finish_robust(args.training_data,args.development_data,args.blind_test,args.output)
    else:
        if args.fresh_test is None or args.stress_test is None:
            parser.error('Use --fresh-test and --stress-test, or --development-data and --blind-test')
        run(args.training_data,args.fresh_test,args.stress_test,args.output,window_s=args.window_s)
    print(f'Report: {args.output / "report.json"}',flush=True)


if __name__=='__main__':
    main()
