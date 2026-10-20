"""Export loss comparisons and per-trial predictions for ML baselines."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _target_label(target):
    return {'contact_location_m':('Contact location',1000.,'mm'),
            'contact_force_N':('Contact force',1.,'N'),
            'load_cell_readout_N':('Load-cell target',1.,'N')}.get(target,(target,1.,'units'))


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'):
        fig.savefig(str(stem)+suffix,dpi=160)
    plt.close(fig)


def plot_training_loss(report, stem):
    """Compare fitted candidates, rather than claiming an epoch history."""
    classification = report.get('task','regression') == 'classification'
    targets = list(report['targets'])
    fig,axes = plt.subplots(1,len(targets),figsize=(7*len(targets),4.8),squeeze=False,constrained_layout=True)
    for ax,target in zip(axes[0],targets):
        result = report['targets'][target]
        title,scale,unit = _target_label(target)
        names = list(result['training_candidates'])
        positions = np.arange(len(names))
        for key,label,color in (('training_candidates','Training','#2563eb'),
                                ('validation_candidates','Validation','#e07a16')):
            loss = [result[key][name]['log_loss'] if classification else result[key][name]['mse']*scale**2
                    for name in names]
            ax.plot(positions,loss,'o-',label=label,color=color,lw=1.8)
        chosen = names.index(result['selected_model'])
        ax.axvline(chosen,color='#555555',ls=':',lw=1,
                   label='Selected by validation log loss' if classification else 'Selected by validation RMSE')
        labels = [name.replace('ridge_alpha_','Ridge α=').replace('logistic_C_','Logistic C=')
                  .replace('random_forest_depth_','RF depth=') for name in names]
        ax.set_xticks(positions,labels,rotation=28,ha='right')
        ax.set(title=title,xlabel='Fitted model candidate',
               ylabel='Cross-entropy / log loss' if classification else f'Mean squared error [{unit}²]')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    family = {'ridge':'Ridge','linear':'Linear','mlp':'MLP'}.get(report.get('regressor_family'),'Linear/Ridge')
    fig.suptitle('Training and validation loss — classifier comparison' if classification else
                 f'Training and validation loss — {family} candidate comparison')
    _save(fig,stem)


def plot_mlp_training_loss(report,stem):
    classification=report['task']=='classification'
    targets=list(report['targets'])
    fig,axes=plt.subplots(1,len(targets),figsize=(7*len(targets),4.8),squeeze=False,constrained_layout=True)
    for ax,target in zip(axes[0],targets):
        result=report['targets'][target]
        selected=result['selected_model'];history=result['epoch_histories'][selected]
        title,scale,unit=_target_label(target)
        factor=1. if classification else scale**2
        epochs=np.arange(1,history['epochs_run']+1)
        ax.plot(epochs,np.asarray(history['training_loss'])*factor,label='Training',color='#2563eb')
        ax.plot(epochs,np.asarray(history['validation_loss'])*factor,label='Validation',color='#e07a16')
        ax.axvline(history['selected_epoch'],color='#555555',ls=':',label='Saved validation checkpoint')
        ax.set(title=f'{title} — {selected}',xlabel='Training epoch',
               ylabel='Cross-entropy / log loss' if classification else f'Mean squared error [{unit}²]')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('MLP training history — separate location and force networks')
    _save(fig,stem)


def plot_classification_predictions(rows, class_values, stem):
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(len(class_values), 2, figsize=(13, 4*len(class_values)),
                             squeeze=False, constrained_layout=True)
    for index, (target, classes) in enumerate(class_values.items()):
        title, scale, unit = _target_label(target)
        lookup = {value: i for i, value in enumerate(classes)}
        predicted = np.asarray([lookup[row['predicted_'+target]] for row in rows])
        labels = [f'{value*scale:g} {unit}' for value in classes]
        x = np.arange(1,len(rows)+1)
        ax, right = axes[index]
        labeled = all(target in row for row in rows)
        # Unseen evaluation labels are rejected by evaluate, not snapped to a trained class.
        known = labeled and all(float(row[target]) in lookup for row in rows)
        if known:
            actual = np.asarray([lookup[float(row[target])] for row in rows])
            ax.plot(x,actual,'o',mfc='none',ms=9,mew=1.5,color='#334155',label='Actual')
        ax.plot(x,predicted,'x--',color='#2563eb',lw=1.2,label='Predicted class')
        ax.set(title=title,xlabel='Prediction window / trial index',ylabel='Class',
               yticks=np.arange(len(classes)),yticklabels=labels,ylim=(-.4,len(classes)-.6))
        ax.grid(alpha=.2)
        ax.legend()
        if known:
            matrix = confusion_matrix(actual,predicted,labels=np.arange(len(classes)))
            right.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
            for i in range(len(classes)):
                for j in range(len(classes)):
                    right.text(j,i,str(matrix[i,j]),ha='center',va='center',
                               color='white' if matrix[i,j] > matrix.max()/2 else '#111827')
            right.set(title=f'{title} — accuracy {np.mean(actual==predicted):.1%}',
                      xlabel='Predicted class',ylabel='Actual class',
                      xticks=np.arange(len(classes)),xticklabels=labels,
                      yticks=np.arange(len(classes)),yticklabels=labels)
        else:
            for i,label in enumerate(labels):
                right.plot(x,[row[f'probability_{target}_class_{i}'] for row in rows],'o-',label=label)
            right.set(title=f'{title} — class probabilities',xlabel='Prediction window / trial index',
                      ylabel='Predicted probability',ylim=(-.05,1.05))
            right.legend()
            right.grid(alpha=.2)
    fig.suptitle(f'Classification inference — {len(rows)} windows')
    _save(fig,stem)


def plot_predictions(rows, targets, stem):
    """Actual/predicted per window plus parity plots when labels are available."""
    labeled_targets = {target:all(target in row for row in rows) for target in targets}
    cols = 2 if any(labeled_targets.values()) else 1
    fig,axes = plt.subplots(len(targets),cols,figsize=(7*cols,3.8*len(targets)),
                            squeeze=False,constrained_layout=True)
    for index,target in enumerate(targets):
        labeled = labeled_targets[target]
        title,scale,unit = _target_label(target)
        predicted = np.asarray([row['predicted_'+target] for row in rows])*scale
        x = np.arange(1,len(rows)+1)
        ax = axes[index,0]
        if labeled:
            actual = np.asarray([float(row[target]) for row in rows])*scale
            ax.plot(x,actual,'o',mfc='none',ms=8,mew=1.5,color='#334155',label='Actual')
        ax.plot(x,predicted,'x--',color='#2563eb',label='Predicted',lw=1.2)
        ax.set(title=title,xlabel='Prediction window / trial index',ylabel=f'{title} [{unit}]')
        ax.grid(alpha=.2)
        ax.legend()
        if labeled:
            parity = axes[index,1]
            low,high = min(actual.min(),predicted.min()),max(actual.max(),predicted.max())
            padding = max((high-low)*.1,1e-6)
            parity.plot([low-padding,high+padding],[low-padding,high+padding],ls=':',color='#555555',label='Ideal')
            parity.scatter(actual,predicted,color='#2563eb',s=40,alpha=.8)
            rmse = np.sqrt(np.mean((predicted-actual)**2))
            parity.set(title=f'{title} — RMSE {rmse:.4g} {unit}',
                       xlabel=f'Actual [{unit}]',ylabel=f'Predicted [{unit}]')
            parity.grid(alpha=.2)
            parity.legend()
        elif cols == 2:
            axes[index,1].set_axis_off()
            axes[index,1].text(.5,.5,'No reference labels for this target',ha='center',va='center',
                               transform=axes[index,1].transAxes)
    domains = ', '.join(sorted({row['source'] for row in rows}))
    fig.suptitle(f'Inference results — {domains}, {len(rows)} windows')
    _save(fig,stem)
