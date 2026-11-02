"""CPU-only plots and localhost dashboard from the completed native MLP run."""
import argparse
import csv
import json
from pathlib import Path

NAMES = ('vx', 'vy', 'yaw')
DIRECTIONS = ('forward', 'backward', 'left', 'right', 'standing', 'other')
COLORS = ('#2865ce', '#8c5ac4', '#159575', '#d43e55', '#788397', '#ad8340')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--distributions-only', action='store_true', help='Plot existing recorded samples without re-reading checkpoints.')
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.distributions_only:
        output = root/'command_statistics'
        data = json.loads((output/'data.json').read_text())
        distribution_plots(output, data)
        (output/'index.html').write_text(Path(__file__).with_name('command_statistics_dashboard.html').read_text())
        return
    runs = list((root/'training').glob('*/*/command_audit_status.json'))
    if len(runs) != 1:
        raise ValueError('Expected one native MLP training run.')
    run = runs[0].parent
    output = root/'command_statistics'
    output.mkdir(exist_ok=True)
    audit = json.loads(runs[0].read_text())
    metadata = json.loads((run/'command_audit_metadata.json').read_text())
    if metadata['symmetry_compatible_io'] or not metadata['obs_normalization']:
        raise ValueError('This inspection expects the native empirical-normalization run.')
    if metadata['csv_envs'] != 1:
        raise ValueError('Expected the current env-0 CSV logging contract.')

    import numpy as np
    import torch
    checkpoints = []
    for path in sorted(run.glob('model_*.pt')):
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        row = dict(iteration=int(path.stem.split('_')[-1]), file=path.name)
        for role, key, expected, indices in (
            ('actor', 'obs_normalizer_state', 100, (33,34,32)),
            ('critic', 'critic_obs_normalizer_state', 103, (36,37,35))):
            norm = state[key]
            if norm['_mean'].numel() != expected:
                raise ValueError(f'Unexpected native {role} observation layout.')
            row[role] = dict(count=int(norm['count']), command={})
            for name, index in zip(NAMES, indices):
                row[role]['command'][name] = dict(index=index,
                    mean=float(norm['_mean'].flatten()[index]),
                    variance=float(norm['_var'].flatten()[index]),
                    std=float(norm['_std'].flatten()[index]))
        checkpoints.append(row)
    if [r['iteration'] for r in checkpoints] != list(range(1000,30001,1000)):
        raise ValueError('Expected all 30 already-completed checkpoints.')

    steps, raw, normalized = [], [], []
    with (run/'commands_policy.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['env_id']) != 0:
                continue
            steps.append(int(row['step']))
            raw.append([float(row[k]) for k in NAMES])
            normalized.append([float(row['normalized_'+k]) for k in NAMES])
    x = np.asarray(raw)
    z = np.asarray(normalized)
    if (x[:,1] < 0).any():
        raise ValueError('Unexpected negative vy in env-0 CSV.')
    if sum(audit['command_counts'].values()) != audit['policy_command_samples']:
        raise ValueError('Direction count mismatch.')

    # These are cumulative all-environment ratios written during training.
    # Read only the needed tags, retaining one point per 100 steps.
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    series = {name: {} for name in DIRECTIONS}
    for event_path in sorted(run.glob('events.out.tfevents.*')):
        for event in EventFileLoader(str(event_path)).Load():
            if event.step % 100 and event.step != audit['last_step']:
                continue
            for value in event.summary.value:
                prefix = 'CommandRatio/'
                if value.tag.startswith(prefix) and value.tag[len(prefix):] in series:
                    number = (float(value.simple_value) if value.HasField('simple_value')
                              else float(value.tensor.float_val[0]))
                    series[value.tag[len(prefix):]][int(event.step)] = number
    series = {k: sorted(v.items()) for k,v in series.items()}
    ratio = {k: audit['command_counts'][k]/audit['policy_command_samples'] for k in DIRECTIONS}
    sample_stats = {}
    for i,name in enumerate(NAMES):
        sample_stats[name] = dict(mean=float(x[:,i].mean()), variance=float(x[:,i].var()),
            std=float(x[:,i].std()), minimum=float(x[:,i].min()), maximum=float(x[:,i].max()),
            normalized_mean=float(z[:,i].mean()), normalized_std=float(z[:,i].std()))
    data = dict(audit=audit, metadata=dict(run=str(run), csv_envs=1,
        description='All-environment policy-input counters; CSV distributions are env 0 only.',
        normalizer_source='Saved running statistics fitted during replay updates; not env-0 sample statistics.',
        epsilon=.01, command_scales=[1.,1.,1.], actor_indices=[33,34,32],
        critic_indices=[36,37,35]), checkpoints=checkpoints, ratio=ratio,
        ratio_series=series, sample=dict(steps=steps,
            raw=np.round(x,7).tolist(), normalized=np.round(z,7).tolist(), statistics=sample_stats))
    (output/'data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    with (output/'normalizer_statistics.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(('iteration','role','channel','index','mean','variance','std','count','epsilon'))
        for row in checkpoints:
            for role in ('actor','critic'):
                for name,stat in row[role]['command'].items():
                    writer.writerow((row['iteration'],role,name,stat['index'],stat['mean'],
                        stat['variance'],stat['std'],row[role]['count'],.01))
    plots(output, data, x, z)
    template=Path(__file__).with_name('command_statistics_dashboard.html')
    (output/'index.html').write_text(template.read_text())
    print(json.dumps(dict(output=str(output),checkpoints=len(checkpoints),
        sample_rows=len(steps),ratios=ratio,env0_statistics=sample_stats),indent=2))


def plots(output, data, x, z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,facecolor='white')
        fig.savefig(output/(name+'.svg'),facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.8))
    bars=axes[0].bar(DIRECTIONS,[data['ratio'][k]*100 for k in DIRECTIONS],color=COLORS)
    axes[0].bar_label(bars,labels=[f"{data['ratio'][k]*100:.3f}%\nn={data['audit']['command_counts'][k]:,}" for k in DIRECTIONS],fontsize=9,padding=5)
    axes[0].set(ylabel='Share of actual policy command inputs (%)',ylim=(0,34),
        title=f"All environments: {data['audit']['policy_command_samples']:,} inputs")
    for name,color in zip(DIRECTIONS,COLORS):
        series=data['ratio_series'][name]
        if series:
            a=np.array(series)
            axes[1].plot(a[:,0],a[:,1]*100,label=name,color=color)
    axes[1].set(xlabel='Training iteration',ylabel='Cumulative share (%)',
        title='Measured cumulative command ratios')
    axes[1].legend(ncol=3,fontsize=8)
    fig.suptitle('Actual sampled commands | rightward vy < 0: zero',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'command_ratios')

    fig,axes=plt.subplots(3,3,figsize=(14,9),sharex=True)
    for i,name in enumerate(NAMES):
        for j,key in enumerate(('mean','variance','std')):
            for role,style in (('actor','-'),('critic','--')):
                axes[i,j].plot([r['iteration'] for r in data['checkpoints']],
                    [r[role]['command'][name][key] for r in data['checkpoints']],
                    style,label=role,color=('#2865ce' if role=='actor' else '#dc8622'))
            axes[i,j].set_title(name+' / '+key)
            axes[i,j].grid(alpha=.2)
            if i==2:axes[i,j].set_xlabel('Checkpoint iteration')
    axes[0,0].legend()
    fig.suptitle('Saved normalizer statistics | 1k to 30k | no retraining',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'normalizer_statistics')

    distribution_plots(output, data)

    checkpoint=next(r for r in data['checkpoints'] if r['iteration']==10000)
    stat=checkpoint['actor']['command']['vy'];mean,std=stat['mean'],stat['std']
    fig,ax=plt.subplots(figsize=(11,5))
    projected=(x[:,1]-mean)/(std+.01)
    ax.hist(projected,bins=65,color='#2865ce',weights=np.ones(len(x))*100/len(x),
        label='Env-0 raw training commands projected with frozen 10k statistics')
    boundary=-mean/(std+.01)
    ax.axvline(boundary,color='#788397',linestyle='--',label=f'Raw vy=0 boundary: {boundary:.3f}')
    for command,color in ((-.3,'#d43e55'),(-.6,'#8c5ac4')):
        value=(command-mean)/(std+.01)
        ax.axvline(value,color=color,linewidth=2,label=f'Right vy={command:+.1f}: z={value:.3f}')
    ax.set(xlabel='Normalized vy using fixed 10k statistics',ylabel='Env-0 samples (%)',
        title='Negative normalized input does not imply a negative physical command')
    ax.legend(fontsize=9);fig.tight_layout();save(fig,'vy_fixed_10k_mapping')


def distribution_plots(output, data):
    """Frequency polygons from the recorded samples; no KDE or artificial samples."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    samples = {key: np.asarray(data['sample'][key]) for key in ('raw','normalized')}
    curves = []
    for col,key in enumerate(('raw','normalized')):
        for index,name in enumerate(NAMES):
            values=samples[key][:,index]
            counts,edges=np.histogram(values,bins=80)
            width=np.diff(edges)
            curves.append(dict(column=col,key=key,index=index,name=name,
                x=(edges[:-1]+edges[1:])/2,ratio=100*counts/len(values),
                density=counts/(len(values)*width),width=width))
    for field,stem in (('ratio','command_distributions'),('density','command_probability_density')):
        fig,axes=plt.subplots(3,2,figsize=(14,10))
        for curve in curves:
            name,key=curve['name'],curve['key']
            ax=axes[curve['index'],curve['column']]
            ax.plot(curve['x'],curve[field],color='#2865ce' if key=='raw' else '#159575',linewidth=1.8)
            ax.axvline(0,color='#aab8cb',linestyle='--',linewidth=.8)
            if name=='vy' and key=='raw':
                ax.set_xlim(left=-.12)
            unit='(dimensionless)' if key=='normalized' else '(rad/s)' if name=='yaw' else '(m/s)'
            ax.set(title=name+' | '+('Before normalization' if key=='raw' else 'After normalization (actually logged)'),
                xlabel=name+' '+unit,ylabel='Samples per bin (%)' if field=='ratio' else 'Probability density')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.18)
        fig.suptitle('Actual command distributions | env 0 | before / after normalization',fontweight='bold',fontsize=16)
        fig.text(.05,.012,'80 bins; bin values joined by straight lines, no KDE. Repeated timestep inputs included. Normalized values use the statistics at collection time.',fontsize=9,color='#53647c')
        fig.tight_layout(rect=(0,.035,1,.95))
        for extension in ('png','svg'):
            fig.savefig(output/(stem+'.'+extension),dpi=180,facecolor='white')
        plt.close(fig)
    with (output/'command_distribution_curves.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(('representation','channel','bin_center','bin_width','sample_ratio_percent','probability_density'))
        for curve in curves:
            for values in zip(curve['x'],curve['width'],curve['ratio'],curve['density']):
                writer.writerow((curve['key'],curve['name'],*values))
    template=Path(__file__).with_name('command_distributions_dashboard.html')
    (output/'distributions.html').write_text(template.read_text())
    print('Created six before/after line distributions and probability-density exports',flush=True)


if __name__=='__main__':
    main()
