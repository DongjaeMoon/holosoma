"""Plots of every trial and a transparently selected matched highlight case."""
import json
from pathlib import Path
import numpy as np

MODELS=('biHAM','sym_aug_HAM','HAM','MLP')
RIGHT=('right','forward_right','backward_right')

def plot_results(output, iteration=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    data=[json.loads(p.read_text()) for p in output.glob('videos/*/model_*/*/metrics.json') if p.parts[-4] in MODELS]
    plots=output/'plots'; plots.mkdir(exist_ok=True)
    fields=[('mean_linear_rmse','Planar RMSE (m/s)','tracking_rmse'),('mean_vy_rmse','Lateral RMSE (m/s)','lateral_rmse'),('mean_survival_s','Observed duration (s)','survival_duration')]
    for field,ylabel,name in fields:
        fig,axes=plt.subplots(3,2,figsize=(11,10))
        for i,direction in enumerate(RIGHT):
            for j,speed in enumerate((.3,.6)):
                ax=axes[i,j]
                for model in MODELS:
                    points=sorted([d for d in data if d['model']==model and d['direction']==direction and d['speed']==speed],key=lambda d:d['iteration'])
                    if points: ax.plot([d['iteration']/1000 for d in points],[d[field] if d[field] is not None else np.nan for d in points],'.-',label=model)
                ax.set(title=f'{direction} | {speed} m/s',xlabel='Training iterations (thousands)',ylabel=ylabel)
                ax.grid(alpha=.25)
        h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=4)
        fig.suptitle('All trials; errors before first failure, after 2 s warmup',y=.96,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('png','svg'): fig.savefig(plots/f'{name}.{ext}',dpi=150)
        plt.close(fig)
    iterations=sorted({d['iteration'] for d in data}) if iteration is None else [iteration]
    for iteration in iterations:
        fig,axes=plt.subplots(3,3,figsize=(14,10))
        for i,direction in enumerate(RIGHT):
            for model in MODELS:
                matches=[d for d in data if d['model']==model and d['iteration']==iteration and d['direction']==direction and d['speed']==.6]
                if not matches: continue
                d=matches[0]; c=d['curve']; t=[x['time_s'] for x in c]
                for j in (0,1):
                    axes[i,j].plot(t,[x['mean_velocity'][j] if x['mean_velocity'] is not None else np.nan for x in c],label=model)
                    axes[i,j].axhline(d['command'][j],color='black',linestyle='--',alpha=.2)
                axes[i,2].plot(t,[x['active_trials']/d['trials'] for x in c],label=model)
            for j,label in enumerate(('vx (m/s)','vy (m/s)','Fraction still active')):
                axes[i,j].set(title=f'{direction}: {label}',xlabel='Time (s)'); axes[i,j].grid(alpha=.25)
        h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=4)
        fig.suptitle(f'{iteration:,} iterations | Active-episode mean; dashed = command',y=.96)
        fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(plots/f'velocity_curves_{iteration:07d}.png',dpi=150); plt.close(fig)
    select_highlight(data,output)

def select_highlight(data,output):
    candidates=[]
    for iteration, case in {(d['iteration'],d['name']) for d in data if d['direction'] in RIGHT}:
        rows={d['model']:d for d in data if d['iteration']==iteration and d['name']==case}
        if set(rows)!=set(MODELS): continue
        for trial in range(rows['biHAM']['trials']):
            values={m:rows[m]['per_trial'][trial] for m in MODELS}
            errors={m:v['linear_rmse'] for m,v in values.items()}
            if any(v is None for v in errors.values()) or values['biHAM']['survived_s']<19.99: continue
            reference=values['biHAM']['initial_state']
            state_keys=('root','dof','actor_obs') + (('domi_clock',) if reference.get('domi_clock') is not None else ())
            if not all(all(np.allclose(values[m]['initial_state'][k],reference[k],atol=1e-4,rtol=1e-4) for k in state_keys) for m in MODELS): continue
            ordered=errors['biHAM']<errors['sym_aug_HAM']<min(errors['HAM'],errors['MLP'])
            candidates.append(dict(case=case,iteration=iteration,trial=trial,errors=errors,requested_order_observed=ordered,rows={m:dict(checkpoint=rows[m]['checkpoint'],initial_state=values[m]['initial_state']) for m in MODELS}))
    if not candidates: return
    chosen=min(candidates,key=lambda c:(c['errors']['biHAM'],c['iteration'],c['case'],c['trial']))
    chosen['selection']='Post-hoc illustration: lowest biHAM error among survived matched rightward trials across all checkpoints; other models use the same checkpoint and initial state. Aggregate plots include ALL trials. Ranking was not a selection criterion.'
    temp=output/'highlight_selection.tmp'; temp.write_text(json.dumps(chosen,indent=2)+'\n'); temp.replace(output/'highlight_selection.json')
