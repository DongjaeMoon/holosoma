"""Replace only the 10k MLP in the existing four-model comparison; CPU plotting."""
import argparse
import json
from pathlib import Path
import statistics
import time

MODELS = ('biHAM', 'sym_aug_HAM', 'HAM', 'MLP')
LABELS = ('biHAM', 'HAM + symmetry augmentation', 'HAM', 'MLP (native IO, IsaacSim)')
COLORS = ('#1769aa', '#e39328', '#60944e', '#8a659e')
SEEN = ('forward', 'backward', 'left')
HELD = ('right', 'forward_right', 'backward_right')


def render(old, new, output):
    reference = json.loads((old/'summary.json').read_text())
    expected = [row for row in reference if row['model'] == 'MLP' and row['iteration'] == 10000]
    if len(expected) != 16:
        raise ValueError('Expected all 16 cases for the original MLP at 10k.')
    paths = [new/'videos'/'MLP'/'model_0010000'/row['name']/'metrics.json' for row in expected]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'New MLP 10k evaluation pending: {len(missing)}/16 cases.')
    replacement = [json.loads(path.read_text()) for path in paths]
    if any(row['model'] != 'MLP' or row['iteration'] != 10000 for row in replacement):
        raise ValueError('Replacement must use the new MLP checkpoint at exactly 10k.')
    rows = [row for row in reference if row['model'] != 'MLP' and row['iteration'] == 10000] + replacement
    if len(rows) != 64 or len({(r['model'], r['name']) for r in rows}) != 64:
        raise ValueError('Expected four models times 16 unique cases.')
    def average(model, directions, key='mean_linear_rmse'):
        selected = [row for row in rows if row['model'] == model and row['direction'] in directions]
        if len(selected) != 2*len(directions) or any(row.get(key) is None for row in selected):
            raise ValueError(f'Missing metric: {model}, {directions}, {key}')
        return statistics.mean(row[key] for row in selected)
    values = {model: dict(training_directions=average(model, SEEN),
        held_out_directions=average(model, HELD),
        **{direction: average(model, (direction,)) for direction in SEEN+HELD}) for model in MODELS}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios':(1, 1.35)})
    width = .18
    for index, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        offset = (index-1.5)*width
        for ax, directions in ((axes[0], (SEEN, HELD)), (axes[1], tuple((d,) for d in SEEN+HELD))):
            heights = [average(model, d) for d in directions]
            bars = ax.bar(np.arange(len(directions))+offset, heights, width, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:.3f}' for v in heights], fontsize=9, padding=3)
    axes[0].set(title='A   Mean tracking error at the 10k checkpoint',
        xticks=(0, 1), xticklabels=('Training directions: forward / backward / left',
                                  'Held-out directions: right / forward-right / backward-right'))
    axes[1].axvspan(2.5, 5.5, color='#edf5fb', zorder=0)
    axes[1].set(title='B   Direction-by-direction comparison at 10k',
        xticks=range(6), xticklabels=('Forward','Backward','Left','Right','Forward-right','Backward-right'))
    for ax in axes:
        ax.set_ylabel('Planar velocity RMSE (m/s)')
        ax.set_ylim(0, max(bar.get_height() for container in ax.containers for bar in container)*1.25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('10k comparison: previous MLP replaced with the native-IO MLP', fontsize=17, fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.94), ncol=4, frameon=False)
    fig.subplots_adjust(top=.84, bottom=.19, hspace=.42)
    fig.text(.08,.105,'All models: FastSAC / G1 / seed 43; 0.3 and 0.6 m/s, 32 trials per speed; 20 s, first 2 s excluded.',
             fontsize=10, color='#475569')
    fig.text(.08,.068,'biHAM/HAM: previous IsaacGym + domi phase + reflection normalization. MLP: new IsaacSim + native phase + empirical normalization.',
             fontsize=9, color='#475569')
    fig.text(.08,.035,'Errors are conditional on the first episode before failure. Survival/failure rates are saved in the accompanying sources JSON.',
             fontsize=9, color='#475569')
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.svg'):
        fig.savefig(output.with_suffix(ext), dpi=220, facecolor='white')
    plt.close(fig)
    output.with_name(output.name+'_values.json').write_text(json.dumps(values, indent=2)+'\n')
    sources = dict(old_evaluation=str(old), new_mlp_evaluation=str(new), iteration=10000,
        replacement_checkpoints=sorted({r['checkpoint'] for r in replacement}),
        conditions='Old three models retain IsaacGym/domi/reflection; MLP uses IsaacSim/native/empirical.',
        failure_rates={model:average(model, tuple(dict.fromkeys(r['direction'] for r in expected)), 'failure_rate')
                       for model in MODELS},
        mean_survival_s={model:average(model, tuple(dict.fromkeys(r['direction'] for r in expected)), 'mean_survival_s')
                         for model in MODELS})
    output.with_name(output.name+'_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(json.dumps(dict(plot=str(output.with_suffix('.png')), values=values), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--mlp-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output stem without an extension.')
    parser.add_argument('--wait', action='store_true', help='Wait for the already-running 10k evaluation; never starts GPU work.')
    args = parser.parse_args()
    while True:
        try:
            render(args.evaluation, args.mlp_evaluation, args.output)
            return
        except FileNotFoundError:
            if not args.wait:
                raise
            state_path = args.mlp_evaluation.parent/'status.json'
            if state_path.exists() and json.loads(state_path.read_text())['phase'] in ('failed','interrupted'):
                raise RuntimeError('Evaluation stopped; no plot made from incomplete replacement data.')
            time.sleep(10)


if __name__ == '__main__':
    main()

