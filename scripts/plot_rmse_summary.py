"""One publication-style PNG: trained directions versus held-out directions."""
import argparse
import json
from pathlib import Path
import statistics

MODELS = ('biHAM', 'sym_aug_HAM', 'HAM', 'MLP')
LABELS = ('biHAM', 'HAM + symmetry augmentation', 'HAM', 'MLP')
COLORS = ('#1769aa', '#e39328', '#60944e', '#8a659e')
SEEN = ('forward', 'backward', 'left')
HELD = ('right', 'forward_right', 'backward_right')


def main():
    global MODELS, LABELS, COLORS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--exclude-mlp', action='store_true')
    args = parser.parse_args()
    if args.exclude_mlp:
        MODELS, LABELS, COLORS = MODELS[:3], LABELS[:3], COLORS[:3]
    rows = json.loads((args.evaluation / 'summary.json').read_text())
    assert len(rows) == 640, 'Expected 4 models x 10 checkpoints x 16 cases.'
    assert len({(r['model'], r['iteration'], r['name']) for r in rows}) == 640
    def average(model, iteration, directions):
        values = [r for r in rows if r['model'] == model and r['iteration'] == iteration
                  and r['direction'] in directions]
        assert len(values) == 2 * len(directions)
        assert all(r['mean_linear_rmse'] is not None for r in values)
        return statistics.mean(r['mean_linear_rmse'] for r in values)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
                         'text.color': '#182b42', 'axes.edgecolor': '#a4afbd'})
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.12), hspace=.48, wspace=.23)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    iterations = range(1000, 10001, 1000)
    for ax, directions, title in zip(axes, (SEEN, HELD),
            ('A   Training-command directions', 'B   Held-out command directions')):
        for model, label, color in zip(MODELS, LABELS, COLORS):
            ax.plot(range(1, 11), [average(model, i, directions) for i in iterations],
                    color=color, marker='o', markersize=4, linewidth=2.2, label=label)
        ax.set_title(title, loc='left', pad=28)
        ax.set(xlabel='Training iterations (thousands)',
               ylabel='Planar velocity RMSE (m/s)', xticks=range(1, 11), xlim=(.7, 10.3))
        ax.set_ylim(0, .60 if args.exclude_mlp else 1.85)
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
        ax.set_axisbelow(True)
        ax.text(0, 1.02, 'Forward / backward / left' if directions == SEEN else
                'Right / forward-right / backward-right', transform=ax.transAxes,
                fontsize=10, color='#64748b')
    ax = fig.add_subplot(grid[1, :])
    directions = SEEN + HELD
    positions = np.arange(6)
    width = .18
    ax.axvspan(2.5, 5.5, color='#edf5fb', zorder=0)
    for index, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        values = [average(model, 10000, (d,)) for d in directions]
        bars = ax.bar(positions + (index - (len(MODELS) - 1) / 2) * width, values, width,
                      color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], fontsize=9, padding=4)
    ax.set(title='C   Final checkpoint (10k): direction-by-direction comparison',
           ylabel='Planar velocity RMSE (m/s)', xticks=positions,
           xticklabels=('Forward', 'Backward', 'Left', 'Right', 'Forward-right', 'Backward-right'),
           ylim=(0, .47), xlim=(-.6, 5.6))
    ax.axvline(2.5, color='#b9c9d8', linewidth=1, linestyle='--')
    ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.text(1, .445, 'TRAINING DIRECTIONS', ha='center', fontsize=10, color='#64748b')
    ax.text(4, .445, 'HELD-OUT DIRECTIONS', ha='center', fontsize=10, color='#1769aa')
    fig.suptitle('Command tracking after the corrected network transfer', x=.07, ha='left',
                 y=.99, fontsize=19, fontweight='bold')
    fig.text(.07, .944, 'FastSAC / G1 locomotion   |   Training seed 43   |   Lower RMSE is better',
             fontsize=12, color='#64748b')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .918), ncol=len(MODELS), frameon=False)
    fig.subplots_adjust(top=.80, bottom=.20, left=.07, right=.98)
    fig.text(.07, .083,
        'Each direction: speeds 0.3 and 0.6 m/s, 32 trials per speed; mean of per-trial planar RMSE.\n'
        'Nominal flat ground; 20 s episodes; first 2 s excluded; errors counted only before the first failure.\n'
        'At 10k all trials survived 20 s. Earlier curves must be read with survival results; one training seed only.',
        fontsize=10, linespacing=1.5, color='#475569')
    fig.text(.07, .025,
        'Shared preprocessing: domi phase + reflection-compatible normalization. Only HAM + augmentation receives mirrored replay.',
        fontsize=10, color='#475569')
    plots = args.evaluation / 'plots'
    plots.mkdir(exist_ok=True)
    stem = 'rmse_summary_no_mlp' if args.exclude_mlp else 'rmse_summary'
    path = plots / (stem + '.png')
    fig.savefig(path, dpi=220, facecolor='white')
    plt.close(fig)
    table = {m: {'training_directions': average(m, 10000, SEEN),
                 'held_out_directions': average(m, 10000, HELD),
                 **{d: average(m, 10000, (d,)) for d in directions}} for m in MODELS}
    (plots / (stem + '_values.json')).write_text(json.dumps(table, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
