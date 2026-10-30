"""Validate saved experiment metrics/configs and link the single summary PNG."""
import argparse
import json
import math
from pathlib import Path
import statistics
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    root = parser.parse_args().campaign
    evaluation = root / 'evaluation'
    rows, maximum, trials = [], 0., 0
    for path in evaluation.glob('videos/*/model_*/*/metrics.json'):
        row = json.loads(path.read_text())
        values = [x for x in row['per_trial'] if x['linear_rmse'] is not None]
        average = statistics.mean(x['linear_rmse'] for x in values) if values else None
        assert average is None or abs(average - row['mean_linear_rmse']) < 1e-9, path
        for value in values:
            error = abs(value['linear_rmse'] - math.hypot(value['vx_rmse'], value['vy_rmse']))
            maximum = max(error, maximum)
            assert error < 1e-5, path
        if row['iteration'] == 10000:
            assert all(x['survived_s'] > 19.99 and not x['failure'] and not x['early_timeout']
                       and x['measured_steps'] == 900 for x in row['per_trial']), path
        trials += len(row['per_trial'])
        rows.append(row)
    assert len(rows) == 640
    flags = {}
    for model, run in json.loads((root / 'runs.json').read_text()).items():
        path = next(Path(run).rglob('holosoma_config.yaml'))
        config = yaml.safe_load(path.read_text())
        algo = config['algo']['config']
        assert algo['symmetry_compatible_io'] and algo['num_learning_iterations'] == 10000
        setup = config['command']['setup_terms']
        assert setup['locomotion_gait']['func'].endswith(':DomiGait')
        assert setup['locomotion_command']['func'].endswith(':ThreeDirectionCommand')
        flags[model] = algo['use_symmetry']
    assert flags == {'biHAM': False, 'MLP': False, 'sym_aug_HAM': True, 'HAM': False}
    report = dict(metric_cases=640, trials=trials, rmse_axis_identity_max_error=maximum,
                  all_10k_trials_complete=True, augmentation_flags=flags)
    (evaluation / 'results_integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    page = evaluation / 'index.html'
    html = page.read_text()
    link = ('<p><a href="plots/rmse_summary.png" download>Download: RMSE summary PNG '
            '(all checkpoints + training / held-out directions)</a></p>\n')
    if 'Download: RMSE summary PNG' not in html:
        offset = html.index('<h1')
        page.write_text(html[:offset] + link + html[offset:])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
