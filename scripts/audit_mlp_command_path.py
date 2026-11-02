"""CPU-only audit of saved MLP command inputs, outputs and existing rollout scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args()
    root=args.campaign.resolve()
    output=root/'command_bug_audit'
    output.mkdir(exist_ok=True)
    run=next((root/'training').glob('*/*/holosoma_config.yaml')).parent
    import torch
    from holosoma.agents.fast_sac.fast_sac import Actor
    from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization
    from holosoma.utils.rotations import quat_rotate_inverse
    from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    result=dict(scope='CPU only: existing checkpoints, saved initial observations and saved rollout logs; no new simulation or training.',
        audit=json.loads((run/'command_audit_status.json').read_text()),checkpoints=[])

    # Independent analytic cases: identity and a +90-degree yaw.
    q=torch.tensor([[0.,0.,0.,1.],[0.,0.,math.sqrt(.5),math.sqrt(.5)]])
    world=torch.tensor([[0.,1.,0.],[0.,1.,0.]])
    expected=torch.tensor([[0.,1.,0.],[1.,0.,0.]])
    native=torch.zeros(2,13);native[:,3:7]=q[:,[3,0,1,2]];native[:,7:10]=world
    proxy=RootStatesProxy(native)
    body=quat_rotate_inverse(proxy[:,3:7],proxy[:,7:10],w_last=True)
    result['quaternion_known_cases_max_error']=float((body-expected).abs().max())
    assert torch.allclose(body,expected,atol=1e-6)

    # This known native running-variance issue is tested separately from OOD behavior.
    n=EmpiricalNormalization(1,'cpu')
    n(torch.tensor([[0.],[0.]]));n(torch.tensor([[2.],[2.]]))
    all_samples=torch.tensor([0.,0.,2.,2.])
    result['native_running_variance_counterexample']=dict(saved_mean=float(n._mean[0,0]),
        expected_mean=float(all_samples.mean()),saved_variance=float(n._var[0,0]),
        expected_variance=float(all_samples.var(unbiased=False)),
        note='Known native variance-merge error; this test does not establish a cause of OOD locomotion.')

    for iteration in (10000,30000):
        checkpoint=run/('model_%07d.pt'%iteration)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        ac=state['experiment_config']['algo']['config']
        assert not ac['use_symmetry'] and not ac['symmetry_compatible_io'] and ac['bilateral'] is None
        ns=state['obs_normalizer_state']
        dim=ns['_mean'].numel();assert dim==100
        actor=Actor({'actor_obs':dict(start=0,end=dim,size=dim)},['actor_obs'],29,1,
            ac['actor_hidden_dim'],ac['log_std_max'],ac['log_std_min'],
            use_tanh=ac['use_tanh'],use_layer_norm=ac['use_layer_norm'],device='cpu')
        actor.load_state_dict(state['actor_state_dict'],strict=True);actor.eval()
        norm=EmpiricalNormalization(dim,'cpu');norm.load_state_dict(ns);norm.eval()
        before={k:v.clone() for k,v in norm.state_dict().items()}
        samples=[];metrics=[]
        for path in sorted((root/'evaluation/videos/MLP'/checkpoint.stem).glob('*/metrics.json')):
            case=json.loads(path.read_text())
            samples.extend(t['initial_state']['actor_obs'] for t in case['per_trial'])
            if case['direction'] in ('right','left','standing'):
                trials=[t for t in case['per_trial'] if t['linear_rmse'] is not None]
                recomputed=sum(t['linear_rmse'] for t in trials)/len(trials)
                assert abs(recomputed-case['mean_linear_rmse'])<1e-8
                metrics.append(dict(case=case['name'],command=case['command'],
                    rmse=recomputed,zero_velocity_baseline_rmse=math.hypot(*case['command'][:2]),
                    trials=case['trials'],successes=case['successes'],failure_rate=case['failure_rate'],
                    mean_velocity=[sum(t['mean_velocity'][k] for t in trials)/len(trials) for k in (0,1)]))
        assert len(samples)==544
        observations=torch.tensor(samples)
        values=(-.9,-.6,-.3,0.,.3,.6,.9)
        actions={};sweep=[]
        with torch.no_grad():
            for vy in values:
                x=observations.clone();x[:,[33,34,32]]=torch.tensor([0.,vy,0.])
                normalized=norm(x,update=False)
                manual=(x-norm._mean)/(norm._std+norm.eps)
                assert torch.equal(normalized,manual)
                action=actor(normalized)[0];assert torch.isfinite(action).all()
                actions[vy]=action
                sweep.append(dict(command_vy=vy,normalized_vy=float(normalized[0,34]),
                    action_rms=float(action.square().mean().sqrt())))
            for row in sweep:
                delta=actions[row['command_vy']]-actions[0.]
                row.update(action_change_from_zero_rms=float(delta.square().mean().sqrt()),
                    fraction_states_with_changed_action=float((delta.abs().amax(-1)>1e-6).float().mean()))
            left_delta=actions[.6]-actions[0.]
            right_delta=actions[0.]-actions[-.6]
            cosine=torch.nn.functional.cosine_similarity(left_delta,right_delta,dim=-1)
        frozen=all(torch.equal(before[k],v) for k,v in norm.state_dict().items())
        assert frozen
        result['checkpoints'].append(dict(iteration=iteration,checkpoint=str(checkpoint),
            actor_class=type(actor).__name__,strict_weight_load=True,
            use_symmetry=ac['use_symmetry'],symmetry_compatible_io=ac['symmetry_compatible_io'],
            bilateral=ac['bilateral'],state_count=len(samples),normalizer_unchanged=frozen,
            command_indices_vx_vy_yaw=[33,34,32],sweep=sweep,
            opposite_command_deltas_cosine_mean=float(cosine.mean()),
            opposite_command_deltas_cosine_median=float(cosine.median()),
            rollout_metrics=metrics,
            limitation='Frozen-state action sensitivity is not a closed-loop command-clamp intervention.'))
        print('AUDIT_CHECKPOINT '+str(iteration)+' '+json.dumps(result['checkpoints'][-1]),flush=True)

    clips=[]
    for name in ('left','right','switch'):
        folder=root/'video_preview_10k'/name
        with (folder/'commands.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        if name=='switch':
            clips.append(dict(name=name,rows=len(rows),commands=sorted({
                tuple(round(float(r[k]),5) for k in ('command_vx','command_vy','command_yaw')) for r in rows})))
            continue
        measured=[r for r in rows if float(r['time_s'])>=2.]
        raw=lambda r,k:float(r[k])
        error=sum((raw(r,'actual_vx')-raw(r,'command_vx'))**2+
                  (raw(r,'actual_vy')-raw(r,'command_vy'))**2 for r in measured)
        mse=error/len(measured)
        mean=[sum(raw(r,k) for r in measured)/len(measured) for k in ('actual_vx','actual_vy')]
        meta=json.loads((folder/'video.json').read_text())
        clips.append(dict(name=name,rows=len(rows),csv_recomputed_planar_rmse=math.sqrt(mse),
            video_metric_planar_rmse=meta['linear_rmse'],
            mean_body_velocity=mean,
            manager_policy_command_checked_each_step=meta['actual_policy_commands_verified'],
            note='CSV is pre-action; video metric is post-action, giving a one-step timing offset.'))
    result['existing_video_command_checks']=clips
    source_paths=[Path(__file__),Path('scripts/evaluate_command_holdout.py'),
        Path('src/holosoma/holosoma/agents/fast_sac/fast_sac.py'),
        Path('src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py')]
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
