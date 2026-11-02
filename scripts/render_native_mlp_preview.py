"""High-contrast IsaacSim replay of saved MLP or biHAM checkpoints."""
import argparse
import csv
from dataclasses import replace
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import quote
from evaluate_command_holdout import atomic_json,build_config,load_actor,render_case,refresh_observations


def style(env):
    import omni.usd
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
    stage=omni.usd.get_context().get_stage()
    def material(name,color):
        mat=UsdShade.Material.Define(stage,'/World/PreviewMaterials/'+name)
        shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        return mat
    blue=material('RobotBlue',(.025,.25,.72))
    dark=material('FeetHands',(.035,.055,.09))
    ground=material('Ground',(.79,.82,.86))
    # Bind above instance roots; changing instanceability after physics setup
    # deletes collision prims and invalidates the PhysX tensor view.
    robot=stage.GetPrimAtPath('/World/envs/env_0/Robot')
    UsdShade.MaterialBindingAPI.Apply(robot).Bind(blue,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    count=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        path=str(prim.GetPath())
        if '/Robot/' in path:
            count+=1
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(ground,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    light=UsdLux.DomeLight.Define(stage,'/World/PreviewDome')
    light.CreateIntensityAttr().Set(650.)
    light.CreateColorAttr().Set(Gf.Vec3f(.88,.93,1.))
    origin=env.terrain_manager.get_state('locomotion_terrain').env_origins[0].cpu().tolist()
    points=[]
    for v in range(-30,31):
        points += [(origin[0]+v,origin[1]-30,.006),(origin[0]+v,origin[1]+30,.006),
                   (origin[0]-30,origin[1]+v,.006),(origin[0]+30,origin[1]+v,.006)]
    grid=UsdGeom.BasisCurves.Define(stage,'/World/PreviewGrid')
    grid.CreateTypeAttr().Set('linear')
    grid.CreateCurveVertexCountsAttr().Set([2]*(len(points)//2))
    grid.CreatePointsAttr().Set(points)
    grid.CreateWidthsAttr().Set([.012])
    grid.SetWidthsInterpolation('constant')
    grid.CreateDisplayColorAttr().Set([(.47,.52,.59)])
    return dict(robot_meshes=count,visual_only=True,robot='blue',ground='light gray 1m grid')


def hud(image,env,title,t,model_label='MLP 10k'):
    import cv2
    import numpy as np
    from holosoma.managers.observation.terms.locomotion import base_lin_vel,base_ang_vel
    frame=image.copy(); h,w=frame.shape[:2]
    command=env.command_manager.commands[0,:3].cpu().tolist()
    actual=[*base_lin_vel(env)[0,:2].cpu().tolist(),float(base_ang_vel(env)[0,2])]
    cv2.rectangle(frame,(0,0),(w,88),(13,22,35),-1)
    cv2.putText(frame,model_label+' | '+title+' | t=%.2fs'%t,(22,27),0,.67,(245,248,252),1,cv2.LINE_AA)
    cv2.putText(frame,'TARGET vx=%+.2f vy=%+.2f m/s yaw=%+.2f rad/s'%tuple(command),(22,53),0,.62,(255,182,65),1,cv2.LINE_AA)
    cv2.putText(frame,'ACTUAL vx=%+.2f vy=%+.2f m/s yaw=%+.2f rad/s'%tuple(actual),(22,78),0,.62,(78,220,255),1,cv2.LINE_AA)
    cx,cy=w-135,h-125
    cv2.rectangle(frame,(w-265,h-253),(w-8,h-8),(13,22,35),-1)
    cv2.putText(frame,'BODY FRAME / TOP VIEW',(w-255,h-231),0,.43,(230,237,245),1,cv2.LINE_AA)
    cv2.line(frame,(cx-78,cy),(cx+78,cy),(100,120,145),1)
    cv2.line(frame,(cx,cy-78),(cx,cy+78),(100,120,145),1)
    cv2.putText(frame,'+vx',(cx+5,cy-81),0,.42,(200,210,225),1,cv2.LINE_AA)
    cv2.putText(frame,'+vy LEFT',(cx-118,cy-8),0,.35,(200,210,225),1,cv2.LINE_AA)
    for value,color,width in ((command,(255,182,65),4),(actual,(78,220,255),2)):
        end=(cx-int(np.clip(value[1],-1,1)*105),cy-int(np.clip(value[0],-1,1)*105))
        cv2.arrowedLine(frame,(cx,cy),end,color,width,cv2.LINE_AA,tipLength=.22)
    cv2.putText(frame,'Orange: target | Cyan: actual',(w-255,h-25),0,.39,(230,237,245),1,cv2.LINE_AA)
    return frame


def page(output,records,model='MLP'):
    cards=[]
    for row in records:
        name=row['name']; path=name+'/'+row['web_file']
        metric=('RMSE %.3f m/s'%row['linear_rmse']) if row['linear_rmse'] is not None else '명령 전환 재생'
        cards.append('<section><h2>'+html.escape(row['label'])+'</h2><video controls playsinline preload="metadata" poster="'+name+'/preview.png" src="'+quote(path)+'"></video><p>'+str(round(row['simulated_seconds'],2))+' s / '+metric+'</p><a href="'+name+'/commands.csv">Actual policy commands and body velocity CSV</a></section>')
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>MLP 10k command replay</title><style>body{margin:30px auto;padding:0 22px;max-width:1500px;font:16px system-ui;background:#111b2a;color:#e8eef7}p{line-height:1.7;color:#bdcbdc}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:#1c293b;padding:18px;border-radius:14px}video{width:100%;background:#ddd;border-radius:8px}a{color:#80c6ff}button{padding:10px;font:inherit}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>MLP 10k · 명령과 실제 움직임</h1><p>기본 empirical 정규화 / 기본 phase / 대칭 증강 OFF. 파란 로봇과 밝은 격자 바닥.<br>주황색=목표 속도, 하늘색=실제 속도. 작은 화살표는 로봇 좌표계를 위에서 본 방향입니다.<br>저장된 10k 정책을 별도 1env에서 재생한 영상입니다. 학습 당시 녹화본이나 32trial 평균 영상은 아닙니다.</p><button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">모든 영상 처음부터 재생</button><p><a href="status.json">렌더 상태</a></p><div class="grid">'''
    text=text.replace('MLP 10k',model+' 10k')
    if model=='biHAM':
        text=text.replace('기본 empirical 정규화 / 기본 phase / 대칭 증강 OFF.','기존 v2 biHAM / reflection 정규화 / domi phase / IsaacGym 학습 정책의 IsaacSim 재생.')
    (output/'index.html').write_text(text+''.join(cards)+'</div></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--model',choices=['MLP','biHAM'],default='MLP')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',DISPLAY='')
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.eval_utils import CheckpointConfig,load_saved_experiment_config
    from holosoma.utils.sim_utils import setup_simulation_environment,simulation_resource_session
    from holosoma.utils.helpers import get_class
    from holosoma.config_types.video import CartesianCameraConfig
    from isaaclab.app import AppLauncher
    root=args.campaign.resolve()
    output=args.output.resolve() if args.output else root/('video_preview_10k_probe' if args.probe else 'video_preview_10k')
    output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint.resolve() if args.checkpoint else next((root/'training').rglob('model_0010000.pt'))
    evaluation=args.evaluation.resolve() if args.evaluation else root/'evaluation'
    protocol=json.loads((evaluation/'manifest.json').read_text())['protocol']
    protocol=dict(protocol,video_duration_s=2. if args.probe else 20.,video_decimation=2)
    if args.model=='biHAM':
        # This is the user's own historical checkpoint, containing config dataclasses.
        from holosoma.utils.eval_utils import _experiment_config_from_saved
        checkpoint_state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        saved=_experiment_config_from_saved(checkpoint_state['experiment_config'])
        del checkpoint_state
    else:
        saved,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    cfg=build_config(saved,protocol,output,True)
    cfg=replace(cfg,logger=replace(cfg.logger,video=replace(cfg.logger.video,width=1280,height=720,
        playback_rate=.5,show_command_overlay=False,
        camera=CartesianCameraConfig(offset=[2.,-3.,1.6],target_offset=[0.,0.,-.1],smoothing=.7))),
        simulator=replace(cfg.simulator,config=replace(cfg.simulator.config,
          sim=replace(cfg.simulator.config.sim,render_interval=8,max_episode_length_s=45.),
          isaacsim=replace(cfg.simulator.config.isaacsim,kit_args=f'--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu={args.gpu}'))))
    cfg.save_config(str(output/'render_config.yaml'))
    original=AppLauncher._resolve_device_settings
    def devices(self,values):
        original(self,values)
        values['active_gpu']=args.gpu
        values['physics_gpu']=0
    AppLauncher._resolve_device_settings=devices
    records=[]
    def status(phase,**fields):
        atomic_json(output/'status.json',dict(phase=phase,model=args.model,gpu=args.gpu,checkpoint=str(checkpoint),clips=records,updated=time.time(),**fields))
    status('starting');page(output,records,args.model)
    try:
        env,device,app=setup_simulation_environment(cfg,device='cuda:0')
        AppLauncher._resolve_device_settings=original
        cleanup_timer=None
        with simulation_resource_session(env,app):
            env._randomize_ctrl_delay=False
            from holosoma.simulator.base_simulator.hooks import Phase
            for hook in env.simulator.hooks._records[Phase.FRAME_END]:
                if hook.name=='video.capture_frame':hook.every=2;hook._counter=0
            atomic_json(output/'visuals.json',style(env))
            algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,device=device,log_dir=str(output/'runtime'))
            algo.setup();policy=load_actor(algo,checkpoint)
            lookup,offset={},0
            for term in env.observation_manager.get_observation_layout(algo.config.actor_obs_keys):
                lookup[term['name']]=(list(range(offset,offset+term['dim'])),term['scale']);offset+=term['dim']
            indices=lookup['command_lin_vel'][0]+lookup['command_ang_vel'][0]
            scales=torch.tensor([lookup['command_lin_vel'][1]]*2+[lookup['command_ang_vel'][1]],device=env.device)
            from holosoma.managers.observation.terms.locomotion import base_lin_vel,base_ang_vel
            cases=[('left',.6,'LEFT +0.6 m/s')]
            if not args.probe:cases.extend([('right',-.6,'RIGHT -0.6 m/s'),('switch',0.,'COMMAND SWITCHING')])
            for name,vy,label in cases:
                destination=output/name;destination.mkdir(exist_ok=True)
                if (destination/'video.json').exists():
                    records.append(json.loads((destination/'video.json').read_text()));page(output,records,args.model);continue
                status('rendering',current=name)
                step=[0];context={'label':label}
                recorder=env.simulator.video_recorder
                recorder._apply_command_overlay=lambda image:hud(image,env,context['label'],step[0]*env.dt,args.model+' 10k')
                add=recorder._add_frame;count=[0]
                def capture(frame):
                    import cv2
                    count[0]+=1
                    if count[0]==20:cv2.imwrite(str(destination/'preview.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    add(frame)
                recorder._add_frame=capture
                with (destination/'commands.csv').open('w') as stream:
                    writer=csv.writer(stream);writer.writerow(('time_s','command_vx','command_vy','command_yaw','actual_vx','actual_vy','actual_yaw'))
                    def logged_policy(obs):
                        if name=='switch':
                            t=step[0]*env.dt
                            sequence=[(3.,[0.,0.,0.],'STAND'),(8.,[.6,0.,0.],'FORWARD +0.6'),
                                (13.,[-.6,0.,0.],'BACKWARD -0.6'),(19.,[0.,.6,0.],'LEFT +0.6'),
                                (25.,[0.,-.6,0.],'RIGHT -0.6'),(99.,[0.,0.,0.],'STAND')]
                            _,target,context['label']=next(item for item in sequence if t<item[0])
                            env.command_manager.commands[:,:3]=torch.tensor([target],device=env.device)
                            refresh_observations(env)
                            obs['actor_obs']=torch.cat([env.obs_buf_dict[k] for k in algo.config.actor_obs_keys],-1)
                        raw=obs['actor_obs'][:,indices]/scales
                        torch.testing.assert_close(raw,env.command_manager.commands[:,:3],atol=1e-6,rtol=1e-6)
                        writer.writerow((step[0]*env.dt,*raw[0].cpu().tolist(),*base_lin_vel(env)[0,:2].cpu().tolist(),float(base_ang_vel(env)[0,2])))
                        step[0]+=1
                        return policy(obs)
                    case_protocol=dict(protocol,video_duration_s=28.) if name=='switch' else protocol
                    row=render_case(env,algo,logged_policy,dict(command=[0.,vy,0.]),destination,case_protocol)
                    if name=='switch':row.update(linear_rmse=None,yaw_rmse=None)
                recorder._add_frame=add
                source=destination/row['files'][0];web_file=name+'.mp4'
                subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(source),'-c','copy','-movflags','+faststart',str(destination/web_file)],check=True)
                row.update(name=name,label=label,web_file=web_file,actual_policy_commands_verified=True)
                atomic_json(destination/'video.json',row);records.append(row);page(output,records,args.model)
                print('VIDEO_READY '+json.dumps(row),flush=True)
            status('complete')
            # Kit may block during headless shutdown after every video is saved.
            # This timer belongs only to this standalone replay process.
            def shutdown_timeout():
                print('All videos saved; exiting stalled IsaacSim shutdown.',flush=True)
                sys.stdout.flush();sys.stderr.flush();os._exit(0)
            cleanup_timer=threading.Timer(30.,shutdown_timeout)
            cleanup_timer.daemon=True;cleanup_timer.start()
        if cleanup_timer is not None:cleanup_timer.cancel()
        status('complete')
    except BaseException as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':
    main()

