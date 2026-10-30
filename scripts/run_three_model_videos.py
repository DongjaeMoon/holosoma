"""Render six commands for three final policies, sequentially on GPU 3."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluate_command_holdout import atomic_json

PAGE = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Three policy videos</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f5f7fb;color:#182b42}#cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:white;padding:18px;border-radius:12px}video{width:100%;background:#111}select,button{font:inherit;padding:9px;margin:10px 12px 18px 0}p{line-height:1.6}@media(max-width:900px){#cards{grid-template-columns:1fr}}</style>
<h1>biHAM · sym-aug HAM · HAM 명령 추종 영상</h1>
<p>최종 10k · 0.6 m/s · yaw 명령 0 · 동일한 평가 seed · 최대 20초 · GPU 3</p>
<p>학습 당시 녹화본이 아닌 저장된 정책 재생입니다. 평지, 관측 노이즈·외란 OFF. 잘된 rollout을 선별하지 않았습니다.<br>sym-aug HAM은 증강을 통해 오른쪽 명령도 학습했습니다.</p>
<p><a href="plots/rmse_summary_no_mlp.png">세 모델 RMSE PNG</a> · <a href="./">전체 수치 평가</a></p>
<select id="direction"><option value="left">학습 명령: 왼쪽</option><option value="forward">학습 명령: 앞</option><option value="backward">학습 명령: 뒤</option><option value="right">제외된 명령: 오른쪽</option><option value="forward_right">제외된 명령: 오른쪽 앞</option><option value="backward_right">제외된 명령: 오른쪽 뒤</option></select>
<button id="play">세 영상 처음부터 재생</button><span id="status"></span><div id="cards"></div>
<script>
const models=[['biHAM','biHAM'],['sym_aug_HAM','sym-aug HAM'],['HAM','HAM']], box=document.querySelector('#cards'), direction=document.querySelector('#direction');
async function json(path){try{let r=await fetch(path,{cache:'no-store'});return r.ok?await r.json():null}catch{return null}}
async function refresh(){let state=await json('three_model_video_status.json');document.querySelector('#status').textContent=state?`${state.status}: ${state.complete}/18 완료`:'';if([...box.querySelectorAll('video')].some(v=>!v.paused))return;box.replaceChildren();
for(let [model,label] of models){let base=`videos/${model}/model_0010000/${direction.value}_0.6mps/`,data=await json(base+'video.json'),card=document.createElement('div');card.className='card';let h=document.createElement('h2');h.textContent=label;card.append(h);
if(data?.files?.length){let v=document.createElement('video');v.controls=true;v.preload='metadata';v.src=base+data.files[0];card.append(v);let p=document.createElement('p');p.textContent=`이 영상 RMSE: ${data.linear_rmse.toFixed(3)} m/s · ${data.simulated_seconds.toFixed(1)}초 · ${data.terminated?'조기 종료':'완주'}`;card.append(p);let a=document.createElement('a');a.href=v.src;a.download='';a.textContent='MP4 다운로드';card.append(a)}else{let p=document.createElement('p');p.textContent='렌더링 대기 중';card.append(p)}box.append(card)}}
direction.onchange=()=>{for(let v of box.querySelectorAll('video'))v.pause();refresh()};document.querySelector('#play').onclick=()=>{for(let v of box.querySelectorAll('video')){v.currentTime=0;v.play().catch(()=>{})}};refresh();setInterval(refresh,15000);
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.campaign/'evaluation'
    lock=(output/'three_model_video.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=('biHAM','sym_aug_HAM','HAM')
    names=[d+'_0.6mps' for d in ('left','right','forward','backward','forward_right','backward_right')]
    spec=json.loads((output/'manifest.json').read_text())
    spec['protocol'].update(video_cases=names,video_duration_s=20.)
    manifest=output/'three_model_video_manifest.json'
    atomic_json(manifest,spec)
    (output/'three_model_videos.html').write_text(PAGE)
    started=time.time()
    def status(state,**fields):
        complete=sum((output/'videos'/m/'model_0010000'/n/'video.json').exists() for m in models for n in names)
        atomic_json(output/'three_model_video_status.json',dict(status=state,complete=complete,total=18,
                    gpu=3,started=started,updated=time.time(),**fields))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID',DISPLAY='',OMP_NUM_THREADS='1')
    runs=json.loads((args.campaign/'runs.json').read_text())
    try:
        for model in models:
            checkpoint=next(Path(runs[model]).rglob('model_0010000.pt'))
            command=[sys.executable,'scripts/evaluate_command_holdout.py','--manifest',str(manifest),
                     '--model',model,'--checkpoint',str(checkpoint),'--phase','video','--gpu','3']
            with (output/f'{model}_six_command_video.log').open('a') as log:
                process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    status('running',model=model,pid=process.pid)
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f'{model} render failed with exit {process.returncode}')
        status('complete')
    except Exception as error:
        status('failed',error=str(error))
        raise


if __name__=='__main__': main()
