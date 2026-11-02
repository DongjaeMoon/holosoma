# Media integrity, historical source parity, and presentation provenance.
from concurrent.futures import ThreadPoolExecutor,as_completed
import cv2,shutil,difflib
movie_paths=sorted(Q.rglob('*.mp4'))
def video_check(p):
 rec=dict(path=str(p.relative_to(R)),bytes=p.stat().st_size)
 try:
  probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames','-show_entries','format=duration','-of','json',str(p)],capture_output=True,text=True,timeout=30)
  if probe.returncode:raise RuntimeError(probe.stderr[:800])
  info=json.loads(probe.stdout);rec['probe']=info
  decode=subprocess.run(['ffmpeg','-v','error','-nostdin','-hwaccel','none','-threads','1','-i',str(p),'-map','0:v:0','-f','null','-'],capture_output=True,text=True,timeout=90)
  rec['full_cpu_decode_returncode']=decode.returncode
  rec['decode_errors']=decode.stderr[:1200]
  cap=cv2.VideoCapture(str(p));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  measurements=[]
  for fraction in (.1,.5,.9):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames*fraction));ok,img=cap.read()
   if not ok:measurements.append(dict(fraction=fraction,read=False));continue
   h,w=img.shape[:2];roi=img[max(90,int(h*.2)):int(h*.82),int(w*.15):int(w*.75)]
   measurements.append(dict(fraction=fraction,read=True,scene_mean=float(roi.mean()),scene_std=float(roi.std()),near_black_fraction=float(np.mean(np.max(roi,axis=-1)<4))))
  cap.release();rec['frames_checked']=measurements
  meta=p.parent/'video.json'
  if meta.exists():
   data=json.loads(meta.read_text())
   rec['metadata_simulated_seconds']=data.get('simulated_seconds')
   rec['duration_minus_simulation']=float(info['format']['duration'])-data.get('simulated_seconds',float(info['format']['duration']))
   rec['metadata_selection']=data.get('selection')
 except Exception as e:rec['error']=repr(e)
 return rec
video_reports=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(video_check,p):p for p in movie_paths}
 for future in as_completed(futures):
  video_reports.append(future.result())
  if len(video_reports)%20==0:print('VIDEO_PROGRESS',len(video_reports),'/',len(movie_paths),flush=True)
video_reports.sort(key=lambda r:r['path'])
write_json('video_integrity.json',video_reports)
# Compare numerics separately from illustrative replays, including backend differences.
replay_comparisons=[]
for label,folder,cohort,model in [('biHAM',Q/'biham_mlp_command_replay/biHAM','v2','biHAM'),('native_MLP',Q/'vanilla_mlp_30k_native_io_gpu2/video_preview_10k','native','MLP')]:
 for name in ('left','right'):
  meta=json.loads((folder/name/'video.json').read_text())
  with (folder/name/'commands.csv').open() as stream:rows=list(csv.DictReader(stream))
  stable=[r for r in rows if float(r['time_s'])>=2.]
  rmse=(sum((float(r['actual_vx'])-float(r['command_vx']))**2+(float(r['actual_vy'])-float(r['command_vy']))**2 for r in stable)/len(stable))**.5
  measured=metric_sources[(cohort,model,10000,name+'_0.6mps')]
  render_cfg=yaml.safe_load((folder/'render_config.yaml').read_text())
  replay_comparisons.append(dict(model=label,direction=name,video_metric=meta['linear_rmse'],csv_rmse=rmse,
    trial_32_mean_metric=measured['mean_linear_rmse'],replay_backend=render_cfg['simulator']['config']['name'],
    training_backend='isaacgym' if cohort=='v2' else 'isaacsim',
    numerical_eval_backend='isaacgym' if cohort=='v2' else 'isaacsim'))
write_json('video_vs_numeric_protocols.json',replay_comparisons)
# Compare methods with the preserved IsaacGym checkout without executing it.
parity=[]
archive=R/'.isaacgym/holosoma/src/holosoma/holosoma'
def methods(src,cls):
 tree=ast.parse(src)
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 return {n.name:n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for relative,cls,names in [('agents/fast_sac/fast_sac.py','Actor',('forward','get_actions_and_log_probs','explore','process_obs')),('agents/fast_sac/fast_sac_utils.py','EmpiricalNormalization',('forward','update'))]:
 try:
  before=methods((archive/relative).read_text(),cls);after=methods((R/'src/holosoma/holosoma'/relative).read_text(),cls)
  for name in names:
   same=ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False)
   parity.append(dict(file=relative,method=cls+'.'+name,ast_identical=same))
 except Exception as e:parity.append(dict(file=relative,error=repr(e)))
write_json('archived_vs_current_method_parity.json',parity)
# Record post-hoc highlight choice and stale presentation labels.
presentation=[]
for cohort,folder in cohorts.items():
 p=folder/'highlight_selection.json'
 if p.exists():
  data=json.loads(p.read_text())
  presentation.append(dict(cohort=cohort,selection=data,index_mentions_final_checkpoint='Matched highlight · final checkpoint' in (folder/'index.html').read_text()))
write_json('presentation_provenance.json',presentation)
print('VIDEOS',len(video_reports),'decode issues',[(r['path'],r.get('error'),r.get('full_cpu_decode_returncode')) for r in video_reports if r.get('error') or r.get('full_cpu_decode_returncode')])
print('VIDEO_DURATION_MISMATCH',[(r['path'],r.get('duration_minus_simulation')) for r in video_reports if abs(r.get('duration_minus_simulation',0))>.2])
print('REPLAY_PROTOCOLS',replay_comparisons)
print('METHOD_PARITY',parity)
print('PRESENTATION',presentation)
