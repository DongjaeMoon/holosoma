# Final preservation and report-link checks. Existing files are read only.
from html.parser import HTMLParser
before=json.loads((O/'source_hashes_before.json').read_text())
after={p:hashlib.sha256((R/p).read_bytes()).hexdigest() if (R/p).is_file() else None for p in before}
source_changes=[p for p,h in before.items() if after[p]!=h]
additional=json.loads((O/'additional_source_hashes.json').read_text())
extra_changes=[r['path'] for r in additional if not (R/r['path']).is_file() or hashlib.sha256((R/r['path']).read_bytes()).hexdigest()!=r['sha256']]
session=json.loads((O/'session.json').read_text())
integrity=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_checked=len(before),source_files_changed=source_changes,additional_text_files_checked=len(additional),additional_text_files_changed=extra_changes,git_head_before=session['git_head'],git_head_after=git('rev-parse','HEAD').strip(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_available=torch.cuda.is_available(),note='1425 src/scripts hashes were taken at audit start. Additional text hashes were taken in the late coverage phase. Only new audit-directory files were written; no training/simulator/GPU execution.')
write_json('source_integrity_final.json',integrity)
cp=json.loads((O/'checkpoint_validation.json').read_text())
cp_changed=[]
for i,rec in enumerate(cp):
 p=R/rec['path'];h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 if h.hexdigest()!=rec['file_sha256']:cp_changed.append(rec['path'])
 if (i+1)%50==0:print('FINAL_CHECKPOINT_HASHES',i+1,flush=True)
logs=json.loads((O/'log_inventory.json').read_text())
logs_changed=[r['path'] for r in logs if not (R/r['path']).is_file() or (R/r['path']).stat().st_size!=r['bytes']]
events=json.loads((O/'tensorboard_validation.json').read_text())
events_changed=[r['path'] for r in events if 'bytes' in r and (R/r['path']).stat().st_size!=r['bytes']]
statuses=json.loads((O/'campaign_statuses.json').read_text())
status_changed=[r['path'] for r in statuses if 'data' in r and json.loads((R/r['path']).read_text())!=r['data']]
write_json('artifact_integrity_final.json',dict(checkpoints_sha256_checked=len(cp),checkpoints_changed=cp_changed,terminal_logs_size_checked=len(logs),terminal_logs_size_changed=logs_changed,tensorboard_sizes_checked=len(events),tensorboard_sizes_changed=events_changed,campaign_statuses_changed=status_changed,note='Checkpoint contents verified against hashes taken during the audit. Text/event checks are size comparisons, not claims of initial bytewise hashes. All existing artifacts were only opened for reading.'))
assert not source_changes and not extra_changes and not cp_changed and not logs_changed and not events_changed and not status_changed
assert integrity['git_head_before']==integrity['git_head_after'] and not integrity['cuda_available']
for p in O.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):
  for name,value in attrs:
   if name in ('href','src') and value and not value.startswith(('http:','https:','#','data:')):self.urls.append(value)
parser=Links();parser.feed((O/'index.html').read_text())
missing=[u for u in parser.urls if not (O/u.split('#')[0]).exists()]
assert not missing,missing
for name in ('READ_ME_FIRST.txt','FULL_AUDIT_REPORT_KO.txt','comparison_lateral_current.csv','index.html'):
 assert (O/name).stat().st_size>100
for p in O.glob('*.json'):json.loads(p.read_text())
write_json('report_validation.json',dict(local_links_checked=len(parser.urls),missing_links=missing,audit_python_files_syntax_ok=True,audit_json_files_parse_ok=True,existing_source_unchanged=True,checkpoint_contents_unchanged=True))
probe=O/'writer_probe.txt'
if probe.exists():probe.unlink()
for parent,dirs,files in os.walk(O):
 os.chown(parent,1005,1005)
 for name in files:os.chown(Path(parent)/name,1005,1005)
manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='audit_output_manifest.json']
write_json('audit_output_manifest.json',manifest);os.chown(O/'audit_output_manifest.json',1005,1005)
print('AUDIT_FINAL_OK',dict(source_changed=len(source_changes),checkpoints_changed=len(cp_changed),missing_links=len(missing),output_files=len(manifest)+1,report_lines=len((O/'FULL_AUDIT_REPORT_KO.txt').read_text().splitlines())),flush=True)
