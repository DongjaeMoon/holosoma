"""Reproduce the audit in a NEW output directory, without simulation or training.

Run inside the existing hssim container environment at the repository root:
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python \
    audit_reports/20261102_overnight/reproduce_cpu_audit.py \
    --output audit_reports/NEW_UNIQUE_DIRECTORY

Reads trusted local torch checkpoints with weights_only=False. No optimizer steps.
ffmpeg and ffprobe must be available. Existing output directories are rejected.
"""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
import argparse,json,hashlib,subprocess,datetime,platform,ast,re,collections,math,time,csv,dataclasses,gc
from pathlib import Path
import yaml,torch,numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
phase_root=Path(__file__).resolve().parent
R=phase_root.parent.parent
O=args.output.resolve()
if not O.is_relative_to(R/'audit_reports'):
 raise SystemExit('Output must be a NEW directory under this repository/audit_reports.')
O.mkdir(parents=True,exist_ok=False)
os.chdir(R)
Q=R/'logs/bilateral_queue'
assert not torch.cuda.is_available()
torch.set_num_threads(1)
def write_json(name,data):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(p.name+'.tmp')
 temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 temp.replace(p)
def git(*arguments):
 result=subprocess.run(['git','-c','safe.directory='+str(R),'-C',str(R),*arguments],capture_output=True,text=True,check=True)
 return result.stdout
names=set(git('ls-files','--','src','scripts').splitlines())|set(git('ls-files','--others','--exclude-standard','--','src','scripts').splitlines())
baseline_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in sorted(names) if (R/n).is_file()}
write_json('source_hashes_before.json',baseline_hashes)
write_json('session.json',dict(root=str(R),executable=sys.executable,python=sys.version,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],git_head=git('rev-parse','HEAD').strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode='CPU reproduction; no training, simulator startup, optimizer steps or existing source edits'))
for source in sorted(phase_root.glob('[0-9][0-9]_*.py')):
 print('AUDIT_PHASE',source.name,flush=True)
 exec(compile(source.read_text(),str(source),'exec'),globals())
changes=[n for n,digest in baseline_hashes.items() if not (R/n).is_file() or hashlib.sha256((R/n).read_bytes()).hexdigest()!=digest]
write_json('reproduction_source_integrity.json',dict(files_checked=len(baseline_hashes),changed_files=changes))
assert not changes,changes
print('COMPLETE',O,flush=True)
