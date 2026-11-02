# Log and provenance scan; files are read only.
severity_patterns={
 'traceback':re.compile(r'Traceback \(most recent call last\)'),
 'cuda_oom':re.compile(r'out of memory|CUDA.*(?:error|failed)',re.I),
 'nonfinite':re.compile(r'(?<![A-Za-z_])(?:nan|NaN)(?![A-Za-z_])'),
 'error':re.compile(r'\[Error\]|RuntimeError:|AssertionError:|ValueError:|Exception:',re.I),
}
flag_pattern=re.compile(r'use_symmetry|symmetry_compatible_io|ThreeDirectionCommand|LocomotionCommand|DomiGait|num_learning_iterations|command_audit|Loaded.*checkpoint|Loading.*checkpoint')
log_reports=[]
for meta in json.loads((O/'log_inventory.json').read_text()):
 p=R/meta['path'];counts=collections.Counter();examples=collections.defaultdict(list);flags=[]
 with p.open(errors='replace') as stream:
  for i,line in enumerate(stream,1):
   for tag,pattern in severity_patterns.items():
    if pattern.search(line):
     counts[tag]+=1
     if len(examples[tag])<12:examples[tag].append(dict(line=i,text=line.rstrip()[:900]))
   if flag_pattern.search(line) and len(flags)<45:flags.append(dict(line=i,text=line.rstrip()[:900]))
 log_reports.append(dict(**meta,severity_counts=dict(counts),examples=dict(examples),config_mentions=flags))
write_json('log_scan.json',log_reports)
canonical_log_alerts=[r for r in log_reports if any(k in r['path'] for k in ('corrected_v2_20261029_0513','vanilla_mlp_30k_native_io_gpu2')) and r['severity_counts']]
write_json('canonical_log_alerts.json',canonical_log_alerts)
native=Q/'vanilla_mlp_30k_native_io_gpu2'
actual=next((native/'training').glob('*/*/command_audit_metadata.json')).parent
meta=json.loads((actual/'command_audit_metadata.json').read_text())
source_checks=[]
for original,digest in meta['source_sha256'].items():
 name=Path(original).name
 snapshot=actual/'command_audit_sources'/name
 root_path=R/Path(original).relative_to('/workspace/holosoma')
 source_checks.append(dict(recorded_path=original,expected_sha256=digest,
   snapshot_exists=snapshot.exists(),
   snapshot_matches=snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest,
   current_matches=root_path.exists() and hashlib.sha256(root_path.read_bytes()).hexdigest()==digest))
write_json('native_source_provenance.json',source_checks)
print('LOG_SCAN',len(log_reports),'canonical files with alerts',len(canonical_log_alerts))
print('ALERTS',[(r['path'],r['severity_counts']) for r in canonical_log_alerts])
print('SOURCE_SNAPSHOTS',source_checks)
print('LEGACY_FILES',[(str(p.relative_to(R)),p.stat().st_size) for p in Q.rglob('*terminal*') if p.is_file()])
