# Inventory outside active source, including preserved Gym/migration code, without imports.
coverage=[];additional_source=[];additional_errors=[]
skip={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.venv','node_modules','audit_reports','logs'}
for parent,dirs,files in os.walk(R):
 dirs[:]=[n for n in dirs if n not in skip]
 for name in files:
  p=Path(parent)/name
  if p.is_symlink():continue
  relative=str(p.relative_to(R))
  try:
   size=p.stat().st_size
   coverage.append(dict(path=relative,bytes=size,suffix=p.suffix))
   if relative not in baseline_hashes and p.suffix in ('.py','.sh','.toml','.yaml','.yml','.md','.txt','.html','.json'):
    data=p.read_bytes();item=dict(path=relative,sha256=hashlib.sha256(data).hexdigest(),bytes=size)
    if p.suffix=='.py':
     try:ast.parse(data,filename=relative);item['python_ast_ok']=True
     except Exception as e:item['python_ast_ok']=False;additional_errors.append(dict(path=relative,error=repr(e)))
    additional_source.append(item)
  except OSError as e:coverage.append(dict(path=relative,error=repr(e)))
write_json('repository_file_inventory.json',coverage)
write_json('additional_source_hashes.json',additional_source)
write_json('additional_python_syntax_issues.json',additional_errors)
cover_summary=dict(inventoried_files=len(coverage),bytes=sum(r.get('bytes',0) for r in coverage),additional_text_files_hashed=len(additional_source),additional_python_files_parsed=sum('python_ast_ok' in r for r in additional_source),additional_python_syntax_issues=len(additional_errors),directory_file_counts=dict(collections.Counter(r['path'].split('/')[0] for r in coverage)),excluded_directories=sorted(skip),scope_note='Logs/checkpoints/videos/events are inventoried and validated separately. Archived source is parsed, never imported or executed. These extra source hashes were taken late in the read-only audit; the original 1425 src/scripts hashes were taken at audit start.')
write_json('repository_coverage_summary.json',cover_summary)
print('REPOSITORY_COVERAGE',cover_summary,flush=True)
