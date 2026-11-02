# Additional read-only evidence checks. Run after phases 00 through 08.
import io, contextlib, runpy, traceback
os.environ['PYTHONDONTWRITEBYTECODE']='1'
evidence=Q/'corrected_v2_20261029_0513/evidence/MLP_training_terminal.txt'
em=json.loads((evidence.parent/'MLP_training_evidence.json').read_text())
source=R/em['source_terminal_log']
raw=source.read_bytes()
clean=re.sub(rb'\x1b\[[0-9;]*m',b'',raw).decode().replace('\r\n','\n').replace('\r','\n').encode()
terminal_check=dict(source=str(source.relative_to(R)),export=str(evidence.relative_to(R)),source_sha256=hashlib.sha256(raw).hexdigest(),metadata_sha256=em['source_log_sha256'],source_hash_matches_metadata=hashlib.sha256(raw).hexdigest()==em['source_log_sha256'],ansi_removed_and_newlines_normalized_exact_match=clean==evidence.read_bytes(),exposure_log_present=em['sampled_command_history_logged'])
assert terminal_check['source_hash_matches_metadata'] and terminal_check['ansi_removed_and_newlines_normalized_exact_match']
write_json('terminal_evidence_copy.json',terminal_check)
checks=[]
old_path=list(sys.path); old_argv=list(sys.argv)
sys.path.insert(0,str(R/'scripts'))
try:
 for name in ('check_vanilla_mlp_30k.py','check_corrected_io.py','check_holdout_metrics.py'):
  out=io.StringIO(); sys.argv=[str(R/'scripts'/name)]
  try:
   with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
    runpy.run_path(str(R/'scripts'/name),run_name='__main__')
   checks.append(dict(script='scripts/'+name,passed=True,output=out.getvalue()))
  except Exception:
   checks.append(dict(script='scripts/'+name,passed=False,output=out.getvalue(),traceback=traceback.format_exc()))
finally:
 sys.path[:]=old_path; sys.argv=old_argv
write_json('cpu_regression_tests.json',checks)
print('TERMINAL_EXPORT_MATCH',terminal_check['ansi_removed_and_newlines_normalized_exact_match'])
print('CPU_TESTS',[(x['script'],x['passed']) for x in checks],flush=True)
