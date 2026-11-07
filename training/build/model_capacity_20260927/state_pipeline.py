"""Sequential, guarded refit/confirmation/export after all declared state trials."""
# Training record: the driver that produced the shipped CAP heads. The reproduction does not
# run it. Each step runs a program of this folder with the same Python, from the development
# repository root (`common.ROOT`), and appends its output to `<tag>.log` here; a failing step
# stops the pipeline.
# Order: wait for the last declared trial (`state/C6_long/trial.json`) ->
#   `state_eval.py --freeze` (when no selection exists) -> `state_train.py selected --refit`
#   (January refit) -> `state_confirm.py` (February check of the January refit) -> February
#   gate -> `state_train.py prepare_final` -> `state_train.py selected --final` (the shipped
#   `state/C6_long_final/`) -> `state_march.py` (March gate) -> build, verify and validate
#   the submission candidate. The final fit runs before the March gate; the March gate
#   decides only whether the candidate is built.
# Usage: `python state_pipeline.py`.
import sys,time,json,subprocess
from pathlib import Path
from common import HERE,ROOT,dump

def step(tag,args):
    """Runs one program as a logged subprocess and stops the pipeline when it fails.

    Args:
        tag: Step name; output is appended to `<tag>.log` in this folder.
        args: Program path and its arguments.
    """
    print('START',tag,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    with (HERE/(tag+'.log')).open('a')as f:
        r=subprocess.run([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    print('END',tag,'exit',r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)

def run():
    """Runs the selection, refit, February gate, final fit, March gate and export in order."""
    while not(HERE/'state/C6_long/trial.json').exists():time.sleep(10)
    if not(HERE/'state/selection.json').exists():step('state_freeze',[HERE/'state_eval.py','--freeze'])
    step('state_refit',[HERE/'state_train.py','selected','--refit'])
    if not(HERE/'state/confirmation/report.json').exists():step('state_feb',[HERE/'state_confirm.py'])
    feb=json.loads((HERE/'state/confirmation/report.json').read_text())
    # February gate: positive total_B change against the complete DK50 state in both the
    # strict and the serve background, and a positive lower 95% bound of the strict
    # date bootstrap.
    go=all(feb['results'][mode]['deltas']['candidate']['total_B']>0 for mode in ['strict','serve'])and feb['uncertainty']['candidate']['ci95']['total_B'][0]>0
    dump(HERE/'state/feb_gate.json',dict(pass_to_production=go,rule='positive complete-DK50 delta in strict and serve backgrounds and positive strict date-bootstrap CI lower bound'))
    if not go:print('STOP: February gate failed; model investigation required',flush=True);return
    step('state_prepare_final',[HERE/'state_train.py','prepare_final'])
    step('state_final',[HERE/'state_train.py','selected','--final'])
    if not(HERE/'state/march/results.json').exists():step('state_march',[HERE/'state_march.py'])
    # March gate on released March observations: `promote` must be true.
    march=json.loads((HERE/'state/march/results.json').read_text())
    if not march['promote']:print('STOP: March confirmation gate failed',flush=True);return
    dest=ROOT/'build/deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv'
    if not dest.exists():step('state_build',[HERE/'build_state_candidate.py'])
    step('state_verify',[HERE/'verify_state_candidate.py'])
    step('state_validate',[ROOT/'scripts/61_validate_submission.py',dest])
    dump(HERE/'state/pipeline_complete.json',dict(candidate=str(dest),status='fully_verified_local_candidate_not_submitted',completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))

if __name__=='__main__':run()
