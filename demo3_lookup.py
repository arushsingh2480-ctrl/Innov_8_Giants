"""Candidate lookup. First run: python demo3_lookup.py --load"""
from pathlib import Path
import sys, pickle, warnings
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / 'pipeline_state.pkl'
def build_state():
    import runpy
    print('Running the actual production pipeline once; this may take a minute...')
    old = Path.cwd()
    try:
        import os
        os.chdir(ROOT)
        ns = runpy.run_path(str(ROOT / 'code' / 'main.py'), run_name='__main__')
    finally:
        os.chdir(old)
    state = {
        'v': ns['v'][['candidate_id','inst','role','tech','apt','rating','kpi','core_hit','exp','notice','pcc','new_college','base','network','b_code','b_new','score','fake','inflated','late','dup_loser','excluded']].copy(),
        'sub': ns['sub'].copy(),
    }
    with STATE_FILE.open('wb') as f: pickle.dump(state, f)
    print(f'Cache created: {STATE_FILE.name}')
def lookup(cid):
    if not STATE_FILE.exists():
        print('First run: python demo3_lookup.py --load'); return
    with STATE_FILE.open('rb') as f: state = pickle.load(f)
    v, sub = state['v'], state['sub']
    row = v[v.candidate_id.astype(str) == str(cid)]
    if row.empty:
        print(f'{cid!r} is not present in test.csv.'); return
    r = row.iloc[0]
    print('=' * 90)
    print(f"CANDIDATE: {cid} | Role: {r.role} | Institute: {r.inst}")
    print('=' * 90)
    for label, col in [('technical assessment','tech'),('aptitude','apt'),('rating','rating'),('KPI met','kpi'),('role-fit core_hit','core_hit'),('experience years','exp'),('notice days','notice'),('public code contributions','pcc'),('new-college applicant','new_college')]:
        print(f' {label:28}: {r[col]}')
    if bool(r.excluded):
        reasons=[]
        if r.fake: reasons.append('FABRICATED-profile rule')
        if r.inflated: reasons.append('TITLE-INFLATION rule')
        if r.late: reasons.append('NOTICE > 60 DAYS')
        if r.dup_loser: reasons.append('DUPLICATE: another row in group retained')
        print('RESULT: EXCLUDED — ' + '; '.join(reasons))
    elif (sub.candidate_id.astype(str) == str(cid)).any():
        rank = int(sub.loc[sub.candidate_id.astype(str) == str(cid), 'rank'].iloc[0])
        print(f'RESULT: IN TOP 500 — rank {rank}/500')
        print(f"Score: base {r.base:.2f} + network {r.network:.2f} + code {r.b_code:.2f} + new-college {r.b_new:.2f} = {r.score:.2f}")
    else:
        cutoff = float(v[v.candidate_id.isin(sub.candidate_id)].score.min())
        print(f'RESULT: NOT IN TOP 500 — score {r.score:.2f}; rank-500 cutoff score {cutoff:.2f}')
        print(f"Score breakdown: base {r.base:.2f} + network {r.network:.2f} + code {r.b_code:.2f} + new-college {r.b_new:.2f} = {r.score:.2f}")
    print('=' * 90)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == '--load':
        build_state()
    else:
        lookup(sys.argv[1])
