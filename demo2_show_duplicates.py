"""Show duplicate groups detected by the production dedup.py."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import dedup as dd
raw = pd.read_csv(ROOT / 'test.csv')
raw = raw.copy()
raw['grp'] = dd.person_groups(raw)
sizes = raw.groupby('grp').size()
dup_groups = sizes[sizes > 1].index
print('=' * 100)
print(f"DEMO 2 — {len(dup_groups)} duplicate-person groups; {int(sizes[sizes > 1].sum())} rows in those groups")
print('=' * 100)
cols = ['candidate_id','full_name','email','phone','institute','total_experience']
shown = 0
for g in dup_groups:
    pair = raw[raw.grp == g]
    if len(pair) != 2:
        continue
    a, b = pair.iloc[0], pair.iloc[1]
    reasons = []
    pa, pb = dd._phone_key(a['phone']), dd._phone_key(b['phone'])
    ea, eb = dd._email_key(a['email']), dd._email_key(b['email'])
    if pa and pa == pb: reasons.append('normalized phone')
    if ea and ea == eb: reasons.append('normalized email')
    print(f"\nMatch #{shown+1} (matched on: {', '.join(reasons) or 'shared identity key'})")
    print(pair[cols].to_string(index=False))
    shown += 1
    if shown >= 5: break
if shown == 0:
    print('No exact two-row duplicate groups available to display.')
print('\nThe production pipeline retains one most-complete row per detected group.')
