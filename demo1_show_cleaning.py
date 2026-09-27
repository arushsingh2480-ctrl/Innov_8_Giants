"""Show raw values beside the actual cleaned features used by the model."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import heist_clean as hc
raw = pd.read_csv(ROOT / 'test.csv')
cleaned = hc.clean(raw)
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 36)
mask = (
    raw['total_experience'].astype(str).str.contains(r'Fresher|months|\+', case=False, na=False)
    | raw['technical_assessment'].astype(str).str.contains(r'/|absent|not taken', case=False, na=False)
    | raw['current_ctc'].astype(str).str.contains(r'\$|Cr|L$', case=False, na=False)
)
sample = raw.loc[mask]
if len(sample) < 6:
    sample = raw
sample = sample.sample(min(6, len(sample)), random_state=1)
print('=' * 110)
print('DEMO 1 — RAW DATA -> CLEANED DATA')
print('=' * 110)
cols = [('total_experience','exp','years'),('technical_assessment','tech','0–100'),('current_ctc','ctc','INR/year'),('institute','inst','canonical name'),('notice_period','notice','days')]
for i, row in sample.iterrows():
    print(f"\nCandidate: {row['candidate_id']}")
    for raw_col, clean_col, unit in cols:
        print(f"  {raw_col:23} RAW={str(row[raw_col])[:32]:34} -> CLEANED={cleaned.loc[i, clean_col]} ({unit})")
print('\nMissing/unparseable values remain NaN; they are not guessed.')
