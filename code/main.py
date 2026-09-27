"""
The Corporate Heist - Innov8 4.0 (Eightfold.ai x ARIES, IIT Delhi)
Reproduces submission.csv.   Run:  python code/main.py     (from the folder that contains train.csv, dev.csv, dev_winners.csv, test.csv)
CPU only, ~1-2 minutes.  All random seeds fixed (SEED=7).
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heist_clean as hc
import dedup as dd

SEED = 7
K = 500
t0 = time.time()
rng = np.random.RandomState(SEED)

# ----------------------------------------------------------------------------- load + clean
tr = pd.read_csv('train.csv'); dv = pd.read_csv('dev.csv'); dw = pd.read_csv('dev_winners.csv'); te = pd.read_csv('test.csv')
ct, cd, ce = hc.clean(tr), hc.clean(dv), hc.clean(te)
ct['y'] = tr['post_hire_score'].values
for c, raw in ((ct, tr), (cd, dv), (ce, te)):
    c['grp'] = dd.person_groups(raw)              # same person, different candidate_id -> same group
    c['n_missing'] = raw.isna().sum(axis=1).values # used only to pick the most complete duplicate

# ----------------------------------------------------------------------------- role -> core skills (learned from applicants)
SK = [c for c in ct.columns if c.startswith('sk_')]
prev = ct.groupby('role')[SK].mean()
CORE = {r: [s for s in SK if prev.loc[r, s] > 0.30] for r in prev.index}
def add_fit(d):
    S = d[SK].values; idx = {s: i for i, s in enumerate(SK)}
    d['core_hit'] = [S[i, [idx[s] for s in CORE[r]]].sum() if r in CORE else 0 for i, r in enumerate(d.role.values)]
    return d
for d in (ct, cd, ce): add_fit(d)

# ----------------------------------------------------------------------------- fabricated-profile rule (learned on train, see documentation)
# Fabricated profiles: assessment score >=85 together with two stacked positive recruiter notes.
# In train these rows average post_hire_score 9 (median 0) vs 68 for equally high-scoring genuine candidates.
# Every row with impossible ages / experience (worked longer than since graduation, degree before age 18 ...) is also inside this set.
def fake_flag(d):
    pattern = (d.tech >= 85) & (d.note_pos == 2)
    impossible = (d.exp > d.since_grad + 1) | ((d.age - d.exp) < 18) | ((d.age - d.since_grad) < 18)
    return (pattern | impossible).values
for d in (ct, cd, ce): d['fake'] = fake_flag(d)

# ----------------------------------------------------------------------------- features
GAP = lambda d: (d.since_grad - d.exp > 2).astype(int)        # 'CV with a gap' (old panel penalised it)
for d in (ct, cd, ce): d['gap'] = GAP(d)
TRUE = ['tech', 'tech_missing', 'apt', 'rating', 'kpi', 'awards', 'core_hit', 'n_skills', 'note_pos', 'note_neg', 'note_oncall',
        'note_vac', 'note_offers', 'exp', 'ctc', 'ectc', 'cur_lv', 'n_emp', 'overtime', 'ncerts', 'trainings', 'train_hours',
        'enrolled', 'role', 'ljc']
BIAS = ['inst', 'city', 'channel', 'degree', 'ctype', 'csize', 'gap']     # old committee's pet preferences (inflated in Archive)
CATS = ['enrolled', 'role', 'inst', 'city', 'channel', 'degree', 'ctype', 'csize']
ALL = TRUE + BIAS
cat_levels = {c: sorted(set(ct[c].astype(str)) | set(cd[c].astype(str)) | set(ce[c].astype(str))) for c in CATS}
def mat(d):
    X = d[ALL].copy()
    for c in CATS: X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
    return X

# ----------------------------------------------------------------------------- training set: genuine, unique people
tr_ok = ct[(~ct.fake)].sort_values('n_missing').drop_duplicates('grp')
Xtr, ytr = mat(tr_ok), tr_ok.y.values
PARAMS = dict(n_estimators=650, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, cat_smooth=30, min_data_per_group=60, cat_l2=20, verbose=-1, n_jobs=2)
models = [lgb.LGBMRegressor(random_state=SEED + s, **PARAMS).fit(Xtr, ytr) for s in range(3)]
predict = lambda X: np.mean([m.predict(X) for m in models], axis=0)
print(f'[{time.time()-t0:5.1f}s] models fitted on {len(tr_ok)} genuine unique archive rows')

# ----------------------------------------------------------------------------- validation on the Ledger (old standard)
def ndcg_at(scores, winners_rank, ids, k):
    order = np.argsort(-scores)[:k]; rel = {c: 1.0 / np.log2(r + 1) for c, r in zip(winners_rank.candidate_id, winners_rank['rank'])}
    dcg = sum(rel.get(ids[i], 0) / np.log2(j + 2) for j, i in enumerate(order))
    return dcg
dv_ok = cd[~cd.fake].sort_values('n_missing').drop_duplicates('grp')
win = set(dw.candidate_id)
# a winner that was collapsed into its twin still counts
grp_win = set(cd[cd.candidate_id.isin(win)].grp)
p_full = predict(mat(dv_ok))
top = np.argsort(-p_full)[:150]
hits = dv_ok.iloc[top].grp.isin(grp_win).sum()
print(f'[Ledger check] old-standard model (with old-panel preferences): {hits}/150 of Ledger top-5% recovered in its top 150')
# neutralised model on the same Ledger (should be a bit worse: the Ledger *was* biased)
def neutral_predict(d, n_draw=24):
    """Average model output over the empirical distribution of the old-panel preference features (partial dependence):
    every candidate is scored as if college tier, city, channel, degree, employer type/size and CV-gap were the same for everyone."""
    ref = tr_ok[BIAS].sample(n_draw, random_state=SEED).reset_index(drop=True)
    base = d[TRUE].reset_index(drop=True); out = np.zeros(len(d))
    for j in range(n_draw):
        dd_ = base.copy()
        for c in BIAS: dd_[c] = ref.loc[j, c]
        out += predict(mat(dd_))
    return out / n_draw
p_neu = neutral_predict(dv_ok)
top_n = np.argsort(-p_neu)[:150]
print(f'[Ledger check] preference-neutralised model: {dv_ok.iloc[top_n].grp.isin(grp_win).sum()}/150 recovered')
print(f'[{time.time()-t0:5.1f}s] validation done')

# ----------------------------------------------------------------------------- institute effects -> old-boys network
# ridge regression of (score) on the true drivers + institute dummies, genuine archive rows only
def inst_effects(d):
    X = pd.DataFrame(index=d.index)
    X['tech'] = d.tech.fillna(d.tech.median()); X['tech2'] = np.maximum(X.tech - 70, 0); X['tech_na'] = d.tech_missing
    X['apt'] = d.apt.fillna(d.apt.median()); X['rating'] = d.rating.fillna(3); X['kpi'] = d.kpi; X['awards'] = d.awards
    X['core'] = d.core_hit; X['np'] = (d.note_pos == 1).astype(int); X['nn'] = d.note_neg; X['exp'] = d.exp; X['exp2'] = d.exp ** 2 / 10
    X['lctc'] = np.log(d.ctc); X['ot'] = d.overtime; X['gap'] = d.gap
    for c in ['role', 'city', 'channel', 'degree', 'ctype', 'csize']:
        X = pd.concat([X, pd.get_dummies(d[c], prefix=c, drop_first=True, dtype=float)], axis=1)
    I = pd.get_dummies(d.inst, dtype=float)
    A = np.hstack([np.ones((len(d), 1)), X.values, I.values]); lam = np.zeros(A.shape[1]); lam[1 + X.shape[1]:] = 3.0
    b = np.linalg.solve(A.T @ A + np.diag(lam) + 1e-6 * np.eye(A.shape[1]), A.T @ d.y.values)
    res = d.y.values - A @ b
    eff = pd.Series(b[1 + X.shape[1]:], index=I.columns); n = d.inst.value_counts().reindex(eff.index)
    se = res.std() / np.sqrt(n)
    return pd.DataFrame({'eff': eff, 'n': n, 'z': eff / se})
ie = inst_effects(tr_ok)
import re
PRESTIGE = re.compile(r'^(indian institute of|national institute of technology|nit |birla institute of technology and science|delhi technological|netaji subhas|international institute of information)')
tier1 = ie[(ie.eff >= 4) & (ie.n >= 150)]                       # 'big-name' colleges (inflated by the old panel, ignored now)
base_t1 = np.average(tier1.eff, weights=tier1.n)
NETWORK = {}
for inst, r in ie.iterrows():
    if inst in tier1.index:
        z_ex = (r.eff - base_t1) / (tr_ok.y.std() / np.sqrt(r.n))
        if z_ex >= 2.5: NETWORK[inst] = 0.5 * (r.eff - base_t1)          # tier-1 college that also is a leadership alma mater
    elif r.n >= 30 and r.z >= 2.5 and r.eff >= 4 and not PRESTIGE.match(inst): NETWORK[inst] = 5.0   # unremarkable college with a tier-1-sized, significant leg-up
print('old-boys network colleges detected:', {k: round(v, 1) for k, v in NETWORK.items()})

# ----------------------------------------------------------------------------- the Vault
v = ce.copy()
v['base'] = neutral_predict(v)                                   # expected performance, old-panel preferences neutralised
v['network'] = v.inst.map(NETWORK).fillna(0.0)
# (a) public code contributions: sustained record (dozens of merged contributions) is fast-tracked
v['pcc_fill'] = v.pcc.fillna(0)
# the vault shows a separate population with 90-400 contributions (uniform tail, ~1.3% of profiles) whose other attributes are 'good, not great':
# these are the sustained contributors -> decisive bonus. Ordinary counts (natural tail, clipped at 60) get a moderate ramp: nothing below ~12, full credit from 30.
v['b_code'] = np.where(v.pcc_fill >= 90, 35.0, 14.0 * np.clip((v.pcc_fill - 12) / 18.0, 0, 1))
# (b) colleges Nightingale has never hired from: top assessment + genuine role fit
seen = set(ct.inst) | set(cd.inst)
v['new_college'] = ~v.inst.isin(seen) & (v.inst != 'unk')
# the new-college applicants carry 3-4x the usual share of assessment scores >=85 (an injected group of assessment toppers) -> bonus grows from tech 78 to 85 and needs role fit
v['b_new'] = np.where(v.new_college, 20.0 * np.clip((v.tech.fillna(0) - 78) / 7, 0, 1) * np.clip((v.core_hit - 2) / 2.5, 0, 1), 0.0)
v['score'] = v.base + v.network + v.b_code + v.b_new

# ----------------------------------------------------------------------------- exclusions
# title inflation: current title reached faster than anyone in the archive ever did (per seniority level)
lvl_min_sg = ct[~ct.fake].groupby('cur_lv').since_grad.min(); lvl_min_exp = ct[~ct.fake].groupby('cur_lv').exp.min()
def inflated(d):
    lim_sg = d.cur_lv.map(lvl_min_sg).where(d.cur_lv >= 4, 0); lim_ex = d.cur_lv.map(lvl_min_exp).where(d.cur_lv >= 4, 0)
    fast_path = d.cp_yrs_before_lead < 3.0                          # first leadership title after <3 years of career
    return ((d.since_grad < lim_sg) | (d.exp < lim_ex - 1) | fast_path).values
v['inflated'] = inflated(v)
v['late'] = (v.notice > 60).values                                   # cannot join within two months
# duplicates: keep exactly one profile per person (the most complete row; ties -> higher id string for determinism)
order = v.sort_values(['n_missing', 'candidate_id'], ascending=[True, True])
keep_idx = order.drop_duplicates('grp').index
v['dup_loser'] = ~v.index.isin(keep_idx)
v['excluded'] = v.fake | v.inflated | v.late | v.dup_loser
print('Vault exclusions -> fabricated:', int(v.fake.sum()), '| title inflation:', int(v.inflated.sum()), '| notice>60d:', int(v.late.sum()),
      '| duplicate twins:', int(v.dup_loser.sum()), '| any:', int(v.excluded.sum()))

# ----------------------------------------------------------------------------- final ranking
elig = v[~v.excluded].sort_values(['score', 'candidate_id'], ascending=[False, True])
sub = elig.head(K)[['candidate_id']].copy()
sub.insert(0, 'rank', np.arange(1, K + 1))
sub.to_csv('submission.csv', index=False)
print(f'[{time.time()-t0:5.1f}s] wrote submission.csv  ({len(sub)} rows).  code-fast-tracked in list: {(elig.head(K).b_code>=30).sum()} (+{((elig.head(K).b_code>0)&(elig.head(K).b_code<30)).sum()} moderate), '
      f'new-college toppers in list: {(elig.head(K).b_new>3).sum()}, network-college in list: {(elig.head(K).network>0).sum()}')
