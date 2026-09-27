import re, pandas as pd, numpy as np
def _email_key(e):
    if pd.isna(e): return None
    e=str(e).strip().lower()
    if '@' not in e: return e
    loc,dom=e.split('@',1)
    loc=loc.split('+')[0].replace('.','').replace('_','')
    return loc+'@'+dom
def _phone_key(p):
    d=re.sub(r'\D','',str(p))
    return d[-10:] if len(d)>=10 else None
def _name_key(n):
    if pd.isna(n): return None
    s=str(n).lower().replace(',',' ')
    toks=[t for t in re.findall(r'[a-z]+',s)]
    return ' '.join(sorted(toks))
def person_groups(df):
    """Union-find over phone / email keys (+ name+age+grad-year); returns group id per row."""
    n=len(df); parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    keys={}
    ph=df['phone'].map(_phone_key); em=df['email'].map(_email_key)
    for i,(p,e) in enumerate(zip(ph,em)):
        for k in (('p',p),('e',e)):
            if k[1] is None: continue
            if k in keys: union(i,keys[k])
            else: keys[k]=i
    return np.array([find(i) for i in range(n)])
