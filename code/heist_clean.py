"""Cleaning / parsing helpers for The Corporate Heist (Nightingale Systems)."""
import re
import numpy as np
import pandas as pd

REF_YEAR = 2026

# ---------------------------------------------------------------- numbers
def _f(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_experience(x):
    """total_experience -> years (float). 'Fresher'->0, '<1 year'->0.5, '20+ years'->20.5, '22 months'->1.83"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('fresher', 'nil', 'none'):
        return 0.0
    m = re.match(r'^<\s*(\d+\.?\d*)', s)
    if m:
        return float(m.group(1)) / 2
    m = re.match(r'^>\s*(\d+\.?\d*)', s)
    if m:
        return float(m.group(1)) + 1
    m = re.match(r'^(\d+\.?\d*)\s*\+', s)
    if m:
        return float(m.group(1)) + 0.5
    m = re.match(r'^(\d+\.?\d*)\s*(months?|mos?|mo)\b', s)
    if m:
        return float(m.group(1)) / 12
    m = re.match(r'^(\d+\.?\d*)', s)
    if m:
        return float(m.group(1))
    return np.nan

def parse_tech(x):
    """technical_assessment -> 0..100 or NaN (not taken)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('absent', 'not taken', 'na', 'n/a', '-', ''):
        return np.nan
    m = re.match(r'^(\d+\.?\d*)\s*/\s*(\d+\.?\d*)$', s)
    if m:
        return 100 * float(m.group(1)) / float(m.group(2))
    s = s.replace('%', '').strip()
    v = _f(s)
    if np.isnan(v):
        return np.nan
    if v <= 1.0 and '.' in s:
        return v * 100
    return v

def parse_apt(x):
    """aptitude_score -> 0..10"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.match(r'^(\d+\.?\d*)\s*/\s*10$', s)
    if m:
        return float(m.group(1))
    if s.endswith('%'):
        return float(s[:-1]) / 10
    v = _f(s)
    return v

RATING_MAP = {'outstanding': 5, 'exceeds expectations': 4, 'meets expectations': 3,
              'needs improvement': 2, 'unsatisfactory': 1}
def parse_rating(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in RATING_MAP:
        return float(RATING_MAP[s])
    if s.startswith('new joiner'):
        return np.nan
    m = re.match(r'^(\d+\.?\d*)\s*/\s*5$', s)
    if m:
        return float(m.group(1))
    return _f(s)

def parse_bool(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('y', 'yes', '1', 'true', '1.0'):
        return 1.0
    if s in ('n', 'no', '0', 'false', '0.0'):
        return 0.0
    return np.nan

def parse_awards(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().lower()
    if s in ('0', '-', '', 'none', 'no'):
        return 0.0
    return 1.0

def parse_notice(x):
    """notice_period -> days; also flag 'serving notice'."""
    if pd.isna(x):
        return np.nan, 0
    s = str(x).strip().lower()
    serving = int('serving' in s)
    if any(k in s for k in ('immediate', 'available now')):
        return 0.0, serving
    m = re.search(r'(\d+\.?\d*)\s*(month|day)', s)
    if m:
        v = float(m.group(1))
        return (v * 30 if m.group(2) == 'month' else v), serving
    return np.nan, serving

USD_INR = 83.0
def parse_ctc(x):
    """-> INR per annum"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    sl = s.lower()
    if s.startswith('$'):
        return float(re.sub(r'[^\d.]', '', s)) * USD_INR
    if s.startswith('₹'):
        return float(re.sub(r'[^\d.]', '', s))
    m = re.match(r'^(?:inr\s*)?(\d+\.?\d*)\s*(cr|crore|lakh|lakhs|lpa|l)?$', sl)
    if m:
        v = float(m.group(1)); u = m.group(2)
        if u in ('cr', 'crore'):
            return v * 1e7
        if u in ('lakh', 'lakhs', 'lpa', 'l'):
            return v * 1e5
        return v
    return np.nan

def parse_contrib(x):
    """public_code_contributions -> (count or NaN, tracked flag)"""
    if pd.isna(x):
        return np.nan, 0
    s = str(x).strip().lower()
    if s == 'not tracked':
        return np.nan, 0
    m = re.search(r'(\d+\.?\d*)', s)
    if m:
        v = float(m.group(1))
        if s.endswith('+'):
            v += 0.5
        return v, 1
    return np.nan, 0

# ---------------------------------------------------------------- city
CITY_MAP = {'madras': 'chennai', 'gurgaon': 'gurugram', 'calcutta': 'kolkata', 'bombay': 'mumbai',
            'new delhi': 'delhi ncr', 'delhi': 'delhi ncr', 'noida': 'delhi ncr', 'gurugram': 'delhi ncr'}
def canon_city(x):
    if pd.isna(x):
        return 'unk'
    s = str(x).strip().lower()
    return CITY_MAP.get(s, s)

# ---------------------------------------------------------------- degree
def canon_degree(x):
    s = str(x).strip().lower().replace('.', '')
    if s in ('btech', 'bachelor of technology', 'be', 'bachelor of engineering'):
        return 'btech'
    if s in ('mtech', 'master of technology'):
        return 'mtech'
    if s in ('bsc', 'bca'):
        return s
    if s in ('msc', 'mca', 'mba', 'ms', 'phd', 'diploma', 'polytechnic diploma', '12th'):
        return 'diploma' if s == 'polytechnic diploma' else s
    return s

# ---------------------------------------------------------------- institute
def _norm(s):
    s = str(s).strip().lower()
    s = s.replace('&', ' and ').replace('–', ' ').replace('-', ' ')
    s = re.sub(r'\(.*?\)', lambda m: ' ' + m.group(0)[1:-1] + ' ' if 'iit' in m.group(0).lower() else ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

IIT_NAMES = ['delhi', 'madras', 'kanpur', 'bombay', 'kharagpur', 'roorkee', 'guwahati', 'hyderabad',
             'jodhpur', 'bhubaneswar', 'patna', 'indore', 'gandhinagar', 'ropar', 'bhu varanasi']
ALIASES = {
    'iit d': 'indian institute of technology delhi', 'iitb': 'indian institute of technology bombay',
    'i i t delhi': 'indian institute of technology delhi', 'iit delhi': 'indian institute of technology delhi',
    'iit bombay': 'indian institute of technology bombay', 'iit bhubaneswar': 'indian institute of technology bhubaneswar',
    'indian institute of technology roorkee': 'indian institute of technology roorkee',
    'university of roorkee': 'indian institute of technology roorkee',
    'indian institute of technology ropar iit ropar': 'indian institute of technology ropar',
    'indian institute of technology bhu varanasi': 'indian institute of technology bhu varanasi',
    'nit trichy': 'national institute of technology tiruchirappalli', 'nit tiruchirappalli': 'national institute of technology tiruchirappalli',
    'national institute of technology trichy': 'national institute of technology tiruchirappalli',
    'national institute of technology surathkal': 'national institute of technology karnataka surathkal',
    'bits pilani': 'birla institute of technology and science pilani',
    'bits pilani ': 'birla institute of technology and science pilani',
    'birla institute of technology and science': 'birla institute of technology and science pilani',
    'dtu': 'delhi technological university', 'delhi college of engineering dce': 'delhi technological university',
    'delhi technological university formerly dce': 'delhi technological university',
    'nsit delhi': 'netaji subhas university of technology', 'nsut': 'netaji subhas university of technology',
    'vit university': 'vellore institute of technology', 'vit vellore': 'vellore institute of technology',
    'manipal university': 'manipal institute of technology',
    'srm university': 'srm institute of science and technology',
    'saveetha university': 'saveetha school of engineering',
    'indian school of mines ism': 'indian school of mines',
    'bengal engineering college': 'indian institute of engineering science and technology shibpur',
}
def canon_institute(x):
    if pd.isna(x):
        return 'unk'
    n = _norm(x)
    n = n.replace('i i t', 'iit')
    return ALIASES.get(n, n)

# ---------------------------------------------------------------- skills
SKILL_GROUPS = {
 'ab_test': ['a/b testing', 'ab testing', 'split testing', 'experimentation'],
 'excel': ['advanced excel', 'excel', 'ms excel'],
 'airflow': ['airflow', 'apache airflow', 'workflow orchestration (airflow)'],
 'aws': ['amazon web services', 'aws', 'aws (ec2/s3)'],
 'android': ['android', 'android sdk'],
 'angular': ['angular', 'angularjs'],
 'kafka': ['apache kafka', 'kafka', 'event streaming (kafka)'],
 'spark': ['apache spark', 'spark', 'pyspark'],
 'api': ['api design', 'rest', 'rest api', 'rest apis', 'restful services'],
 'azure': ['azure', 'ms azure'],
 'bash': ['bash/linux', 'linux', 'unix/linux'],
 'cpp': ['c plus plus', 'c++', 'cpp'],
 'cicd': ['ci-cd', 'ci/cd', 'cicd', 'jenkins', 'github actions'],
 'cv': ['computer vision', 'cv (images)'],
 'docker': ['containers (docker)', 'docker', 'dockerization'],
 'css': ['css', 'css3', 'scss'],
 'cypress': ['cypress', 'cypress.io'],
 'flutter': ['dart/flutter', 'flutter'],
 'django': ['django'],
 'k8s': ['eks/gke', 'k8s', 'kubernetes'],
 'js': ['es6', 'java script', 'javascript', 'js'],
 'express': ['express/node', 'node', 'node js', 'node.js', 'nodejs'],
 'fastapi': ['fastapi'],
 'flask': ['flask'],
 'gcp': ['gcp', 'google cloud', 'google cloud platform'],
 'git': ['git', 'github', 'version control'],
 'go': ['go', 'golang'],
 'graphql': ['graph ql', 'graphql'],
 'hadoop': ['hadoop', 'hdfs'],
 'html': ['html', 'html5'],
 'stats': ['hypothesis testing', 'statistical modelling', 'statistics', 'stats'],
 'terraform': ['hcl', 'iac (terraform)', 'terraform'],
 'java': ['java', 'java 8', 'core java'],
 'kotlin': ['kotlin'],
 'mlops': ['ml ops', 'mlflow', 'mlops', 'model deployment'],
 'microservices': ['micro-services', 'microservice architecture', 'microservices', 'msa'],
 'mongo': ['mongo', 'mongodb'],
 'mysql': ['mysql'],
 'nlp': ['natural language processing', 'nlp'],
 'numpy': ['numpy'], 'pandas': ['pandas'],
 'postgres': ['postgre sql', 'postgres', 'postgresql', 'psql'],
 'bi': ['power bi', 'powerbi', 'tableau'],
 'python': ['python', 'python 3', 'python3'],
 'pytorch': ['pytorch', 'torch'],
 'react': ['react', 'react js', 'react.js', 'reactjs'],
 'rn': ['react native', 'react-native', 'rn'],
 'redis': ['redis'],
 'sklearn': ['scikit learn', 'scikit-learn', 'sklearn'],
 'selenium': ['selenium', 'selenium webdriver'],
 'spring': ['spring', 'spring boot', 'springboot'],
 'sql': ['sql', 'structured query language', 't-sql'],
 'swift': ['swift', 'swiftui'],
 'tf': ['tensorflow', 'tf/keras'],
 'ts': ['ts', 'type script', 'typescript'],
 'vue': ['vue', 'vue.js', 'vuejs'],
}
SKILL_LOOKUP = {a: k for k, v in SKILL_GROUPS.items() for a in v}
SKILL_KEYS = list(SKILL_GROUPS)

def parse_skills(x):
    if pd.isna(x):
        return set()
    out = set()
    for t in re.split(r'[;,|]', str(x)):
        t = t.strip().lower()
        if t in SKILL_LOOKUP:
            out.add(SKILL_LOOKUP[t])
    return out

# ---------------------------------------------------------------- career path
SEP = re.compile(r'\s*(?:->|→|>|\|)\s*')
DUR = re.compile(r'^(.*?)\s*[\(\[]\s*(\d+\.?\d*)\s*(y|yrs|mo)\s*[\)\]]\s*$')
SENIORITY_ORDER = [
    ('intern', 0), ('trainee', 0), ('apprentice', 0),
    ('associate', 1), ('junior', 1), ('jr', 1),
    ('head of', 8), ('vp', 9), ('director', 7), ('engineering manager', 6), ('manager', 6),
    ('quality manager', 6), ('principal', 5), ('staff', 5), ('tech lead', 4), ('lead', 4),
    ('senior', 3), ('sr.', 3), ('sr ', 3),
]
def title_level(t):
    """0 entry, 2 mid, 3 senior, 4 lead, 5 staff/principal, 6 manager, 7 director, 8 head, 9 VP"""
    s = str(t).strip().lower()
    for k, v in SENIORITY_ORDER:
        if k in s:
            return v
    if re.search(r'\b(iii|3)\b', s):
        return 3
    return 2

def parse_career(x):
    """-> list of (title, years)"""
    if pd.isna(x):
        return []
    res = []
    for part in SEP.split(str(x)):
        m = DUR.match(part.strip())
        if m:
            v = float(m.group(2))
            if m.group(3) == 'mo':
                v /= 12
            res.append((m.group(1).strip(), v))
    return res

def career_features(x, cur_title):
    steps = parse_career(x)
    n = len(steps)
    tot = sum(d for _, d in steps)
    lv = [title_level(t) for t, _ in steps]
    maxlv = max(lv) if lv else np.nan
    # years spent before first reaching a leadership (>=6) title
    yrs_to_lead = np.nan
    acc = 0.0
    for (t, d), l in zip(steps, lv):
        if l >= 6:
            yrs_to_lead = acc
            break
        acc += d
    return dict(cp_n=n, cp_total=tot if n else np.nan, cp_maxlv=maxlv,
                cp_yrs_before_lead=yrs_to_lead, cp_lastlv=lv[-1] if lv else np.nan,
                cur_lv=title_level(cur_title) if not pd.isna(cur_title) else np.nan)

# ---------------------------------------------------------------- notes
POS_NOTES = ['strong ownership', 'led the migration', 'consistently delivered', 'shipped a feature',
             'mentored two', 'excellent system-design']
NEG_NOTES = ['lukewarm', 'struggled with the debugging', 'missed two sprint', 'needed frequent guidance',
             'communication in the panel']
ONCALL_NOTES = ['on-call', 'pager', 'runbooks', '24x7']
def note_features(x):
    s = '' if pd.isna(x) else str(x).lower()
    return dict(note_pos=sum(k in s for k in POS_NOTES), note_neg=sum(k in s for k in NEG_NOTES),
                note_oncall=int(any(k in s for k in ONCALL_NOTES)),
                note_vac=int('vacation' in s), note_offers=int('other offers' in s))

def canon_role(x):
    return str(x).strip()

# ---------------------------------------------------------------- master
def clean(df):
    o = pd.DataFrame(index=df.index)
    o['candidate_id'] = df['candidate_id']
    o['age'] = pd.to_numeric(df['age'], errors='coerce')
    o['grad_year'] = pd.to_numeric(df['graduation_year'], errors='coerce')
    o['since_grad'] = REF_YEAR - o['grad_year']
    o['inst'] = df['institute'].map(canon_institute)
    o['degree'] = df['degree'].map(canon_degree)
    o['major'] = df['major'].fillna('NA')
    o['city'] = df['current_city'].map(canon_city)
    o['role'] = df['applied_role'].map(canon_role)
    o['title'] = df['current_title']
    o['exp'] = df['total_experience'].map(parse_experience)
    o['n_emp'] = pd.to_numeric(df['num_employers'], errors='coerce')
    o['ctype'] = df['company_type'].fillna('NA')
    o['csize'] = df['company_size'].fillna('NA').replace({'10/49': '10-49'})
    o['tech'] = df['technical_assessment'].map(parse_tech)
    o['tech_missing'] = o['tech'].isna().astype(int)
    o['apt'] = df['aptitude_score'].map(parse_apt)
    o['rating'] = df['last_rating'].map(parse_rating)
    o['kpi'] = df['kpi_met'].map(parse_bool)
    o['awards'] = df['awards'].map(parse_awards)
    o['trainings'] = pd.to_numeric(df['trainings_last_year'], errors='coerce')
    o['train_hours'] = pd.to_numeric(df['training_hours'], errors='coerce')
    o['channel'] = df['recruitment_channel'].replace({'Employee Referral': 'Referral'}).fillna('NA')
    lj = df['last_job_change'].astype(str).str.strip().str.lower()
    o['ljc'] = lj.map({'never': -1, '>4': 5}).fillna(pd.to_numeric(lj, errors='coerce'))
    o['enrolled'] = df['currently_enrolled'].fillna('NA')
    o['overtime'] = (df['overtime_history'].astype(str).str.lower() == 'yes').astype(int)
    o['ctc'] = df['current_ctc'].map(parse_ctc)
    o['ectc'] = df['expected_ctc'].map(parse_ctc)
    nd = df['notice_period'].map(parse_notice)
    o['notice'] = [a for a, b in nd]
    o['serving'] = [b for a, b in nd]
    o['ncerts'] = df['certifications'].fillna('').map(lambda s: len([t for t in s.split(';') if t.strip()]))
    sk = df['skills'].map(parse_skills)
    o['n_skills'] = sk.map(len)
    for k in SKILL_KEYS:
        o['sk_' + k] = sk.map(lambda s, k=k: int(k in s))
    cf = pd.DataFrame([career_features(a, b) for a, b in zip(df['career_path'], df['current_title'])], index=df.index)
    o = pd.concat([o, cf], axis=1)
    nf = pd.DataFrame([note_features(x) for x in df['recruiter_note']], index=df.index)
    o = pd.concat([o, nf], axis=1)
    if 'public_code_contributions' in df.columns:
        pc = df['public_code_contributions'].map(parse_contrib)
        o['pcc'] = [a for a, b in pc]
        o['pcc_tracked'] = [b for a, b in pc]
    return o
