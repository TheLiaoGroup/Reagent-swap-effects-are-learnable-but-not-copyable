"""
hte_common.py - single source of truth for every definition that affects data partitioning.

Two errors in this project came from re-implementing the same concept in different scripts:
block labels written as "[blk2]" in one place and "b2" in another, and the sealed-set
similarity computed from one substrate in one script and from both in another. Anything that
decides which rows go where lives here and is imported, never re-written."""
import numpy as np, pandas as pd
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs, Descriptors, Crippen
RDLogger.DisableLog('rdApp.*')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
REAG = ['reagent%d_id' % i for i in range(1, 7)]
FILL, JAC = (0.7, 0.9)
MIN_SUB, MIN_COND = (50, 12)
TOPK, USABLE, DEAD = (3, 0.2, 0.05)
SEED = 20260818
HOLD = 0.2
FRAC_NONGO = 0.7
GO_SCREENS = {'Buchwald-Hartwig|1', 'Buchwald-Hartwig|2', 'Chan-Lam|1', 'Grignard|1', 'Kumada|1', 'SNAr|1', 'Sonogashira|1', 'Suzuki|1'}

def screen_key(cls, blk):
    """the ONE spelling of a screen identifier"""
    return '%s|%d' % (cls, blk)

def blocks(sub):
    """substrates whose condition sets agree at Jaccard >= JAC, greedy on signatures"""
    sig = sub.groupby('subpair')['condition_id'].apply(lambda s: frozenset(s.unique()))
    cnt = Counter(sig.values)
    sigs = [s for s, _ in cnt.most_common()]
    used, res = (set(), [])
    for i, a in enumerate(sigs):
        if i in used:
            continue
        grp = [i]
        used.add(i)
        for j in range(i + 1, len(sigs)):
            if j in used:
                continue
            if len(a & sigs[j]) / max(len(a | sigs[j]), 1) >= JAC:
                grp.append(j)
                used.add(j)
        keep = {sigs[k] for k in grp}
        res.append(([sp for sp, s in sig.items() if s in keep], sorted(set().union(*keep))))
    res.sort(key=lambda x: -len(x[0]))
    return res

def _descr(m):
    if m is None:
        return [np.nan] * 10
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), Descriptors.TPSA(m), Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), Descriptors.NumAromaticRings(m), Descriptors.FractionCSP3(m), Descriptors.NumHeteroatoms(m), Descriptors.RingCount(m)]

def features(subpairs):
    """ECFP4 of both partners + physchem. THE feature definition."""
    f1, f2, ds = ([], [], [])
    for sp in subpairs:
        parts = [x for x in sp.split(' >> ') if x and x != 'nan']
        ms = [Chem.MolFromSmiles(x) for x in parts[:2]]
        while len(ms) < 2:
            ms.append(None)
        for slot, mm in zip((f1, f2), ms):
            a = np.zeros(2048, dtype=np.int8)
            if mm is not None:
                DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(mm), a)
            slot.append(a)
        ds.append(_descr(ms[0]) + _descr(ms[1]))
    return (np.array(f1, dtype=np.float32), np.array(f2, dtype=np.float32), np.nan_to_num(np.array(ds, dtype=np.float32)))

def similarity(F1, F2):
    """mean Tanimoto over BOTH partners. THE similarity definition - used for the sealed
    set, for the kNN model, and for eligibility. Never compute it any other way."""

    def bulk(F):
        s = F @ F.T
        c = F.sum(axis=1)
        return s / np.clip(c[:, None] + c[None, :] - s, 1e-09, None)
    S = 0.5 * (bulk(F1) + bulk(F2)) if F2.sum() > 0 else bulk(F1)
    np.fill_diagonal(S, -1)
    return S

def build_screens(U):
    """every qualifying screen, keyed by screen_key"""
    out = {}
    for cls, sub in U.groupby('rxn_class'):
        for bi, (subs, conds) in enumerate(blocks(sub), 1):
            if len(subs) < MIN_SUB or len(conds) < MIN_COND:
                continue
            s2 = sub[sub['subpair'].isin(subs) & sub['condition_id'].isin(conds)]
            p = s2.pivot_table(index='subpair', columns='condition_id', values='yield', aggfunc='mean')
            p = p.loc[p.notna().mean(axis=1) >= FILL, p.notna().mean(axis=0) >= FILL]
            if p.shape[0] < MIN_SUB or p.shape[1] < MIN_COND:
                continue
            sps = list(p.index)
            F1, F2, Dsc = features(sps)
            ok = F1.sum(axis=1) > 0
            if ok.sum() < MIN_SUB:
                continue
            out[screen_key(cls, bi)] = dict(cls=cls, blk=bi, sps=[s for s, o in zip(sps, ok) if o], M=p.values.astype(float)[ok], F1=F1[ok], F2=F2[ok], Dsc=Dsc[ok], conds=list(p.columns))
    return _independent_only(_drop_qc_twin(out))

def _independent_only(out):
    """Blocks of the same transformation must be independent campaigns.

    Greedy signature clustering separates two blocks whenever their condition sets agree at less
    than JAC, which can leave a pair that still shares most of its conditions and differs only in
    the substrates run. Two such blocks are not two experiments and must not be counted as two.
    Where any pair of blocks of one transformation shares a condition, the larger is kept."""
    drop = set()
    keys = sorted(out)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if out[a]['cls'] != out[b]['cls'] or a in drop or b in drop:
                continue
            if set(out[a]['conds']) & set(out[b]['conds']):
                drop.add(a if len(out[a]['sps']) < len(out[b]['sps']) else b)
    return {k: v for k, v in out.items() if k not in drop}
QC_TWINS = {'Hiyama': [('C43', 'C46')], 'Suzuki': [('C74', 'C77')], 'Finkelstein': [('C34', 'C37')]}

def _drop_qc_twin(out):
    for key, v in out.items():
        pairs = QC_TWINS.get(v['cls'], [])
        if not pairs:
            continue
        conds = [str(c) for c in v['conds']]
        drop = set()
        for a, b in pairs:
            if a in conds and b in conds:
                drop.add(conds.index(b))
        if not drop:
            continue
        keep = [i for i in range(len(conds)) if i not in drop]
        v['M'] = v['M'][:, keep]
        v['conds'] = [v['conds'][i] for i in keep]
    return out

def allocate(screens, seed=SEED, hold=HOLD, frac_nongo=FRAC_NONGO):
    """THE partition. GO screens: seal the least-similar `hold` fraction, analyse the rest.
    non-GO screens: random `frac_nongo` sample, floor MIN_SUB."""
    rs = np.random.RandomState(seed)
    alloc = {}
    for key in sorted(screens):
        S = screens[key]
        n = len(S['sps'])
        if key in GO_SCREENS:
            nn = similarity(S['F1'], S['F2']).max(axis=1)
            seal = sorted(np.argsort(nn)[:int(round(n * hold))].tolist())
            an = sorted(set(range(n)) - set(seal))
        else:
            seal = []
            k = min(max(int(round(n * frac_nongo)), MIN_SUB), n)
            an = sorted(rs.choice(n, k, replace=False).tolist()) if k < n else list(range(n))
        alloc[key] = dict(analysis=an, sealed=seal, reserve=sorted(set(range(n)) - set(an) - set(seal)))
    return alloc

def drop_duplicated_cells(U):
    """Remove every (transformation, substrate pair, condition) that appears more than once.

    These are NOT replicate measurements. They arise from the same substrate/condition label
    being used in two separate campaigns whose execution differed, so the two values are not
    comparable and neither can be trusted. They are excluded from the analysis and moved to
    the reserve. Returns (clean_frame, dropped_frame)."""
    key = ['rxn_class', 'subpair', 'condition_id']
    n = U.groupby(key)['yield'].transform('size')
    dup = n > 1
    return (U[~dup].copy(), U[dup].copy())
