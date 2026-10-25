"""Training-only response spectra and anchor-based new-pair prediction."""
import numpy as np
from scipy.linalg import svd

def centred(m, view='interaction'):
    x = np.asarray(m, dtype=float)
    mask = np.isfinite(x)
    row = np.nanmean(x, axis=1, keepdims=True)
    if view == 'row':
        return np.where(mask, x - row, 0.0)
    col = np.zeros((1, x.shape[1]))
    for _ in range(30):
        row = np.nanmean(x - col, axis=1, keepdims=True)
        col = np.nanmean(x - row, axis=0, keepdims=True)
    return np.where(mask, x - row - col, 0.0)

def spectrum(m, view='interaction'):
    r = centred(m, view)
    s = svd(r, compute_uv=False, check_finite=False)
    e = s * s
    if e.sum() < 1e-20:
        return dict(k_var90=0, k_var95=0, participation_rank=0.0, ev6=0.0, ev12=0.0, energy=0.0)
    p = e / e.sum()
    cum = p.cumsum()
    return dict(k_var90=int(np.searchsorted(cum, 0.9) + 1), k_var95=int(np.searchsorted(cum, 0.95) + 1), participation_rank=float(1 / np.sum(p * p)), ev6=float(cum[min(5, len(cum) - 1)]), ev12=float(cum[min(11, len(cum) - 1)]), energy=float(e.sum()))

def fit_basis(train):
    mu = np.nanmean(train, axis=0)
    r = centred(train, 'interaction')
    _, s, vt = svd(r, full_matrices=False, check_finite=False)
    en = s * s
    rank = int(np.searchsorted(np.cumsum(en) / en.sum(), 0.9) + 1) if en.sum() > 1e-20 else 0
    return dict(mu=mu, vt=vt, rank=rank)

def select_panel(train, k, method, rng):
    k = min(k, train.shape[1])
    if method == 'random':
        return rng.permutation(train.shape[1])[:k]
    if method == 'high_mean':
        return np.argsort(-np.nanmean(train, axis=0), kind='stable')[:k]
    if method != 'variance_greedy':
        raise ValueError(method)
    r = centred(train, 'interaction')
    cov = r.T @ r / max(len(r) - 1, 1)
    c = cov.copy()
    lam = max(float(np.trace(c) / len(c)) * 0.05, 1e-08)
    chosen = []
    for _ in range(k):
        d = np.diag(c).copy()
        d[chosen] = -np.inf
        j = int(np.argmax(d))
        chosen.append(j)
        v = c[:, j].copy()
        c -= np.outer(v, v) / (c[j, j] + lam)
        c = (c + c.T) * 0.5
    return np.array(chosen)

def reconstruct(train, test, anchors, method, fitted=None):
    fit = fit_basis(train) if fitted is None else fitted
    mu = fit['mu']
    out = np.tile(mu, (len(test), 1))
    for i, y in enumerate(test):
        a = np.asarray(anchors)[np.isfinite(y[anchors])]
        if not len(a):
            continue
        if method == 'additive':
            out[i] = mu + np.mean(y[a] - mu[a])
        elif method == 'affine':
            x = mu[a] - np.mean(mu[a])
            slope = x @ (y[a] - y[a].mean()) / max(x @ x, 1e-12)
            out[i] = y[a].mean() + slope * (mu - mu[a].mean())
        elif method == 'low_rank':
            r = min(len(a) - 1, fit['rank'])
            if r < 1:
                out[i] = mu + np.mean(y[a] - mu[a])
                continue
            v = fit['vt'][:r].T
            design = np.column_stack([np.ones(len(a)), v[a]])
            penalty = np.diag([0.0] + [0.01] * r)
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ (y[a] - mu[a]))
            out[i] = mu + coef[0] + v @ coef[1:]
        else:
            raise ValueError(method)
    return np.clip(out, 0.0, 1.0)
