"""Getis–Ord Gi* and reproducible unconditional random-label inference.

The standardized Gi* formula follows the official Esri method documentation.
Permutation inference randomizes the entire value vector over fixed grid locations,
including the focal value; this is unconditional, not PySAL's conditional permutation.
No normal-tail p-values are used for the maps.
"""
import numpy as np
from scipy import sparse
from scipy.stats import false_discovery_control


def queen_weights(rows, cols):
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    if len(lookup) != len(rows):
        raise ValueError()
    ii, jj = [], []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = lookup.get((int(r + dr), int(c + dc)))
                if j is not None:
                    ii.append(i)
                    jj.append(j)
    w = sparse.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(rows), len(rows)))
    if (w != w.T).nnz or not np.all(w.diagonal() == 1):
        raise ValueError()
    return w


def gi_star_z(values, weights):
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    if n < 30 or not np.isfinite(y).all():
        raise ValueError()
    counts = np.asarray(weights.sum(axis=1)).ravel()
    squared = np.asarray(weights.power(2).sum(axis=1)).ravel()
    mean = y.mean(axis=0)
    sd = np.sqrt(np.maximum(np.mean(y ** 2, axis=0) - mean ** 2, 0))
    denominator = np.sqrt(np.maximum((n * squared - counts ** 2) / (n - 1), 0))[:, None] * sd[None, :]
    deviation = weights @ y - counts[:, None] * mean[None, :]
    z = np.full_like(deviation, np.nan)
    np.divide(deviation, denominator, out=z, where=denominator > 1e-15)
    # A focal cell with no other valid neighbor is not a spatial cluster test.
    z[counts <= 1] = np.nan
    return z, counts.astype(int) - 1, denominator, deviation


def permutation_hotspots(values, weights, permutations=9999, seed=20261012,
                         alpha=0.05, fdr="by", batch_size=32, progress=None):
    y = np.asarray(values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if permutations < 99 or not (0 < alpha < 1) or fdr not in {"bh", "by"}:
        raise ValueError()
    z, neighbors, denominator, observed = gi_star_z(y, weights)
    counts = neighbors + 1
    mean = y.mean(axis=0)
    eligible = np.isfinite(z)
    extreme = np.zeros(y.shape, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n, scenarios = y.shape
    # Reuse each random location permutation across scenarios to preserve pairing.
    for start in range(0, permutations, batch_size):
        b = min(batch_size, permutations - start)
        indices = np.stack([rng.permutation(n) for _ in range(b)])
        randomized = y[indices].transpose(1, 0, 2).reshape(n, b * scenarios)
        lag = (weights @ randomized).reshape(n, b, scenarios)
        random_deviation = lag - counts[:, None, None] * mean[None, None, :]
        # Inclusive ties and the +1 correction prevent anti-conservative zero p-values.
        extreme += np.count_nonzero(random_deviation >= observed[:, None, :] - 1e-12, axis=1)
        if progress and ((start + b) % 2048 == 0 or start + b == permutations):
            progress(start + b, permutations)
    p = (extreme + 1.0) / (permutations + 1.0)
    p[~eligible] = np.nan
    q_bh, q_by = np.full_like(p, np.nan), np.full_like(p, np.nan)
    for scenario in range(scenarios):
        selected = eligible[:, scenario]
        if selected.any():
            q_bh[selected, scenario] = false_discovery_control(p[selected, scenario], method="bh")
            q_by[selected, scenario] = false_discovery_control(p[selected, scenario], method="by")
    q = q_by if fdr == "by" else q_bh
    hot = eligible & (z > 0) & (q <= alpha)
    return {"z": z, "p_upper": p, "q_bh": q_bh, "q_by": q_by, "q": q,
            "hotspot": hot, "neighbor_count": neighbors, "tested": eligible}
