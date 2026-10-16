import numpy as np
from crown_labels.core.background import apply as add_background_cost

def prepare_unary(model, data):
    unary = model['unary']
    lookup = {p['source_id']: k for k, p in enumerate(model['pool'], 1) if p['source'].startswith('local_')}
    for k, p in enumerate(model['pool'], 1):
        if not p['source'].startswith('joint'):
            continue
        left, right = [lookup[s] for s in p['source_members']]
        allowed = np.flatnonzero(np.isfinite(unary[:, left]) | np.isfinite(unary[:, right]))
        unary[:, k] = np.inf
        unary[allowed, k] = (-np.logaddexp(-8 * unary[allowed, left], -8 * unary[allowed, right]) + np.log(2)) / 8
    best = unary[:, 1:].min(1)
    good = np.isfinite(best)
    for k in range(1, unary.shape[1]):
        selected = np.flatnonzero(good & np.isfinite(unary[:, k]))
        unary[selected, k] -= best[selected]
    return add_background_cost(model, data)

def initialize(model):
    leaves = [k for k, p in enumerate(model['pool'], 1) if p['source'].startswith('local_')]
    allowed = np.asarray([0] + leaves)
    return allowed[np.argmin(model['unary'][:, allowed], axis=1)].astype(np.int32)
