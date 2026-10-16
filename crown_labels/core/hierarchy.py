import numpy as np

def candidate_families(records):
    local = {r['source_id']: r['id'] for r in records if r['source'].startswith('local_')}
    parents = {}
    owner = list(range(len(records) + 1))

    def root(k):
        while owner[k] != k:
            owner[k] = owner[owner[k]]
            k = owner[k]
        return k
    for rec in records:
        if rec['source'].startswith('local_'):
            continue
        children = sorted({local[x] for x in rec['source_members'] if x in local})
        if children:
            parents[rec['id']] = children
            for child in children:
                owner[root(child)] = root(rec['id'])
    groups = {}
    for rec in records:
        groups.setdefault(root(rec['id']), []).append(rec['id'])
    families = []
    for labels in sorted(groups.values(), key=lambda x: min(x)):
        leaves = sorted(set(labels) - set(parents))
        modes = [leaves]
        for parent in sorted(set(labels) & set(parents)):
            modes.append(sorted([parent] + list(set(leaves) - set(parents[parent]))))
        families.append(dict(id=len(families), labels=sorted(labels), modes=modes))
    return (families, parents)

def conditional_problem(unary, labels, roi, allowed, adjacency, costs):
    """Exact ROI objective, including fixed boundary and outside activation."""
    lut = np.full(len(labels), -1, np.int32)
    lut[roi] = np.arange(len(roi))
    edges = adjacency[roi].tocoo()
    neighbour = lut[edges.col]
    internal = (neighbour >= 0) & (edges.row < neighbour)
    a, b, w = (edges.row[internal], neighbour[internal], edges.data[internal])
    boundary = neighbour < 0
    external = labels[edges.col[boundary]]
    data = unary[np.ix_(roi, allowed)].astype(np.float64)
    if np.any(allowed == 0):
        empty = ~np.isfinite(data).any(axis=1)
        if np.any(empty):
            data[empty, np.flatnonzero(allowed == 0)[0]] = 0.0
    for j, identity in enumerate(allowed):
        data[:, j] += np.bincount(edges.row[boundary], weights=edges.data[boundary] * (external != identity), minlength=len(roi))
    full_count = np.bincount(labels, minlength=unary.shape[1])
    local_count = np.bincount(labels[roi], minlength=unary.shape[1])
    local_costs = costs[allowed].copy()
    local_costs[(full_count - local_count)[allowed] > 0] = 0
    index = np.full(unary.shape[1], -1, int)
    index[allowed] = np.arange(len(allowed))
    initial = index[labels[roi]]
    displaced = initial < 0
    if np.any(displaced):
        initial[displaced] = np.argmin(data[displaced], axis=1)
    rows = np.arange(len(initial))
    invalid = ~np.isfinite(data[rows, initial])
    if np.any(invalid):
        finite_rows = np.isfinite(data[invalid]).any(axis=1)
        if np.any(finite_rows):
            bad_rows = np.flatnonzero(invalid)[finite_rows]
            initial[bad_rows] = np.argmin(data[bad_rows], axis=1)
        no_finite = np.flatnonzero(invalid)[~finite_rows]
        if len(no_finite):
            unknown = np.flatnonzero(allowed == 0)
            if len(unknown):
                initial[no_finite] = unknown[0]
            else:
                raise ValueError('No feasible local assignment')
    return (data, a, b, w, local_costs, initial)
