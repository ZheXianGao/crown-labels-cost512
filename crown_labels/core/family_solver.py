import time
import numpy as np
from scipy.sparse import coo_matrix
from crown_labels.core.hierarchy import conditional_problem, candidate_families
from crown_labels.core.joint_solver import optimize, energy
from crown_labels.core.compatible import validate

def delta_energy(labels, proposal, roi, unary, adjacency, costs):
    lookup = np.full(len(labels), -1, np.int32)
    lookup[roi] = np.arange(len(roi))
    edge = adjacency[roi].tocoo()
    j = lookup[edge.col]
    take = (j < 0) | (edge.row < j)
    row = edge.row[take]
    col = edge.col[take]
    weights = edge.data[take]
    old = labels[roi]
    new_neighbors = labels[col].copy()
    inside = lookup[col] >= 0
    new_neighbors[inside] = proposal[lookup[col[inside]]]
    data = float(np.sum(unary[roi, proposal].astype(float) - unary[roi, old].astype(float)))
    pair = float(np.sum(weights * ((proposal[row] != new_neighbors).astype(float) - (old[row] != labels[col]).astype(float))))
    counts = np.bincount(labels, minlength=len(costs))
    after = counts - np.bincount(old, minlength=len(costs)) + np.bincount(proposal, minlength=len(costs))
    activation = float(costs[after > 0].sum() - costs[counts > 0].sum())
    return data + pair + activation

def solve(model, costs, initial, max_passes=30, time_limit=3600.0, callback=None):
    start = time.monotonic()
    unary = model['unary']
    families, parents = candidate_families(model['candidates'])
    parents = {int(k): set(v) for k, v in parents.items()}
    leaves = set(range(1, unary.shape[1])) - set(parents)
    labels = initial.copy()
    validate(labels, families, parents)
    a, b, w = (model['edge_i'], model['edge_j'], model['weights'])
    adj = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(labels), len(labels))).tocsr()
    supports = {k: np.flatnonzero(np.isfinite(unary[:, k])) for k in range(1, unary.shape[1])}
    current = energy(labels, unary, a, b, w, costs)
    trace = [dict(iteration=0, **current)]
    moves = []
    converged = False
    inner_incomplete = 0
    tasks = [('refine', k) for k in sorted(leaves)] + [v for k in sorted(parents) for v in [('merge', k), ('split', k)]]
    for iteration in range(1, max_passes + 1):
        accepted = 0
        incomplete = False
        for operation, k in tasks:
            if time.monotonic() - start >= time_limit:
                incomplete = True
                break
            active = set(np.unique(labels))
            active_parents = active & set(parents)
            if operation == 'refine':
                if any((k in parents[p] for p in active_parents)):
                    continue
                removed = set()
                added = {k}
                affected = {k}
            elif operation == 'split':
                if k not in active:
                    continue
                removed = {k}
                added = parents[k]
                affected = removed | added
            else:
                members = parents[k]
                conflict = {p for p in active_parents if p != k and parents[p] & members}
                removed = members | conflict
                restored = (set().union(*(parents[p] for p in conflict)) if conflict else set()) - members
                added = {k} | restored
                affected = removed | added
            roi = np.unique(np.concatenate([supports[x] for x in affected]))
            roi = np.union1d(roi, np.flatnonzero(np.isin(labels, list(removed))))
            if not len(roi):
                continue
            allowed = np.array(sorted(({0} | set(labels[roi]) | added) - removed), int)
            data, la, lb, lw, lc, seed = conditional_problem(unary, labels, roi, allowed, adj, costs)
            result, log = optimize(data, la, lb, lw, lc, initial=seed, max_passes=50, time_limit=60.0)
            if not log['converged_full_pass']:
                incomplete = True
                inner_incomplete += 1
            candidate = allowed[result]
            if np.array_equal(candidate, labels[roi]):
                continue
            delta = delta_energy(labels, candidate, roi, unary, adj, costs)
            if delta < -1e-06:
                proposal = labels.copy()
                proposal[roi] = candidate
                validate(proposal, families, parents)
                value = energy(proposal, unary, a, b, w, costs)
                assert abs(value['total'] - current['total'] - delta) < 0.0001, (value['total'], current['total'], delta)
                assert value['total'] < current['total'] - 1e-06
                moves.append(dict(iteration=iteration, operation=operation, candidate=k, delta=delta, changed_nodes=int((candidate != labels[roi]).sum())))
                labels = proposal
                current = value
                accepted += 1
        trace.append(dict(iteration=iteration, accepted=accepted, **current))
        print('PASS', iteration, 'moves', accepted, 'energy', current['total'], 'elapsed', round(time.monotonic() - start), flush=True)
        if callback:
            callback(labels, dict(trace=trace, moves=moves, seconds=time.monotonic() - start))
        if accepted == 0 and (not incomplete):
            converged = True
            break
        if time.monotonic() - start >= time_limit:
            break
    return (labels, dict(converged_full_pass=converged, trace=trace, moves=moves, inner_unconverged=inner_incomplete, families=families, parents={k: sorted(v) for k, v in parents.items()}, seconds=time.monotonic() - start))
