import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

def energy(labels, unary, edge_i, edge_j, weights, costs):
    data = float(unary[np.arange(len(labels)), labels].sum(dtype=np.float64))
    pair = float(weights[labels[edge_i] != labels[edge_j]].sum(dtype=np.float64))
    label = float(costs[np.unique(labels)].sum())
    return dict(data=data, pair=pair, label=label, total=data + pair + label, active_labels=int(np.count_nonzero(np.unique(labels))))

def expansion(labels, alpha, unary, edge_i, edge_j, weights, costs, scale=10000.0):
    n = len(labels)
    variable = (labels != alpha) & np.isfinite(unary[:, alpha])
    selected = np.flatnonzero(variable)
    nv = len(selected)
    if not nv:
        return (labels.copy(), dict(variables=0, scale=scale, auxiliaries=0))
    lut = np.full(n, -1, np.int32)
    lut[selected] = np.arange(nv)
    u0 = unary[selected, labels[selected]].astype(np.float64)
    u1 = unary[selected, alpha].astype(np.float64)
    rows, cols, capacity = ([], [], [])
    a, b = (lut[edge_i], lut[edge_j])
    both = (a >= 0) & (b >= 0)
    diff = labels[edge_i[both]] != labels[edge_j[both]]
    c = weights[both] * (1 - 0.5 * diff)
    offset = -0.5 * weights[both] * diff
    np.add.at(u1, a[both], offset)
    np.add.at(u1, b[both], offset)
    rows.extend([a[both], b[both]])
    cols.extend([b[both], a[both]])
    capacity.extend([c, c])
    for src, dst in ((edge_i, edge_j), (edge_j, edge_i)):
        one = variable[src] & ~variable[dst]
        local = lut[src[one]]
        np.add.at(u0, local, weights[one] * (labels[src[one]] != labels[dst[one]]))
        np.add.at(u1, local, weights[one] * (alpha != labels[dst[one]]))
    aux0, aux1, implications = ([], [], [])
    if not np.any(labels == alpha) and costs[alpha] > 0:
        auxiliary = nv + len(aux0)
        aux0.append(0.0)
        aux1.append(float(costs[alpha]))
        implications.append((np.full(nv, auxiliary), np.arange(nv)))
    for old in np.unique(labels[selected]):
        if costs[old] <= 0 or np.any((labels == old) & ~variable):
            continue
        auxiliary = nv + len(aux0)
        aux0.append(float(costs[old]))
        aux1.append(0.0)
        group = np.flatnonzero(labels[selected] == old)
        implications.append((group, np.full(len(group), auxiliary)))
    u0, u1 = (np.r_[u0, aux0], np.r_[u1, aux1])
    common = np.minimum(u0, u1)
    u0, u1 = (u0 - common, u1 - common)
    finite_sum = float(u0.sum() + u1.sum() + sum((x.sum() for x in capacity)))
    infinity = finite_sum + 1.0
    actual_scale = min(scale, 2 ** 28 / infinity)
    for src, dst in implications:
        rows.append(src)
        cols.append(dst)
        capacity.append(np.full(len(src), infinity))
    source, sink = (len(u0), len(u0) + 1)
    vertices = np.arange(len(u0))
    rows.extend([np.full(len(u0), source), vertices])
    cols.extend([vertices, np.full(len(u0), sink)])
    capacity.extend([u1, u0])
    scaled = np.rint(np.concatenate(capacity) * actual_scale).astype(np.int64)
    graph = coo_matrix((scaled, (np.concatenate(rows), np.concatenate(cols))), shape=(sink + 1, sink + 1)).tocsr()
    graph.eliminate_zeros()
    flow = maximum_flow(graph, source, sink, method='dinic')
    residual = graph - flow.flow
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    reachable = np.zeros(sink + 1, bool)
    reachable[breadth_first_order(residual, source, directed=True, return_predecessors=False)] = True
    proposal = labels.copy()
    proposal[selected[~reachable[:nv]]] = alpha
    return (proposal, dict(variables=nv, scale=actual_scale, auxiliaries=len(aux0), capacity_rounding_bound=0.5 / actual_scale))

def optimize(unary, edge_i, edge_j, weights, costs, initial=None, max_passes=3, time_limit=180.0):
    if np.isnan(unary).any() or np.any(weights < 0) or np.any(costs < 0):
        raise ValueError('Require finite-or-infinite unary, nonnegative Potts edges and label costs')
    labels = np.argmin(unary, axis=1).astype(np.int32) if initial is None else initial.copy()
    current = energy(labels, unary, edge_i, edge_j, weights, costs)
    if not np.isfinite(current['total']):
        raise ValueError('Every current assignment must be feasible')
    start = time.time()
    trace, moves = ([dict(iteration=0, **current)], [])
    converged, timed_out = (False, False)
    min_scale = 10000.0
    support = np.isfinite(unary).sum(0)
    order = sorted(range(1, unary.shape[1]), key=lambda k: (-support[k], k)) + [0]
    for iteration in range(1, max_passes + 1):
        accepted = 0
        for alpha in order:
            if time.time() - start > time_limit:
                timed_out = True
                break
            proposal, cut = expansion(labels, alpha, unary, edge_i, edge_j, weights, costs)
            min_scale = min(min_scale, cut['scale'])
            value = energy(proposal, unary, edge_i, edge_j, weights, costs)
            if value['total'] < current['total'] - 1e-07:
                changed = int((proposal != labels).sum())
                moves.append(dict(iteration=iteration, alpha=int(alpha), changed_nodes=changed, before=current['total'], after=value['total'], **cut))
                labels, current = (proposal, value)
                accepted += 1
        trace.append(dict(iteration=iteration, accepted=accepted, **current))
        if timed_out:
            break
        if accepted == 0:
            converged = True
            break
    return (labels, dict(trace=trace, moves=moves, converged_full_pass=converged, time_limit_reached=timed_out, max_passes=max_passes, integer_scale_min=min_scale, seconds=round(time.time() - start, 3), objective='sum unary + weighted Potts + cost of every nonempty label', guarantee='Every accepted floating-point objective strictly decreases; no global optimum guarantee'))
