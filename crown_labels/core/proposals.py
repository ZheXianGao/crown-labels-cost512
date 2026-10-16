import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist

def get_groups(model, support_extension_m=5.0, min_seed_nodes=None):
    started = time.monotonic()
    h = np.asarray(model['heights'])
    n = len(h)
    a = model['edge_i']
    b = model['edge_j']
    height = np.minimum(h[a], h[b])
    assert (height > 0).all() and np.isfinite(height).all()
    graph = coo_matrix((-height, (a, b)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(graph).tocoo()
    order = np.argsort(mst.data, kind='stable')
    aa = mst.row[order]
    bb = mst.col[order]
    hh = -mst.data[order]
    min_seed_nodes = int(model['base'].min_seed_nodes if min_seed_nodes is None else min_seed_nodes)
    leaves = {int(p['source_id']): (k, p) for k, p in enumerate(model['pool'], 1) if p['source'].startswith('local_')}
    representative = {s: int(p['core'][h[p['core']].argmax()]) for s, (_, p) in leaves.items()}
    assert len(set(representative.values())) == len(representative), 'Seed cores have duplicate highest node; explicitly handle coincident seeds before hierarchy'
    members = {node: (s,) for s, node in representative.items()}
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    events = []
    for a, b, saddle in zip(aa, bb, hh):
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = (rb, ra)
        ma = members.pop(ra, ())
        mb = members.pop(rb, ())
        parent[rb] = ra
        size[ra] += size[rb]
        if ma and mb:
            merged = tuple(sorted(ma + mb))
            members[ra] = merged
            lower = min((float(h[representative[s]]) for s in merged))
            events.append(dict(source_members=list(merged), member_count=len(merged), children=[list(ma), list(mb)], saddle_height_m=float(saddle), lower_peak_height_m=lower, valley_depth_m=lower - float(saddle), relative_depth=(lower - float(saddle)) / lower))
        elif ma or mb:
            members[ra] = ma or mb
    support = {s: np.flatnonzero(np.isfinite(model['unary'][:, k])) for s, (k, p) in leaves.items()}
    groups = []
    for event in events:
        ss = event['source_members']
        xy = np.array([leaves[s][1]['center'] for s in ss])
        span = float(pdist(xy).max()) if len(ss) > 1 else 0.0
        common = None
        for s in sorted(ss, key=lambda s: len(support[s])):
            common = support[s] if common is None else np.intersect1d(common, support[s], assume_unique=True)
            if not len(common):
                break
        common_count = len(common) if common is not None else 0
        reasons = []
        if len(ss) < 3:
            reasons.append('binary_group_original_pairs_retained')
        if span > 2 * support_extension_m:
            reasons.append('peak_span_exceeds_existing_2x_support_extension')
        if common_count < min_seed_nodes:
            reasons.append('common_finite_support_below_existing_min_seed_nodes')
        event.update(common_support_nodes=common_count, max_peak_span_m=span, retained=not reasons, rejection_reasons=reasons)
        if event['retained']:
            groups.append(event)
    audit = dict(method='seed-component hierarchy in existing 3D maximin normalized-height filtration', nodes=n, edges=len(model['edge_i']), local_seed_count=len(leaves), hierarchy_events=len(events), retained_groups=len(groups), retained_arity_histogram={str(k): sum((len(g['source_members']) == k for g in groups)) for k in sorted(set((len(g['source_members']) for g in groups)))}, constraints=dict(max_peak_span_m=2 * support_extension_m, min_common_finite_support_nodes=min_seed_nodes, new_valley_threshold=None), all_events=events, seconds=time.monotonic() - started, ground_truth_read=False, existing_pair_candidates_removed=False, labels_modified=False)
    return (groups, audit)
