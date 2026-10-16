import numpy as np
from scipy import ndimage as nd

def repair(ids, valid, forbidden=None, res=0.1, closing_m=0.3, hole_m2=1.0, distance_m=0.4, add_fraction=0.3):
    out = ids.copy()
    hard = np.zeros(ids.shape, bool) if forbidden is None else forbidden.astype(bool)
    assert not np.any((ids > 0) & hard), 'Remove policy-forbidden positives before repair'
    domain = valid & ~hard
    best = np.full(ids.shape, np.inf, np.float32)
    owner = np.zeros_like(ids)
    records = []
    radius = int(np.ceil(closing_m / res))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= (closing_m / res) ** 2 + 1e-08
    margin = int(np.ceil((2 * closing_m + distance_m) / res)) + 2
    for k, sl0 in enumerate(nd.find_objects(ids), 1):
        if sl0 is None:
            continue
        sl = tuple((slice(max(0, s.start - margin), min(n, s.stop + margin)) for s, n in zip(sl0, ids.shape)))
        local = ids[sl]
        own = local == k
        regular = nd.binary_closing(own, structure=disk)
        holes = nd.binary_fill_holes(own) & ~own
        cc, num = nd.label(holes, np.ones((3, 3)))
        sizes = np.bincount(cc.ravel())
        keep = sizes * res ** 2 <= hole_m2
        keep[0] = False
        occupied = np.unique(cc[local > 0])
        keep[occupied] = False
        holefill = keep[cc]
        distance = nd.distance_transform_edt(~own) * res
        proposal = (regular | holefill) & (local == 0) & domain[sl] & (distance <= distance_m + 1e-08)
        ix = np.flatnonzero(proposal)
        cap = int(own.sum() * add_fraction)
        if len(ix) > cap:
            ix = ix[np.argsort(distance.ravel()[ix], kind='stable')[:cap]]
            proposal[:] = False
            proposal.ravel()[ix] = True
        win = proposal & (distance < best[sl] - 1e-07)
        best[sl][win] = distance[win]
        owner[sl][win] = k
        records.append(dict(id=k, original_pixels=int(own.sum()), proposed_pixels=int(proposal.sum()), cap_pixels=cap))
    add = (ids == 0) & (owner > 0) & domain
    out[add] = owner[add]
    assert np.array_equal(np.unique(out[out > 0]), np.unique(ids[ids > 0]))
    assert np.array_equal(out[ids > 0], ids[ids > 0])
    assert not np.any(out[~domain])
    for r in records:
        assert int(((out == r['id']) & (ids == 0)).sum()) <= r['cap_pixels']
    return (out, dict(added_pixels=int(add.sum()), config=dict(resolution_m=res, closing_m=closing_m, max_hole_m2=hole_m2, max_distance_m=distance_m, addition_fraction=add_fraction), instances=records))
