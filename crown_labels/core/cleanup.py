from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi

@dataclass(frozen=True)
class CleanupConfig:
    resolution: float = 0.1
    sigma_m: float = 0.3
    opening_m: float = 0.3
    closing_m: float = 0.4
    max_expansion_m: float = 0.4
    protected_core_m: float = 0.6
    max_island_m2: float = 2.0
    max_hole_m2: float = 0.8
    max_change_fraction: float = 0.25
    boundary_margin: float = 1e-06
    boundary_erosion_m: float = 0.2
CONNECT = np.ones((3, 3), bool)

def component_count(mask):
    return int(ndi.label(mask, CONNECT)[1])

def disk(radius):
    n = int(np.ceil(radius))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return x * x + y * y <= radius * radius + 1e-08

def perimeter_m(ids, res):
    total = 0
    padded = np.pad(ids, 1)
    for a, b in [(padded[1:, :], padded[:-1, :]), (padded[:, 1:], padded[:, :-1])]:
        changed = a != b
        total += np.count_nonzero(changed & (a > 0)) + np.count_nonzero(changed & (b > 0))
    return float(total * res)

def clean_partition(ids, cfg=CleanupConfig(), allowed=None):
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('Expected nonnegative 2D instance IDs')
    identities = np.unique(ids[ids > 0])
    out = np.zeros_like(ids)
    best = np.full(ids.shape, 0.5, np.float32)
    own_score = np.zeros(ids.shape, np.float32)
    protected = np.zeros(ids.shape, bool)
    before = {}
    islands = {}
    if allowed is None:
        allowed = np.ones(ids.shape, bool)
    if allowed.shape != ids.shape:
        raise ValueError('Domain shape differs')
    domain = allowed.astype(bool) | (ids > 0)
    domain &= ndi.distance_transform_edt(ids == 0) * cfg.resolution <= cfg.max_expansion_m
    for ident in identities:
        mask = ids == ident
        area = int(mask.sum())
        cc, n = ndi.label(mask, CONNECT)
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        tiny = [i for i in range(1, n + 1) if i != largest and sizes[i] * cfg.resolution ** 2 <= cfg.max_island_m2]
        kept = mask & ~np.isin(cc, tiny)
        depth = _sparse_unpadded_edt(kept) * cfg.resolution
        protected |= depth > cfg.protected_core_m
        anchor = np.unravel_index(depth.argmax(), depth.shape)
        protected[anchor] = True
        yy, xx = np.where(mask)
        pad = int(np.ceil((4 * cfg.sigma_m + 2 * cfg.closing_m) / cfg.resolution)) + 1
        y0 = max(0, int(yy.min()) - pad)
        y1 = min(ids.shape[0], int(yy.max()) + pad + 1)
        x0 = max(0, int(xx.min()) - pad)
        x1 = min(ids.shape[1], int(xx.max()) + pad + 1)
        sl = np.s_[y0:y1, x0:x1]
        local = kept[sl]
        equivalent_radius = np.sqrt(local.sum() * cfg.resolution ** 2 / np.pi)
        radius = min(cfg.opening_m, 0.2 * equivalent_radius)
        opened = ndi.binary_opening(local, structure=disk(radius / cfg.resolution))
        if not opened.any():
            opened = local.copy()
        regular = ndi.binary_closing(opened, structure=disk(cfg.closing_m / cfg.resolution))
        holes = ndi.binary_fill_holes(regular) & ~regular
        hole_ids, nh = ndi.label(holes, CONNECT)
        for hole in range(1, nh + 1):
            region = hole_ids == hole
            if region.sum() * cfg.resolution ** 2 <= cfg.max_hole_m2 and (not np.any((ids[sl] > 0) & (ids[sl] != ident) & region)):
                regular[region] = True
        regular &= domain[sl]
        score = ndi.gaussian_filter(regular.astype(np.float32), cfg.sigma_m / cfg.resolution, mode='constant')
        own = ids[sl] == ident
        own_score[sl][own] = score[own]
        win = score > best[sl]
        out[sl][win] = ident
        best[sl][win] = score[win]
        before[int(ident)] = dict(area=area, components=n)
        islands[int(ident)] = sum((int(sizes[i]) for i in tiny))
    out[~domain] = 0
    keep = protected | (ids > 0) & (out > 0) & (best - own_score < cfg.boundary_margin)
    out[keep] = ids[keep]
    locked = set()
    rollbacks = []
    while True:
        newly = []
        for ident in identities:
            k = int(ident)
            if k in locked:
                continue
            old = ids == ident
            new = out == ident
            delta = int(np.count_nonzero(old ^ new))
            reason = None
            if not new.any():
                reason = 'preserve_identity'
            elif delta > cfg.max_change_fraction * before[k]['area']:
                reason = 'change_budget'
            elif component_count(new) > before[k]['components']:
                reason = 'avoid_new_fragments'
            if reason:
                newly.append(k)
                rollbacks.append(dict(instance_id=k, reason=reason))
        if not newly:
            break
        locked.update(newly)
        touch = np.isin(ids, newly) | np.isin(out, newly)
        out[touch] = ids[touch]
    changed = out != ids
    assert np.array_equal(np.unique(out[out > 0]), identities), 'Identity set changed'
    assert np.all(out[protected] == ids[protected]), 'Reliable interior moved'
    assert not np.any((out > 0) & ~domain), 'Foreground outside permitted domain'
    return (out, dict(config=asdict(cfg), changed_pixels=int(changed.sum()), removed_pixels=int(((ids > 0) & (out == 0)).sum()), reassigned_pixels=int(((out > 0) & changed).sum()), added_pixels=int(((out > 0) & (ids == 0)).sum()), before_instances=len(identities), after_instances=len(identities), before_components=sum((x['components'] for x in before.values())), after_components=sum((component_count(out == i) for i in identities)), protected_core_pixels=int(protected.sum()), rollbacks=rollbacks, island_pixels_proposed=sum(islands.values()), before_perimeter_m=perimeter_m(ids, cfg.resolution), after_perimeter_m=perimeter_m(out, cfg.resolution)))

from .bbox_edt import _bbox, sparse_unpadded_edt as _sparse_unpadded_edt
def component_count(mask):
    box = _bbox(mask)
    if box is None:
        return 0
    y0,y1,x0,x1 = box
    return int(ndi.label(mask[y0:y1,x0:x1], CONNECT)[1])
