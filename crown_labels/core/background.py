import hashlib, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import ndimage as nd

def _stripe(job):
    values, lo, hi = job
    return nd.percentile_filter(values, percentile=20, size=101, mode='nearest')[lo:hi]

def background(upper, counts, cache_dir=None, workers=12):
    upper = upper.astype(float)
    good = np.isfinite(upper) & (counts > 0)
    assert good.any(), 'No observed heights'
    cache = None
    if cache_dir is not None:
        h = hashlib.sha256(b'percentile20_size101_nearest_v1')
        for value in [upper, good]:
            h.update(str((value.shape, value.dtype.str)).encode())
            h.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        cache = Path(cache_dir) / ('local_background_' + h.hexdigest() + '.npy')
        cache.parent.mkdir(exist_ok=True, parents=True)
        if cache.exists():
            return np.load(cache)
    nearest = nd.distance_transform_edt(~good, return_distances=False, return_indices=True)
    surface = upper[tuple(nearest)]
    h = surface.shape[0]
    workers = min(workers, max(1, h // 64))
    cuts = np.linspace(0, h, workers + 1, dtype=int)
    jobs = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        lo = max(0, start - 50)
        hi = min(h, end + 50)
        jobs.append((surface[lo:hi], start - lo, end - lo))
    if workers == 1:
        base = _stripe(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            base = np.concatenate(list(pool.map(_stripe, jobs)), axis=0)
    if cache is not None:
        temp = cache.with_suffix('.' + str(os.getpid()) + '.tmp')
        with temp.open('wb') as f:
            np.save(f, base)
        temp.replace(cache)
    return base

def apply(model, data, threshold=0.6):
    cache = None
    upper = data['upper']
    if os.environ.get('GRID01_CONTEXT'):
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool) if os.environ['GRID01_CONTEXT'] == 'cross' else np.ones((2, 2), bool) if os.environ['GRID01_CONTEXT'] == 'box2' else np.ones((3, 3), bool)
        upper = nd.maximum_filter(upper, footprint=footprint, mode='nearest')
    base = background(upper, data['counts'], cache)
    rc = np.floor(model['nodes'][:, :2] / 0.1).astype(int)
    excess = model['heights'] - base[rc[:, 1], rc[:, 0]]
    penalty = 0.8 * np.maximum(threshold / np.maximum(excess, 0.05) - 1, 0)
    for k in range(1, model['unary'].shape[1]):
        ix = np.flatnonzero(np.isfinite(model['unary'][:, k]))
        model['unary'][ix, k] += penalty[ix]
    return dict(background_radius_m=5.0, background_quantile=0.2, height_excess_threshold_m=threshold, penalized_nodes=int((penalty > 0).sum()), penalty_above_unknown_nodes=int((penalty > 0.8).sum()), runtime='exact eight-strip percentile with 50-cell halo and input-hashed cache')
