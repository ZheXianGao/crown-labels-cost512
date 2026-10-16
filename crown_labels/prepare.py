import os
from pathlib import Path
import json, time, warnings
import numpy as np, rasterio
from affine import Affine
from rasterio.windows import Window
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from . import science
P = Path(os.environ.get('CROWN_RUN_ROOT','.')).resolve()

def validity(w):
    n = w['size']
    x0, y0 = w['origin_xy']
    with rasterio.open(w['dom']) as src:
        tf = src.transform
        assert tf.b == tf.d == 0 and tf.a > 0 and (tf.e < 0)
        xs = x0 + (np.arange(n) + 0.5) * 0.1
        ys = y0 - (np.arange(n) + 0.5) * 0.1
        cols = np.floor((xs - tf.c) / tf.a).astype(int)
        rows = np.floor((ys - tf.f) / tf.e).astype(int)
        c0, c1, r0, r1 = (int(cols.min()), int(cols.max() + 1), int(rows.min()), int(rows.max() + 1))
        win = Window(c0, r0, c1 - c0, r1 - r0)
        alpha = [i + 1 for i, c in enumerate(src.colorinterp) if c.name == 'alpha']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', rasterio.errors.NodataShadowWarning)
            amask = src.read(alpha[0], window=win, boundless=True, fill_value=0) > 0 if alpha else src.read_masks(1, window=win, boundless=True) > 0
            valid = amask[np.ix_(rows - r0, cols - c0)]
            mask = np.any(src.read_masks([1, 2, 3], window=win, boundless=True) > 0, axis=0) & amask
            rgbvalid = mask[np.ix_(rows - r0, cols - c0)]
        return (valid, rgbvalid, dict(native_resolution_m=[tf.a, -tf.e], native_transform=list(tf)[:6], RGB_resampled=False))

def prepare(item, out):
    start = time.monotonic()
    w = item['window']
    name = w['scene']
    inp = P / 'inputs' / name
    raw = np.load(inp / 'raw_world.npy')
    receipt = json.load(open(inp / 'raw_receipt.json'))
    assert len(raw) == receipt['raw_points']
    with np.load(receipt['ground']) as z:
        dtm = z['dtm']
        gt = z['transform']
        gobs = z['ground_observed']
    ground = nd.map_coordinates(dtm, [(raw[:, 1] - gt[5]) / gt[4] - 0.5, (raw[:, 0] - gt[2]) / gt[0] - 0.5], order=1, mode='constant', cval=np.nan, prefilter=False)
    heights = raw[:, 2] - ground
    x0, y0 = w['origin_xy']
    xy = np.c_[raw[:, 0] - x0, y0 - raw[:, 1]]
    size = w['size']
    valid, rgbvalid, domlog = validity(w)
    rc = np.floor(xy / 0.1).astype(int)
    inside = ((rc >= 0) & (rc < size)).all(1)
    assert inside.all()
    selected = np.flatnonzero(np.isfinite(heights) & (heights > 0.5) & (heights <= 60) & valid[rc[:, 1], rc[:, 0]])
    pts = np.c_[xy[selected], heights[selected]]
    if len(pts):
        q = np.floor(pts / 0.1).astype(np.int64)
        q -= q.min(0)
        stride = q.max(0) + 1
        keys = (q[:, 0] * stride[1] + q[:, 1]) * stride[2] + q[:, 2]
        _, inv, count = np.unique(keys, return_inverse=True, return_counts=True)
        centers = np.column_stack([np.bincount(inv, weights=pts[:, j]) / count for j in range(3)])
        zmean = np.bincount(inv, weights=raw[selected, 2] - raw[:, 2].min()) / count
        nodes = np.c_[centers[:, :2], zmean]
        maxh = np.full(len(count), -np.inf)
        np.maximum.at(maxh, inv, pts[:, 2])
        tops = np.flatnonzero(pts[:, 2] == maxh[inv])
        topidx = np.full(len(count), len(pts), np.int64)
        np.minimum.at(topidx, inv[tops], tops)
        highest = pts[topidx]
        use = highest[:, 2] >= 2.5
        peaks, diag = science.detector()(highest[use], selected[topidx[use]])
        admitted, admission = science.admit_peaks(peaks, highest)
        tree = cKDTree(nodes)
        support = np.empty(len(nodes), np.uint16)
        for s in range(0, len(nodes), 20000):
            distances = tree.query(nodes[s:s + 20000], k=min(16, len(nodes)), distance_upper_bound=0.8, workers=1)[0]
            support[s:s + 20000] = np.isfinite(distances).sum(axis=1) if distances.ndim == 2 else np.isfinite(distances).astype(np.uint16)
    else:
        inv = np.zeros(0, np.int32)
        count = np.zeros(0, np.int64)
        centers = nodes = np.empty((0, 3))
        support = np.zeros(0, np.uint16)
        peaks = admitted = []
        diag = {}
        admission = []
    flat = rc[:, 1] * size + rc[:, 0]
    order = np.lexsort((-raw[:, 2], flat))
    cell, first = np.unique(flat[order], return_index=True)
    topraw = order[first]
    r2v = np.full(len(raw), -1, np.int32)
    r2v[selected] = inv
    pv = np.full(size * size, -1, np.int32)
    pv[cell] = r2v[topraw]
    pv = pv.reshape(size, size)
    ph = np.full(size * size, np.nan, np.float32)
    ph[cell] = heights[topraw]
    ph = ph.reshape(size, size)
    counts = np.bincount(flat, minlength=size * size).reshape(size, size).astype(np.uint32)
    upper = np.full(size * size, -np.inf, np.float32)
    np.maximum.at(upper, flat, heights)
    upper = upper.reshape(size, size)
    xs = x0 + (np.arange(size) + 0.5) * 0.1
    ys = y0 - (np.arange(size) + 0.5) * 0.1
    gc = np.floor((xs - gt[2]) / gt[0]).astype(int)
    gr = np.floor((ys - gt[5]) / gt[4]).astype(int)
    groundvalid = (gr[:, None] >= 0) & (gr[:, None] < gobs.shape[0]) & (gc[None, :] >= 0) & (gc[None, :] < gobs.shape[1])
    groundvalid &= gobs[np.ix_(np.clip(gr, 0, gobs.shape[0] - 1), np.clip(gc, 0, gobs.shape[1] - 1))]
    data = dict(nodes=nodes, heights=centers[:, 2], local_support=support, upper=upper, counts=counts, valid=valid, rgb_valid=rgbvalid, projection_voxel=pv, projection_height=ph, voxel_count=count, ground_support=groundvalid)
    audit = dict(name=name, raw_points=len(raw), vegetation_points=len(selected), nodes=len(nodes), raw_peaks=len(peaks), admitted_peaks=len(admitted), detector=diag, admission=admission, DOM=domlog, voxel_m=0.1, peak_radius_m=2.0, auxiliary_grid_m=0.1, projection_grid_m=0.1, old_generator_called=False, one_voxelization=True, ground_sha256=receipt['ground_sha256'], raw_array_sha256=receipt['raw_array_sha256'], seconds=time.monotonic() - start)
    science.write(out / 'preparation.json', audit)
    science.write(out / 'peaks.json', admitted)
    return (data, admitted, audit)
