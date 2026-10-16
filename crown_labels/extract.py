import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import json, time, hashlib, shutil, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, laspy, rasterio
P = Path(os.environ.get('CROWN_RUN_ROOT','.')).resolve()

def write(p, j):
    p = Path(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(j, ensure_ascii=False, indent=2))
    q.replace(p)

def flight(job):
    u, items = job
    start = time.monotonic()
    parts = {x['window']['scene']: [] for x in items}
    sources = {}
    records = []
    for item in items:
        for src in item['window']['las_sources']:
            sources[src['path']] = src
    for path, ref in sources.items():
        before = Path(path).stat()
        digest = hashlib.sha256()
        with laspy.open(path) as f:
            with rasterio.open(items[0]['window']['dom']) as dom:
                assert f.header.parse_crs() == dom.crs
            header = dict(point_count=f.header.point_count, scales=f.header.scales.tolist(), offsets=f.header.offsets.tolist(), mins=f.header.mins.tolist(), maxs=f.header.maxs.tolist(), crs=str(f.header.parse_crs()))
            for c in f.chunk_iterator(2000000):
                digest.update(memoryview(np.ascontiguousarray(c.array)).cast('B'))
                xx, yy, zz = (np.asarray(c.x), np.asarray(c.y), np.asarray(c.z))
                for item in items:
                    w = item['window']
                    l, b, r, t = w['bounds']
                    ok = (xx >= l) & (xx < r) & (yy > b) & (yy <= t)
                    if ok.any():
                        parts[w['scene']].append(np.c_[xx[ok], yy[ok], zz[ok]])
        after = Path(path).stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        records.append(dict(path=path, bytes=before.st_size, mtime_ns=before.st_mtime_ns, point_record_stream_sha256=digest.hexdigest(), header=header))
    for item in items:
        w = item['window']
        name = w['scene']
        d = P / 'inputs' / name
        d.mkdir(exist_ok=True, parents=True)
        raw = np.concatenate(parts[name]) if parts[name] else np.empty((0, 3), np.float64)
        parts[name].clear()
        assert len(raw) > 0, (name, 'No raw points')
        dest = d / 'raw_world.npy'
        with dest.with_suffix('.tmp').open('wb') as f:
            np.save(f, raw)
        dest.with_suffix('.tmp').replace(dest)
        gp = P / 'ground' / f'UAV_{u}.npz'
        gp.parent.mkdir(exist_ok=True)
        if not gp.exists():
            shutil.copy2(item['ground'], gp)
        audit = dict(name=name, uav=u, raw_points=len(raw), raw_array_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(raw)).cast('B')).hexdigest(), raw_sources=records, ground=str(gp), ground_sha256=item['ground_sha256'], bounds=w['bounds'], array_file=str(dest), seconds=time.monotonic() - start)
        write(d / 'raw_receipt.json', audit)
        print('RAW_READY', name, len(raw), round(time.monotonic() - start, 1), flush=True)
    return dict(uav=u, windows=len(items), seconds=time.monotonic() - start)
