from collections import OrderedDict
import hashlib, json
import numpy as np

def make(original, budget_bytes=512 * 1024 ** 2):
    cache = OrderedDict()
    stats = dict(hits=0, misses=0, bytes=0)

    def optimize(data, a, b, w, costs, initial=None, **kwargs):
        h = hashlib.sha256()
        for x in [data, a, b, w, costs, initial]:
            if x is None:
                h.update(b'None')
                continue
            v = np.ascontiguousarray(x)
            h.update(str((v.shape, v.dtype.str)).encode())
            h.update(memoryview(v).cast('B'))
        h.update(json.dumps(kwargs, sort_keys=True).encode())
        key = h.digest()
        if key in cache:
            stats['hits'] += 1
            result, log = cache.pop(key)
            cache[key] = (result, log)
            return (result.copy(), dict(log))
        stats['misses'] += 1
        result, log = original(data, a, b, w, costs, initial=initial, **kwargs)
        if log['converged_full_pass']:
            size = result.nbytes + 1024
            while cache and stats['bytes'] + size > budget_bytes:
                _, (old, _) = cache.popitem(last=False)
                stats['bytes'] -= old.nbytes + 1024
            if size <= budget_bytes:
                cache[key] = (result.copy(), dict(log))
                stats['bytes'] += size
        return (result, log)
    optimize.stats = stats
    return optimize
