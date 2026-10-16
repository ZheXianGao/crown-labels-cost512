import numpy as np
from scipy.spatial import cKDTree

def admit_peaks(peaks, highest_points, radius_m=2.0, min_vertical_ratio=0.7):
    points = np.asarray(highest_points)
    tree = cKDTree(points[:, :2])
    accepted, audit = ([], [])
    for source_id, peak in enumerate(peaks, 1):
        indices = tree.query_ball_point([peak['x'], peak['y']], radius_m)
        local = points[indices]
        if len(local) >= 3:
            variance = np.var(local, axis=0, ddof=1)
            ratio = float(np.sqrt(variance[2] / max(variance[:2].mean(), 1e-12)))
        else:
            ratio = 0.0
        keep = ratio >= min_vertical_ratio
        if keep:
            accepted.append(dict(peak))
        audit.append(dict(source_seed_id=source_id, retained=bool(keep), local_voxels=len(local), vertical_to_horizontal_std=ratio, reason='admitted' if keep else 'insufficient_vertical_structure'))
    return (accepted, audit)
