import numpy as np
from scipy import ndimage as ndi

def _bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return None
    return (int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1)

def sparse_unpadded_edt(mask):
    box = _bbox(mask)
    if box is None:
        return np.zeros(mask.shape, float)
    y0, y1, x0, x1 = box
    if y0 == 0 or x0 == 0 or y1 == mask.shape[0] or (x1 == mask.shape[1]):
        return ndi.distance_transform_edt(mask)
    result = np.zeros(mask.shape, float)
    result[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = ndi.distance_transform_edt(mask[y0 - 1:y1 + 1, x0 - 1:x1 + 1])
    return result
