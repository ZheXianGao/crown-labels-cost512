"""Optional user-supplied geographic background exclusions; no data bundled."""
import hashlib,json
import numpy as np
from affine import Affine
from rasterio.warp import reproject,Resampling

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def policy_masks(policy, uav, transform, crs, shape):
    hard = np.zeros(shape, bool)
    whole = np.zeros(shape, bool)
    records = []
    locked = {r['mask_path']: r['mask_sha256'] for section in ('ignore_masks','background_masks','whole_crown_ignore_regions') for r in policy.get(section, [])}
    # Mask coordinates, not historical ID numbers, define the corrections.
    for section in ('ignore_masks', 'background_masks', 'whole_crown_ignore_regions'):
        for row in policy.get(section, []):
            if int(row['uav']) != int(uav):
                continue
            path = row['mask_path']
            assert sha(path) == locked[path], ('Changed policy asset', path)
            with np.load(path) as z:
                src = z['mask'].astype(np.uint8)
            assert list(src.shape) == row['shape']
            dst = np.zeros(shape, np.uint8)
            reproject(src, dst, src_transform=Affine(*row['transform']),
                      src_crs=row['crs'], dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
            mask = dst.astype(bool)
            hard |= mask
            remove_whole = (section == 'whole_crown_ignore_regions'
                            or row.get('whole_crown_exclusion', False)
                            or 'remove_power_lines' in row.get('meaning', [])
                           )
            if remove_whole:
                whole |= mask
            records.append(dict(scene=row['scene'], section=section,
                                pixels=int(mask.sum()), whole_instance=remove_whole,
                                asset_sha256=locked[path]))
    return hard, whole, records

def finish(raw, data, rgb_valid, hard, whole, v):
    # Entire objects are selected BEFORE cutting pixels and before any 2-D cleanup.
    removed = np.unique(raw[whole]); removed = removed[removed > 0]
    forbidden = hard | np.isin(raw, removed)
    ids = raw.copy(); ids[forbidden | ~rgb_valid] = 0
    domain = data['valid'].astype(bool) & rgb_valid & ~forbidden
    ids[~domain] = 0
    ids, gaplog = v.repair(ids, domain, forbidden=forbidden)
    observed = data['projection_voxel'] >= 0
    allowed = domain & ((observed & (data['projection_height'] > .5)) | (ids > 0))
    clean = v.make_fast_cleanup()
    mild = v.CleanupConfig(sigma_m=.15, opening_m=.1, closing_m=.2, max_expansion_m=.2,
                          max_island_m2=.08, max_hole_m2=.08, max_change_fraction=.1)
    ids, mildlog = clean(ids, mild, allowed=allowed)
    ids[~domain] = 0
    ids, compactlog = clean(ids, v.CleanupConfig(**v.CONFIGS['compact_medium']), allowed=domain)
    ids[~domain] = 0
    assert not np.any(ids[forbidden | ~rgb_valid])
    assert set(np.unique(ids)).issubset(set(np.unique(raw)))
    height = data['projection_height'].copy()
    old_ignore = data.get('height_ignore_pixels', np.zeros(raw.shape, bool)).astype(bool)
    height_valid = observed & np.isfinite(height) & rgb_valid & ~forbidden & ~old_ignore
    arrays = dict(candidate_ids=ids, segmentation_valid=rgb_valid.copy(),
                  foreground=(ids > 0) & rgb_valid, background=(ids == 0) & rgb_valid,
                  training_ignore_pixels=np.zeros(raw.shape, bool),
                  geographic_forbidden=forbidden, height=height,
                  height_valid=height_valid, height_ignore_pixels=~height_valid)
    assert np.array_equal(arrays['foreground'] | arrays['background'], rgb_valid)
    assert not np.any(arrays['foreground'] & arrays['background'])
    assert np.array_equal(arrays['height'], data['projection_height'], equal_nan=True)
    return arrays, dict(removed_whole_ids=removed.tolist(), forbidden_pixels=int(forbidden.sum()),
                       gap=gaplog, mild=mildlog, compact=compactlog)
