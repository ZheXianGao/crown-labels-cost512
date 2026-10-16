from crown_labels.core.seed_config import LabelConfig

def geometry_seed_config(base):
    return LabelConfig(target_height=base.target_height_m, grid_resolution=base.raster_m, seed_scales=(0.7, 1.1, 1.5), seed_min_separation=2.0)
