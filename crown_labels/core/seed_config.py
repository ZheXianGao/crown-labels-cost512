from dataclasses import dataclass

@dataclass
class LabelConfig:
    target_height: float = 2.5
    height_mode: str = 'object'
    vegetation_floor: float = 0.5
    object_height_quantile: float = 0.95
    seed_height_radius: float = 0.8
    height_min_points: int = 6
    height_min_xy_cells: int = 4
    low_crown_max_path: float = 1.2
    near_ground_background_height: float = 0.5
    background_crown_buffer: float = 0.6
    min_feret: float = 1.5
    grid_resolution: float = 0.2
    seed_scales: tuple = (0.4, 0.7, 1.0)
    seed_match_radius: float = 1.0
    seed_min_separation: float = 1.5
    seed_min_prominence: float = 0.3
    seed_min_scale_fraction: float = 2 / 3
    chm_nodata: float = -9999.0
    voxel_xy: float = 0.35
    voxel_z: float = 0.5
    max_graph_nodes: int = 35000
    graph_neighbors: int = 10
    graph_radius: float = 1.2
    graph_sigma: float = 0.65
    graph_lambda: float = 1.0
    graph_iterations: int = 50
    graph_tolerance: float = 0.0001
    visibility_depth: float = 0.65
    cleanup_sigma: float = 0.22
    cleanup_max_change: float = 0.4
    confidence_threshold: float = 0.5
    core_erosion: float = 0.2
    background_height_margin: float = 0.4
    background_min_returns: int = 3
