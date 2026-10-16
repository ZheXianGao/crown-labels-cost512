from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, distance
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

@dataclass(frozen=True)
class Config:
    voxel_m: float = 0.1
    raster_m: float = 0.1
    resolution: float = 0.1
    graph_radius_m: float = 0.8
    check_radius_m: float = 0.6
    neighbours: int = 12
    vegetation_floor_m: float = 0.5
    chm_core_radius_m: float = 0.6
    ff_anchor_radius_m: float = 0.8
    ff_confidence: float = 0.65
    ff_min_views: int = 3
    ff_min_core_area_m2: float = 1.0
    min_candidate_separation_m: float = 1.5
    visible_depth_m: float = 0.6
    min_seed_nodes: int = 6
    path_limit_m: float = 8.0
    core_inset_m: float = 0.2
    min_margin: float = 0.2
    target_height_m: float = 2.5
    min_feret_m: float = 1.5
    background_buffer_m: float = 0.4

def make_graph(nodes, radius, neighbours):
    if not len(nodes):
        return coo_matrix((0, 0)).tocsr()
    distance, index = cKDTree(nodes).query(nodes, k=min(neighbours + 1, len(nodes)), distance_upper_bound=radius, workers=4)
    if distance.ndim == 1:
        return coo_matrix((len(nodes), len(nodes))).tocsr()
    src = np.repeat(np.arange(len(nodes)), distance.shape[1])
    dst, dist = (index.ravel(), distance.ravel())
    keep = np.isfinite(dist) & (src != dst)
    pairs = np.unique(np.sort(np.c_[src[keep], dst[keep]], axis=1), axis=0)
    weights = np.linalg.norm(nodes[pairs[:, 0]] - nodes[pairs[:, 1]], axis=1)
    return coo_matrix((np.r_[weights, weights], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(len(nodes), len(nodes))).tocsr()

def clean_3d(ids, graph, seed_ids, nodes, heights, cfg):
    result = ids.copy()
    records = []
    for identity in np.unique(ids[ids > 0]):
        indices = np.flatnonzero(ids == identity)
        _, component = connected_components(graph[indices][:, indices], directed=False)
        sizes = np.bincount(component)
        direct_seed = seed_ids == identity
        seed_nodes = np.flatnonzero(direct_seed)
        if len(seed_nodes):
            path = dijkstra(graph, directed=False, indices=seed_nodes, min_only=True, limit=cfg.path_limit_m)
            connected = np.isfinite(np.asarray(path))[indices]
        else:
            connected = np.zeros(len(indices), bool)
        anchored = np.bincount(component, weights=seed_ids[indices] == identity) > 0
        supported = np.bincount(component, weights=connected) > 0
        keep = supported[component] & (sizes[component] >= cfg.min_seed_nodes)
        result[indices[~keep]] = 0
        member = indices[keep]
        q95 = float(np.quantile(heights[member], 0.95)) if len(member) else 0.0
        if len(member):
            corners = np.concatenate([nodes[member, :2] + cfg.voxel_m * np.array([dx, dy]) for dx, dy in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))])
            hull = corners[ConvexHull(corners).vertices]
            width = float(distance.pdist(hull).max())
        else:
            width = 0.0
        eligible = q95 > cfg.target_height_m and width >= cfg.min_feret_m
        if not eligible:
            result[member] = 0
        records.append(dict(id=int(identity), before_voxels=len(indices), removed_fragment_voxels=int((~keep).sum()), components=len(sizes), retained_seeded_components=int((anchored & (sizes >= cfg.min_seed_nodes)).sum()), retained_path_supported_components=int((supported & (sizes >= cfg.min_seed_nodes)).sum()), height_q95_m=q95, feret_m=width, eligible=bool(eligible)))
    return (result, records)
