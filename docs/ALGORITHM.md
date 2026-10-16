# 实例代价512：算法伪代码


## 1. 总流程

```python
def generate_training_labels(lidar, dtm, dom, window, crop_plan):
    # 1. 在带缓冲的处理窗口读取点云，采样地面高程。
    raw = read_points(lidar, region=window.with_halo)
    ground = sample_raster(dtm, xy=raw.xy, interpolation="bilinear")
    points = attach_height(raw, height=raw.z-ground)
    domain = read_spatial_domain(dom, window)

    # 2. 只做一次0.1m体素化；保留均值、最高真实回波及原始点映射。
    vegetation = select_points(points, height_range=(0.5, 60), domain=domain)
    voxels = voxelize(vegetation, size=0.1)
    graph = make_graph(voxels.xyz, radius=0.8, neighbours=12)
    edge_weight = potts_weights(graph, strength=1.5, length_scale=0.45)
    observation = rasterize_observations(points, resolution=0.1)

    # 3. 从真实回波找树顶，检查突出度和三维支撑。
    peaks = detect_treetops(voxels.highest_returns)
    peaks = admit_by_vertical_structure(peaks, voxels.highest_returns, ratio=0.7)
    leaves, paths = build_single_crowns(voxels, peaks, graph, observation)

    if leaves:
        # 4. 每个单冠拟合一个稳健高斯；先加入可竞争的双冠假设。
        pairs = propose_pairs(leaves, paths, max_pairs=300)
        candidates = leaves+pairs
        unary = initial_gaussian_costs(voxels, candidates, paths)

        # 5. 用现有三维图的高度层级提出多冠假设。
        groups = propose_hierarchy(graph, voxels.height, leaves, unary)
        unary = replace_pair_costs(unary, leaves, pairs)
        unary = shift_foreground_minimum(unary)
        unary = add_local_background_penalty(unary, voxels, observation)
        candidates, unary = append_group_costs(candidates, unary, groups)

        # 6. 共同优化体素归属、实例数量和父子候选选择。
        model = GeometryModel(candidates, unary, graph, edge_weight,
                              background_cost=0.8, instance_cost=512)
        labels, converged = solve_until_full_pass(model)
        require(converged)
        labels = clean_3d(labels, model, min_nodes=6,
                          path_limit=8.0, height_q95=2.5, min_feret=1.5)
    else:
        labels = background_labels(voxels)

    # 7. 按每格最高原始回波投影；不向下搜索被遮挡实例。
    ids = project_top_returns(points, voxels.point_mapping, labels, resolution=0.1)
    ids, domain = apply_geographic_exclusions(ids, domain, window.review_masks)
    ids = repair_small_gaps(ids, max_area=1.0, max_distance=0.4)
    ids = clean_partition(ids, domain, recipe="mild_then_compact")
    ids = finish_surface(ids, domain)

    # 8. 实例、高度和有效域分别保存；高度缺测不写成0。
    height, height_valid = build_height_target(observation, domain)
    save_parent(ids, height, height_valid, domain, window)

    # 9. 按固定位置裁512块；保留所有与裁块相交的实例。
    samples = []
    for crop in crop_plan:
        sample = make_dense_crop(dom, ids, height, height_valid, domain, crop)
        samples.append(sample)
    return save_manifest(samples, grid_m=0.1, crop_size=512, model_size=1008)
```


## 2. 点云归一化与体素化

```python
def voxelize(points, size=0.1):
    # 体素按局部(x,y,离地高度)分组；整数编码和分组索引封装在group_by_voxel。
    groups = group_by_voxel(points.local_xy, points.height, size)
    center_xy = group_mean(points.local_xy, groups)
    mean_height = group_mean(points.height, groups)
    highest = group_argmax(points.height, groups)

    # 图的第三坐标保留原高程几何，只减去统一常数；并非直接改成离地高度。
    relative_z = points.z-points.window_min_z
    xyz = column_stack(center_xy, group_mean(relative_z, groups))
    support = count_neighbours(xyz, radius=0.8, max_neighbours=16, include_self=True)

    return Voxels(
        xyz=xyz, height=mean_height, highest_returns=points[highest],
        local_support=support, point_mapping=groups.point_to_voxel,
    )


def rasterize_observations(points, resolution):
    # 统计所有原始回波，而不只统计已筛出的植被点。
    cells = group_by_grid(points.xy, resolution)
    return Observation(
        upper=group_max(points.height, cells),
        count=group_count(cells),
        top_return=group_argmax(points.z, cells),
        ground_observed=read_dtm_observation_support(cells),
    )
```


## 3. 树顶检测与准入

```python
def detect_treetops(highest_returns):
    points = highest_returns[highest_returns.height >= 2.5]
    tree = cKDTree(points.xy)
    shortlist = suppress_neighbours(points, radius=0.5, order="height_descending")
    peaks = []

    for point in shortlist:
        local = points[tree.query_ball_point(point.xy, r=2.0)]
        if any(local.height > point.height+1e-5):
            continue
        top = within_radius(local, point.xy, 0.35)
        if count(top.height >= point.height-0.5) < 3:
            continue

        # 环带分12扇区；至少6区各有3个点，比较各区95分位高度。
        ring = annulus(local, point.xy, inner=1.08, outer=1.8)
        sector_top = sector_quantiles(ring, sectors=12, quantile=0.95, min_points=3)
        if len(sector_top) < 6:
            continue
        prominence = point.height-np.percentile(sector_top, 35)
        if prominence >= 0.3:
            peaks.append(point)

    return suppress_neighbours(peaks, radius=2.0, order="height_descending")


def admit_by_vertical_structure(peaks, highest_returns, ratio):
    accepted = []
    for peak in peaks:
        local = within_radius(highest_returns, peak.xy, 2.0)
        if len(local) < 3:
            continue
        variance = np.var(local.xy_height, axis=0, ddof=1)
        vertical_ratio = np.sqrt(variance[2]/max(variance[:2].mean(), 1e-12))
        if vertical_ratio >= ratio:
            accepted.append(peak)
    return accepted
```


## 4. 三维邻接图

```python
def make_graph(nodes, radius, neighbours):
    if len(nodes) < 2:
        return empty_graph(len(nodes))

    distance, neighbour = cKDTree(nodes).query(
        nodes, k=min(neighbours+1, len(nodes)), distance_upper_bound=radius,
    )
    # 去掉自身和无穷距离，将i→j与j→i合并为同一无向边。
    edges = unique_undirected_edges(neighbour, distance, exclude_self=True)
    length = np.linalg.norm(nodes[edges.start]-nodes[edges.end], axis=1)
    return sparse_graph(node_count=len(nodes), edges=edges,
                        values=length, symmetric=True)


def potts_weights(graph, strength, length_scale):
    # 最短路用欧氏边长；分割能量使用另算的Potts权重。
    length = graph.undirected_edge_lengths
    return strength*np.exp(-0.5*(length/length_scale)**2)
```


## 5. 单冠支持与稳健高斯

```python
def build_single_crowns(voxels, peaks, graph, observation):
    upper = maximum_filter(observation.upper, footprint="2x2")
    visible = (voxels.height >= sample_grid(upper, voxels.xy)-0.6)
    visible &= voxels.local_support >= 3
    leaves, paths = [], {}

    for peak in peaks:
        core = within_radius(voxels, peak.xy, 0.6).where(visible)
        if len(core) < 6 or np.quantile(core.height, 0.95) <= 2.5:
            continue

        path = multi_source_dijkstra(graph, sources=core.indices, limit=5.0)
        fit = within_radius(voxels, peak.xy, 2.0)
        fit = fit.where((fit.local_support >= 3) & is_finite(path[fit.indices]))
        if len(fit) < 6:
            continue

        # core锚定树顶；fit保留周围所有合格高度层，不只取上表面。
        candidate = Crown(core=core.indices, fit=fit.indices,
                          center=peak.xy, members=[peak.id])
        leaves.append(candidate)
        paths[peak.id] = path
    return leaves, paths


def robust_gaussian(sample):
    center = np.median(sample, axis=0)
    offset = sample-center
    distance = np.linalg.norm(offset, axis=1)
    cutoff = max(np.quantile(distance, 0.9), 0.35)
    offset = clip_vector_norm(offset, maximum=cutoff)

    covariance = offset.T @ offset/len(sample)
    covariance = shrink_covariance(covariance, isotropic_weight=0.1)
    eigenvalue, rotation = np.linalg.eigh(covariance)
    eigenvalue = np.clip(eigenvalue, 0.35**2, 8.0**2)
    precision = matrix_from_eigenpairs(rotation, 1/eigenvalue)
    logdet = np.log(eigenvalue).sum()
    return Gaussian(center, precision, logdet)


def gaussian_cost(voxels, candidate, path):
    gaussian = robust_gaussian(voxels.xyz[candidate.fit])
    mahal2 = mahalanobis_squared(voxels.xyz, gaussian)
    fit_distance = distance_to_points(voxels.xyz, voxels.xyz[candidate.fit])
    inside = ((mahal2 <= 16) & (fit_distance <= 5.0)
              & within_gaussian_search_radius(voxels, gaussian, maximum=24.0))
    cost = full_cost(len(voxels), value=np.inf)
    cost[inside] = 0.5*(mahal2[inside]+gaussian.logdet)

    # 无5m三维路径的点不可分给该冠；有路径的边缘允许有限延伸。
    reachable = is_finite(path)
    cost[~reachable] = np.inf
    fringe = reachable & ~is_finite(cost)
    cost[fringe] = 1.0+0.15*path[fringe]
    return cost


def initial_gaussian_costs(voxels, candidates, paths):
    unary = cost_matrix(len(voxels), len(candidates)+1, fill=np.inf)
    unary[:, 0] = 0.8
    for label, candidate in enumerate(candidates, start=1):
        path = minimum_member_distance(paths, candidate.members)
        unary[:, label] = gaussian_cost(voxels, candidate, path)
    return normalize_foreground(unary, scale=8.0, clip_range=(0, 1))
```


## 6. 双冠和层级多冠候选

```python
def propose_pairs(leaves, paths, max_pairs):
    proposals = []
    for left, right in unordered_pairs(leaves):
        separation = np.linalg.norm(left.center-right.center)
        shared = intersect(finite_support(paths[left.id]),
                           finite_support(paths[right.id]))
        if separation > 10.0 or len(shared) < 6:
            continue
        overlap = len(shared)/min(support_size(left, paths), support_size(right, paths))
        proposals.append(Pair(left, right, overlap, separation))

    proposals = sorted(proposals, key=lambda p: (-p.overlap, p.separation))[:max_pairs]
    return [union_candidate(p.left, p.right) for p in proposals]


def propose_hierarchy(graph, height, leaves, unary):
    edge_height = minimum_endpoint_height(graph, height)
    forest = minimum_spanning_forest(graph, edge_cost=-edge_height)
    components = SeedComponents(graph.node_count)
    for leaf in leaves:
        # 一个树顶对应core内的最高节点。
        components.attach_seed(argmax_node(height, leaf.core), leaf.id)

    groups = []
    for edge in sort_edges(forest, by="height_descending"):
        left, right = components.find(edge.start), components.find(edge.end)
        if left == right:
            continue
        members = union(left.seeds, right.seeds)
        both_seeded = bool(left.seeds) and bool(right.seeds)
        components.merge(left, right)
        if not both_seeded or len(members) < 3:
            continue

        span = maximum_peak_distance(leaves, members)
        common = common_finite_support(unary, members)
        if span <= 10.0 and len(common) >= 6:
            groups.append(union_candidate(*select_leaves(leaves, members)))
    return unique_candidates(groups)


def equal_mixture_cost(child_costs):
    # 等权soft-min，β=8；所有子冠都不可达的位置仍是无穷代价。
    return (-logsumexp(-8*child_costs, axis=1)+np.log(child_costs.shape[1]))/8


def replace_pair_costs(unary, leaves, pairs):
    for pair in pairs:
        unary[:, pair.label] = equal_mixture_cost(unary[:, pair.child_labels])
    return unary


def append_group_costs(candidates, unary, groups):
    for group in groups:
        if already_present(group, candidates):
            continue
        cost = equal_mixture_cost(unary[:, group.child_labels])
        candidates.append(group)
        unary = append_column(unary, cost)
    return candidates, unary
```


## 7. 局部背景项与三项能量

```python
def add_local_background_penalty(unary, voxels, observation):
    upper = maximum_filter(observation.upper, footprint="2x2")
    surface = nearest_observed_fill(upper, observed=observation.count > 0)
    base = percentile_filter(surface, percentile=20, size=101, mode="nearest")
    excess = voxels.height-sample_grid(base, voxels.xy)
    penalty = 0.8*np.maximum(0.6/np.maximum(excess, 0.05)-1, 0)
    return add_to_finite_foreground(unary, penalty)


def energy(labels, model):
    data = sum_selected_cost(model.unary, labels)
    different = labels[model.edges.start] != labels[model.edges.end]
    smooth = model.edge_weight[different].sum()
    active = unique(labels[labels > 0])
    instances = model.activation_costs[active].sum()
    return data+smooth+instances
```


## 8. 图割与父子假设联合优化

```python
def solve_until_full_pass(model):
    # 只用背景和单冠初始化；联合假设通过合并/拆分进入。
    labels = argmin_labels(model.unary, allowed=[0]+model.leaf_labels)
    for continuation in range(30):
        for outer_pass in range(30):
            accepted, inner_complete = 0, True
            tasks = refine_tasks(model.leaves)+merge_split_tasks(model.parents)

            for operation, candidate in tasks:
                move = plan_family_move(operation, candidate, labels, model)
                if move is None:
                    continue
                # 合并时移除成员子冠和冲突父冠，恢复冲突父冠的其余子冠。
                # 拆分时移除该父冠，重新放开其所有成员子冠。
                problem = conditional_subgraph(model, labels, move)
                local_labels, complete = alpha_expansion(problem, max_passes=50)
                inner_complete &= complete
                proposal = replace_roi(labels, problem.roi, local_labels)

                if not compatible_parent_children(proposal, model.parents):
                    continue
                if energy(proposal, model) < energy(labels, model)-1e-6:
                    labels = proposal
                    accepted += 1

            if accepted == 0 and inner_complete:
                return labels, True
            if outer_time_limit_reached(seconds=3600):
                break
        save_solver_checkpoint(labels)
    return labels, False


def conditional_subgraph(model, labels, move):
    roi = union_candidate_supports(model, move.affected)
    roi = union(roi, nodes_assigned_to(labels, move.removed))
    allowed = ([0]+labels_present(labels, roi)+move.added)-move.removed
    unary = select_costs(model.unary, nodes=roi, labels=allowed)

    # 固定子图外的标签，把跨边界的Potts代价加进局部unary。
    boundary = edges_crossing(model.graph, roi)
    unary = add_fixed_boundary_cost(unary, boundary, labels, allowed)
    costs = activation_costs(allowed, foreground=512, background=0)
    costs[active_outside_roi(labels, roi, allowed)] = 0
    return LocalProblem(roi, allowed, unary, internal_edges(model, roi), costs,
                        initial=feasible_current_labels(labels, roi, allowed, unary))


def alpha_expansion(problem, max_passes):
    labels = problem.initial
    order = sort_by_finite_support(problem.unary, background_last=True)
    for iteration in range(max_passes):
        accepted = 0
        for alpha in order:
            if inner_time_limit_reached(seconds=60):
                return labels, False
            binary = BinaryEnergy(choices=("keep_current", "switch_to_alpha"))
            binary.add_unary(problem.unary, labels, alpha)
            binary.add_potts(problem.edges, labels, alpha)
            # 辅助节点表达新实例激活及旧实例消失的代价。
            binary.add_label_costs(problem.activation_costs, labels, alpha)
            network = binary.to_nonnegative_integer_capacities(scale=10000)
            flow = maximum_flow(network.graph, network.source, network.sink, method="dinic")
            proposal = decode_min_cut(network, flow, labels, alpha)

            # 图割容量会取整；只有原浮点能量严格下降才接受。
            if energy(proposal, problem) < energy(labels, problem)-1e-7:
                labels = proposal
                accepted += 1
        if accepted == 0:
            return labels, True
    return labels, False
```


## 9. 清理、投影与独立高度目标

```python
def clean_3d(labels, model, min_nodes, path_limit, height_q95, min_feret):
    for identity in active_foreground(labels):
        nodes = nodes_assigned_to(labels, identity)
        core = intersect(model.candidates[identity].core, nodes)
        supported = multi_source_dijkstra(model.graph, sources=core, limit=path_limit)
        parts = connected_components(model.graph.subgraph(nodes))
        for part in parts:
            if len(part) < min_nodes or not any(is_finite(supported[part])):
                labels[part] = 0
        retained = nodes_assigned_to(labels, identity)
        if (height_quantile(model, retained, 0.95) <= height_q95
                or feret_width(model, retained) < min_feret):
            labels[retained] = 0
    return labels


def project_top_returns(points, point_mapping, labels, resolution):
    cells = group_by_grid(points.xy, resolution)
    top = group_argmax(points.z, cells)
    voxel = point_mapping[top]
    ids = empty_label_raster(cells.shape)
    observed = voxel >= 0
    ids[cells[top[observed]]] = labels[voxel[observed]]
    return ids


def build_height_target(observation, domain):
    valid = domain.rgb_valid & ~domain.forbidden
    valid &= observation.ground_observed & (observation.count >= 3)
    valid &= is_finite(observation.upper) & between(observation.upper, -2, 60)
    height = where(valid, np.maximum(observation.upper, 0), np.nan)
    return height, valid
```


## 10. 训练裁块

```python
def make_dense_crop(dom, ids, height, height_valid, domain, crop):
    # GDAL采到0.1m网格，512像素对应51.2m；实例ID直接裁取，不插值。
    rgb, rgb_valid = warp_rgb(dom, crop, resolution=0.1, size=512,
                              interpolation="bilinear")
    target = crop_array(ids, crop)
    valid = rgb_valid & crop_array(domain.allowed, crop)
    valid = ignore_border(valid, pixels=8)
    objects = unique(target[valid & (target > 0)])
    require(len(objects) <= 200)             # 超出接口容量就拒绝，不截断标签

    return TrainingSample(
        rgb=rgb, rgb_model=resize(rgb, size=1008, interpolation="bilinear"),
        instance_ids=target,
        masks=[target == identity for identity in objects],
        valid_pixels=valid, ignore_pixels=~valid,
        background=valid & (target == 0),
        height=crop_array(height, crop),
        height_valid=crop_array(height_valid, crop),
    )
```
