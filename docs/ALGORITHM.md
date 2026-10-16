# 实例代价512：算法伪代码

## 1. 总流程

```python
def generate_training_labels(lidar, dtm, dom, window, crop_plan):
    # 1. 原始LAS/LAZ提供世界XYZ，不提供已归一化树高。
    # DTM、DOM、点云需处于相同平面坐标系，并采用相容的高程基准。
    raw_xyz = read_points(lidar, region=window.with_halo)  # [N,3]，单位m
    points = initialize_points(raw_xyz, dtm, window)
    domain = read_spatial_domain(dom, window)

    # 2. 体素输入只保留0.5<h<=60m且影像空间有效的点。
    # points仍保留所有原始回波，用于二维计数及最高回波投影。
    # 一个体素是0.1m×0.1m×0.1m的小盒子，不是一个二维像元。
    vegetation = select_vegetation(points, domain)
    voxels = voxelize(vegetation, size=0.1)
    graph = make_graph(voxels.xyz, radius=0.8, neighbours=12)
    edge_weight = potts_weights(graph, strength=1.5, length_scale=0.45)
    observation = rasterize_observations(points, resolution=0.1)

    # 3. 每体素选一条真实最高回波；树顶检测再从这些代表点中找峰。
    # 最高代表点不等于树顶；只有通过局部最高、突出度、支撑检查才成为种子。
    peaks = detect_treetops(voxels.highest_returns)
    peaks = admit_by_vertical_structure(peaks, voxels.highest_returns, ratio=0.7)
    leaves, paths = build_single_crowns(voxels, peaks, graph, observation)

    if leaves:
        # 4. 每个单冠拟合一个稳健高斯；先加入可竞争的双冠假设。
        pairs = propose_pairs(leaves, paths, max_pairs=300)
        candidates = leaves+pairs
        unary = initial_gaussian_costs(voxels, candidates, paths)

        # 5. 用现有三维图的高度层级提出多冠假设。
        groups = propose_hierarchy(graph, voxels.height_agl, leaves, unary)
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

    # 7. 三维标签投到0.1m二维格。这里每格按原始world_z选最高点。
    # 若最高点没有植被体素或已被分到背景，该格保持背景，不改取较低点。
    ids = project_top_returns(points, voxels.point_mapping, labels, resolution=0.1)
    ids, domain = apply_geographic_exclusions(ids, domain, window.review_masks)
    ids = repair_small_gaps(ids, max_area=1.0, max_distance=0.4)
    ids = clean_partition(ids, domain, recipe="mild_then_compact")
    ids = finish_surface(ids, domain)

    # 8. 实例标签与高度目标各自构建。高度来自原始观测，不从树冠ID反推。
    # 有效0m与缺测不同：前者保留0，后者保存NaN并将height_valid设为False。
    height, height_valid = build_height_target(observation, domain)
    save_parent(ids, height, height_valid, domain, window)

    # 9. 按固定位置裁512块；保留所有与裁块相交的实例。
    samples = []
    for crop in crop_plan:
        sample = make_dense_crop(dom, ids, height, height_valid, domain, crop)
        samples.append(sample)
    return save_manifest(samples, grid_m=0.1, crop_size=512, model_size=1008)
```

## 2. 原始点云初始化与字段定义

```python
def initialize_points(raw_xyz, dtm, window):
    # raw_xyz[n] = [X,Y,Z]：第n条原始回波的地图坐标及原始高程，均为m。
    # Z是测得的高程，不是树高；N是该处理窗口内原始回波数量。
    world_xy = raw_xyz[:, :2]                       # [N,2]，世界平面坐标
    world_z = raw_xyz[:, 2]                         # [N]，原始高程

    # DTM给出每条回波所在位置的地面高程。
    # 按DTM自身网格双线性采样；没有地面值的位置返回NaN，不补成0。
    ground_z = sample_raster(dtm, xy=world_xy, interpolation="bilinear")
    height_agl = world_z-ground_z                   # [N]，离地高度
    # AGL = above ground level。例：Z=112m、地面=100m，则离地12m。
    # 这里是每条回波的离地高度，不是已经估计好的整棵树高。

    # 将世界平面坐标平移到处理窗口左上角；没有缩放坐标，也没有重采样点云。
    # 局部x向右增大，局部y向下增大，与栅格列/行方向一致。
    x0, y0 = window.top_left_world_xy
    local_xy = column_stack(world_xy[:, 0]-x0, y0-world_xy[:, 1])
    # local_xy[n]=[3.2,5.7]：该点位于左上角向右3.2m、向下5.7m。

    # 建图保留三维空间的高程差，仅减去全窗口统一常数以减小数值量级。
    # relative_z与height_agl不同：坡地上地面高度随位置变，统一常数不随点变。
    relative_z = world_z-world_z.min()

    return PointCloud(
        point_id=np.arange(len(raw_xyz)),          # 回波原始行号，筛选后不重编号
        raw_count=len(raw_xyz),                    # 原始N，作为整窗映射的长度
        world_xy=world_xy, world_z=world_z,         # 原坐标，用于DTM采样/最高点投影
        local_xy=local_xy,                         # 局部平面位置，用于体素/栅格定位
        ground_z=ground_z, height_agl=height_agl,   # 地面高程与离地高度分别保留
        relative_z=relative_z,                     # 用于建图的第三坐标
    )


def select_vegetation(points, domain):
    # points[keep]表示同时筛选所有逐点字段；point_id仍指向原始回波行号。
    # 不把地面或缺测点删出原始points，只决定哪些点参加植被体素建模。
    keep = is_finite(points.height_agl)
    keep &= (points.height_agl > 0.5) & (points.height_agl <= 60)
    keep &= sample_grid(domain.valid, points.local_xy)
    return points[keep]
```

## 3. 体素与二维观测

```python
def voxelize(points, size=0.1):
    # points是筛选后的植被回波，所有逐点字段一一对应。
    # 按(local_x,local_y,height_agl)分0.1m三维体素；分组索引封装在helper中。
    groups = group_by_voxel(points.local_xy, points.height_agl, size)
    mean_xy = group_mean(points.local_xy, groups)          # [M,2]，体素内平均位置
    mean_height_agl = group_mean(points.height_agl, groups) # [M]，体素内平均离地高度

    # highest_index是M个索引：每个体素内height_agl最大的原始点；并列取首个。
    # points[highest_index]保留这些真实点的坐标、高度、point_id等字段。
    # 它不是M个高度数，也不是每棵树的最高点，更不是整个窗口的最高点。
    highest_index = group_argmax(points.height_agl, groups, ties="first")
    highest_returns = points[highest_index]
    # 例：同体素3点height_agl=[10.00,10.04,10.08]m：
    # mean_height_agl=10.04m；highest_returns保留离地10.08m的那条真实回波。

    # 节点是体素内点的均值，不是体素盒子的几何中心，也不使用最高点代替均值。
    # 前两坐标为mean_xy，第三坐标为平均relative_z；距离和高斯在此XYZ上计算。
    xyz = column_stack(mean_xy, group_mean(points.relative_z, groups))  # [M,3]
    support = count_neighbours(xyz, radius=0.8, max_neighbours=16, include_self=True)
    # support统计附近体素节点（包含自身）；不是0.1m二维格内的原始回波数。

    # 总长N的映射：入选原始点→所属体素编号，未入选点→-1。
    # 后续投影先选最高原始回波，再用这个映射找到它的体素标签。
    point_mapping = map_raw_points_to_voxels(
        point_ids=points.point_id, group_ids=groups.point_to_voxel,
        raw_count=points.raw_count, missing=-1,
    )

    return Voxels(
        xyz=xyz, local_xy=mean_xy, height_agl=mean_height_agl,
        highest_returns=highest_returns, local_support=support,
        point_mapping=point_mapping,
    )


def rasterize_observations(points, resolution):
    # 输入整窗全部原始points，不只取植被体素内的点。
    # 这是二维0.1m格，同一格可能有很多不同高度层的三维体素。
    cells = group_by_grid(points.local_xy, resolution)
    return Observation(
        # 每二维格最大的离地高度；用于上表面支撑检查及独立高度监督。
        upper_height_agl=group_max(points.height_agl, cells),
        # 每格原始回波数量；不同于上面的三维邻居数support。
        count=group_count(cells),
        # 每格原始高程Z最大的回波索引，供二维实例投影。
        # 坡地上最大world_z与最大height_agl不保证是同一条回波。
        top_return_index=group_argmax(points.world_z, cells, ties="first"),
        # 读取对应DTM格的真实观测支持，不以插值值非空冒充真实观测。
        ground_observed=read_dtm_observation_support(cells),
    )
```

## 4. 树顶检测与准入

```python
def detect_treetops(highest_returns):
    # 只从离地至少2.5m的体素代表点找峰，mean_height_agl不参与峰值替代。
    points = highest_returns[highest_returns.height_agl >= 2.5]
    tree = cKDTree(points.local_xy)
    # 按离地高度降序，用0.5m短名单减少重复邻域查询；它不是最后的2m树顶去重。
    shortlist = suppress_neighbours(points, radius=0.5, order="height_descending")
    peaks = []

    for point in shortlist:
        local = points[tree.query_ball_point(point.local_xy, r=2.0)]
        # 2m范围内存在更高代表点则拒绝；1e-5只是浮点比较容差。
        if any(local.height_agl > point.height_agl+1e-5):
            continue
        top = within_radius(local, point.local_xy, 0.35)
        # 0.35m内至少3个代表点距峰高不超过0.5m，避免孤立尖点直接成为树顶。
        if count(top.height_agl >= point.height_agl-0.5) < 3:
            continue

        # 环带分12扇区；至少6区各有3个点，比较各区95分位高度。
        ring = annulus(local, point.local_xy, inner=1.08, outer=1.8)
        sector_top = sector_quantiles(ring, sectors=12, quantile=0.95, min_points=3)
        if len(sector_top) < 6:
            continue
        # 突出度=当前峰高−环带扇区顶高的35分位，至少0.3m才保留。
        prominence = point.height_agl-np.percentile(sector_top, 35)
        if prominence >= 0.3:
            peaks.append(point)

    return suppress_neighbours(peaks, radius=2.0, order="height_descending")


def admit_by_vertical_structure(peaks, highest_returns, ratio):
    accepted = []
    for peak in peaks:
        local = within_radius(highest_returns, peak.local_xy, 2.0)
        if len(local) < 3:
            continue
        # 局部坐标与离地高度组成[x,y,h]，使用样本方差；不是原始高程Z的方差。
        variance = np.var(column_stack(local.local_xy, local.height_agl), axis=0, ddof=1)
        vertical_ratio = np.sqrt(variance[2]/max(variance[:2].mean(), 1e-12))
        if vertical_ratio >= ratio:
            accepted.append(peak)
    return assign_sequential_ids(accepted, start=1)
```

## 5. 三维邻接图

```python
def make_graph(nodes, radius, neighbours):
    if len(nodes) < 2:
        return empty_graph(len(nodes))

    # k包含自身；每个节点先查0.8m内最多12个其他节点，未找到的邻居距离为inf。
    distance, neighbour = cKDTree(nodes).query(
        nodes, k=min(neighbours+1, len(nodes)), distance_upper_bound=radius,
    )
    # 去掉自身和无穷距离，将i→j与j→i合并为同一无向边。
    # 对称合并是双向kNN的并集，因此最终某些节点的度可能超过12。
    edges = unique_undirected_edges(neighbour, distance, exclude_self=True)
    length = np.linalg.norm(nodes[edges.start]-nodes[edges.end], axis=1)
    return sparse_graph(node_count=len(nodes), edges=edges,
                        values=length, symmetric=True)


def potts_weights(graph, strength, length_scale):
    # 最短路用欧氏边长；分割能量使用另算的Potts权重。
    length = graph.undirected_edge_lengths
    return strength*np.exp(-0.5*(length/length_scale)**2)
```

## 6. 单冠支持与稳健高斯

```python
def build_single_crowns(voxels, peaks, graph, observation):
    upper = maximum_filter(observation.upper_height_agl, footprint="2x2")
    # 离局部上表面最多0.6m、且有邻域支撑的节点可作core；fit不会只限于这一层。
    visible = (voxels.height_agl >= sample_grid(upper, voxels.local_xy)-0.6)
    visible &= voxels.local_support >= 3
    leaves, paths = [], {}

    for peak in peaks:
        # core取树顶平面半径0.6m内的可见节点，至少6个且离地高度95分位>2.5m。
        core = within_radius(voxels, peak.local_xy, 0.6).where(visible)
        if len(core) < 6 or np.quantile(core.height_agl, 0.95) <= 2.5:
            continue

        # 沿现有三维边找距core累计路径≤5m的节点，隔空接近但不连通的冠层不算支持。
        path = multi_source_dijkstra(graph, sources=core.indices, limit=5.0)
        fit = within_radius(voxels, peak.local_xy, 2.0)
        fit = fit.where((fit.local_support >= 3) & is_finite(path[fit.indices]))
        if len(fit) < 6:
            continue

        # core锚定树顶；fit保留周围所有合格高度层，不只取上表面。
        candidate = Crown(id=peak.id, core=core.indices, fit=fit.indices,
                          center=peak.local_xy, members=[peak.id])
        leaves.append(candidate)
        paths[peak.id] = path
    return leaves, paths


def robust_gaussian(sample):
    # sample是拟合节点的三维XYZ：局部x、局部y、原始Z减全窗常数。
    center = np.median(sample, axis=0)
    offset = sample-center
    distance = np.linalg.norm(offset, axis=1)
    cutoff = max(np.quantile(distance, 0.9), 0.35)
    # 以90%分位半径截幅远端偏移，中心仍用原中位数，不再重新估计。
    offset = clip_vector_norm(offset, maximum=cutoff)

    covariance = offset.T @ offset/len(sample)
    # 协方差=0.9×原协方差+0.1×trace(cov)/3×I，避免瘦长/退化点集数值不稳。
    covariance = shrink_covariance(covariance, isotropic_weight=0.1)
    eigenvalue, rotation = np.linalg.eigh(covariance)
    # 特征值是方差，所以下限和上限分别为0.35²、8²，单位m²。
    eigenvalue = np.clip(eigenvalue, 0.35**2, 8.0**2)
    precision = matrix_from_eigenpairs(rotation, 1/eigenvalue)
    logdet = np.log(eigenvalue).sum()
    return Gaussian(center, precision, logdet)


def gaussian_cost(voxels, candidate, path):
    gaussian = robust_gaussian(voxels.xyz[candidate.fit])
    # 马氏距离平方衡量节点相对该冠中心的偏离，考虑各方向尺度与相关性。
    mahal2 = mahalanobis_squared(voxels.xyz, gaussian)
    fit_distance = distance_to_points(voxels.xyz, voxels.xyz[candidate.fit])
    inside = ((mahal2 <= 16) & (fit_distance <= 5.0)
              & within_gaussian_search_radius(voxels, gaussian, maximum=24.0))
    cost = full_cost(len(voxels), value=np.inf)
    # 高斯负对数密度去掉公共常数：0.5×马氏距离平方+0.5×log(det(cov))。
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
        candidate.label = label                 # 0为背景，候选从第1列开始
        path = minimum_member_distance(paths, candidate.members)
        unary[:, label] = gaussian_cost(voxels, candidate, path)
    # 逐节点减去最优有限前景代价、除8并裁到[0,1]；背景列仍保持0.8。
    return normalize_foreground(unary, scale=8.0, clip_range=(0, 1))
```

## 7. 双冠和层级多冠候选

```python
def propose_pairs(leaves, paths, max_pairs):
    proposals = []
    # 双冠只是“这两个峰也许是一棵冠”的备选，不会立即覆盖两个单冠。
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
    # 边高度取两端较低离地高度；以负高度求最小生成森林，相当于先连接高处。
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
        # 成员峰最大平面跨度≤10m且所有单冠共有至少6个可达节点；不额外按谷深筛选。
        if span <= 10.0 and len(common) >= 6:
            children = select_leaves(leaves, members)
            group = union_candidate(*children)
            group.child_labels = [child.label for child in children]
            groups.append(group)
    return unique_candidates(groups)


def equal_mixture_cost(child_costs):
    # 等权soft-min，β=8；所有子冠都不可达的位置仍是无穷代价。
    # 仅在成员单冠的支持域并集上有有限代价，不额外创造可达区域。
    return (-logsumexp(-8*child_costs, axis=1)+np.log(child_costs.shape[1]))/8


def replace_pair_costs(unary, leaves, pairs):
    leaf_label = {leaf.id: leaf.label for leaf in leaves}
    for pair in pairs:
        children = [leaf_label[seed_id] for seed_id in pair.members]
        unary[:, pair.label] = equal_mixture_cost(unary[:, children])
    return unary


def append_group_costs(candidates, unary, groups):
    for group in groups:
        if already_present(group, candidates):
            continue
        cost = equal_mixture_cost(unary[:, group.child_labels])
        group.label = unary.shape[1]
        candidates.append(group)
        unary = append_column(unary, cost)
    return candidates, unary
```

## 8. 局部背景项与三项能量

```python
def add_local_background_penalty(unary, voxels, observation):
    upper = maximum_filter(observation.upper_height_agl, footprint="2x2")
    # 最近值填充只为估计局部背景曲面；不据此填补训练高度或树冠实例。
    surface = nearest_observed_fill(upper, observed=observation.count > 0)
    base = percentile_filter(surface, percentile=20, size=101, mode="nearest")
    # 与5m半径局部20分位上表面比较，不把这个局部曲面叫作DTM地面。
    excess = voxels.height_agl-sample_grid(base, voxels.local_xy)
    penalty = 0.8*np.maximum(0.6/np.maximum(excess, 0.05)-1, 0)
    return add_to_finite_foreground(unary, penalty)


def energy(labels, model):
    data = sum_selected_cost(model.unary, labels)
    different = labels[model.edges.start] != labels[model.edges.end]
    smooth = model.edge_weight[different].sum()
    active = unique(labels[labels > 0])
    # 全图每个活跃前景实例收512一次；条件子图对已在外部激活的实例收0。
    instances = model.activation_costs[active].sum()
    return data+smooth+instances
```

## 9. 图割与父子假设联合优化

```python
def solve_until_full_pass(model):
    # 只用背景和单冠初始化；联合假设通过合并/拆分进入。
    labels = argmin_labels(model.unary, allowed=[0]+model.leaf_labels)
    # 一次外层调用未完整收敛可保存后继续；达到上限仍不收敛就不能导出。
    for continuation in range(30):
        for outer_pass in range(30):
            accepted, inner_complete = 0, True
            # 固定顺序先单冠细化，再逐个联合候选尝试merge/split，不按人工精度挑操作。
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

                # 父候选不能与它的成员单冠同时活跃；两个父候选也不能共享成员。
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
    # 实例在ROI外还有节点时激活费已经支付，ROI内重复收费会歪曲全图目标。
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
            # 候选数不会硬限定成最终树数；512激活项让过多实例受到代价惩罚。
            binary.add_label_costs(problem.activation_costs, labels, alpha)
            network = binary.to_nonnegative_integer_capacities(scale=10000)
            # 最大流给出二元最小割，再从残量网络的源端可达性解码keep/switch。
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

## 10. 清理、投影与独立高度目标

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
        # 删掉无支撑小分量后再次检验整冠；Feret是最大平面宽度，不是半径。
        retained = nodes_assigned_to(labels, identity)
        if (height_quantile(model, retained, 0.95) <= height_q95
                or feret_width(model, retained) < min_feret):
            labels[retained] = 0
    return labels


def project_top_returns(points, point_mapping, labels, resolution):
    cells = group_by_grid(points.local_xy, resolution)
    # 现在按二维格取原始Z最高点，区别于体素内按height_agl挑代表点。
    top = group_argmax(points.world_z, cells, ties="first")
    voxel = point_mapping[top]
    ids = empty_label_raster(cells.shape)
    # 最高原始点未进入植被体素时映射为-1，该格不会向下找另一条回波。
    observed = voxel >= 0
    ids[cells[top[observed]]] = labels[voxel[observed]]
    return ids


def build_height_target(observation, domain):
    valid = domain.rgb_valid & ~domain.forbidden
    # 要求格内至少3条原始回波及DTM观测支持；不是至少3个体素邻居。
    valid &= observation.ground_observed & (observation.count >= 3)
    valid &= is_finite(observation.upper_height_agl) & between(observation.upper_height_agl, -2, 60)
    # 可靠观测允许[-2,60]m；负值在目标中截到0，无可靠观测则保留NaN。
    height = where(valid, np.maximum(observation.upper_height_agl, 0), np.nan)
    return height, valid
```

## 11. 训练裁块

```python
def make_dense_crop(dom, ids, height, height_valid, domain, crop):
    # GDAL采到0.1m网格，512像素对应51.2m；实例ID直接裁取，不插值。
    rgb, rgb_valid = warp_rgb(dom, crop, resolution=0.1, size=512,
                              interpolation="bilinear")
    target = crop_array(ids, crop)
    valid = rgb_valid & crop_array(domain.allowed, crop)
    # 8像素工程边缘只改损失有效域，不把完整实例ID从文件中擦掉。
    valid = ignore_border(valid, pixels=8)
    objects = unique(target[valid & (target > 0)])
    require(len(objects) <= 200)             # 超出接口容量就拒绝，不截断标签

    return TrainingSample(
        rgb=rgb, rgb_model=resize(rgb, size=1008, interpolation="bilinear"),
        instance_ids=target,
        # 每个实例mask保留裁块中的完整形状；valid_pixels另行决定哪些像素计算损失。
        masks=[target == identity for identity in objects],
        valid_pixels=valid, ignore_pixels=~valid,
        background=valid & (target == 0),
        height=crop_array(height, crop),
        height_valid=crop_array(height_valid, crop),
    )
```
