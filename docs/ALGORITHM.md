# Cost512：Python式算法伪代码

按“点云 → 图 → 单冠高斯 → 联合候选 → 图割 → 训练裁块”的顺序阅读。代码使用NumPy/SciPy的实际操作，中文只作注释。

完整教学文件：[algorithm_walkthrough.py](../examples/algorithm_walkthrough.py)。这是讲解版；生产算法、运行队列和已有标签没有修改。

| 库 | 本流程中的用途 |
|---|---|
| NumPy | 体素分组、数组索引、协方差和能量 |
| scipy.spatial | KDTree邻居查询、树顶邻域 |
| scipy.sparse / csgraph | 稀疏图、Dijkstra、生成森林、Dinic最大流 |
| scipy.ndimage / special | DTM采样、局部滤波、logsumexp |
| Rasterio / GDAL / Affine | 坐标变换、DOM采样和有效域 |
| laspy | 原始LAS/LAZ分块读取（工程输入层） |
| OpenCV | RGB从512放大到1008 |


## 1. 总流程

只在这一段调用工程接口；下面展开这些接口里的计算。生产入口仍是 `python -m crown_labels generate`。

```python
def generate_one_window(job, output_dir):
    # 省略锁、断点、计时及文件校验；各函数是仓库中的真实接口。
    from crown_labels import prepare, science, surface, exporter

    data, peaks, preparation_log = prepare.prepare(job, output_dir)
    if len(data["nodes"]):
        model, model_log = science.build(data, peaks, output_dir)
    else:
        model = dict(pool=[], nodes=data["nodes"])
        model_log = dict(empty=True)

    voxel_ids, solver_log = science.solve(model, output_dir)
    assert solver_log["converged_full_pass"]
    assert solver_log["inner_unconverged"] == 0

    cleaned, object_log = science.clean3d(voxel_ids, model)
    raw_projection, arrays, cleanup_log = science.postprocess(
        cleaned, data, job["window"]
    )
    allowed = data["valid"] & data["rgb_valid"] & ~arrays["geographic_forbidden"]
    arrays["candidate_ids"], surface_log = surface.refine(arrays["candidate_ids"], allowed)
    return exporter.export(
        arrays, job["window"], solver_log,
        dict(preparation=preparation_log, model=model_log,
             postprocessing=cleanup_log, surface=surface_log), output_dir,
    )
```


## 2. 库与数组

`raw[N,3]`是世界XYZ；`nodes[M,3]`是体素几何中心；`heights[M]`是离地高度；`unary[M,K+1]`是候选代价，0列为背景。

```python
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix, triu
from scipy.sparse.csgraph import (
    dijkstra, minimum_spanning_tree, maximum_flow, breadth_first_order,
)
from scipy.special import logsumexp
from crown_labels.core.hierarchy import candidate_families
from crown_labels.core.compatible import validate
from crown_labels.core.joint_model import JointConfig
```


## 3. DTM归一化与一次体素化

DTM按自身分辨率双线性采样。体素按(x,y,离地高度)分组，但建图第三坐标仍取原高程减常数，坡地上这两者不能混写。实际分块读取见 `prepare.py`。

```python
def normalize_and_voxelize(raw, dtm, transform, origin_xy, valid):
    # raw: [N,3] 世界坐标；transform为[a,b,c,d,e,f]，仅示例北向上影像。
    a, b, c, d, e, f = transform
    assert b == d == 0 and a > 0 and e < 0
    ground = nd.map_coordinates(
        dtm, [(raw[:, 1]-f)/e-0.5, (raw[:, 0]-c)/a-0.5],
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    h = raw[:, 2] - ground                  # 离地高度，单位米
    x0, y0 = origin_xy
    xy = np.c_[raw[:, 0]-x0, y0-raw[:, 1]]
    col_row = np.floor(xy/0.1).astype(np.int64)
    assert ((col_row >= 0) & (col_row < np.array(valid.shape[::-1]))).all()
    keep = np.isfinite(h) & (h > 0.5) & (h <= 60)
    keep &= valid[col_row[:, 1], col_row[:, 0]]
    selected = np.flatnonzero(keep)
    pts = np.c_[xy[selected], h[selected]]
    if not len(pts):
        return dict(nodes=np.empty((0, 3)), heights=np.empty(0),
                    highest=np.empty((0, 3)), local_support=np.empty(0, int))

    # 以(x,y,离地高度)做一次0.1m体素分组。
    q = np.floor(pts/0.1).astype(np.int64)
    q -= q.min(axis=0)
    stride = q.max(axis=0)+1
    key = (q[:, 0]*stride[1]+q[:, 1])*stride[2]+q[:, 2]
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    center = np.column_stack([
        np.bincount(inverse, weights=pts[:, j])/count for j in range(3)
    ])
    # 建图使用原高程的均值减全窗常数；不能把第三坐标直接换成离地高度。
    z = np.bincount(inverse, weights=raw[selected, 2]-raw[:, 2].min())/count
    nodes = np.c_[center[:, :2], z]

    # 每个体素保留一个最高真实回波，供树顶检测使用。
    maximum = np.full(len(count), -np.inf)
    np.maximum.at(maximum, inverse, pts[:, 2])
    top = np.flatnonzero(pts[:, 2] == maximum[inverse])
    top_index = np.full(len(count), len(pts), np.int64)
    np.minimum.at(top_index, inverse[top], top)
    highest = pts[top_index]

    tree = cKDTree(nodes)
    distance, _ = tree.query(nodes, k=min(16, len(nodes)),
                             distance_upper_bound=0.8, workers=1)
    support = (np.isfinite(distance).sum(axis=1) if distance.ndim == 2
               else np.isfinite(distance).astype(int))
    return dict(nodes=nodes, heights=center[:, 2], highest=highest,
                highest_raw_indices=selected[top_index], local_support=support,
                selected=selected, inverse=inverse)
```


## 4. 树顶检测和候选准入

调用前先取 `highest[:,2] >= 2.5` 的点传给 `detect()`；准入检查 `admit_peaks()` 使用全部体素最高回波。不读取旧标签，不注入兜底树顶。

```python
def detect(pts, rawids):
    x = pts[:, :2]
    h = pts[:, 2]
    tree = cKDTree(x)
    suppressed = np.zeros(len(x), bool)
    short = []
    # 先按高度降序；0.5m短名单减少重复检查。
    for i in np.argsort(-h, kind='stable'):
        if suppressed[i]:
            continue
        short.append(int(i))
        suppressed[tree.query_ball_point(x[i], 0.5)] = True
    peaks = []
    nlocal = 0
    nsupport = 0
    for i in short:
        near = np.asarray(tree.query_ball_point(x[i], 2.0), int)
        if np.any(h[near] > h[i] + 1e-05):
            continue
        nlocal += 1
        dist = np.linalg.norm(x[near] - x[i], axis=1)
        support = int(((dist <= 0.35) & (h[near] >= h[i] - 0.5)).sum())
        if support < 3:
            continue
        nsupport += 1
        # 2m局部最高点还需顶部支持与环带突出度；不是所有局部极值都保留。
        ring = (dist >= 1.08) & (dist <= 1.8)
        delta = x[near[ring]] - x[i]
        rh = h[near[ring]]
        if not len(rh):
            continue
        sectors = np.floor((np.arctan2(delta[:, 1], delta[:, 0]) + np.pi) / (2 * np.pi) * 12).astype(int) % 12
        tops = [float(np.quantile(rh[sectors == s], 0.95)) for s in range(12) if (sectors == s).sum() >= 3]
        if len(tops) < 6:
            continue
        prom = float(h[i] - np.percentile(tops, 35))
        if prom < 0.3:
            continue
        peaks.append(dict(raw_index=int(rawids[i]), x=float(x[i, 0]), y=float(x[i, 1]), height=float(h[i]), prominence=prom, support=support))
    accepted = []
    # 最终树顶之间保持至少2m距离。
    for p in sorted(peaks, key=lambda p: p['height'], reverse=True):
        if any((np.hypot(p['x'] - q['x'], p['y'] - q['y']) < 2.0 for q in accepted)):
            continue
        accepted.append(p)
    return (accepted, dict(shortlist=len(short), local_maxima=nlocal, support_passed=nsupport, prominence_passed=len(peaks)))

def admit_peaks(peaks, highest_points, radius_m=2.0, min_vertical_ratio=0.7):
    points = np.asarray(highest_points)
    tree = cKDTree(points[:, :2])
    accepted, audit = ([], [])
    for source_id, peak in enumerate(peaks, 1):
        indices = tree.query_ball_point([peak['x'], peak['y']], radius_m)
        local = points[indices]
        if len(local) >= 3:
            # 2m邻域内，垂直标准差 / 水平标准差；阈值固定0.7。
            variance = np.var(local, axis=0, ddof=1)
            ratio = float(np.sqrt(variance[2] / max(variance[:2].mean(), 1e-12)))
        else:
            ratio = 0.0
        keep = ratio >= min_vertical_ratio
        if keep:
            accepted.append(dict(peak))
        audit.append(dict(source_seed_id=source_id, retained=bool(keep), local_voxels=len(local), vertical_to_horizontal_std=ratio, reason='admitted' if keep else 'insufficient_vertical_structure'))
    return (accepted, audit)
```


## 5. 三维建图

路径计算用欧氏距离图；能量平滑用另算的Potts权重。kNN边对称化后，节点度数可能大于12。

```python
def make_graph(nodes, radius=0.8, neighbours=12):
    n = len(nodes)
    if n < 2 or neighbours < 1:
        return coo_matrix((n, n)).tocsr()

    # 查询0.8m内最近的12个其他节点；距离按三维XYZ计算。
    dist, idx = cKDTree(nodes).query(
        nodes, k=min(neighbours+1, n),
        distance_upper_bound=radius, workers=4,
    )
    src = np.repeat(np.arange(n), dist.shape[1])
    dst = idx.ravel()
    keep = np.isfinite(dist.ravel()) & (src != dst)

    # 将i→j和j→i合为一条无向边，去重后再双向写入CSR。
    pairs = np.unique(np.sort(np.c_[src[keep], dst[keep]], axis=1), axis=0)
    i, j = pairs.T
    length = np.linalg.norm(nodes[i]-nodes[j], axis=1)
    return coo_matrix(
        (np.r_[length, length], (np.r_[i, j], np.r_[j, i])), shape=(n, n)
    ).tocsr()


def potts_edges(graph):
    edge = triu(graph, k=1).tocoo()          # 每条无向边只计一次
    weight = 1.5*np.exp(-0.5*(edge.data/0.45)**2)
    return edge.row, edge.col, weight
```


## 6. 单冠锚点与高斯拟合域

`data["upper"]`是所有原始回波的每格最大离地高度；`local_CHM`是历史变量名，本版树顶来自真实点。`paths`为示意保存完整距离向量，生产版只存有限索引和距离。

```python
def single_candidates(data, peaks, graph):
    nodes, h = data["nodes"], data["heights"]
    xy_tree = cKDTree(nodes[:, :2])
    col_row = np.floor(nodes[:, :2]/0.1).astype(int)
    col, row = col_row.T
    upper = nd.maximum_filter(data["upper"], footprint=np.ones((2, 2), bool),
                              mode="nearest")
    visible = data["valid"][row, col] & (h > 0.5)
    visible &= (h >= upper[row, col]-0.6) & (data["local_support"] >= 3)
    leaves, paths = [], {}
    for seed_id, peak in enumerate(peaks, 1):
        center = np.array([peak["x"], peak["y"]])
        core = np.asarray(xy_tree.query_ball_point(center, 0.6), int)
        core = core[visible[core]]
        if len(core) < 6 or np.quantile(h[core], 0.95) <= 2.5:
            continue
        distance = dijkstra(graph, indices=core, directed=True,
                            min_only=True, limit=5.0)
        fit = np.asarray(xy_tree.query_ball_point(center, 2.0), int)
        fit = fit[(h[fit] > 0.5) & (data["local_support"][fit] >= 3)
                  & np.isfinite(distance[fit])]
        if len(fit) < 6:
            continue
        # core只作锚点；拟合使用2m内所有合格冠层回波，不只取顶层。
        leaves.append(dict(source="local_CHM", source_id=seed_id,
                           source_members=[seed_id], center=center, core=core, fit=fit))
        paths[seed_id] = distance
    return leaves, paths
```


## 7. 稳健高斯与单冠数据项

高斯只提供相对空间代价，不直接给最终冠界；有限支持还受5m三维路径约束。

```python
def gaussian_model(sample, cfg):
    # 中位数作中心；对最外10%的偏移截幅，再估计协方差。
    center = np.median(sample, axis=0)
    delta = sample - center
    radius = np.linalg.norm(delta, axis=1)
    cap = max(float(np.quantile(radius, 0.9)), cfg.covariance_floor_m)
    delta = delta * np.minimum(1.0, cap / np.maximum(radius, 1e-08))[:, None]
    cov = delta.T @ delta / max(len(delta), 1)
    # 10%各向同性收缩；特征值限定到0.35²～8²，防止奇异。
    cov = 0.9 * cov + 0.1 * np.eye(3) * np.trace(cov) / 3
    value, vectors = np.linalg.eigh(cov)
    value = np.clip(value, cfg.covariance_floor_m ** 2, cfg.covariance_max_m ** 2)
    cov = vectors * value @ vectors.T
    inverse = vectors / value @ vectors.T
    return (center, cov, inverse, float(np.log(value).sum()))


def gaussian_column(nodes, candidate, distance, cfg=JointConfig()):
    mu, covariance, inverse, logdet = gaussian_model(nodes[candidate["fit"]], cfg)
    radius = min(24.0, np.sqrt(np.linalg.eigvalsh(covariance).max()*16.0))
    possible = np.asarray(cKDTree(nodes).query_ball_point(mu, radius), int)
    delta = nodes[possible]-mu
    mahal2 = np.einsum("ni,ij,nj->n", delta, inverse, delta)
    nearest_fit, _ = cKDTree(nodes[candidate["fit"]]).query(nodes[possible])
    feasible = (mahal2 <= 16.0) & (nearest_fit <= 5.0)
    column = np.full(len(nodes), np.inf, np.float32)
    column[possible[feasible]] = (0.5*mahal2[feasible]+0.5*logdet).astype(np.float32)

    # 高斯不能跨到无3D路径支持的另一片表面；有路径的边缘允许有限延伸。
    reachable = np.isfinite(distance)
    column[~reachable] = np.inf
    missing = reachable & ~np.isfinite(column)
    column[missing] = (1.0+0.15*distance[missing]).astype(np.float32)
    return column
```


## 8. 双冠假设与初始代价矩阵

双冠最多300个，指生成标签的候选池，不是SAM3的query数。先以单冠和双冠各自高斯生成初值，再执行第10节替换双冠数据项。

```python
def pair_candidates(leaves, paths):
    proposals = []
    for i, left in enumerate(leaves):
        ls = np.flatnonzero(np.isfinite(paths[left["source_id"]]))
        for right in leaves[i+1:]:
            separation = np.linalg.norm(left["center"]-right["center"])
            if separation > 10.0:
                continue
            rs = np.flatnonzero(np.isfinite(paths[right["source_id"]]))
            shared = np.intersect1d(ls, rs, assume_unique=True)
            if len(shared) < 6:
                continue
            overlap = len(shared)/min(len(ls), len(rs))
            proposals.append((-overlap, separation, left, right))

    pairs = []
    for index, (_, _, left, right) in enumerate(
            sorted(proposals, key=lambda p: (p[0], p[1]))[:300]):
        pairs.append(dict(
            source="joint_CHM", source_id=100000+index,
            source_members=[left["source_id"], right["source_id"]],
            center=(left["center"]+right["center"])/2,
            core=np.union1d(left["core"], right["core"]),
            fit=np.union1d(left["fit"], right["fit"]),
        ))
    return pairs


def initial_unary(nodes, pool, paths):
    unary = np.full((len(nodes), len(pool)+1), np.inf, np.float32)
    unary[:, 0] = 0.8                      # 0号背景参与竞争
    if not pool:
        return unary
    for k, candidate in enumerate(pool, 1):
        distance = np.minimum.reduce([paths[s] for s in candidate["source_members"]])
        unary[:, k] = gaussian_column(nodes, candidate, distance)
    best = unary[:, 1:].min(axis=1)
    for k in range(1, unary.shape[1]):
        good = np.isfinite(best) & np.isfinite(unary[:, k])
        unary[good, k] = np.clip((unary[good, k]-best[good])/8.0, 0, 1)
    return unary
```


## 9. 层级多冠假设

实际调用 `groups, audit = get_groups(model)`。先在已归一化的初始矩阵上取有限支持，再改写双冠/背景代价。谷深只记录，不另设谷深阈值。

```python
def get_groups(model, support_extension_m=5.0, min_seed_nodes=None):
    started = time.monotonic()
    h = np.asarray(model['heights'])
    n = len(h)
    a = model['edge_i']
    b = model['edge_j']
    # 边高度取两个端点的较低高度。
    height = np.minimum(h[a], h[b])
    assert (height > 0).all() and np.isfinite(height).all()
    graph = coo_matrix((-height, (a, b)), shape=(n, n)).tocsr()
    # 负高度的最小生成森林，相当于按高→低连接已有节点。
    mst = minimum_spanning_tree(graph).tocoo()
    order = np.argsort(mst.data, kind='stable')
    aa = mst.row[order]
    bb = mst.col[order]
    hh = -mst.data[order]
    min_seed_nodes = int(model['base'].min_seed_nodes if min_seed_nodes is None else min_seed_nodes)
    leaves = {int(p['source_id']): (k, p) for k, p in enumerate(model['pool'], 1) if p['source'].startswith('local_')}
    # 每个树顶用core内最高节点表示；同一代表节点不允许重复。
    representative = {s: int(p['core'][h[p['core']].argmax()]) for s, (_, p) in leaves.items()}
    assert len(set(representative.values())) == len(representative), 'Seed cores have duplicate highest node; explicitly handle coincident seeds before hierarchy'
    members = {node: (s,) for s, node in representative.items()}
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    events = []
    # 并查集记录树顶集合相遇的层级，不添加新的树顶。
    for a, b, saddle in zip(aa, bb, hh):
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = (rb, ra)
        ma = members.pop(ra, ())
        mb = members.pop(rb, ())
        parent[rb] = ra
        size[ra] += size[rb]
        if ma and mb:
            merged = tuple(sorted(ma + mb))
            members[ra] = merged
            lower = min((float(h[representative[s]]) for s in merged))
            events.append(dict(source_members=list(merged), member_count=len(merged), children=[list(ma), list(mb)], saddle_height_m=float(saddle), lower_peak_height_m=lower, valley_depth_m=lower - float(saddle), relative_depth=(lower - float(saddle)) / lower))
        elif ma or mb:
            members[ra] = ma or mb
    support = {s: np.flatnonzero(np.isfinite(model['unary'][:, k])) for s, (k, p) in leaves.items()}
    groups = []
    for event in events:
        ss = event['source_members']
        xy = np.array([leaves[s][1]['center'] for s in ss])
        span = float(pdist(xy).max()) if len(ss) > 1 else 0.0
        # 多冠假设要求所有成员有共同可达支持，不能只靠二维靠近。
        common = None
        for s in sorted(ss, key=lambda s: len(support[s])):
            common = support[s] if common is None else np.intersect1d(common, support[s], assume_unique=True)
            if not len(common):
                break
        common_count = len(common) if common is not None else 0
        reasons = []
        if len(ss) < 3:
            reasons.append('binary_group_original_pairs_retained')
        # 最大树顶跨度10m；至少3成员、至少6个共同支持节点。
        if span > 2 * support_extension_m:
            reasons.append('peak_span_exceeds_existing_2x_support_extension')
        if common_count < min_seed_nodes:
            reasons.append('common_finite_support_below_existing_min_seed_nodes')
        event.update(common_support_nodes=common_count, max_peak_span_m=span, retained=not reasons, rejection_reasons=reasons)
        if event['retained']:
            groups.append(event)
    audit = dict(method='seed-component hierarchy in existing 3D maximin normalized-height filtration', nodes=n, edges=len(model['edge_i']), local_seed_count=len(leaves), hierarchy_events=len(events), retained_groups=len(groups), retained_arity_histogram={str(k): sum((len(g['source_members']) == k for g in groups)) for k in sorted(set((len(g['source_members']) for g in groups)))}, constraints=dict(max_peak_span_m=2 * support_extension_m, min_common_finite_support_nodes=min_seed_nodes, new_valley_threshold=None), all_events=events, seconds=time.monotonic() - started, ground_truth_read=False, existing_pair_candidates_removed=False, labels_modified=False)
    return (groups, audit)
```


## 10. 联合候选数据项

联合列用成员单冠的等权soft-min（β=8），并非重新拟合一个大高斯替代全部成员。多冠元数据仍记录联合高斯；其代价由下面的混合式给出。背景罚项的完整内核见 `core/background.py`。

```python
def local_background_penalty(model, data):
    upper = nd.maximum_filter(data["upper"].astype(float),
                              footprint=np.ones((2, 2), bool), mode="nearest")
    observed = np.isfinite(upper) & (data["counts"] > 0)
    assert observed.any()
    # 只为估计局部背景曲面填最近值；不填实例或高度监督缺测。
    nearest = nd.distance_transform_edt(
        ~observed, return_distances=False, return_indices=True
    )
    surface = upper[tuple(nearest)]
    base = nd.percentile_filter(surface, percentile=20, size=101, mode="nearest")
    col, row = np.floor(model["nodes"][:, :2]/0.1).astype(int).T
    excess = model["heights"]-base[row, col]
    penalty = 0.8*np.maximum(0.6/np.maximum(excess, 0.05)-1, 0)
    for k in range(1, model["unary"].shape[1]):
        ok = np.isfinite(model["unary"][:, k])
        model["unary"][ok, k] += penalty[ok]
    return penalty


def finalize_unary(model, data, groups):
    # 顺序与science.build一致：双冠替换 → 重定零点 → 背景罚项 → 多冠追加。
    unary = model["unary"]
    lookup = {p["source_id"]: k for k, p in enumerate(model["pool"], 1)
              if p["source"].startswith("local_")}
    for k, p in enumerate(model["pool"], 1):
        if not p["source"].startswith("joint"):
            continue
        left, right = [lookup[s] for s in p["source_members"]]
        ok = np.isfinite(unary[:, left]) | np.isfinite(unary[:, right])
        unary[:, k] = np.inf
        unary[ok, k] = (-np.logaddexp(-8*unary[ok, left],
                                    -8*unary[ok, right])+np.log(2))/8

    best = unary[:, 1:].min(axis=1)
    for k in range(1, unary.shape[1]):
        ok = np.isfinite(best) & np.isfinite(unary[:, k])
        unary[ok, k] -= best[ok]             # 第二次只平移，不再除8或截幅

    # 真实库内核：box2上表面、101×101局部20分位数和逐节点罚项。
    # penalty = 0.8*max(0.6/max(height-local_base, 0.05)-1, 0)
    local_background_penalty(model, data)
    seen = {tuple(sorted(p["source_members"])) for p in model["pool"]}
    for group in groups:
        members = tuple(sorted(group["source_members"]))
        if members in seen or len(members) < 3:
            continue
        seen.add(members)
        children = [lookup[s] for s in members]
        parts = [model["pool"][k-1] for k in children]
        column = (-logsumexp(-8*unary[:, children], axis=1)+np.log(len(children)))/8
        candidate = dict(
            source="joint_CHM", source_id=200000+len(model["pool"]),
            source_members=list(members),
            center=np.mean([p["center"] for p in parts], axis=0),
            core=np.unique(np.concatenate([p["core"] for p in parts])),
            fit=np.unique(np.concatenate([p["fit"] for p in parts])),
        )
        model["pool"].append(candidate)
        unary = np.column_stack([unary, column.astype(np.float32)])
    model["unary"] = unary
    model["costs"] = np.r_[0.0, np.full(len(model["pool"]), 512.0)]
    return model
```


## 11. 三项能量

`costs[0]=0`、其余512。512是每个活跃实例的一次激活代价，不是每个体素收512，也不是训练损失权重。

```python
def energy(labels, unary, edge_i, edge_j, weights, costs):
    # 第一项：每个体素选择一个候选的代价。
    data = float(unary[np.arange(len(labels)), labels].sum(dtype=np.float64))
    # 第二项：相邻体素标签不同，付一次Potts边代价。
    pair = float(weights[labels[edge_i] != labels[edge_j]].sum(dtype=np.float64))
    # 第三项：每个实际占用节点的实例只付一次512，背景为0。
    label = float(costs[np.unique(labels)].sum())
    return dict(data=data, pair=pair, label=label, total=data + pair + label, active_labels=int(np.count_nonzero(np.unique(labels))))
```


## 12. α-expansion与整数最大流

一个α步决定每个节点保留旧标签还是换为α；用SciPy Dinic最大流求二元切分。每次接受前重新计算浮点目标，不宣称全局最优。

```python
def expansion(labels, alpha, unary, edge_i, edge_j, weights, costs, scale=10000.0):
    n = len(labels)
    # 每个可变节点只有两种选择：保留原标签，或改成alpha。
    variable = (labels != alpha) & np.isfinite(unary[:, alpha])
    selected = np.flatnonzero(variable)
    nv = len(selected)
    if not nv:
        return (labels.copy(), dict(variables=0, scale=scale, auxiliaries=0))
    lut = np.full(n, -1, np.int32)
    lut[selected] = np.arange(nv)
    u0 = unary[selected, labels[selected]].astype(np.float64)
    u1 = unary[selected, alpha].astype(np.float64)
    rows, cols, capacity = ([], [], [])
    a, b = (lut[edge_i], lut[edge_j])
    # 把Potts项变成二元图割容量，保持同一能量差。
    both = (a >= 0) & (b >= 0)
    diff = labels[edge_i[both]] != labels[edge_j[both]]
    c = weights[both] * (1 - 0.5 * diff)
    offset = -0.5 * weights[both] * diff
    np.add.at(u1, a[both], offset)
    np.add.at(u1, b[both], offset)
    rows.extend([a[both], b[both]])
    cols.extend([b[both], a[both]])
    capacity.extend([c, c])
    for src, dst in ((edge_i, edge_j), (edge_j, edge_i)):
        one = variable[src] & ~variable[dst]
        local = lut[src[one]]
        np.add.at(u0, local, weights[one] * (labels[src[one]] != labels[dst[one]]))
        np.add.at(u1, local, weights[one] * (alpha != labels[dst[one]]))
    # 辅助节点处理实例激活与消失的512代价。
    aux0, aux1, implications = ([], [], [])
    if not np.any(labels == alpha) and costs[alpha] > 0:
        auxiliary = nv + len(aux0)
        aux0.append(0.0)
        aux1.append(float(costs[alpha]))
        implications.append((np.full(nv, auxiliary), np.arange(nv)))
    for old in np.unique(labels[selected]):
        if costs[old] <= 0 or np.any((labels == old) & ~variable):
            continue
        auxiliary = nv + len(aux0)
        aux0.append(float(costs[old]))
        aux1.append(0.0)
        group = np.flatnonzero(labels[selected] == old)
        implications.append((group, np.full(len(group), auxiliary)))
    u0, u1 = (np.r_[u0, aux0], np.r_[u1, aux1])
    # 减共同常数，确保容量非负；只改变图割常数项。
    common = np.minimum(u0, u1)
    u0, u1 = (u0 - common, u1 - common)
    finite_sum = float(u0.sum() + u1.sum() + sum((x.sum() for x in capacity)))
    infinity = finite_sum + 1.0
    # SciPy最大流使用整数容量；缩放并取整，随后回到浮点能量验收。
    actual_scale = min(scale, 2 ** 28 / infinity)
    for src, dst in implications:
        rows.append(src)
        cols.append(dst)
        capacity.append(np.full(len(src), infinity))
    source, sink = (len(u0), len(u0) + 1)
    vertices = np.arange(len(u0))
    rows.extend([np.full(len(u0), source), vertices])
    cols.extend([vertices, np.full(len(u0), sink)])
    capacity.extend([u1, u0])
    scaled = np.rint(np.concatenate(capacity) * actual_scale).astype(np.int64)
    graph = coo_matrix((scaled, (np.concatenate(rows), np.concatenate(cols))), shape=(sink + 1, sink + 1)).tocsr()
    graph.eliminate_zeros()
    # Dinic最大流；残量网络中从source可达的变量保留原标签。
    flow = maximum_flow(graph, source, sink, method='dinic')
    residual = graph - flow.flow
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    reachable = np.zeros(sink + 1, bool)
    reachable[breadth_first_order(residual, source, directed=True, return_predecessors=False)] = True
    proposal = labels.copy()
    proposal[selected[~reachable[:nv]]] = alpha
    return (proposal, dict(variables=nv, scale=actual_scale, auxiliaries=len(aux0), capacity_rounding_bound=0.5 / actual_scale))
def optimize(unary, edge_i, edge_j, weights, costs, initial=None, max_passes=3, time_limit=180.0):
    if np.isnan(unary).any() or np.any(weights < 0) or np.any(costs < 0):
        raise ValueError('Require finite-or-infinite unary, nonnegative Potts edges and label costs')
    labels = np.argmin(unary, axis=1).astype(np.int32) if initial is None else initial.copy()
    current = energy(labels, unary, edge_i, edge_j, weights, costs)
    if not np.isfinite(current['total']):
        raise ValueError('Every current assignment must be feasible')
    start = time.time()
    trace, moves = ([dict(iteration=0, **current)], [])
    converged, timed_out = (False, False)
    min_scale = 10000.0
    support = np.isfinite(unary).sum(0)
    # 先按有限支持数处理前景候选，背景最后。
    order = sorted(range(1, unary.shape[1]), key=lambda k: (-support[k], k)) + [0]
    for iteration in range(1, max_passes + 1):
        accepted = 0
        for alpha in order:
            if time.time() - start > time_limit:
                timed_out = True
                break
            proposal, cut = expansion(labels, alpha, unary, edge_i, edge_j, weights, costs)
            min_scale = min(min_scale, cut['scale'])
            value = energy(proposal, unary, edge_i, edge_j, weights, costs)
            # 只有原始浮点目标严格下降才接受；图割取整本身不作验收。
            if value['total'] < current['total'] - 1e-07:
                changed = int((proposal != labels).sum())
                moves.append(dict(iteration=iteration, alpha=int(alpha), changed_nodes=changed, before=current['total'], after=value['total'], **cut))
                labels, current = (proposal, value)
                accepted += 1
        trace.append(dict(iteration=iteration, accepted=accepted, **current))
        if timed_out:
            break
        if accepted == 0:
            converged = True
            break
    return (labels, dict(trace=trace, moves=moves, converged_full_pass=converged, time_limit_reached=timed_out, max_passes=max_passes, integer_scale_min=min_scale, seconds=round(time.time() - start, 3), objective='sum unary + weighted Potts + cost of every nonempty label', guarantee='Every accepted floating-point objective strictly decreases; no global optimum guarantee'))
```


## 13. 条件子图

子图外标签固定。跨界边和子图外已激活实例都必须计入，否则局部下降不等于全图下降。

```python
def conditional_problem(unary, labels, roi, allowed, adjacency, costs):
    """Exact ROI objective, including fixed boundary and outside activation."""
    lut = np.full(len(labels), -1, np.int32)
    lut[roi] = np.arange(len(roi))
    edges = adjacency[roi].tocoo()
    neighbour = lut[edges.col]
    internal = (neighbour >= 0) & (edges.row < neighbour)
    a, b, w = (edges.row[internal], neighbour[internal], edges.data[internal])
    # 子图外标签固定；跨边界边的代价加入子图unary。
    boundary = neighbour < 0
    external = labels[edges.col[boundary]]
    data = unary[np.ix_(roi, allowed)].astype(np.float64)
    if np.any(allowed == 0):
        empty = ~np.isfinite(data).any(axis=1)
        if np.any(empty):
            data[empty, np.flatnonzero(allowed == 0)[0]] = 0.0
    for j, identity in enumerate(allowed):
        data[:, j] += np.bincount(edges.row[boundary], weights=edges.data[boundary] * (external != identity), minlength=len(roi))
    full_count = np.bincount(labels, minlength=unary.shape[1])
    local_count = np.bincount(labels[roi], minlength=unary.shape[1])
    local_costs = costs[allowed].copy()
    # 如果同一实例在子图外已激活，子图内不能再收一次512。
    local_costs[(full_count - local_count)[allowed] > 0] = 0
    index = np.full(unary.shape[1], -1, int)
    index[allowed] = np.arange(len(allowed))
    # 尽量保留当前标签；已被禁止或不可行的标签才换成局部argmin。
    initial = index[labels[roi]]
    displaced = initial < 0
    if np.any(displaced):
        initial[displaced] = np.argmin(data[displaced], axis=1)
    rows = np.arange(len(initial))
    invalid = ~np.isfinite(data[rows, initial])
    if np.any(invalid):
        finite_rows = np.isfinite(data[invalid]).any(axis=1)
        if np.any(finite_rows):
            bad_rows = np.flatnonzero(invalid)[finite_rows]
            initial[bad_rows] = np.argmin(data[bad_rows], axis=1)
        no_finite = np.flatnonzero(invalid)[~finite_rows]
        if len(no_finite):
            unknown = np.flatnonzero(allowed == 0)
            if len(unknown):
                initial[no_finite] = unknown[0]
            else:
                raise ValueError('No feasible local assignment')
    return (data, a, b, w, local_costs, initial)
```


## 14. 单冠细化、合并和拆分

`candidate_families()`与`validate()`是仓库真实辅助函数，分别构造父子关系、检查活跃假设兼容。生产包装最多续求30次；只有最终完整pass收敛且无内层未收敛才导出。

```python
def delta_energy(labels, proposal, roi, unary, adjacency, costs):
    lookup = np.full(len(labels), -1, np.int32)
    lookup[roi] = np.arange(len(roi))
    edge = adjacency[roi].tocoo()
    j = lookup[edge.col]
    take = (j < 0) | (edge.row < j)
    row = edge.row[take]
    col = edge.col[take]
    weights = edge.data[take]
    old = labels[roi]
    new_neighbors = labels[col].copy()
    inside = lookup[col] >= 0
    new_neighbors[inside] = proposal[lookup[col[inside]]]
    data = float(np.sum(unary[roi, proposal].astype(float) - unary[roi, old].astype(float)))
    pair = float(np.sum(weights * ((proposal[row] != new_neighbors).astype(float) - (old[row] != labels[col]).astype(float))))
    counts = np.bincount(labels, minlength=len(costs))
    after = counts - np.bincount(old, minlength=len(costs)) + np.bincount(proposal, minlength=len(costs))
    activation = float(costs[after > 0].sum() - costs[counts > 0].sum())
    return data + pair + activation

def solve(model, costs, initial, max_passes=30, time_limit=3600.0, callback=None):
    start = time.monotonic()
    unary = model['unary']
    families, parents = candidate_families(model['candidates'])
    parents = {int(k): set(v) for k, v in parents.items()}
    leaves = set(range(1, unary.shape[1])) - set(parents)
    labels = initial.copy()
    validate(labels, families, parents)
    a, b, w = (model['edge_i'], model['edge_j'], model['weights'])
    adj = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(labels), len(labels))).tocsr()
    supports = {k: np.flatnonzero(np.isfinite(unary[:, k])) for k in range(1, unary.shape[1])}
    current = energy(labels, unary, a, b, w, costs)
    trace = [dict(iteration=0, **current)]
    moves = []
    converged = False
    inner_incomplete = 0
    # 固定调度：单冠细化，再逐个父候选尝试合并和拆分。
    tasks = [('refine', k) for k in sorted(leaves)] + [v for k in sorted(parents) for v in [('merge', k), ('split', k)]]
    for iteration in range(1, max_passes + 1):
        accepted = 0
        incomplete = False
        for operation, k in tasks:
            if time.monotonic() - start >= time_limit:
                incomplete = True
                break
            active = set(np.unique(labels))
            active_parents = active & set(parents)
            # 禁止父候选与其子冠同时活跃；有冲突时先移除相关假设。
            if operation == 'refine':
                if any((k in parents[p] for p in active_parents)):
                    continue
                removed = set()
                added = {k}
                affected = {k}
            elif operation == 'split':
                if k not in active:
                    continue
                removed = {k}
                added = parents[k]
                affected = removed | added
            else:
                members = parents[k]
                conflict = {p for p in active_parents if p != k and parents[p] & members}
                removed = members | conflict
                restored = (set().union(*(parents[p] for p in conflict)) if conflict else set()) - members
                added = {k} | restored
                affected = removed | added
            # 只优化受影响候选的支持域，但保留精确的跨界边与激活代价。
            roi = np.unique(np.concatenate([supports[x] for x in affected]))
            roi = np.union1d(roi, np.flatnonzero(np.isin(labels, list(removed))))
            if not len(roi):
                continue
            allowed = np.array(sorted(({0} | set(labels[roi]) | added) - removed), int)
            data, la, lb, lw, lc, seed = conditional_problem(unary, labels, roi, allowed, adj, costs)
            # 局部alpha-expansion：50轮/60秒上限；未收敛必须记录。
            result, log = optimize(data, la, lb, lw, lc, initial=seed, max_passes=50, time_limit=60.0)
            if not log['converged_full_pass']:
                incomplete = True
                inner_incomplete += 1
            candidate = allowed[result]
            if np.array_equal(candidate, labels[roi]):
                continue
            delta = delta_energy(labels, candidate, roi, unary, adj, costs)
            # 计算全图真实能量变化，下降才接受；随后检查父子兼容。
            if delta < -1e-06:
                proposal = labels.copy()
                proposal[roi] = candidate
                validate(proposal, families, parents)
                value = energy(proposal, unary, a, b, w, costs)
                assert abs(value['total'] - current['total'] - delta) < 0.0001, (value['total'], current['total'], delta)
                assert value['total'] < current['total'] - 1e-06
                moves.append(dict(iteration=iteration, operation=operation, candidate=k, delta=delta, changed_nodes=int((candidate != labels[roi]).sum())))
                labels = proposal
                current = value
                accepted += 1
        trace.append(dict(iteration=iteration, accepted=accepted, **current))
        print('PASS', iteration, 'moves', accepted, 'energy', current['total'], 'elapsed', round(time.monotonic() - start), flush=True)
        if callback:
            callback(labels, dict(trace=trace, moves=moves, seconds=time.monotonic() - start))
        # 一整轮无接受且所有内层完整收敛，才标记full-pass收敛。
        if accepted == 0 and (not incomplete):
            converged = True
            break
        if time.monotonic() - start >= time_limit:
            break
    return (labels, dict(converged_full_pass=converged, trace=trace, moves=moves, inner_unconverged=inner_incomplete, families=families, parents={k: sorted(v) for k, v in parents.items()}, seconds=time.monotonic() - start))
```


## 15. 投影、dense目标与RGB采样

投影前的三维清理见 `science.clean3d()`；二维清理见 `science.postprocess()`与`surface.refine()`。二维轮廓整理可能在已有ID间重分边界，不能称作只删碎片。高度/H另存，缺测不填0，人工标注不参与生成。

```python
def project_top_return(raw, raw_to_voxel, cleaned_ids, origin_xy, shape):
    x0, y0 = origin_xy
    xy = np.c_[raw[:, 0]-x0, y0-raw[:, 1]]
    col_row = np.floor(xy/0.1).astype(int)
    flat = col_row[:, 1]*shape[1]+col_row[:, 0]
    order = np.lexsort((-raw[:, 2], flat))   # 每格按原始高程从高到低
    _, first = np.unique(flat[order], return_index=True)
    top_raw = order[first]
    voxel = raw_to_voxel[top_raw]           # 不合格的最高点映射为-1
    good = voxel >= 0
    ids = np.zeros(np.prod(shape), np.int32)
    ids[flat[top_raw[good]]] = cleaned_ids[voxel[good]]
    return ids.reshape(shape)              # 不向下寻找被遮挡的树冠


def dense_crop(ids, rgb_valid, geographic_valid, edge=8):
    # ids保留完整实例；D只决定哪些像素参与训练。
    domain = rgb_valid & geographic_valid
    if edge:
        domain = domain.copy()
        domain[:edge] = domain[-edge:] = False
        domain[:, :edge] = domain[:, -edge:] = False
    keys = np.unique(ids[domain & (ids > 0)])
    return dict(
        instance_ids=ids.copy(), object_ids=keys,
        masks=np.asarray([ids == k for k in keys], bool).reshape(-1, *ids.shape),
        valid_pixels=domain, ignore_pixels=~domain,
        reliable_background=domain & (ids == 0),
    )


def sample_rgb(dom_path, transform, crs):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    import cv2
    # transform固定0.1m网格；512像素对应51.2m，不是100.8m。
    with rasterio.open(dom_path) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=512, height=512,
                       src_nodata=None, resampling=Resampling.bilinear) as vrt:
            rgb = vrt.read([1, 2, 3]).transpose(1, 2, 0)
            alpha = [i+1 for i, kind in enumerate(src.colorinterp) if kind.name == "alpha"]
            valid = (vrt.read(alpha[0], resampling=Resampling.nearest) > 0 if alpha else
                     np.all(vrt.read_masks([1, 2, 3], resampling=Resampling.nearest) > 0, axis=0))
    rgb[~valid] = 0
    rgb_model = cv2.resize(rgb, (1008, 1008), interpolation=cv2.INTER_LINEAR)
    return rgb, valid, rgb_model            # uint8；模型内只归一化一次
```


## 源码对应

| 讲解步骤 | 生产实现 |
|---|---|
| DTM、体素、最高回波 | [`prepare.py`](../crown_labels/prepare.py) |
| 树顶、准入、图 | [`peaks.py`](../crown_labels/core/peaks.py)、[`admission.py`](../crown_labels/core/admission.py)、[`algorithm.py`](../crown_labels/core/algorithm.py) |
| 高斯与初始候选 | [`joint_model.py`](../crown_labels/core/joint_model.py)、[`builder_kernel.py.txt`](../crown_labels/core/builder_kernel.py.txt) |
| 层级与联合代价 | [`proposals.py`](../crown_labels/core/proposals.py)、[`unary.py`](../crown_labels/core/unary.py)、[`science.py`](../crown_labels/science.py) |
| 条件图割与父子假设 | [`joint_solver.py`](../crown_labels/core/joint_solver.py)、[`hierarchy.py`](../crown_labels/core/hierarchy.py)、[`family_solver.py`](../crown_labels/core/family_solver.py) |
| 清理、导出、训练读取 | [`policy.py`](../crown_labels/policy.py)、[`exporter.py`](../crown_labels/exporter.py)、[`datasets.py`](../crown_labels/datasets.py) |

当前输出使用完整实例ID和dense有效域，不包含旧V筛选或unknown类别。0号背景竞争仍是求解器的一部分。历史paired640的RGB、有效域、高度由模板保留；新生成器的高度按0.1m原始上表面观测构建，两者分别记录。
