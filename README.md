# LiDAR树冠标签：实例代价512版

从原始LiDAR点云生成树冠实例标签，再导出RGB模型使用的dense训练裁块。这是2026-10-15生成批次的独立代码整理版，不是旧r6/FF3D流程。

固定参数：0.1米体素与投影网格、2米树顶邻域、候选准入比0.7、实例激活代价512、图项系数1.5。512是实例代价；训练裁块也恰好为512像素，两者不是同一个参数。

## 给老师讲流程

先读[流程与详细伪代码](docs/ALGORITHM.md)：展开了三维建图、高斯拟合、联合候选和图割优化。代码入口[`worker.py::run()`](crown_labels/worker.py)也按六步重新排开。

## 方法

1. 用DTM将原始点高程归一化，进行一次0.1米体素化；提取真实点树顶并筛选候选。
2. 构建单冠、双冠和层级多冠假设，以高斯数据项、图平滑项和实例激活项联合优化。
3. 满足完整收敛检查后，进行三维清理、投影及二维轮廓整理。
4. 保留完整实例ID。RGB有效域内使用前景/背景dense监督；无效影像忽略。旧V筛选和unknown分区不包含在本仓库。

高度为相对DTM的离地高度，不是原始海拔。高度可靠域单独保存，不由实例mask推断。当前生成器按0.1米格的上表面回波、至少3次观测及DTM观测支持构造高度。历史paired640实验的RGB、高度和有效域由提供的模板原样保留，不能把两种高度来源混称为同一批。

## 安装

建议Linux、Python 3.11或3.12。标签生成仅使用CPU，不需要SAM3、GPU或模型权重。

```bash
pip install -e .
python -m crown_labels check
python -m unittest discover -s tests -v
```

## 输入与生成

输入为LAS/LAZ、同坐标系DOM，以及DTM缓存。DTM缓存NPZ需包含`dtm`、六参数`transform`和布尔`ground_observed`。这里不重新估计地面模型。

复制`examples/plan.example.json`填写自己的文件、坐标和SHA256。示例坐标是虚构值。核心1024×1024、每边160像素缓冲时，处理尺寸1344×1344；运行配置须明确记录。

```bash
python -m crown_labels generate --plan /data/plan.json --out /data/generated --workers 4
```

每个窗口写入阶段状态、候选、能量轨迹、收敛结果及完整实例标签。相同输入重跑时读取已完成结果；失败结果保留。`--workers`自行按内存设置。不要向现有运行目录换入另一份计划。

可选的农田/电线等地理排除区放在输出目录的`review_policy.json`；默认为空。它们是额外数据质检，不应被写成算法自动判断能力。规则中每个外部mask都需提供`mask_sha256`；仓库不包含项目的真实质检mask或坐标。

## 训练数据导出

```bash
# 每个1024母窗导出4个不重叠512裁块，按整母窗做质量检查
python -m crown_labels build-crops --run /data/generated --out /data/dense

# 复现历史640位置：提供原来的裁块模板，只替换实例标签
python -m crown_labels build-crops --run /data/generated --out /data/paired640 --paired-template /data/template/manifest.json
```

RGB先用GDAL采到0.1米、512×512像素，对应51.2×51.2米。`datasets.load_crop`再将RGB双线性放大至1008；该函数返回原始uint8 RGB，模型自己的归一化只执行一次。实例ID不做双线性插值。

默认保留历史8像素工程边缘忽略；它不是不确定区域筛选。所有与有效域相交的实例保留，不截断实例列表；超过200目标时拒绝进入当前训练接口。新增1024母窗按4块导出；历史640来自128个母窗各5个重叠位置，须使用模板，不能冒称640个独立区域。

主要数组：`rgb`、`rgb_valid`、`instance_ids`、`valid_pixels`、`ignore_pixels`、`reliable_background`、`height`、`height_valid`、`height_bin`。背景只在有效域内定义为ID=0。高度缺测不会被填成0。

仓库包含标签生成、数据导出和数据读取接口，不包含SAM3官方代码、微调权重、实验数据或训练队列。SAM3训练期可使用高度辅助损失；推理输入是否仅RGB取决于模型实现，不能由标签生成器保证。

## 来源与限制

`SOURCE_MANIFEST.json`记录提取函数及生产源码指纹；`SOURCE_LOCK.json`绑定公开文件。整理只移除了实验运行入口和旧监督导出接口，求解器的0号背景状态仍保留，不能删除背景竞争来模拟“去unknown”。

本代码不承诺在新区域上的精度；参数曾在开发区域调整。应使用完整标注、空间隔离的新区域评价。公开代码不包含真实影像、点云、人工标注、地块坐标或访问凭据。原始输入和质量排除记录需由使用者保存。

代码可公开查看；具体使用许可证由作者另行确定。NumPy、SciPy、Rasterio等依赖按各自许可证安装，未复制其实现。
