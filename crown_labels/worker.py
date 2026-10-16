"""单窗口的标签生成流程。

阅读顺序：run()的六个步骤 → science.py → core中的具体算法。
详细伪代码见 docs/ALGORITHM.md。本文件只编排步骤，不修改算法参数。
"""

from pathlib import Path
import fcntl
import json
import os
import resource
import time
import traceback
from contextlib import contextmanager

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from . import science, prepare, exporter, surface

P = Path(os.environ.get("CROWN_RUN_ROOT", ".")).resolve()


def run(window_name):
    """生成一个窗口的完整实例标签，校验成功后返回导出记录。

    输入：plan.json中的窗口记录、原始点缓存、DTM和DOM。
    输出：母窗标签、四个裁块、阶段审计与result.json。
    """
    science.check_sources()
    job = _load_job(window_name)
    window = job["window"]
    output_dir = P / "runs" / window_name
    output_dir.mkdir(parents=True, exist_ok=True)

    assert job["active_instance_cost"] == 512
    assert job["candidate_admission_min_vertical_ratio"] == 0.7

    # 同一窗口只允许一个进程写入；已完成的结果直接返回。
    with (output_dir / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = _completed_result(output_dir)
        if previous is not None:
            return previous

        progress = _Progress(window_name, output_dir)

        try:
            # 1. 点云准备：DTM归一化、0.1米体素化、树顶检测和候选准入。
            with progress.step("prepare_raw", "prepare"):
                points, treetops, preparation_audit = prepare.prepare(job, output_dir)

            # 2. 建模：三维邻接图、单冠高斯、双冠及层级多冠假设。
            with progress.step(
                "build_new_method", "build",
                nodes=len(points["nodes"]), peaks=len(treetops),
            ):
                if len(points["nodes"]):
                    model, model_audit = science.build(points, treetops, output_dir)
                else:
                    model = dict(pool=[], nodes=points["nodes"])
                    model_audit = dict(empty=True, leaf=0, pair=0, hierarchy=0)

            # 3. 联合优化：数据项 + 邻接平滑项 + 每个活跃实例512的代价。
            with progress.step(
                "optimize", "solve",
                nodes=len(points["nodes"]), candidates=len(model["pool"]),
            ):
                voxel_labels, solver_log = science.solve(model, output_dir)

            assert solver_log["converged_full_pass"]
            assert solver_log["inner_unconverged"] == 0

            # 4. 三维清理：去掉缺少路径支持的小分量和不合格实例。
            with progress.step("clean_3d", "clean3d"):
                cleaned_labels, object_audit = science.clean3d(voxel_labels, model)

            np.save(output_dir / "cleaned_voxel_ids.npy", cleaned_labels)
            science.write(output_dir / "clean3d.json", object_audit)
            del model  # 后续只需要体素标签，及时释放候选模型。

            # 5. 最高回波投影到0.1米平面；应用地理质检，再整理轮廓。
            with progress.step("cleanup_2d_and_policy", "postprocess"):
                raw_projection, arrays, postprocessing_audit = science.postprocess(
                    cleaned_labels, points, window
                )
                allowed = (
                    points["valid"]
                    & points["rgb_valid"]
                    & ~arrays["geographic_forbidden"]
                )
                final_ids, surface_audit = surface.refine(arrays["candidate_ids"], allowed)

                exporter.save_npz(
                    output_dir / "projection_and_cleaned.npz",
                    dict(
                        raw_projection=raw_projection,
                        pre_surface_ids=arrays["candidate_ids"],
                        cleaned_ids=final_ids,
                    ),
                )
                arrays["candidate_ids"] = final_ids
                science.write(output_dir / "surface_audit.json", surface_audit)

            # 6. 导出母窗及裁块；逐块检查ID对应关系和背景定义。
            with progress.step("export_native_DOM", "export"):
                audits = dict(
                    preparation=preparation_audit,
                    model=model_audit,
                    postprocessing=postprocessing_audit,
                    surface=surface_audit,
                    stage_seconds=progress.timings,
                )
                result = exporter.export(arrays, window, solver_log, audits, output_dir)

            with progress.step("verify_export", "verify"):
                _verify_crops(result, output_dir)

            result.update(
                stage_seconds=progress.timings,
                peak_RSS_GiB=_peak_memory_gib(),
                seconds=time.monotonic() - progress.started,
                final_surface_cleaned=True,
                file_audit_passed=True,
            )
            science.write(output_dir / "result.json", result)
            science.write(exporter.O / "review" / window_name / "summary.json", result)
            progress.mark("complete", IDs=result["core_instances"], seconds=result["seconds"])
            return result

        except BaseException:
            science.write(
                output_dir / "error.json",
                dict(name=window_name, traceback=traceback.format_exc()),
            )
            progress.mark("failed")
            raise


# 以下是运行记录和文件校验，不参与几何建模。

def _load_job(window_name):
    with (P / "plan.json").open() as stream:
        plan = json.load(stream)
    return next(job for job in plan["selected"] if job["window"]["scene"] == window_name)


def _completed_result(output_dir):
    status_path = output_dir / "status.json"
    if status_path.exists():
        with status_path.open() as stream:
            status = json.load(stream)
        if status["stage"] == "complete":
            with (output_dir / "result.json").open() as stream:
                return json.load(stream)
    return None


def _verify_crops(result, output_dir):
    checks = []
    with np.load(result["native_path"]) as mother:
        for record in result["crops"]:
            with np.load(record["native_path"]) as crop:
                row, col = record["origin"]
                crop_to_mother = np.r_[0, np.asarray(record["core_instance_ids"], np.int32)]
                assert np.array_equal(
                    crop_to_mother[crop["instance_ids"]],
                    mother["instance_ids"][row:row + 512, col:col + 512],
                )
                assert np.array_equal(
                    crop["reliable_background"],
                    crop["valid_pixels"] & (crop["instance_ids"] == 0),
                )
                checks.append(dict(
                    name=record["name"],
                    core_ID_mapping_exact=True,
                    targets=record["targets"],
                    query_capacity_ok=record["query_capacity_ok"],
                ))
    science.write(output_dir / "file_audit.json", dict(crops=checks, passed=True))


def _peak_memory_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class _Progress:
    def __init__(self, window_name, output_dir):
        self.window_name = window_name
        self.output_dir = output_dir
        self.started = time.monotonic()
        self.timings = {}

    def mark(self, stage, **details):
        elapsed = time.monotonic() - self.started
        science.write(self.output_dir / "status.json", dict(
            name=self.window_name,
            stage=stage,
            elapsed_seconds=elapsed,
            peak_RSS_GiB=_peak_memory_gib(),
            **details,
        ))
        print("STAGE", self.window_name, stage, round(elapsed, 1), flush=True)

    @contextmanager
    def step(self, stage, timing_key, **details):
        self.mark(stage, **details)
        started = time.monotonic()
        yield
        self.timings[timing_key] = time.monotonic() - started
