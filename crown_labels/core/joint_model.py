from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class JointConfig:
    fit_radius_m: float = 2.0
    support_extension_m: float = 5.0
    covariance_floor_m: float = 0.35
    covariance_max_m: float = 8.0
    max_mahalanobis2: float = 16.0
    gaussian_cost_scale: float = 8.0
    background_cost: float = 0.8
    graph_strength: float = 1.5
    ff_edge_attenuation: float = 0.5
    label_cost_voxels: float = 512.0
    max_joint_candidates: int = 300
    max_passes: int = 3
    solver_seconds: float = 120.0
    core_margin: float = 0.1

def gaussian_model(sample, cfg):
    center = np.median(sample, axis=0)
    delta = sample - center
    radius = np.linalg.norm(delta, axis=1)
    cap = max(float(np.quantile(radius, 0.9)), cfg.covariance_floor_m)
    delta = delta * np.minimum(1.0, cap / np.maximum(radius, 1e-08))[:, None]
    cov = delta.T @ delta / max(len(delta), 1)
    cov = 0.9 * cov + 0.1 * np.eye(3) * np.trace(cov) / 3
    value, vectors = np.linalg.eigh(cov)
    value = np.clip(value, cfg.covariance_floor_m ** 2, cfg.covariance_max_m ** 2)
    cov = vectors * value @ vectors.T
    inverse = vectors / value @ vectors.T
    return (center, cov, inverse, float(np.log(value).sum()))
