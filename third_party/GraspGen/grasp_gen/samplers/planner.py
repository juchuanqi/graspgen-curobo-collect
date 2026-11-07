# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Planner dispatch: ``--planner diffusion`` (the original sampler) or
# ``--planner graspmoe`` (diffusion union OBB-swept candidates).

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from grasp_gen.grasp_server import GraspGenSampler
from grasp_gen.samplers.graspmoe import run_graspmoe, run_graspmoe_batch
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_planner_on_object(
    object_pc: np.ndarray,
    grasp_sampler: GraspGenSampler,
    planner: str = "graspmoe",
    grasp_threshold: float = -1.0,
    num_grasps: int = 200,
    topk_num_grasps: int = -1,
    moe_num_yaws: int = 36,
    moe_z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    moe_outlier_threshold: float = 0.014,
    moe_outlier_k: int = 20,
    moe_obb_mode: str = "advanced",
    moe_skip_obb_rule: str = "auto",
    moe_obb_density: str = "sparse",
    moe_obb_position_spacing_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[dict]]:
    """Run a planner on one object's PC and return concatenated world-frame
    grasps + per-grasp branch tags.

    Returns:
        grasps_world: (K, 4, 4) np.float32  (diffusion grasps then OBB grasps)
        scores:       (K,)      np.float32
        branch_tags:  list[str] of length K, each "diff" or "obb"
        obb_dict:     {"center", "half_extent", "R"} or None
    """
    if planner == "diffusion":
        grasps_t, scores_t = GraspGenSampler.run_inference(
            object_pc,
            grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
        )
        grasps = grasps_t.cpu().numpy().astype(np.float32) if len(grasps_t) else np.zeros((0, 4, 4), dtype=np.float32)
        scores = scores_t.cpu().numpy().astype(np.float32) if len(scores_t) else np.zeros((0,), dtype=np.float32)
        return grasps, scores, ["diff"] * len(grasps), None

    if planner == "graspmoe":
        out = run_graspmoe(
            object_pc=object_pc,
            grasp_sampler=grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            num_yaws=moe_num_yaws,
            z_offsets_cm=tuple(moe_z_offsets_cm),
            outlier_threshold=moe_outlier_threshold,
            outlier_k=moe_outlier_k,
            obb_mode=moe_obb_mode,
            skip_obb_rule=moe_skip_obb_rule,
            obb_density=moe_obb_density,
            obb_position_spacing_m=moe_obb_position_spacing_m,
        )
        grasps = np.concatenate([out["grasps_diff"], out["grasps_obb"]], axis=0)
        scores = np.concatenate([out["scores_diff"], out["scores_obb"]], axis=0)
        tags = ["diff"] * len(out["grasps_diff"]) + ["obb"] * len(out["grasps_obb"])
        return grasps, scores, tags, out["obb"]

    raise ValueError(f"Unknown planner '{planner}'. Use 'diffusion' or 'graspmoe'.")


def run_planner_on_batch(
    object_pcs: list,
    grasp_sampler: GraspGenSampler,
    planner: str = "graspmoe",
    grasp_threshold: float = -1.0,
    num_grasps: int = 200,
    topk_num_grasps: int = -1,
    moe_num_yaws: int = 36,
    moe_z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    moe_outlier_threshold: float = 0.014,
    moe_outlier_k: int = 20,
    moe_obb_mode: str = "advanced",
    moe_skip_obb_rule: str = "auto",
    moe_obb_density: str = "sparse",
    moe_obb_position_spacing_m: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray, list[str], Optional[dict]]]:
    """Batched form of :func:`run_planner_on_object`. Returns one
    ``(grasps_world, scores, branch_tags, obb_dict_or_None)`` tuple per input
    PC, in input order.

    For the GraspMoE planner this folds the diffusion forward pass into a
    single batched call; the OBB branch and its discriminator scoring stay
    per-object. For the diffusion baseline the inference call is batched via
    :meth:`GraspGenSampler.run_inference_batch`.
    """
    n = len(object_pcs)
    if n == 0:
        return []

    if planner == "graspmoe":
        moe_results = run_graspmoe_batch(
            object_pcs,
            grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            num_yaws=moe_num_yaws,
            z_offsets_cm=tuple(moe_z_offsets_cm),
            outlier_threshold=moe_outlier_threshold,
            outlier_k=moe_outlier_k,
            obb_mode=moe_obb_mode,
            skip_obb_rule=moe_skip_obb_rule,
            obb_density=moe_obb_density,
            obb_position_spacing_m=moe_obb_position_spacing_m,
        )
        results: list = []
        for out in moe_results:
            grasps = np.concatenate([out["grasps_diff"], out["grasps_obb"]], axis=0)
            scores = np.concatenate([out["scores_diff"], out["scores_obb"]], axis=0)
            tags = ["diff"] * len(out["grasps_diff"]) + ["obb"] * len(out["grasps_obb"])
            results.append((grasps, scores, tags, out["obb"]))
        return results

    if planner == "diffusion":
        diff_results = GraspGenSampler.run_inference_batch(
            object_pcs,
            grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
        )
        results = []
        for grasps_t, scores_t in diff_results:
            if len(grasps_t) == 0:
                results.append(
                    (
                        np.zeros((0, 4, 4), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        [],
                        None,
                    )
                )
                continue
            grasps = grasps_t.cpu().numpy().astype(np.float32)
            scores = scores_t.cpu().numpy().astype(np.float32)
            grasps[:, 3, 3] = 1
            results.append((grasps, scores, ["diff"] * len(grasps), None))
        return results

    raise ValueError(f"Unknown planner '{planner}'. Use 'diffusion' or 'graspmoe'.")
