"""Summarize elapsed batch throughput, every attempt and paired initial states."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from common import read, write


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def initial_comparison(a, b):
    if a.get("initial_qpos") is None or b.get("initial_qpos") is None:
        return dict(status="missing")
    joint = max(abs(x-y) for x, y in zip(a["initial_qpos"], b["initial_qpos"]))
    pa, pb = a["initial_object_pose"], b["initial_object_pose"]
    position = math.sqrt(sum((x-y)**2 for x, y in zip(pa[:3], pb[:3])))
    qa, qb = pa[3:], pb[3:]
    dot = abs(sum(x*y for x, y in zip(qa, qb))) / math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
    rotation = 2 * math.acos(min(1., dot))
    return dict(status="matched" if joint <= 1e-5 and position <= 1e-5 and rotation <= 1e-4 else "mismatch",
                max_joint_difference_rad=joint, object_position_difference_m=position,
                object_rotation_difference_rad=rotation)


def summarize(output):
    output = Path(output)
    plan = read(output / "PLAN.json")
    run = read(output / "RUN.json")
    rows, stages, performance = [], [], {}
    for stage in plan["stages"]:
        directory = output / stage["name"]
        stage_receipt = read(directory / "STAGE.json") if (directory / "STAGE.json").exists() else {}
        stage_rows, controls = [], []
        for path in sorted(directory.glob("workers/*/*/METADATA.json")):
            metadata = read(path)
            perf = read(path.with_name("PERFORMANCE.json")) if path.with_name("PERFORMANCE.json").exists() else {}
            performance[(stage["name"], metadata["case_id"])] = perf
            controls.extend(perf.get("execution_step_s", []))
            validations = [path.with_name(name) for name in ("VALIDATION.json", "RAW_FORCE_VALIDATION.json")]
            validated = all(p.exists() and read(p).get("status") == "passed" for p in validations)
            audit = path.with_name("INITIAL_STATE_AUDIT.json")
            audit_status = read(audit).get("status", "missing") if audit.exists() else "missing"
            timing = metadata["timing_s"]
            row = dict(stage=stage["name"], case_id=metadata["case_id"], object_id=metadata["object_id"],
                       success=metadata["success"], validated=validated, initial_state_audit_status=audit_status,
                       reason=metadata["reason"],
                       actions=metadata["actions"], reused_environment=metadata["reused_environment"],
                       initialization_s=metadata["initialization_s"], episode_wall_s=metadata["episode_wall_s"],
                       planning_s=timing["planning"], planner_queue_s=metadata["queue_wait_s"],
                       reset_s=timing["reset_call"], park_s=timing.get("park", 0.),
                       settle_s=timing["settle_controls"], execution_s=timing["physics_camera"],
                       record_s=timing["record"], save_s=timing["save"], episode=str(path.parent))
            stage_rows.append(row)
        rows.extend(stage_rows)
        wall = stage_receipt.get("wall_s")
        successes = sum(r["success"] for r in stage_rows)
        accepted = sum(r["success"] and r["validated"] for r in stage_rows)
        requested_accepted = sum(r["success"] and r["validated"] and
            r["initial_state_audit_status"] == "passed" for r in stage_rows)
        slow = [v for v in controls if v > 1.]
        stages.append(dict(
            stage=stage["name"], status=stage_receipt.get("status", "not_started"),
            planned_slots=len(stage["cases"]), saved_slots=len(stage_rows),
            missing_slots=len(stage["cases"]) - len(stage_rows),
            scored_successes=successes, validated_successes=accepted,
            requested_state_validated_successes=requested_accepted,
            requested_state_validated_successes_per_collection_hour=requested_accepted * 3600 / wall if wall else None,
            initial_state_audits_missing=sum(r["initial_state_audit_status"] == "missing" for r in stage_rows),
            success_fraction_of_planned=successes / len(stage["cases"]),
            stage_wall_s=wall,
            planner_service_startup_s=stage_receipt.get("planner_service_startup_s"),
            completed_slots_per_hour=len(stage_rows) * 3600 / wall if wall else None,
            scored_successes_per_hour=successes * 3600 / wall if wall else None,
            validated_successes_per_collection_hour=accepted * 3600 / wall if wall else None,
            equivalent_collection_minutes_per_scored_success=wall / successes / 60 if wall and successes else None,
            mean_episode_s=mean(r["episode_wall_s"] for r in stage_rows) if stage_rows else None,
            mean_warm_episode_s=mean(warm) if (warm := [r["episode_wall_s"] for r in stage_rows if r["reused_environment"]]) else None,
            mean_initialization_s=mean(first) if (first := [r["initialization_s"] for r in stage_rows if not r["reused_environment"]]) else None,
            planning_queue_s=sum(r["planner_queue_s"] for r in stage_rows),
            controls=len(controls), mean_control_s=mean(controls) if controls else None,
            p50_control_s=percentile(controls, .5), p90_control_s=percentile(controls, .9),
            p99_control_s=percentile(controls, .99),
            over_one_second_time_fraction=sum(slow) / sum(controls) if sum(controls) else None,
            per_object={name: dict(saved=sum(r["object_id"] == name for r in stage_rows),
                                   successes=sum(r["object_id"] == name and r["success"] for r in stage_rows))
                        for name in plan["objects"]},
        ))
    paired = []
    row_lookup = {(r["stage"], r["case_id"]): r for r in rows}
    for case in sorted({r["case_id"] for r in rows if r["stage"] == "cold1"}):
        reference = performance.get(("cold1", case), {})
        for name in ("warm1", "parallel3"):
            if (name, case) in performance:
                before, after = row_lookup[("cold1", case)], row_lookup[(name, case)]
                paired.append(dict(case_id=case, comparison="cold1_vs_" + name,
                    **initial_comparison(reference, performance[(name, case)]),
                    cold_actions=before["actions"], compared_actions=after["actions"],
                    cold_success=before["success"], compared_success=after["success"],
                    cold_episode_plus_attributed_startup_s=before["episode_wall_s"] + before["initialization_s"],
                    compared_episode_plus_attributed_startup_s=after["episode_wall_s"] + after["initialization_s"],
                    cold_execution_s=before["execution_s"], compared_execution_s=after["execution_s"]))
    accepted_unique = sorted({r["case_id"] for r in rows if r["success"] and r["validated"]})
    requested_unique = sorted({r["case_id"] for r in rows if r["success"] and r["validated"] and
                              r["initial_state_audit_status"] == "passed"})
    result = dict(status=run["status"], planned_slots=plan["total_attempt_slots"], saved_slots=len(rows),
                  unique_scenes=plan["unique_scene_count"], unique_validated_successful_scenes=len(accepted_unique),
                  unique_scene_fields_count_case_ids=True,
                  unique_requested_state_validated_successful_case_ids=len(requested_unique),
                  stages=stages, initial_state_pairs=paired,
                  paired_initial_state_mismatches=sum(p["status"] != "matched" for p in paired),
                  collection_wall_s=run["collection_wall_s"], total_with_validation_wall_s=run["total_wall_s"],
                  validated_successful_slots_per_total_hour=sum(r["success"] and r["validated"] for r in rows) * 3600 / run["total_wall_s"],
                  requested_state_validated_successful_slots_per_total_hour=sum(
                      s["requested_state_validated_successes"] for s in stages) * 3600 / run["total_wall_s"],
                  scope=plan.get("scope", "Fixed known-geometry timing comparison; paired repetitions are duplicates"))
    write(output / "SUMMARY.json", result)
    write(output / "EPISODES.json", rows)
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["stage", "case_id"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["GraspGen + cuRobo / Isaac Lab 数据采集", "", f"状态：{run['status']}。固定槽 {plan['total_attempt_slots']}，已保存 {len(rows)}。",
             "", "| 模式 | 保存/计划 | 成功 | 数据验证后成功 | 批次分钟 | 成功条/小时 | 控制步 P50/P90 秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def number(value, digits=2):
        return f"{value:.{digits}f}" if value is not None else "—"
    for stage in stages:
        lines.append(f"| {stage['stage']} | {stage['saved_slots']}/{stage['planned_slots']} | "
                     f"{stage['scored_successes']} | {stage['validated_successes']} | "
                     f"{number(stage['stage_wall_s']/60 if stage['stage_wall_s'] else None)} | "
                     f"{number(stage['scored_successes_per_hour'])} | "
                     f"{number(stage['p50_control_s'],3)}/{number(stage['p90_control_s'],3)} |")
    lines.extend(["", "批次时间包含规划服务和全部仿真进程的启动、执行、原始数据保存及退出。",
                  "CPU数据验证在采集后单独计时，未导出LeRobot或训练视频；SUMMARY.json同时报告包含验证的整体吞吐。",
                  "", f"配对初态检查：{len(paired)} 对，未匹配/缺失 {result['paired_initial_state_mismatches']} 对。",
                  "初态阈值为关节1e-5 rad、物体位置1e-5 m、旋转1e-4 rad；若不匹配，不能只将差异归因于启动复用。",
                  "", f"按案例ID去重的物理验证成功数：{len(accepted_unique)}，不代表实际初态去重数。",
                  f"实际初态核查：通过 {sum(r['initial_state_audit_status'] == 'passed' for r in rows)} / {len(rows)}；"
                  f"缺少核查 {sum(r['initial_state_audit_status'] == 'missing' for r in rows)}。缺少核查不能视为按计划生成。",
                  "测速模式间重复案例不能当成独立新样本。",
                  "", "查看 episodes.csv/EPISODES.json 获得逐物体、逐回合、初始化、落稳、规划排队与执行耗时。",
                  "共享规划服务的排队计入每条规划时间；环境数量是独立进程数，没有声称同进程多环境向量化。"])
    lines.extend(["", "| 模式 | 指定初态且数据验证成功/计划 | 对应条/采集小时 | 复用回合均值秒 | 规划排队均值秒 |",
                  "|---|---:|---:|---:|---:|"])
    for stage in stages:
        lines.append(f"| {stage['stage']} | {stage['requested_state_validated_successes']}/{stage['planned_slots']} | "
                     f"{number(stage['requested_state_validated_successes_per_collection_hour'])} | "
                     f"{number(stage['mean_warm_episode_s'])} | "
                     f"{number(stage['planning_queue_s']/stage['saved_slots'] if stage['saved_slots'] else None)} |")
    if plan["total_attempt_slots"] == 40:
        lines.extend(["", "默认模式：cold1/warm1各8条，parallel3为24条且包含上述8个配对场景。",
                      "三组物体类别比例相同，但全部初态集合不同；不能将整体吞吐差直接称为纯并发因果倍率。",
                      "需完整同场景对照时使用 --full，三组各24条；SUMMARY.json提供逐配对的动作数与耗时。"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.output)
