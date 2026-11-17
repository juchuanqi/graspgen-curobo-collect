"""Exercise export gates without loading Isaac or LeRobot."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/lerobot_export"))
from bridge import CAMERAS, read, write
from select_sources import main as select, verify_trust
from runtime.collector.obstacle_scene import planner_obstacles


class LeRobotSelectionTests(unittest.TestCase):
    def test_strict_gate_rejects_legacy_or_missing_static(self):
        with self.assertRaisesRegex(ValueError, 'unverified_motion_version'):
            verify_trust(Path('/missing'), {}, ROOT / 'runtime')
        with self.assertRaisesRegex(ValueError, 'initial_static_not_passed'):
            verify_trust(Path('/missing'), {'success_contract': {'version': 'com_translation_pose_arc_v3'}}, ROOT / 'runtime')

    def test_selector_checks_raw_start_and_retains_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            runtime = ROOT / "runtime"
            output = run / "selection.json"
            plan = read(runtime / "config/scene.json")
            asset = read(runtime / "assets/MANIFEST.json")["objects"]["block55"]
            pose = np.array([*plan["object_reference_xy_m"], -asset["bounds_min"][2], 1., 0., 0., 0.])
            cases = [
                dict(case_id=f"block55_{i}", object_id="block55", fixed_index=i, scene_seed=i,
                     xy_offset_m=[0., 0.], yaw_deg=0.)
                for i in range(5)
            ]
            write(run / "PLAN.json", dict(control_hz=60, rgb_hz=60, camera_count=2,
                                         camera_resolution=[256, 256],
                                         stages=[dict(name="collection", cases=cases)]))
            write(run / "RUN.json", dict(status="completed", validation_requested=True))
            for entry in cases[:-1]:
                index = entry["fixed_index"]
                episode = run / "collection/workers/lane0" / entry["case_id"]
                (episode / "cameras").mkdir(parents=True)
                write(episode / "ENTRY.json", entry)
                write(episode / "METADATA.json", dict(success=True, accepted=True, reason="stable_grasp_hold", actions=3))
                for name in ("VALIDATION", "RAW_FORCE_VALIDATION", "INITIAL_STATE_AUDIT"):
                    if index != 3 or name != "INITIAL_STATE_AUDIT":
                        write(episode / f"{name}.json", {"status": "passed"})
                for name in ("SCENE", "ENVIRONMENT", "NATIVE_PHYSICS", "SOURCE_AT_LAUNCH"):
                    write(episode / f"{name}.json", {})
                native = pose.copy()
                if index == 2:
                    native[0] += .01
                np.savez(episode / "trajectory.npz", object_pose=pose[None], qpos=np.zeros((1, 9)),
                         contact_substep_object_pose_w=native[None, None])
                for camera in CAMERAS:
                    np.save(episode / "cameras" / f"{camera}.npy", np.empty(0))
            args = Namespace(run=run, runtime=runtime, output=output, stage=None, per_object=None,
                             allow_legacy_unverified=True)
            select(args)
            result = read(output)
            self.assertEqual((result["stage"], result["fixed_slots"], result["selected_episodes"]), ("collection", 5, 1))
            self.assertEqual([r.get("exclusion") for r in result["inspected_slots"]], [
                None, "duplicate_measured_start", "requested_initial_state_mismatch",
                "online_initial_audit_missing_or_failed", "missing_fixed_slot",
            ])
            # Reproduce the historical bug: PLAN requests a wall but every
            # recorded scene is clear. Even legacy format mode must reject it.
            registered = read(run / 'PLAN.json')
            registered['obstacle'] = 'wall'
            write(run / 'PLAN.json', registered)
            args.output = run / 'mismatched_selection.json'
            with self.assertRaisesRegex(ValueError, 'No eligible episodes'):
                select(args)
            rejected = read(args.output)
            self.assertEqual(rejected['fixed_slots'], 5)
            self.assertEqual(rejected['selected_episodes'], 0)
            self.assertEqual([r['exclusion'] for r in rejected['inspected_slots']],
                             ['obstacle_scene_mismatch'] * 4 + ['missing_fixed_slot'])
            source = Path(result['episodes'][0]['source'])
            write(source / 'SCENE.json', dict(obstacle_mode='wall', obstacles=planner_obstacles('wall')))
            args.output = run / 'wall_selection.json'
            select(args)
            selected = read(args.output)
            self.assertEqual(selected['selected_episodes'], 1)
            self.assertEqual(selected['obstacle_mode'], 'wall')
            self.assertEqual(selected['episodes'][0]['obstacle_mode'], 'wall')

    def test_selector_rejects_run_without_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write(run / "RUN.json", dict(status="completed", validation_requested=False))
            output = run / "selection.json"
            with self.assertRaisesRegex(ValueError, "validation enabled"):
                select(Namespace(run=run, runtime=ROOT / "runtime", output=output, stage=None, per_object=None))
            self.assertFalse(output.exists())
