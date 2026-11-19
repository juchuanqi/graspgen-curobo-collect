"""CPU checks for pose pairing, collision screening and planner fallback."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime"), str(ROOT / "runtime/collector")]
import grasp_symmetry as symmetry
from target_provider import screen_scene_candidates, world_targets
from sim_worker import episode_options


def transform(degrees=0, axis="z"):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return value


def candidate(degrees=170, rank=1):
    local = transform(degrees)
    local[:3, 3] = [.02, -.01, .03]
    return dict(T_object_tcp=local.tolist(), T_object_grasp=local.tolist(),
                original_rank=rank, score=.9, execution_quality={"contact_score": .8},
                pad_span=dict(positive_y_m=.025, negative_y_m=-.02,
                              center_y_m=.0025, positive_samples=9, negative_samples=7),
                width_attempts=[{"valid": True}])


class GraspSymmetryTests(unittest.TestCase):
    def test_sixteen_pairs_keep_tcp_approach_and_right_handed_frames(self):
        parents = [candidate(30 + i * 10, i) for i in range(16)]
        untouched = copy.deepcopy(parents)
        obj = transform(42, "y")
        obj[:3, 3] = [.5, .1, .02]
        current = transform(-23, "x")
        near, far = symmetry.split_candidates(parents, obj, current)
        self.assertEqual((len(near), len(far)), (16, 16))
        self.assertEqual(len({c["candidate_id"] for c in near + far}), 32)
        self.assertEqual(parents, untouched)
        for parent, n, f in zip(parents, near, far):
            a, b = np.asarray(n["T_object_tcp"]), np.asarray(f["T_object_tcp"])
            np.testing.assert_array_equal(a[:, 3], b[:, 3])
            np.testing.assert_array_equal(a[:3, 2], b[:3, 2])
            np.testing.assert_allclose(a[:3, :2], -b[:3, :2], atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(a[:3, :3]), 1.)
            self.assertAlmostEqual(np.linalg.det(b[:3, :3]), 1.)
            self.assertLessEqual(n["rotation_from_current_deg"], f["rotation_from_current_deg"] + 1e-9)
            for c in (n, f):
                expected = np.degrees(Rotation.from_matrix(current[:3, :3].T @ (obj @ np.asarray(c["T_object_tcp"]))[:3, :3]).magnitude())
                self.assertAlmostEqual(c["rotation_from_current_deg"], expected, places=9)
                self.assertEqual(c["parent_original_rank"], parent["original_rank"])

    def test_current_world_pose_changes_the_preferred_variant(self):
        parent = candidate(170)
        near, far = symmetry.split_candidates([parent], np.eye(4), np.eye(4))
        self.assertEqual(near[0]["symmetry_variant"], "z180")
        self.assertAlmostEqual(near[0]["rotation_from_current_deg"], 10.)
        self.assertAlmostEqual(far[0]["rotation_from_current_deg"], 170.)
        near, _ = symmetry.split_candidates([parent], transform(20), transform(-170))
        self.assertEqual(near[0]["symmetry_variant"], "original")
        self.assertAlmostEqual(near[0]["rotation_from_current_deg"], 0.)

    def test_ties_and_extreme_angles_are_stable(self):
        near, far = symmetry.split_candidates([candidate(90)], np.eye(4), np.eye(4))
        self.assertEqual(near[0]["symmetry_variant"], "original")
        for degree in (0, 1e-8, 179.99999999, 180):
            self.assertAlmostEqual(symmetry.rotation_distance_deg(np.eye(4), transform(degree)), degree, places=7)
        self.assertEqual(symmetry.split_candidates([], np.eye(4), np.eye(4)), ([], []))

    def test_flipped_pose_has_explicit_parent_evidence_and_swapped_pad_sides(self):
        parent = candidate()
        near, _ = symmetry.split_candidates([parent], np.eye(4), np.eye(4))
        derived = near[0]
        self.assertIsNone(derived["score"])
        self.assertEqual(derived["parent_model_score"], .9)
        self.assertNotIn("T_object_grasp", derived)
        self.assertEqual(derived["parent_T_object_grasp"], parent["T_object_grasp"])
        self.assertNotIn("execution_quality", derived)
        self.assertEqual(derived["parent_execution_quality"], parent["execution_quality"])
        self.assertEqual(derived["pad_span"]["positive_y_m"], .02)
        self.assertEqual(derived["pad_span"]["negative_y_m"], -.025)
        self.assertEqual(derived["pad_span"]["positive_samples"], 7)

    def test_invalid_pose_and_mode_fail_before_planning(self):
        invalid = [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]),
                   np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                symmetry.split_candidates([candidate()], value, np.eye(4))
        with self.assertRaises(ValueError):
            symmetry.mode("typo")

    def test_each_variant_gets_geometry_checks_and_far_survives_near_rejection(self):
        calls = []
        def geometry(path, opening, *args, **kwargs):
            calls.append(path.copy())
            # Near is the -10-degree pose; reject it at every opening.
            return {"valid": path[-1, 0, 0] < 0, "reason": "test_collision"}
        scene = dict(object_id="unused", object_pose=np.eye(4).tolist(),
                     tcp_pose=np.eye(4).tolist(), grasp_symmetry="near-first")
        with patch("geometry.api.check_gripper_path", side_effect=geometry):
            valid, report = screen_scene_candidates([candidate()], scene)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["symmetry_group"], "far")
        self.assertEqual([g["qualified"] for g in report["scene_groups"]], [0, 1])
        self.assertGreater(len(calls), 1)
        self.assertEqual(report["rejections"][0]["symmetry_group"], "near")

    def test_scene_limit_is_per_group_and_off_preserves_original_order(self):
        parents = [candidate(100 + i, i) for i in range(3)]
        scene = dict(object_id="unused", object_pose=np.eye(4).tolist(),
                     tcp_pose=np.eye(4).tolist(), grasp_symmetry="near-first")
        with patch("geometry.api.check_gripper_path", return_value={"valid": True}):
            valid, report = screen_scene_candidates(parents, scene, maximum=2)
            self.assertEqual([c["symmetry_group"] for c in valid], ["near", "near", "far", "far"])
            self.assertEqual(report["scene_candidate_limit"], 4)
            scene["grasp_symmetry"] = "off"
            valid, _ = screen_scene_candidates(parents, scene, maximum=2)
        self.assertEqual([c["original_rank"] for c in valid], [0, 1])
        self.assertTrue(all("symmetry_variant" not in c for c in valid))
        self.assertEqual([c["score"] for c in valid], [.9, .9])

    def test_cli_plan_and_worker_options_record_the_selected_mode(self):
        for selected in symmetry.MODES:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "collect.py"),
                                     "--dry-run", "--grasp-symmetry", selected],
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["grasp_symmetry"], selected)
            self.assertEqual(plan["symmetry_candidate_budget"]["full_planned_total"], 5)
            job = dict(candidate_root="candidates", persistent=True, stage="collection",
                       worker_id="worker", grasp_symmetry=selected)
            self.assertEqual(episode_options(None, None, job, 0).grasp_symmetry, selected)


@unittest.skipUnless(importlib.util.find_spec("curobo") is not None, "requires the cuRobo planning environment")
class PlannerSymmetryTests(unittest.TestCase):
    """Exercise the real select loop with CPU doubles for GPU operations."""

    def run_selection(self, reachable=(True, True), reject_near=False, maximum=5):
        from generic_planner import Planner
        near, far = symmetry.split_candidates([candidate()], np.eye(4), np.eye(4))
        pool = world_targets(near + far, np.eye(4))
        for c in pool:
            c["opening_m"] = .08
        with tempfile.TemporaryDirectory() as folder:
            planner = Planner.__new__(Planner)
            planner.scene = dict(qpos=[0.] * 9)
            planner.output = Path(folder)
            planner.set_width = lambda width: None
            # Far deliberately has less estimated joint travel. Near priority
            # must survive the subsequent IK/motion-cost ordering.
            planner.batch_ik = lambda *args: (np.array(reachable)[:, None],
                                              np.array([[[1.] * 7], [[.1] * 7]]))
            planner.motion = lambda *args, **kwargs: (np.zeros((2, 7)), {"success": True})
            planner.line = lambda *args, **kwargs: (np.zeros((2, 7)), {"valid": True})
            labels = []
            def validate(path, width, label, **kwargs):
                labels.append(label)
                return {"valid": not (reject_near and label.endswith("z180"))}
            planner.validate = validate
            with patch("generic_planner.screen_scene_candidates", return_value=(pool, {})):
                result = planner.select([], maximum=maximum)
            json.dumps(result, allow_nan=False)
            return result, labels

    def test_near_group_precedes_shorter_joint_estimate_in_far_group(self):
        result, _ = self.run_selection()
        self.assertTrue(result["success"])
        self.assertEqual(result["selected"]["symmetry_group"], "near")

    def test_far_is_available_after_near_ik_failure(self):
        result, _ = self.run_selection(reachable=(False, True))
        self.assertTrue(result["success"])
        self.assertEqual(result["selected"]["symmetry_group"], "far")

    def test_far_is_available_after_near_full_path_failure(self):
        result, labels = self.run_selection(reject_near=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["selected"]["symmetry_group"], "far")
        self.assertEqual(len(result["candidate_report"]["full_candidates"]), 2)
        self.assertIn("approach_sym0000_z180", labels)
        self.assertIn("approach_sym0000_original", labels)

    def test_full_planning_limit_is_shared_by_both_groups(self):
        result, _ = self.run_selection(reject_near=True, maximum=1)
        self.assertFalse(result["success"])
        self.assertEqual(len(result["candidate_report"]["full_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
