"""CPU checks for the optional kinematic wall obstacle."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/collector"))
from obstacle_scene import WALL, planner_obstacles, verify_obstacle_scene
sys.path.insert(0, str(ROOT))
from sim_worker import create_environment


class ObstacleSceneTests(unittest.TestCase):
    def test_default_mode_adds_no_planner_obstacle(self):
        self.assertEqual(planner_obstacles("none"), [])
        self.assertEqual(planner_obstacles("clear"), [])

    def test_wall_intrudes_on_approach_corridor(self):
        wall = planner_obstacles("wall")[0]
        self.assertEqual(wall, WALL)
        self.assertEqual(wall["dims"], [0.02, 0.30, 0.22])
        self.assertEqual(wall["pose"][:3], [0.545, 0.0, 0.11])
        # Target rests near x=0.60; the slab must sit in front of it, not behind the base.
        self.assertGreater(wall["pose"][0], 0.50)
        self.assertLess(wall["pose"][0], 0.59)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            planner_obstacles("clutter")

    def test_scene_evidence_requires_matching_mode_and_geometry(self):
        verify_obstacle_scene('none', {})  # Historical clear scene.
        scene = dict(obstacle_mode='wall', obstacles=planner_obstacles('wall'))
        verify_obstacle_scene('wall', scene)
        for wrong in ({}, dict(obstacle_mode='none', obstacles=[]),
                      dict(obstacle_mode='wall', obstacles=[])):
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_obstacle_scene('wall', wrong)
        scene['obstacles'][0]['pose'][0] += .01
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_obstacle_scene('wall', scene)
        self.assertEqual(WALL['pose'][0], .545)

    def test_worker_constructs_requested_simulation_scenario(self):
        def environment(cfg, **kwargs):
            return kwargs
        for job, expected in [({}, 'clear'), ({'obstacle': 'none'}, 'clear'),
                              ({'obstacle': 'wall'}, 'wall')]:
            env = create_environment(environment, None, {'object_id': 'block55'}, {}, job)
            self.assertEqual(env['scenario'], expected)
            self.assertEqual(env['occlusion'], 'none')
        with self.assertRaises(ValueError):
            create_environment(environment, None, {'object_id': 'block55'}, {}, {'obstacle': 'clutter'})

    def test_collect_plan_records_obstacle_choice(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "collect.py"), "--dry-run", "--obstacle", "wall"],
            capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["obstacle"], "wall")


if __name__ == "__main__":
    unittest.main()
