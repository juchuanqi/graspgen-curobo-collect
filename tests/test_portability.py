"""CPU regression checks for portability, scheduling, protocol and unchanged geometry."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "runtime/collector"))
import numpy as np
from common import PlannerClient, collection_plan, gpu_tokens, make_plan, read, write
from collect import Processes, job_waves, run_stage
from generate_candidates import restore_grasps, surface_sample
from initial_state import install_reset_guard, pose_comparison, requested_pose
from planner_service import Engine, Handler, Server
from tools.check_assets import check_assets
from tools.package import source_files


class PortabilityTests(unittest.TestCase):
    def test_gpu_mapping_respects_allocation(self):
        self.assertEqual(gpu_tokens("0", "5,6,7,8"), ["5"]*4)
        self.assertEqual(gpu_tokens("0,1", "5,6"), ["5", "5", "5", "6"])
        self.assertEqual(gpu_tokens("2,0,1,3", "GPU-a,GPU-b,GPU-c,GPU-d"),
                         ["GPU-c", "GPU-a", "GPU-b", "GPU-d"])
        for invalid in ("0,0", "0,1,2", "-1", "3"):
            with self.assertRaises(ValueError):
                gpu_tokens(invalid, "0,1")

    def test_fixed_long_batch_covers_every_slot_once(self):
        plan = collection_plan(["block55", "slender95", "cylinder36h52", "disk56h24"], 30, 3)
        waves = job_waves(plan["stages"][0])
        all_cases = [c for wave in waves for _, job in wave for c in job]
        self.assertEqual(len(waves), 4)
        self.assertTrue(all(len(job) == 10 for wave in waves for _, job in wave))
        self.assertEqual(len({c["case_id"] for c in all_cases}), 120)
        self.assertEqual(len({c["scene_seed"] for c in all_cases}), 120)
        self.assertEqual(plan["total_attempt_slots"], 120)
        self.assertEqual(plan, collection_plan(plan["objects"], 30, 3))

    def test_benchmark_compatibility_counts(self):
        self.assertEqual(make_plan()["total_attempt_slots"], 40)
        self.assertEqual(make_plan(full=True)["total_attempt_slots"], 72)
        self.assertEqual(make_plan(only="parallel3", episodes_per_worker=10)["total_attempt_slots"], 120)

    def test_centering_once_and_fixed_tcp_adapter(self):
        from geometry.api import graspgen_to_tcp
        raw = np.repeat(np.eye(4)[None], 200, axis=0)
        center = np.array([.01, -.02, .03])
        poses = restore_grasps(raw, center)
        np.testing.assert_allclose(raw[:, :3, 3], 0)
        np.testing.assert_allclose(poses[0, :3, 3], center)
        tcp = graspgen_to_tcp(poses)
        np.testing.assert_allclose(tcp[0, :3, 3], center + [0, 0, .1034])
        np.testing.assert_allclose(tcp[0, :3, 2], [0, 0, 1], atol=1e-12)

    def test_surface_points_stay_on_mesh_without_scale_change(self):
        import trimesh
        mesh = trimesh.creation.box([.055, .038, .032])
        a, b = surface_sample(mesh, 2048, 2), surface_sample(mesh, 2048, 2)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.abs(a) <= np.array([.0275, .019, .016])+1e-8))
        distance_to_face = np.min(np.abs(np.abs(a)-np.array([.0275, .019, .016])), axis=1)
        self.assertLess(distance_to_face.max(), 1e-8)

    def test_assets_and_geometry_load_from_moved_path(self):
        check_assets(ROOT / "runtime")
        from geometry.api import sample_surface
        from geometry.robot import description, fk
        for name in collection_plan(["block55", "slender95", "cylinder36h52", "disk56h24"], 1)["objects"]:
            sample = sample_surface(name, 16, 4)
            self.assertEqual(sample["points_object"].shape, (16, 3))
            self.assertTrue(np.isfinite(sample["normals_object"]).all())
        self.assertTrue(description()[0])
        scene = read(ROOT / "runtime/config/scene.json")
        pose = fk(np.array(scene["initial_qpos"]))["panda_hand_tcp"]
        self.assertEqual(pose.shape, (4, 4))

    def test_reset_guard_priming_and_wrong_state_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = collection_plan(["block55"], 1)["stages"][0]["cases"][0]
            asset = {"bounds_min": [-.0275, -.019, -.016]}
            xy = [.60672926787287, .00074124993989]
            expected = requested_pose(entry, asset, xy)
            events = []
            class Tensor:
                def __getitem__(self, item): return self
                def clone(self): return self
            env = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=Tensor())),
                                  joint_ids=[], park_object=lambda: events.append("park"))
            env.reset = lambda: events.append("reset")
            episode = SimpleNamespace(timing={}, row={"object_pose": np.array(expected)}, reason=None, success=False)
            episode.reset = lambda: env.reset()
            raw = expected.copy()
            raw[0] += .02
            install_reset_guard(episode, env, entry, asset, xy, folder, lambda: raw, prime_first=True)
            episode.reset()
            self.assertEqual(events, ["park", "reset"])
            self.assertEqual(episode.reason, "requested_initial_state_mismatch")
            self.assertEqual(read(Path(folder) / "INITIAL_STATE_AUDIT.json")["status"], "failed")
            self.assertEqual(pose_comparison(expected, expected)["status"], "matched")

    def test_shared_planner_serializes_interleaved_clients(self):
        with tempfile.TemporaryDirectory() as folder:
            class Planner:
                def __init__(self, scene, output): self.scene = scene
                def update(self, scene, output, remove_object=False): self.scene = scene
                def select(self, candidates, **kwargs):
                    before = self.scene["id"]
                    time.sleep(.02)
                    return dict(success=True, before=before, after=self.scene["id"])
            with Server(("127.0.0.1", 0), Handler) as server:
                server.token = "test-token"
                server.engine = Engine(Planner, lambda seed: None)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                replies = []
                address = dict(port=server.server_address[1], token=server.token)
                def request(index):
                    client = PlannerClient(address)
                    replies.append(client.request(dict(command="select", scene={"id": index}, seed=index,
                                                       out=str(Path(folder) / str(index)), candidates=[])))
                clients = [threading.Thread(target=request, args=(i,)) for i in range(3)]
                for client in clients: client.start()
                for client in clients: client.join()
                server.shutdown()
                thread.join()
            self.assertEqual(len(replies), 3)
            self.assertTrue(all(r["before"] == r["after"] for r in replies))
            self.assertEqual(sorted(r["request_index"] for r in replies), [1, 2, 3])

    def test_real_subprocess_dispatch_handles_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="grasp code ") as temporary:
            root = Path(temporary)
            scripts, output = root / "scripts with spaces", root / "output with spaces"
            scripts.mkdir(); output.mkdir()
            (scripts / "planner_service.py").write_text(
                "import argparse,json,time\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser();p.add_argument('--ready');p.add_argument('--runtime');a=p.parse_args()\n"
                "Path(a.ready).write_text(json.dumps({'port':1,'token':'test'}))\ntime.sleep(60)\n")
            (scripts / "sim_worker.py").write_text(
                "import argparse,json\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--output');a,unknown=p.parse_known_args()\n"
                "job=json.loads(Path(a.job).read_text());out=Path(a.output);out.mkdir(parents=True)\n"
                "(out/'seen.json').write_text(json.dumps(dict(job=job,args=unknown)))\n")
            args = SimpleNamespace(output=output, runtime=ROOT / "runtime", candidate_root=root / "new candidates",
                                   startup_timeout=10, episode_timeout=10, gui=True)
            stage = collection_plan(["block55"], 2)["stages"][0]
            processes = Processes()
            try:
                with patch("collect.HERE", scripts), patch("collect.environment", lambda *unused: dict(os.environ)):
                    run_stage(stage, args, ["0"]*4, sys.executable, sys.executable, processes)
                receipt = read(output / "collection/STAGE.json")
                worker = Path(receipt["jobs"][0]["output"])
                seen = read(worker / "seen.json")
                self.assertEqual(seen["job"]["candidate_root"], str(args.candidate_root))
                self.assertNotIn("--headless", seen["args"])
                self.assertEqual(receipt["status"], "completed")
            finally:
                processes.close()
            self.assertTrue(all(child.poll() is not None for child in processes.children))

    def test_delivery_has_no_recordings_or_machine_paths(self):
        files = list(source_files())
        self.assertTrue(files)
        paths = {str(relative) for _, relative in files}
        self.assertIn("third_party/GraspGen/grasp_gen/models/grasp_gen.py", paths)
        self.assertIn("third_party/GraspGen/pointnet2_ops/pointnet2_ops/_ext-src/src/sampling_gpu.cu", paths)
        self.assertIn("third_party/GraspGen/assets/panda_gripper/hand.stl", paths)
        self.assertIn("third_party/GraspGen/assets/panda_gripper/finger.stl", paths)
        for path, relative in files:
            self.assertNotIn(relative.name, {"initial.npz", "candidates.npz", "trajectory.npz", "METADATA.json"})
            self.assertNotIn(path.suffix, {".log", ".npy", ".pyc", ".so", ".mp4"})
            if path.suffix == ".py" and "third_party" not in relative.parts and "tests" not in relative.parts:
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertFalse(node.value.startswith(("/mnt/", "/tmp/", "/root/", "/opt/")), str(relative))


if __name__ == "__main__":
    unittest.main()
