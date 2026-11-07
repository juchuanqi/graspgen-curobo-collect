"""Panda episode execution, synchronized recording and unchanged grasp scoring."""
import json
import time
import numpy as np
import torch
from common import write
from collection_env import URDF, NAMES, LINKS
from collection_score import CollectionScore, PROTOCOL, matrix, sanitize
from io_audit import LiveIOAudit
from geometry.api import check_robot_path
from target_provider import load_candidate_source

def create_episode(env, entry, asset, args, casegrid, planner, process_started, loaded_source_hashes, initialization_s):

    class CameraStream:
        """Single append-only NPY, final header length patched without RGB copy."""

        def __init__(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open('wb+')
            self.n = 0
            np.lib.format.write_array_header_1_0(self.file, {'descr': '|u1', 'fortran_order': False, 'shape': (2401, 256, 256, 3)})
            self.offset = self.file.tell()

        def append(self, value):
            value = np.asarray(value)
            if value.shape != (256, 256, 3) or value.dtype != np.uint8:
                raise ValueError('Invalid RGB source')
            self.file.write(value.tobytes(order='C'))
            self.n += 1

        def close(self):
            self.file.seek(0)
            np.lib.format.write_array_header_1_0(self.file, {'descr': '|u1', 'fortran_order': False, 'shape': (self.n, 256, 256, 3)})
            if self.file.tell() != self.offset:
                raise RuntimeError('NPY dynamic header offset changed')
            self.file.flush()
            self.file.close()

    class Episode:

        def __init__(self):
            self.row = None
            self.reason = None
            self.success = False
            self.score = None
            self.rows = []
            self.actions = []
            self.phases = []
            self.phase = 0
            self.streams = {}
            self.timing = {'reset_settle': 0.0, 'reset_call': 0.0, 'settle_controls': 0.0, 'physics_camera': 0.0, 'env_step': 0.0, 'snapshot': 0.0, 'record': 0.0, 'candidate_load': 0.0, 'planning': 0.0, 'save': 0.0}
            self.control_timings = []
            self.io_audit = LiveIOAudit() if args.audit_io else None
            self.io_audit_wall_s = 0.0
            self.selected = None
            self.planner_replies = []
            self.ever_bilateral = False
            self.started = time.monotonic()
            self.lift_enabled = False

        def scene(self):
            return {'object_id': entry['object_id'], 'urdf_path': str(URDF), 'joint_names': NAMES, 'qpos': self.row['qpos'].tolist(), 'tcp_pose': self.row['tcp_pose'].tolist(), 'base_pose': env.robot.data.root_link_pose_w[0].cpu().numpy().tolist(), 'table_pose': [0.495, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0], 'table_shape_local_pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'table_shape_half_size': [0.6, 0.6, 0.025], 'object_pose': self.row['object_pose'].tolist(), 'object_parts': asset['parts'], 'object_size_m': asset['size_m'], 'obstacles': []}

        def status(self, state):
            write(args.output / 'STATUS.json', {'state': state, 'case_id': entry['case_id'], 'method': args.method, 'actions': len(self.actions), 'wall_s': time.monotonic() - self.started, 'reason': self.reason})

        def plan(self, command, **kwargs):
            request = dict(command=command, scene=self.scene(), seed=int(entry['scene_seed']), out=str(args.output / 'planning' / command), **kwargs)
            write(args.output / f'{command}_request.json', request)
            self.status('planning_' + command)
            started = time.monotonic()
            reply = planner.request(request, fresh=args.cold_planner and command == 'select')
            self.timing['planning'] += time.monotonic() - started
            self.planner_replies.append(reply)
            write(args.output / f'{command}_reply.json', reply)
            return reply

        def record(self, row, images, initial=False):
            d = self.score.derive(row) if initial else self.score.update(row, hold_enabled=self.lift_enabled)
            clean = sanitize({**row, **d, 'sim_time': np.float64(len(self.actions) / 60.0)})
            self.rows.append(clean)
            self.row = row
            for name, value in images.items():
                self.streams[name].append(value)
            if (np.asarray(row['pad_object_force_n']) >= 0.5).all():
                self.ever_bilateral = True
            if self.score.reason and (not self.score.success):
                self.reason = self.score.reason
            if self.score.success:
                self.success = True
                self.reason = self.score.reason

        def step(self, action):
            if self.reason:
                return False
            if len(self.actions) >= 2400:
                self.reason = 'control_step_budget_exhausted'
                return False
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (9,) or not np.isfinite(action).all():
                self.reason = 'invalid_action'
                return False
            if np.max(np.abs(action[:7] - self.row['position_targets'][:7])) > 0.002501:
                self.reason = 'joint_target_delta_exceeded'
                return False
            t = time.monotonic()
            env.step(torch.tensor(action[None], device=env.device))
            step_s = time.monotonic() - t
            t_snapshot = time.monotonic()
            row, images = env.snapshot()
            snapshot_s = time.monotonic() - t_snapshot
            self.timing['physics_camera'] += time.monotonic() - t
            self.timing['env_step'] += step_s
            self.timing['snapshot'] += snapshot_s
            self.actions.append(action.copy())
            self.phases.append(self.phase)
            t_record = time.monotonic()
            self.record(row, images)
            record_s = time.monotonic() - t_record
            self.timing['record'] += record_s
            self.control_timings.append((step_s, snapshot_s, record_s, time.monotonic() - t))
            if self.io_audit is not None and (self.phase >= 2 or len(self.actions) % 60 == 0):
                audit_started = time.monotonic()
                self.io_audit.compare(env, row, images)
                self.io_audit_wall_s += time.monotonic() - audit_started
            if len(self.actions) % 60 == 0:
                self.status('executing_phase_' + str(self.phase))
            return self.reason is None

        def reset(self):
            t = time.monotonic()
            park_started = time.monotonic()
            if args.reused_env:
                env.park_object()
            self.timing['park'] = time.monotonic() - park_started
            env.case_xy = list(entry['xy_offset_m'])
            env.case_yaw = np.deg2rad(entry['yaw_deg'])
            env.reset_seed = int(entry['scene_seed'])
            torch.manual_seed(int(entry['scene_seed']))
            np.random.seed(int(entry['scene_seed']))
            env.reset()
            self.timing['reset_call'] = time.monotonic() - t
            t_settle = time.monotonic()
            self.settle_protocol = env.settle(minimal=not args.full_settle) if args.io_mode == 'optimized' else None
            if args.io_mode == 'baseline':
                for _ in range(120):
                    env.step(env.targets.clone())
            self.timing['settle_controls'] = time.monotonic() - t_settle
            self.row, images = env.snapshot()
            self.score = CollectionScore(self.row, entry['task'])
            self.streams = {name: CameraStream(args.output / 'cameras' / f'{name}.npy') for name in images}
            self.record(self.row, images, initial=True)
            self.timing['reset_settle'] = time.monotonic() - t
            initial = check_robot_path(self.row['qpos'], self.row['qpos'][7:].sum(), matrix(self.row['object_pose']), entry['object_id'], base_pose=matrix(env.robot.data.root_link_pose_w[0].cpu().numpy()))
            write(args.output / 'INITIAL_GEOMETRY.json', initial)
            if not initial['valid']:
                self.reason = 'invalid_initial_geometry'
            if not np.isfinite(self.row['qpos']).all():
                self.reason = 'nonfinite_initial_state'
            write(args.output / 'SCENE.json', self.scene())

        def load_candidates(self):
            started = time.monotonic()
            candidates, provenance = load_candidate_source(args.candidate_root, args.method, entry['object_id'])
            write(args.output / 'CANDIDATE_SOURCE.json', provenance)
            self.timing['candidate_load'] = time.monotonic() - started
            if not candidates:
                self.reason = 'no_geometry_qualified_candidates'
            return candidates

        def follow(self, path, opening, settle=0):
            for q in path:
                if not self.step(np.r_[q, opening / 2, opening / 2]):
                    return
            for _ in range(settle):
                if not self.step(np.r_[path[-1], opening / 2, opening / 2]):
                    return

        def rollout(self):
            self.reset()
            if self.reason:
                return
            if args.qualify_only:
                reply = self.plan('qualify')
                self.reason = 'environment_qualification_passed' if reply and reply.get('success') else 'environment_qualification_failed'
                return
            candidates = self.load_candidates()
            if self.reason:
                return
            reply = self.plan('select', candidates=candidates)
            if not reply or not reply.get('success'):
                self.reason = (reply or {}).get('reason', 'candidate_selection_failed')
                return
            self.selected = reply['selected']
            opening = self.selected['opening_m']
            arm = self.row['position_targets'][:7].copy()
            for each in np.linspace(float(self.row['qpos'][7:].mean()), opening / 2, 30):
                if not self.step(np.r_[arm, each, each]):
                    return
            with np.load(self.selected['path_file']) as p:
                approach = p['approach'].copy()
                descend = p['descend'].copy()
            self.phase = 0
            self.follow(approach, opening, settle=20)
            if self.reason:
                return
            error = np.linalg.norm(self.row['tcp_pose'][:3] - np.asarray(self.selected['pregrasp_world'])[:3, 3])
            if error > 0.003:
                self.reason = 'pregrasp_tracking_failed'
                return
            self.phase = 1
            self.follow(descend, opening, settle=20)
            if self.reason:
                return
            target = np.asarray(self.selected['grasp_world'])
            if np.linalg.norm(self.row['tcp_pose'][:3] - target[:3, 3]) > 0.003:
                self.reason = 'grasp_tracking_failed'
                return
            self.phase = 2
            arm = self.row['position_targets'][:7].copy()
            close_begin = len(self.rows) - 1
            for each in np.linspace(opening / 2, 0.0, 90):
                if not self.step(np.r_[arm, each, each]):
                    return
            for _ in range(45):
                if not self.step(np.r_[arm, 0.0, 0.0]):
                    return
            if not (self.row['pad_object_force_n'] >= 0.5).all():
                self.reason = 'bilateral_pad_contact_not_established'
                return
            closure = np.stack([r['qpos'] for r in self.rows[close_begin:]])
            objects = np.stack([matrix(r['object_pose']) for r in self.rows[close_begin:]])
            geometry_started = time.monotonic()
            checked = check_robot_path(closure, closure[:, 7:].sum(axis=1), objects, entry['object_id'], base_pose=matrix(env.robot.data.root_link_pose_w[0].cpu().numpy()), allow_pad_contact=True, pad_penetration_m=0.001)
            checked['contact_mode'] = 'measured_grasp_1mm; raw PhysX depth<=1mm and nonpad<=1N remain hard limits'
            closure_geometry_s = time.monotonic() - geometry_started
            self.timing['closure_geometry'] = closure_geometry_s
            write(args.output / 'CLOSURE_GEOMETRY.json', checked)
            if not checked['valid']:
                self.reason = 'closure_full_geometry_failed'
                return
            self.phase = 3
            reply = self.plan('lift', remaining_budget_s=max(0.0, self.selected['remaining_budget_s'] - closure_geometry_s))
            if not reply or not reply.get('success'):
                self.reason = (reply or {}).get('reason', 'lift_planning_failed')
                return
            with np.load(reply['path_file']) as p:
                lift = p['execution_q'].copy()
            self.lift_enabled = True
            self.follow(lift, 0.0, settle=20)
            if self.reason:
                return
            self.phase = 4
            for _ in range(240):
                if not self.step(self.row['position_targets']):
                    return
            self.reason = 'stable_hold_not_reached'

        def save(self, error=None):
            t = time.monotonic()
            for stream in self.streams.values():
                stream.close()
            if self.rows:
                raw = {key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]}
                raw['action_targets'] = np.asarray(self.actions, np.float32).reshape(-1, 9)
                raw['phase'] = np.asarray(self.phases, np.uint8)
                raw['control_timing_s'] = np.asarray(self.control_timings, np.float64).reshape(-1, 4)
                if len(raw['qpos']) != len(self.actions) + 1:
                    raise RuntimeError('State/action chronology broken')
                np.savez_compressed(args.output / 'trajectory.npz', **raw)
            self.timing['save'] = time.monotonic() - t
            metadata = {'case_id': entry['case_id'], 'object_id': entry['object_id'], 'scene_seed': int(entry['scene_seed']), 'method': args.method, 'role': args.role, 'io_mode': args.io_mode, 'task': entry['task'], 'accepted': bool(self.success), 'success': bool(self.success), 'reason': self.reason, 'terminal_reason': self.reason, 'physical_attempts': int(not args.qualify_only and self.selected is not None), 'actions': len(self.actions), 'observations': len(self.rows), 'control_hz': 60, 'physics_hz': 480, 'sim_seconds': len(self.actions) / 60.0, 'camera_names': list(env.cameras), 'joint_names': NAMES, 'contact_links': LINKS, 'success_contract': PROTOCOL, 'best_stable_window': self.score.best_window if self.score else 0, 'score_diagnostics': self.score.report(self.reason) if self.score else None, 'ever_bilateral_pad_contact': self.ever_bilateral, 'selected_candidate': self.selected, 'candidate_budget': casegrid['candidate_budget'], 'timing_s': self.timing, 'episode_wall_s': time.monotonic() - self.started, 'control_timing_columns': ['env_step', 'snapshot', 'record', 'total'], 'process_wall_s': time.monotonic() - process_started, 'error': error, 'settle_protocol': getattr(self, 'settle_protocol', None), 'source_sha256': loaded_source_hashes, 'asset': asset, 'training_format': '8D after official export: 7 absolute arm q targets + total opening; raw 9D joints retained', 'terminal_observation': 'N actions paired with observations[0:N], terminal at N', 'human_preview': False}
            if self.rows:
                metadata.update(max_lift_m=float(max((float(r['lift_m']) for r in self.rows))), maximum_forbidden_force_n=float(max((float(r['forbidden_force_n']) for r in self.rows))), final_pad_force_n=self.row['pad_object_force_n'].tolist(), final_object_table_force_n=float(self.row['object_table_peak_n']))
            metadata['io_audit_wall_s'] = self.io_audit_wall_s
            if self.io_audit is not None:
                write(args.output / 'LIVE_IO_AUDIT.json', self.io_audit.report())
            metadata.update(benchmark_stage=args.stage, worker_id=args.worker_id, reused_environment=bool(args.reused_env), initialization_s=initialization_s if not args.reused_env else 0.0, queue_wait_s=sum((r.get('queue_wait_s', 0.0) for r in self.planner_replies)), process_wall_is_cumulative=True, primary_scope='GraspGen + cuRobo + Isaac Lab collection')
            write(args.output / 'METADATA.json', metadata)
            write(args.output / 'RESULT.json', metadata)
            self.status('completed')
            return metadata
    return Episode()
