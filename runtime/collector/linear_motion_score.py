"""Measured COM translation and actor rotation at 480 Hz.

No historical labels are changed. The native velocity channels remain required,
finite telemetry. All grasp and safety thresholds are inherited unchanged.
"""
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from pose_motion_score import MotionInputError, rotation, legacy

VERSION = 'com_translation_pose_arc_v3'
PROTOCOL = {**legacy.PROTOCOL, 'version': VERSION,
    'linear_measurement': 'sum of eight world COM position increments / (8/480 s)',
    'angular_measurement': 'sum of eight shortest actor SO(3) increments / (8/480 s)',
    'com_reference': 'explicit finite actor-local COM; no zero-offset fallback',
    'native_velocity': 'required finite telemetry; not a motion threshold',
    'missing_motion_input': 'terminal rejection; no endpoint or native fallback'}
BLOCKERS = (
    'hold_disabled', 'invalid_motion_observation', 'lift_below_60mm',
    'bilateral_pad_contact_missing', 'opening_out_of_range', 'object_on_table',
    'linear_motion_too_fast', 'angular_pose_motion_too_fast',
    'required_handle_contact_missing', 'hard_failure',
    'relative_translation_drift', 'relative_rotation_drift',
)


def checked_pose(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,):
        raise MotionInputError('Expected one XYZ/WXYZ pose')
    rotation(value)
    return value


def checked_com(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise MotionInputError('Explicit finite actor-local COM is required')
    return value


def interval_inputs(row):
    try:
        poses = np.asarray(row['contact_substep_object_pose_w'], dtype=float)
        ticks = np.asarray(row['contact_substep_physics_tick'])
        count = np.asarray(row['contact_substep_sample_count'])
        time = np.asarray(row['sim_time'], dtype=float)
        endpoint = checked_pose(row['object_pose'])
    except (KeyError, TypeError, ValueError) as error:
        raise MotionInputError('Missing or malformed physical interval') from error
    if poses.shape != (8, 7) or ticks.shape != (8,) or count.shape != () or count != 8:
        raise MotionInputError('Exactly eight physical samples required')
    if (not np.issubdtype(ticks.dtype, np.integer) or ticks[0] < 0
            or not np.all(np.diff(ticks) == 1)):
        raise MotionInputError('Consecutive integer physics ticks required')
    if time.shape != () or not np.isfinite(time):
        raise MotionInputError('Finite scalar control time required')
    rotations = rotation(poses)
    if (np.linalg.norm(poses[-1, :3] - endpoint[:3]) > 1e-6
            or (rotations[-1] * rotation(endpoint).inv()).magnitude() > 1e-5):
        raise MotionInputError('Physical/control endpoint misalignment')
    return poses, ticks, float(time)


def interval_motion(previous_pose, previous_tick, previous_time, row, com_local_m):
    previous_pose = checked_pose(previous_pose)
    com = checked_com(com_local_m)
    poses, ticks, time = interval_inputs(row)
    if not np.array_equal(ticks, np.arange(previous_tick + 1, previous_tick + 9)):
        raise MotionInputError('Gap or duplicate in physical ticks')
    if not np.isclose(time - previous_time, 8 / 480, rtol=0, atol=1e-8):
        raise MotionInputError('Control/physics time mismatch')
    all_poses = np.vstack([previous_pose, poses])
    rotations = rotation(all_poses)
    com_w = all_poses[:, :3] + rotations.apply(np.broadcast_to(com, (9, 3)).copy())
    distances = np.linalg.norm(np.diff(com_w, axis=0), axis=-1)
    angles = (rotations[1:] * rotations[:-1].inv()).magnitude()
    return dict(pose_interval_linear_speed_m_s=float(distances.sum() * 60),
                pose_substep_peak_linear_speed_m_s=float(distances.max() * 480),
                pose_interval_angular_speed_rad_s=float(angles.sum() * 60),
                pose_substep_peak_angular_speed_rad_s=float(angles.max() * 480),
                com_path_m=float(distances.sum()), com_endpoint_w=com_w[-1],
                endpoint_pose=poses[-1].copy(), last_tick=int(ticks[-1]), time=time)


class LinearMotionScore(legacy.PhysicalScore):
    def __init__(self, initial_row, com_local_m, task='pick_up_any_stable_region'):
        super().__init__(checked_pose(initial_row['object_pose']), task)
        self.com_local_m = checked_com(com_local_m).copy()
        _, ticks, time = interval_inputs(initial_row)
        self.previous_pose = checked_pose(initial_row['object_pose']).copy()
        self.previous_tick = int(ticks[-1])
        self.previous_time = time
        self.motion_errors = []
        self.blocker_counts = Counter()
        self.last_blockers = []

    def update(self, row, hold_enabled=True):
        # A terminal result is frozen, matching a collector stopping there.
        if self.reason is not None:
            raise RuntimeError('Score is terminal; start a separate diagnostic score to continue')
        self.steps += 1
        p = PROTOCOL
        required = ('qpos', 'qvel', 'position_targets', 'object_pose', 'tcp_pose',
                    'object_linear_velocity', 'object_angular_velocity',
                    'link_contact_force', 'link_contact_peak_n', 'object_table_peak_n',
                    'object_wall_peak_n', 'pad_object_force_n', 'nonpad_object_peak_n',
                    'nonpad_object_force_bound_n', 'max_contact_penetration_m')
        if not all(k in row and np.isfinite(row[k]).all() for k in required):
            self.reason = 'nonfinite_or_missing_state'
            self.window.clear()
            self.last_blockers = ['hard_failure']
            return {'motion_measurement_valid': False}
        try:
            checked_pose(row['object_pose'])
            checked_pose(row['tcp_pose'])
            measured = interval_motion(self.previous_pose, self.previous_tick,
                                       self.previous_time, row, self.com_local_m)
        except (MotionInputError, TypeError, ValueError) as error:
            self.motion_errors.append(dict(step=self.steps, error=str(error)))
            self.reason = 'invalid_motion_observation'
            self.window.clear()
            self.last_blockers = ['invalid_motion_observation', 'hard_failure']
            return {'motion_measurement_valid': False}
        self.previous_pose = measured.pop('endpoint_pose')
        self.previous_tick = measured.pop('last_tick')
        self.previous_time = measured.pop('time')
        d = {**self.derive(row), **measured, 'motion_measurement_valid': True,
             'native_linear_speed_m_s': float(np.linalg.norm(row['object_linear_velocity'])),
             'native_angular_speed_rad_s': float(np.linalg.norm(row['object_angular_velocity']))}
        if d['forbidden_force_n'] > p['max_forbidden_force_n']:
            self.reason = 'forbidden_contact'
        elif float(row['max_contact_penetration_m']) > p['max_contact_penetration_m']:
            self.reason = 'excess_contact_penetration'
        elif row['object_pose'][2] < -.005:
            self.reason = 'object_fell'
        tests = [not hold_enabled, False, d['lift_m'] < p['lift_m'],
                 not (d['pad_force_each_n'] >= p['pad_force_each_n']).all(),
                 not p['opening_range_m'][0] <= d['opening_m'] <= p['opening_range_m'][1],
                 row['object_table_peak_n'] > p['max_table_force_n'],
                 d['pose_interval_linear_speed_m_s'] > p['max_linear_speed_m_s'],
                 d['pose_interval_angular_speed_rad_s'] > p['max_angular_speed_rad_s'],
                 self.task == 'pick_up_outer_handle_bar' and not d['outer_handle_contact'],
                 self.reason is not None, False, False]
        if not any(tests):
            self.window.append(d['object_in_tcp'])
        else:
            self.window.clear()
        self.best_window = max(self.best_window, len(self.window))
        d.update(relative_drift_window_available=len(self.window) == p['hold_steps'],
                 relative_translation_drift_m=0., relative_rotation_drift_deg=0.)
        if len(self.window) == p['hold_steps']:
            relative = np.linalg.inv(self.window[0])[None] @ np.stack(self.window)
            d['relative_translation_drift_m'] = float(np.linalg.norm(relative[:, :3, 3], axis=-1).max())
            d['relative_rotation_drift_deg'] = float(np.rad2deg(Rotation.from_matrix(relative[:, :3, :3]).magnitude().max()))
            tests[10] = d['relative_translation_drift_m'] > p['max_relative_translation_drift_m']
            tests[11] = d['relative_rotation_drift_deg'] > p['max_relative_rotation_drift_deg']
            if not any(tests):
                self.success = True
                self.reason = 'stable_handle_hold' if self.task == 'pick_up_outer_handle_bar' else 'stable_grasp_hold'
                self.first_success_step = self.steps
        self.last_blockers = [name for name, blocked in zip(BLOCKERS, tests) if blocked]
        if hold_enabled:
            self.blocker_counts.update(self.last_blockers)
        return d
