"""Version 2 engineering score: physical pose arc per control interval.

Legacy native velocity is retained as telemetry and checked for finiteness.
No formal result is migrated. Requires an initial snapshot and all 8 substeps.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCORING = Path(__file__).resolve().with_name('legacy_scoring.py')
spec = importlib.util.spec_from_file_location('_immutable_physical_score', SCORING)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
VERSION = 'pose_arc_per_control_v2'
PROTOCOL = {**legacy.PROTOCOL, 'version': VERSION,
    'angular_measurement': 'sum of 8 shortest SO(3) physical pose increments / (8/480 s)',
    'native_angular_velocity': 'recorded and finite; auxiliary legacy score remains separate',
    'missing_motion_input': 'hard reject; no frame-rate or native-velocity fallback'}


class MotionInputError(ValueError):
    pass


def rotation(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise MotionInputError('Nonfinite or malformed pose')
    q = pose[...,3:7]
    if np.any(np.abs(np.linalg.norm(q,axis=-1)-1.) > .001):
        raise MotionInputError('Quaternion is not normalized')
    return Rotation.from_quat(pose[...,[4,5,6,3]])


def pose_interval_motion(previous_pose, previous_tick, row):
    try:
        poses = np.asarray(row['contact_substep_object_pose_w'],dtype=float)
        ticks = np.asarray(row['contact_substep_physics_tick'])
        count = int(row['contact_substep_sample_count'])
    except (KeyError, ValueError, TypeError) as error:
        raise MotionInputError('Missing physical interval') from error
    if poses.shape != (8,7) or ticks.shape != (8,) or count != 8:
        raise MotionInputError('Exactly eight physical samples required')
    if not np.isfinite(ticks).all() or not np.array_equal(ticks, np.arange(previous_tick+1,previous_tick+9)):
        raise MotionInputError('Non-consecutive physical ticks')
    orientations = rotation(np.vstack([previous_pose,poses]))
    endpoint = rotation(row['object_pose'])
    if (np.linalg.norm(poses[-1,:3]-np.asarray(row['object_pose'])[:3]) > 1e-6
        or (orientations[-1]*endpoint.inv()).magnitude() > 1e-5):
        raise MotionInputError('Physical/control endpoint misalignment')
    angles = (orientations[1:]*orientations[:-1].inv()).magnitude()
    return dict(mean_angular_speed_rad_s=float(angles.sum()/(8/480)),
                peak_physics_angular_speed_rad_s=float(angles.max()*480),
                angular_path_rad=float(angles.sum()), last_tick=int(ticks[-1]),
                endpoint_pose=poses[-1].copy(), physical_step_angles_rad=angles)


class PoseMotionScore(legacy.PhysicalScore):
    def __init__(self, initial_row, task='pick_up_any_stable_region'):
        super().__init__(initial_row['object_pose'], task)
        rotation(initial_row['object_pose'])
        self.previous_pose = np.asarray(initial_row['object_pose']).copy()
        ticks = np.asarray(initial_row['contact_substep_physics_tick'])
        if ticks.shape != (8,) or not np.isfinite(ticks).all() or not np.all(np.diff(ticks)==1):
            raise MotionInputError('Initial snapshot requires a consecutive physical interval')
        self.previous_tick = int(ticks[-1])
        self.motion_errors = []

    def update(self, row, hold_enabled=True):
        self.steps += 1
        d = self.derive(row)
        try:
            measured = pose_interval_motion(self.previous_pose,self.previous_tick,row)
            self.previous_pose = measured['endpoint_pose']
            self.previous_tick = measured['last_tick']
            motion_valid = True
            d.update(pose_interval_angular_speed_rad_s=measured['mean_angular_speed_rad_s'],
                     pose_substep_peak_angular_speed_rad_s=measured['peak_physics_angular_speed_rad_s'])
        except MotionInputError as error:
            self.motion_errors.append(dict(step=self.steps, error=str(error)))
            self.reason = 'invalid_motion_observation'
            motion_valid = False
        d['motion_measurement_valid'] = motion_valid
        d['native_angular_speed_rad_s'] = float(np.linalg.norm(row['object_angular_velocity']))
        required = ('qpos','qvel','position_targets','object_pose','tcp_pose',
            'object_linear_velocity','object_angular_velocity','link_contact_force','link_contact_peak_n',
            'object_table_peak_n','pad_object_force_n','nonpad_object_peak_n','nonpad_object_force_bound_n')
        if not all(np.isfinite(row[k]).all() for k in required):
            self.reason = 'nonfinite_state'
        elif d['forbidden_force_n']>1:
            self.reason = 'forbidden_contact'
        elif float(row.get('max_contact_penetration_m',0))>.001:
            self.reason = 'excess_contact_penetration'
        elif row['object_pose'][2]<-.005:
            self.reason = 'object_fell'
        region = self.task!='pick_up_outer_handle_bar' or d['outer_handle_contact']
        valid = bool(self.reason is None and hold_enabled and motion_valid and d['lift_m']>=.06
            and (d['pad_force_each_n']>=.5).all() and .002<=d['opening_m']<=.078
            and row['object_table_peak_n']<=.5
            and np.linalg.norm(row['object_linear_velocity'])<=.02
            and d['pose_interval_angular_speed_rad_s']<=.2 and region)
        if valid:
            self.window.append(d['object_in_tcp'])
        else:
            self.window.clear()
        self.best_window = max(self.best_window,len(self.window))
        if len(self.window)==120:
            relative = np.linalg.inv(self.window[0])[None]@np.stack(self.window)
            if (np.linalg.norm(relative[:,:3,3],axis=-1).max()<=.002
                and Rotation.from_matrix(relative[:,:3,:3]).magnitude().max()<=np.deg2rad(5)):
                self.success = True
                self.reason = 'stable_handle_hold' if self.task=='pick_up_outer_handle_bar' else 'stable_grasp_hold'
                if self.first_success_step is None:
                    self.first_success_step=self.steps
        return d
