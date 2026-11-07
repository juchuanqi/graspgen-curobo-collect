"""Collection integration of the calibrated pose-motion score and legacy telemetry.

The primary score is pose_arc_per_control_v2. No threshold is object-specific.
Legacy success is measured on the same collected prefix, never used to stop it.
"""
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from pose_motion_score import PoseMotionScore, PROTOCOL, VERSION, legacy

matrix, pose, sanitize = legacy.matrix, legacy.pose, legacy.sanitize

# Persist the mapping alongside the trajectory so failures are inspectable.
HOLD_BLOCKERS = (
    'hold_disabled', 'invalid_motion_observation', 'lift_below_60mm',
    'bilateral_pad_contact_missing', 'opening_out_of_range', 'object_on_table',
    'linear_motion_too_fast', 'angular_pose_motion_too_fast',
    'required_handle_contact_missing', 'hard_failure',
    'relative_translation_drift', 'relative_rotation_drift',
)


class CollectionScore:
    def __init__(self, initial_row, task='pick_up_any_stable_region'):
        self.primary = PoseMotionScore(initial_row, task)
        self.legacy = legacy.PhysicalScore(initial_row['object_pose'], task)
        self.blocker_counts = Counter()
        self.legacy_angular_exceedances = 0
        self.hold_observations = 0
        self.last_blockers = []

    @property
    def success(self):
        return self.primary.success

    @property
    def reason(self):
        return self.primary.reason

    @property
    def best_window(self):
        return self.primary.best_window

    @property
    def first_success_step(self):
        return self.primary.first_success_step

    def _augment(self, row, derived, hold_enabled, initial=False):
        # Identical keys on reset and every step are required by NPZ stacking.
        d = {
            'pose_interval_angular_speed_rad_s': 0.,
            'pose_substep_peak_angular_speed_rad_s': 0.,
            'motion_measurement_valid': False,
            'relative_drift_window_available': False,
            'relative_translation_drift_m': 0.,
            'relative_rotation_drift_deg': 0.,
            **derived,
        }
        d.update(
            motion_interval_available=not initial,
            native_angular_speed_rad_s=float(np.linalg.norm(row['object_angular_velocity'])),
            hold_enabled=bool(hold_enabled),
            hold_blocker_mask=np.uint16(sum(1 << HOLD_BLOCKERS.index(k) for k in self.last_blockers)),
            primary_score_success=bool(self.success),
            legacy_score_success=bool(self.legacy.success),
            primary_stable_window=np.int64(len(self.primary.window)),
            legacy_stable_window=np.int64(len(self.legacy.window)),
        )
        return d

    def derive(self, row):
        """Initial snapshot has no preceding control interval to measure."""
        return self._augment(row, self.primary.derive(row), False, initial=True)

    def update(self, row, hold_enabled=True):
        # Collection terminates on the primary result. Freeze the diagnostic
        # legacy result at its own terminal state, as its original collector did.
        if self.legacy.reason is None:
            self.legacy.update(row, hold_enabled=hold_enabled)
        d = self.primary.update(row, hold_enabled=hold_enabled)
        p = PROTOCOL
        lower, upper = p['opening_range_m']
        if len(self.primary.window) == p['hold_steps']:
            relative = np.linalg.inv(self.primary.window[0])[None] @ np.stack(self.primary.window)
            d.update(
                relative_drift_window_available=True,
                relative_translation_drift_m=float(np.linalg.norm(relative[:, :3, 3], axis=-1).max()),
                relative_rotation_drift_deg=float(np.rad2deg(Rotation.from_matrix(relative[:, :3, :3]).magnitude().max())),
            )
        tests = (
            not hold_enabled,
            not d['motion_measurement_valid'],
            d['lift_m'] < p['lift_m'],
            not (d['pad_force_each_n'] >= p['pad_force_each_n']).all(),
            not lower <= d['opening_m'] <= upper,
            row['object_table_peak_n'] > p['max_table_force_n'],
            np.linalg.norm(row['object_linear_velocity']) > p['max_linear_speed_m_s'],
            d.get('pose_interval_angular_speed_rad_s', float('inf')) > p['max_angular_speed_rad_s'],
            self.primary.task == 'pick_up_outer_handle_bar' and not d['outer_handle_contact'],
            self.reason is not None and not self.success,
            d.get('relative_translation_drift_m', 0.) > p['max_relative_translation_drift_m'],
            d.get('relative_rotation_drift_deg', 0.) > p['max_relative_rotation_drift_deg'],
        )
        self.last_blockers = [name for name, blocked in zip(HOLD_BLOCKERS, tests) if blocked]
        if hold_enabled:
            self.hold_observations += 1
            self.blocker_counts.update(self.last_blockers)
            self.legacy_angular_exceedances += int(
                np.linalg.norm(row['object_angular_velocity']) > p['max_angular_speed_rad_s'])
        return self._augment(row, d, hold_enabled)

    def report(self, terminal_reason=None):
        return {
            'primary_version': VERSION,
            'primary_success': bool(self.success),
            'primary_reason': self.reason or terminal_reason or 'incomplete',
            'primary_best_window': self.best_window,
            'primary_first_success_step': self.first_success_step,
            'observed_control_steps': self.primary.steps,
            'motion_errors': self.primary.motion_errors,
            'legacy_success_on_collected_prefix': bool(self.legacy.success),
            'legacy_reason_on_collected_prefix': self.legacy.reason or (
                'not_reached_before_primary_terminal' if self.success else terminal_reason or 'incomplete'),
            'legacy_best_window_on_collected_prefix': self.legacy.best_window,
            'legacy_first_success_step': self.legacy.first_success_step,
            'legacy_angular_exceedances_during_hold': self.legacy_angular_exceedances,
            'hold_observations': self.hold_observations,
            'hold_blocker_counts': dict(self.blocker_counts),
            'last_hold_blockers': self.last_blockers,
            'hold_blocker_mask_bits': list(HOLD_BLOCKERS),
            'legacy_contract': legacy.PROTOCOL,
            'primary_contract': PROTOCOL,
        }
