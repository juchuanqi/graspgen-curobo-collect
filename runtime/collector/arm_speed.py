"""Optional joint-path timing, recorded with each episode.

Only MotionGen approach and held-object paths are retimed. Cartesian descent
and support departure retain 0.0018 rad/step interpolation. Acceleration is
for scalar path progress, not each joint at corners. Geometry is rechecked.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

DT = 1.0 / 60.0
BASELINE_STEP = 0.0018
RECOMMENDED = (1.0, 5.0, 0.3, 2.0)


@dataclass(frozen=True)
class SpeedProfile:
    values: tuple | None = None

    def __post_init__(self):
        if self.values is not None:
            if len(self.values) != 4 or any(not math.isfinite(x) or x <= 0 for x in self.values):
                raise ValueError('Arm speed requires four finite positive v_free,a_free,v_carry,a_carry values')
            if max(self.values[0], self.values[2]) > 2.175:
                raise ValueError('Arm speed exceeds the minimum Panda joint velocity limit, 2.175 rad/s')

    @property
    def enabled(self):
        return self.values is not None

    def environment_value(self):
        return ','.join(str(x) for x in self.values) if self.enabled else ''

    def step_limit(self, phase):
        if not self.enabled or phase not in (0, 3):
            return 0.002501
        return self.values[0 if phase == 0 else 2] * DT + 1e-6

    def describe(self):
        values = self.values or (None,) * 4
        return dict(schema=1, name='baseline' if not self.enabled else (
            'recommended' if self.values == RECOMMENDED else 'custom'),
            enabled=self.enabled, v_free_rad_s=values[0], a_free_rad_s2=values[1],
            v_carry_rad_s=values[2], a_carry_rad_s2=values[3],
            straight_segments_step_rad=BASELINE_STEP, control_dt_s=DT)


def parse(value):
    if not value or value == 'baseline':
        return SpeedProfile()
    if value == 'recommended':
        return SpeedProfile(RECOMMENDED)
    try:
        return SpeedProfile(tuple(float(x) for x in value.split(',')))
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid ARM_SPEED_PROFILE: use baseline, recommended, or v_free,a_free,v_carry,a_carry') from error


def from_environment():
    return parse(os.environ.get('ARM_SPEED_PROFILE', '').strip())


def from_metadata(metadata):
    """Old recordings keep the original bound, regardless of ambient env."""
    saved = metadata.get('arm_speed_profile')
    if saved is None:
        return SpeedProfile()
    if not isinstance(saved, dict) or saved.get('schema') != 1 or type(saved.get('enabled')) is not bool:
        raise ValueError('Invalid saved arm speed profile')
    profile = SpeedProfile(tuple(saved[k] for k in (
        'v_free_rad_s', 'a_free_rad_s2', 'v_carry_rad_s', 'a_carry_rad_s2')) if saved['enabled'] else None)
    if saved != profile.describe():
        raise ValueError('Inconsistent saved arm speed profile')
    return profile


def validate_action_deltas(actions, previous_targets, phases, metadata):
    import numpy as np
    profile = from_metadata(metadata)
    delta = np.abs(actions[:, :7] - previous_targets[:, :7]).max(axis=1)
    bounds = np.asarray([profile.step_limit(int(phase)) for phase in phases]) + 1e-7
    if not np.all(delta <= bounds):
        raise AssertionError('Arm target step exceeds the recorded phase speed profile')


def retime(q, vmax, amax):
    """Resample dense joint paths with the previously tested scalar trapezoid."""
    import numpy as np
    if any(not math.isfinite(x) or x <= 0 for x in (vmax, amax)):
        raise ValueError('Retime limits must be finite and positive')
    q = np.asarray(q, np.float64)
    if q.ndim != 2 or not len(q) or not np.isfinite(q).all():
        raise ValueError('Expected a nonempty finite joint path')
    if len(q) < 2:
        return q.astype(np.float32)
    seg = np.abs(np.diff(q, axis=0)).max(axis=1)
    s = np.concatenate([[0.], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return q[:1].astype(np.float32)
    vs, acceleration = vmax * DT, amax * DT * DT
    positions, pos, vel = [0.], 0., 0.
    while total - pos > 1e-12:
        stop = vel * vel / (2 * acceleration)
        if total - pos <= stop + 1e-12:
            vel = max(vel - acceleration, acceleration)
        else:
            vel = min(vel + acceleration, vs)
        vel = min(vel, vs)
        pos = min(total, pos + vel)
        positions.append(pos)
        if len(positions) > 100000:
            raise RuntimeError('Retime did not converge within 100000 steps')
    out = np.stack([np.interp(positions, s, q[:, j]) for j in range(q.shape[1])], axis=1)
    out[0], out[-1] = q[0], q[-1]
    return out.astype(np.float32)
