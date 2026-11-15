"""Independent vectorized hold verifier, migrated from the 421-record trust audit.

Does not import the collector scorer. Static initialization is checked separately.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def require(condition, message):
    if not condition:
        raise ValueError(message)


def rotations(poses):
    poses = np.asarray(poses, dtype=float)
    require(poses.shape[-1] == 7 and np.isfinite(poses).all(), 'invalid pose')
    require(np.all(np.abs(np.linalg.norm(poses[..., 3:], axis=-1) - 1) <= .001),
            'invalid quaternion')
    return Rotation.from_quat(poses.reshape(-1, 7)[:, [4, 5, 6, 3]])


def physical_motion(data, com):
    """Include all eight substeps and previous endpoint; no net-motion fallback."""
    com = np.asarray(com, dtype=float)
    require(com.shape == (3,) and np.isfinite(com).all(), 'explicit COM required')
    pose = np.asarray(data['object_pose'], dtype=float)
    sub = np.asarray(data['contact_substep_object_pose_w'], dtype=float)
    count = np.asarray(data['contact_substep_sample_count'])
    ticks = np.asarray(data['contact_substep_physics_tick'])
    states = len(pose)
    require(sub.shape == (states, 8, 7), 'eight physical poses required')
    require(count.shape == (states,) and np.all(count == 8), 'missing substeps')
    require(ticks.shape == (states, 8) and ticks.dtype.kind in 'iu', 'integer ticks required')
    require(np.all(np.diff(ticks.reshape(-1)) == 1), 'physical tick gap/duplicate')
    require(np.allclose(data['sim_time'], np.arange(states) / 60, rtol=0, atol=1e-10),
            'control chronology')
    rp = rotations(pose)
    rs = rotations(sub)
    endpoint = sub[:, -1]
    require(np.max(np.linalg.norm(endpoint[:, :3] - pose[:, :3], axis=-1)) <= 1e-6,
            'physical/control position mismatch')
    require(np.max((rotations(endpoint) * rp.inv()).magnitude()) <= 1e-5,
            'physical/control rotation mismatch')
    if states == 1:
        return np.array([0.]), np.array([0.])
    # The initial eight samples are retained, but there is no earlier endpoint
    # for them. The first scored interval is observation 1, as in collection.
    seq = np.concatenate([pose[:-1, None, :], sub[1:]], axis=1)
    r = rotations(seq)
    world_com = seq[..., :3] + r.apply(np.tile(com, ((states - 1) * 9, 1))).reshape(states - 1, 9, 3)
    linear = np.linalg.norm(np.diff(world_com, axis=1), axis=-1).sum(axis=1) * 60
    q = seq.reshape(-1, 7)[:, [4, 5, 6, 3]].reshape(states - 1, 9, 4)
    r0 = Rotation.from_quat(q[:, :-1].reshape(-1, 4))
    r1 = Rotation.from_quat(q[:, 1:].reshape(-1, 4))
    angular = (r1 * r0.inv()).magnitude().reshape(states - 1, 8).sum(axis=1) * 60
    return np.r_[0., linear], np.r_[0., angular]


def hold_evidence(data, contract, com):
    """Independent vectorized implementation, using the saved uniform thresholds."""
    require(contract['control_hz'] == 60 and contract['physics_hz'] == 480, 'unsupported rate')
    require(contract['hold_steps'] == 120, 'unsupported hold window')
    linear, angular = physical_motion(data, com)
    obj = data['object_pose']; tcp = data['tcp_pose']
    states = len(obj)
    require(len(data['phase']) + 1 == states, 'phase chronology')
    require(np.array_equal(data['hold_enabled'], np.r_[False, data['phase'] >= 3]), 'hold phase mismatch')
    rp, rt = rotations(obj), rotations(tcp)
    relative_position = rt.inv().apply(obj[:, :3] - tcp[:, :3])
    relative_rotation = rt.inv() * rp
    forbidden = np.maximum.reduce([
        data['link_contact_peak_n'][:, :8].max(axis=(1, 2)),
        data['link_contact_peak_n'][:, 8:, 1:].max(axis=(1, 2)),
        data['object_wall_peak_n'], data['nonpad_object_force_bound_n'].max(axis=1)])
    hard = ((forbidden > contract['max_forbidden_force_n'])
            | (data['max_contact_penetration_m'] > contract['max_contact_penetration_m'])
            | (obj[:, 2] < -.005))
    # Initial conditions are audited separately; production first scores state 1.
    hard[0] = False
    hard_prefix = np.maximum.accumulate(hard)
    opening = data['qpos'][:, 7:9].sum(axis=1)
    lift = obj[:, 2] - obj[0, 2]
    valid = (data['hold_enabled'] & ~hard_prefix
             & (lift >= contract['lift_m'])
             & (data['pad_object_force_n'] >= contract['pad_force_each_n']).all(axis=1)
             & (opening >= contract['opening_range_m'][0])
             & (opening <= contract['opening_range_m'][1])
             & (data['object_table_peak_n'] <= contract['max_table_force_n'])
             & (linear <= contract['max_linear_speed_m_s'])
             & (angular <= contract['max_angular_speed_rad_s']))
    first = None; window = 0; best = 0; proof = None
    for i in range(1, states):
        window = min(120, window + 1) if valid[i] else 0
        best = max(best, window)
        if window != 120:
            continue
        start = i - 119
        translation = np.linalg.norm(relative_position[start:i + 1] - relative_position[start], axis=1).max()
        angle = (relative_rotation[start:i + 1] * relative_rotation[start].inv()).magnitude().max()
        if (translation <= contract['max_relative_translation_drift_m']
                and np.rad2deg(angle) <= contract['max_relative_rotation_drift_deg']):
            first = i
            sl = slice(start, i + 1)
            proof = dict(first_state=start, last_state=i, controls=120,
                         min_lift_m=float(lift[sl].min()),
                         min_pad_force_each_n=data['pad_object_force_n'][sl].min(axis=0).tolist(),
                         max_com_speed_m_s=float(linear[sl].max()),
                         max_pose_angular_speed_rad_s=float(angular[sl].max()),
                         max_table_force_n=float(data['object_table_peak_n'][sl].max()),
                         max_relative_translation_m=float(translation),
                         max_relative_rotation_deg=float(np.rad2deg(angle)))
            break
    return dict(success_on_recorded_prefix=first is not None, first_success_state=first,
                best_window=best, final_window=window, proof=proof,
                max_lift_m=float(lift.max()), hard_violation_states=int(hard.sum()),
                max_forbidden_force_n=float(forbidden.max()),
                motion_measurement='sum of eight COM translation / SO(3) rotation increments per control interval')
