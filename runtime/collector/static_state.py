"""Uniform pre-planning motion evidence collected inside the existing settle."""
import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = dict(version='static_surface_arc_v1', controls=30, physics_hz=480,
                control_hz=60, max_surface_path_bound_m=.0005,
                max_robot_object_force_n=1e-6,
                measurement='sum actor translation increments + bounding radius * SO(3) arc; 240 physical intervals')


def evaluate(data, bounds_min, bounds_max):
    poses = np.asarray(data['object_pose'], dtype=float)
    samples = np.asarray(data['contact_substep_object_pose_w'], dtype=float)
    ticks = np.asarray(data['contact_substep_physics_tick'])
    count = np.asarray(data['contact_substep_sample_count'])
    forces = np.asarray(data['link_contact_peak_n'])
    if (poses.shape != (31, 7) or samples.shape != (31, 8, 7)
            or ticks.shape != (31, 8) or ticks.dtype.kind not in 'iu'
            or count.shape != (31,) or not np.all(count == 8)
            or ticks.min() < 0
            or not np.all(np.diff(ticks.reshape(-1)) == 1)
            or forces.shape != (31, 10, 3)):
        raise ValueError('Static evidence requires31 consecutive eight-substep observations')
    if not all(np.isfinite(v).all() for v in (poses, samples, forces)):
        raise ValueError('Nonfinite static evidence')
    all_q = samples[..., 3:]
    if (np.max(np.abs(np.linalg.norm(all_q, axis=-1) - 1)) > .001
            or np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=-1) - 1)) > .001):
        raise ValueError('Invalid static quaternion')
    if np.max(np.abs(samples[:, -1, :3] - poses[:, :3])) > 1e-6:
        raise ValueError('Static control endpoint mismatch')
    endpoint = Rotation.from_quat(samples[:, -1, [4, 5, 6, 3]])
    cached = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    if np.max((endpoint * cached.inv()).magnitude()) > 1e-5:
        raise ValueError('Static endpoint rotation mismatch')
    sequence = np.concatenate([poses[:1], samples[1:].reshape(-1, 7)])
    rotations = Rotation.from_quat(sequence[:, [4, 5, 6, 3]])
    translation = float(np.linalg.norm(np.diff(sequence[:, :3], axis=0), axis=-1).sum())
    angular = float((rotations[1:] * rotations[:-1].inv()).magnitude().sum())
    lower, upper = np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        raise ValueError('Finite object bounds required')
    radius = float(np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper))))
    bound = translation + radius * angular
    robot_force = float(forces[:, :, 0].max())
    passed = bound <= PROTOCOL['max_surface_path_bound_m'] and robot_force <= PROTOCOL['max_robot_object_force_n']
    return dict(status='passed' if passed else 'ineligible', protocol=PROTOCOL,
                controls=30, physical_intervals=240, duration_s=.5,
                first_physics_tick=int(ticks[0, -1]), last_physics_tick=int(ticks[-1, -1]),
                translation_path_m=translation, rotation_path_rad=angular,
                bounding_radius_m=radius, surface_path_bound_m=bound,
                max_robot_object_force_n=robot_force,
                reason='static_window_passed' if passed else 'motion_or_prior_robot_contact')
