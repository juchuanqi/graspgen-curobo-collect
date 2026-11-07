"""Public CPU adapter and geometry guards for mesh-driven Panda grasping.

All poses use meters and homogeneous column-vector transforms. GraspGen input
poses MUST already be in the original object frame (centering restored once
by inference). No object-specific grasp offsets or removed obstacles exist.
"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from .primitives import Convex, primitive_convex, pose_matrix, transform, union_sdf, sat
from .robot import GRASPGEN_TO_TCP, collision_bodies, fk, gripper_transforms, classify_pad_contact, pad_depth

ROOT = Path(__file__).resolve().parents[1]
OPENINGS_M = (.08, .06, .04, .03, .02, .01)
DEFAULT_TABLE = {'shape': 'box', 'size': [1.2, 1.2, .05], 'position': [.495, 0, -.025], 'name': 'table'}


@lru_cache(maxsize=1)
def manifest():
    result = json.loads((ROOT / 'assets/MANIFEST.json').read_text())
    for asset in result['objects'].values():
        for key in ('usd_path', 'mesh_path', 'surface_path'):
            asset[key] = str((ROOT / asset[key]).resolve())
    return result


def object_spec(asset_id):
    if isinstance(asset_id, dict):
        return asset_id
    return manifest()['objects'][asset_id]


@lru_cache(maxsize=5)
def _cached_object(asset_id):
    s = object_spec(asset_id)
    parts = [primitive_convex(p, 64, conservative=True) for p in s['parts']]
    with np.load(s['surface_path']) as f:
        points, normals = f['points'].copy(), f['normals'].copy()
    return parts, points, normals


def signed_distance(asset_id, points_object, return_part=False):
    return union_sdf(np.asarray(points_object).reshape(-1, 3), object_spec(asset_id)['parts'], return_part)


def sample_surface(asset_id, count, seed):
    """Deterministic model-size sampling from already exposed union triangles."""
    import trimesh
    s = object_spec(asset_id)
    mesh = trimesh.load(s['mesh_path'], process=False)
    with np.load(s['surface_path']) as a:
        labels = a['face_part_index']
    rng = np.random.default_rng(int(seed))
    face = rng.choice(len(mesh.faces), int(count), p=mesh.area_faces / mesh.area)
    uv = rng.random((int(count), 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    tri = mesh.triangles[face]
    p = tri[:, 0] + uv[:, :1]*(tri[:, 1]-tri[:, 0]) + uv[:, 1:]*(tri[:, 2]-tri[:, 0])
    n, i = mesh.face_normals[face], labels[face]
    return {'points_object': p, 'normals_object': n, 'part_index': i,
            'center_object': p.mean(axis=0), 'seed': int(seed), 'units': 'meters'}


def graspgen_to_tcp(poses):
    poses = np.asarray(poses, float)
    if poses.shape[-2:] != (4, 4):
        raise ValueError('Grasp poses must end in 4x4')
    return poses @ GRASPGEN_TO_TCP


def _pose_array(value, count):
    p = np.asarray(value, float)
    if p.shape in ((4, 4), (7,)):
        return np.repeat(pose_matrix(p)[None], count, axis=0)
    if p.shape != (count, 4, 4):
        raise ValueError(f'Object pose expected static or [{count},4,4]')
    return p


def _moving_hull(body_a, body_b):
    return Convex.hull(np.vstack((body_a.vertices, body_b.vertices)), body_a.name)


def _path_check(transforms, openings, objects, asset_id, table, allow_pad_contact, motion_slack, check_target_table,
                relative_slack=None, target_slack=None, pad_penetration_m=.0005):
    target_parts = _cached_object(asset_id)[0]
    table_body = primitive_convex(DEFAULT_TABLE if table is None else table) if table is not False else None
    bodies = [collision_bodies(t, pad_penetration_m) for t in transforms]
    count = len(bodies)
    checks, allowed = 0, 0
    maximum_slack = 0.
    relative_slack = motion_slack if relative_slack is None else relative_slack
    target_slack = motion_slack if target_slack is None else target_slack
    def near(a, b, obstacle, slack):
        bounds = np.array([np.minimum(a.bounds[0], b.bounds[0]), np.maximum(a.bounds[1], b.bounds[1])])
        other = obstacle.bounds
        return np.maximum(bounds[0] - other[1], other[0] - bounds[1]).max() <= slack
    def fail(reason, pair, segment, overlap, **extra):
        return {'valid': False, 'reason': reason, 'pair': pair, 'segment': int(segment),
                'penetration_m': float(max(overlap, 0)), 'sat_axis_overlap_m': float(overlap),
                'pair_checks': checks, 'allowed_pad_pairs': allowed, 'max_arc_cover_slack_m': maximum_slack, **extra}
    for k in range(max(1, count - 1)):
        j = min(k + 1, count - 1)
        slack = float(motion_slack[k]) if j != k else 0.
        rel_slack = float(relative_slack[k]) if j != k else 0.
        obj_slack = float(target_slack[k]) if j != k else 0.
        maximum_slack = max(maximum_slack, slack, rel_slack, obj_slack)
        # Work in object frame for moving target checks. Attached target motion
        # is naturally accounted for; no target geometry is removed.
        inv_a, inv_b = np.linalg.inv(objects[k]), np.linalg.inv(objects[j])
        rel_a = {n: (b.moved(inv_a), role) for n, (b, role) in bodies[k].items()}
        rel_b = {n: (b.moved(inv_b), role) for n, (b, role) in bodies[j].items()}
        for name, (body_a, role) in bodies[k].items():
            body_b = bodies[j][name][0]
            if table_body is not None:
                checks += 1
                hit, overlap = False, 0.
                if near(body_a, body_b, table_body, slack + .00005):
                    swept_world = body_a if j == k else _moving_hull(body_a, body_b)
                    hit, overlap = sat(swept_world, table_body, clearance=slack + .00005)
                if hit:
                    return fail('robot_table_collision', [name, 'table'], k, overlap)
            ra, rb = rel_a[name][0], rel_b[name][0]
            swept_relative = None
            for part in target_parts:
                checks += 1
                if not near(ra, rb, part, rel_slack + .00005):
                    continue
                if swept_relative is None:
                    swept_relative = ra if j == k else _moving_hull(ra, rb)
                hit, overlap = sat(swept_relative, part, clearance=rel_slack + .00005)
                if hit:
                    if role == 'pad' and allow_pad_contact:
                        allowed += 1
                        continue
                    return fail('robot_object_collision', [name, part.name], k, overlap)
        if table_body is not None and check_target_table:
            for part in target_parts:
                a, b = part.moved(objects[k]), part.moved(objects[j])
                checks += 1
                if not near(a, b, table_body, obj_slack):
                    continue
                swept = a if j == k else _moving_hull(a, b)
                hit, overlap = sat(swept, table_body, clearance=obj_slack)
                # The initial support plane may touch the target. It must not
                # cross table material more than the uniform 0.5mm tolerance.
                if hit and overlap > .0005 - obj_slack:
                    return fail('object_table_penetration', [part.name, 'table'], k, overlap)
    return {'valid': True, 'reason': 'geometry_clear', 'pair': None, 'pair_checks': checks,
            'allowed_pad_pairs': allowed, 'max_arc_cover_slack_m': maximum_slack,
            'coverage': f'URDF collision STL convex hulls for arm/hand; exact URDF finger boxes, inward{pad_penetration_m*1000:g}mm rubber slab only allowed against target; complete target primitives, circumscribed cylinder64; table box; endpoint hull + second-derivative and curve-deviation bounds',
            'uncovered': 'robot self-collision (cuRobo owns it); base link0 fixed fixture; dynamics and contact-force qualification',
            'poses_checked': count, 'opening_m_range': [float(np.min(openings)), float(np.max(openings))]}


def check_gripper_path(T_world_tcp, opening_m, object_pose, asset_id, table=None, allow_pad_contact=False,
                       check_target_table=True, pad_penetration_m=.0005):
    started = time.monotonic()
    pad_penetration_m = pad_depth(pad_penetration_m)
    poses = np.asarray(T_world_tcp, float)
    if poses.shape == (4, 4):
        poses = poses[None]
    n = len(poses)
    openings = np.broadcast_to(np.asarray(opening_m, float), (n,)).copy()
    if not np.isfinite(poses).all() or not np.isfinite(openings).all() or (openings < 0).any() or (openings > .080001).any():
        raise ValueError('Nonfinite pose or opening outside Panda 0..80mm')
    objects = _pose_array(object_pose, n)
    transforms = [gripper_transforms(p, w) for p, w in zip(poses, openings)]
    if n > 1:
        theta = (Rotation.from_matrix(poses[:-1, :3, :3]).inv() * Rotation.from_matrix(poses[1:, :3, :3])).magnitude()
        obj_theta = (Rotation.from_matrix(objects[:-1, :3, :3]).inv() * Rotation.from_matrix(objects[1:, :3, :3])).magnitude()
        # Conservative arc-to-chord cover for rigid gripper geometry radius
        # 0.25m plus combined object-frame motion, with aperture cross term.
        width_delta = np.abs(np.diff(openings))
        speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) + .25 * theta + .5 * width_delta
        obj_speed = np.linalg.norm(np.diff(objects[:, :3, 3], axis=0), axis=1)
        rel_radius = np.maximum(np.linalg.norm(poses[:-1, :3, 3] - objects[:-1, :3, 3], axis=1),
                                np.linalg.norm(poses[1:, :3, 3] - objects[1:, :3, 3], axis=1)) + .25
        slack = (.25 * theta ** 2 + width_delta * theta) / 8
        rel_slack = (8 * slack + 2 * obj_theta * (speed + obj_speed) + obj_theta ** 2 * rel_radius) / 8
        radius = max(np.linalg.norm(b.vertices, axis=1).max() for b in _cached_object(asset_id)[0])
        obj_slack = radius * obj_theta ** 2 / 8
    else:
        slack = rel_slack = obj_slack = np.zeros(1)
    result = _path_check(transforms, openings, objects, asset_id, table, allow_pad_contact, slack, check_target_table,
                         rel_slack, obj_slack, pad_penetration_m)
    result['interpolation'] = 'TCP translation linear and rotation SLERP; opening linear; object translation linear and rotation SLERP between each supplied sample'
    result.update(arm_checked=False, wall_s=time.monotonic() - started, pad_penetration_m=pad_penetration_m)
    return result


def _group_joint_path(q, openings, max_sum_dq=.012, max_curve_deviation_m=.00001):
    """Compress only when the original piecewise joint curve is covered.

    The maximum L1 distance of every original knot to the endpoint joint
    segment bounds every point of the original piecewise-linear joint curve.
    A 2m joint Lipschitz bound plus aperture residual is added to the sweep.
    This is a geometric projection, so nonlinear time stretching is harmless.
    """
    indices, deviations = [0], []
    start, count = 0, len(q)
    fingers = q[:, 7:9] if q.shape[1] == 9 else openings[:, None] * np.array([[.5, .5]])
    while start < count - 1:
        stop = start + 1
        selected_deviation = 0.
        while stop + 1 < count:
            end = stop + 1
            if np.abs(np.diff(q[start:end + 1, :7], axis=0)).sum() > max_sum_dq:
                break
            delta = q[end, :7] - q[start, :7]
            denom = delta @ delta
            if denom > 1e-24:
                alpha = np.clip((q[start:end + 1, :7] - q[start, :7]) @ delta / denom, 0, 1)
            elif np.linalg.norm(fingers[end] - fingers[start]) > 1e-12:
                fd = fingers[end] - fingers[start]
                alpha = np.clip((fingers[start:end + 1] - fingers[start]) @ fd / (fd @ fd), 0, 1)
            else:
                alpha = np.zeros(end - start + 1)
            joint_deviation = np.abs(q[start:end + 1, :7] - (q[start, :7] + alpha[:, None] * delta)).sum(axis=1)
            finger_deviation = np.abs(fingers[start:end + 1] - (fingers[start] + alpha[:, None] * (fingers[end] - fingers[start]))).sum(axis=1)
            bound = float((2 * joint_deviation + finger_deviation).max())
            if bound > max_curve_deviation_m:
                break
            stop, selected_deviation = end, bound
        indices.append(stop); deviations.append(selected_deviation)
        start = stop
    return np.asarray(indices, int), np.asarray(deviations)


def _check_robot_path_once(qpos, opening_m, object_pose, asset_id, base_pose=None, table=None,
                     allow_pad_contact=False, check_target_table=True, attached_to_tcp=None,
                     compress=True, pad_penetration_m=.0005):
    started = time.monotonic()
    pad_penetration_m = pad_depth(pad_penetration_m)
    q = np.asarray(qpos, float)
    if q.ndim == 1:
        q = q[None]
    if q.shape[1] not in (7, 9):
        raise ValueError('Robot path requires N x 7 or N x 9')
    n = len(q)
    openings = np.broadcast_to(np.asarray(opening_m, float), (n,)).copy()
    if not np.isfinite(q).all() or not np.isfinite(openings).all() or (openings < 0).any() or (openings > .080001).any():
        raise ValueError('Invalid qpos/opening')
    if q.shape[1] == 9 and ((q[:, 7:9] < -.000001).any() or (q[:, 7:9] > .040001).any()
                            or not np.allclose(q[:, 7:9].sum(axis=1), openings, atol=.000002, rtol=0)):
        raise ValueError('Actual 9D finger joints must be0..40mm and sum to supplied total opening')
    base = np.eye(4) if base_pose is None else pose_matrix(base_pose)
    objects = _pose_array(object_pose, n)
    static_target = np.allclose(objects, objects[0], atol=1e-12, rtol=0)
    original_n = n
    if compress and (static_target or attached_to_tcp is not None):
        indices, deviation = _group_joint_path(q, openings)
    else:
        indices, deviation = np.arange(n), np.zeros(max(0, n - 1))
    q, openings, objects = q[indices], openings[indices], objects[indices]
    n = len(q)
    transforms = [fk(v, None if len(v) == 9 else w, base) for v, w in zip(q, openings)]
    attachment = None if attached_to_tcp is None else pose_matrix(attached_to_tcp)
    if attachment is not None:
        objects = np.stack([t['panda_hand_tcp'] @ attachment for t in transforms])
    if n > 1:
        dq = np.abs(np.diff(q[:, :7], axis=0)).sum(axis=1)
        # Product-of-rotations second derivative bound: every point is within
        # 2m of each arm joint in the Panda reachable workspace. This bounds
        # the joint-linear FK arc away from its endpoint vertex chord.
        width_delta = np.abs(np.diff(q[:, 7:9], axis=0)).sum(axis=1) if q.shape[1] == 9 else np.abs(np.diff(openings))
        finger_speed = width_delta if q.shape[1] == 9 else width_delta / 2
        slack = (2 * dq ** 2 + 2 * finger_speed * dq) / 8 + deviation
        if attachment is not None:
            # In this serial chain inv(T_tcp(q))*T_link(q) cancels common
            # ancestors. Remaining inverse joint chain has length <=2m;
            # no arbitrary independent object interpolation is assumed.
            rel_slack = slack.copy()
            target_radius = max(np.linalg.norm(transform(b.vertices, attachment), axis=1).max()
                                for b in _cached_object(asset_id)[0])
            obj_slack = (2 + target_radius) * dq ** 2 / 8 + deviation * (2 + target_radius) / 2
        else:
            da = (Rotation.from_matrix(objects[:-1, :3, :3]).inv() * Rotation.from_matrix(objects[1:, :3, :3])).magnitude()
            obj_speed = np.linalg.norm(np.diff(objects[:, :3, 3], axis=0), axis=1)
            radius = 2 + np.maximum(np.linalg.norm(base[:3, 3] - objects[:-1, :3, 3], axis=1),
                                    np.linalg.norm(base[:3, 3] - objects[1:, :3, 3], axis=1))
            rel_slack = slack + (2 * da * (2 * dq + finger_speed + obj_speed) + da ** 2 * radius) / 8
            target_radius = max(np.linalg.norm(b.vertices, axis=1).max() for b in _cached_object(asset_id)[0])
            obj_slack = target_radius * da ** 2 / 8
    else:
        slack = rel_slack = obj_slack = np.zeros(1)
    result = _path_check(transforms, openings, objects, asset_id, table, allow_pad_contact, slack, check_target_table,
                         rel_slack, obj_slack, pad_penetration_m)
    result.update(arm_checked=True, wall_s=time.monotonic() - started, original_poses=original_n,
                  compressed_poses=n, max_joint_curve_deviation_cover_m=float(deviation.max(initial=0)),
                  finger_positions='actual independent qpos7/qpos8' if q.shape[1] == 9 else 'symmetric opening/2',
                  attached_to_tcp=attachment is not None,
                  pad_penetration_m=pad_penetration_m, _checked_indices=indices.tolist(),
                  interpolation='original q/opening piecewise linear; grouped endpoint hull + 2m joint-chain second derivative and original-curve residual cover; object coupled to TCP' if attachment is not None else 'original q/opening piecewise linear; static target, or supplied target translation linear and rotation SLERP; arbitrary moving targets not compressed')
    return result


def _mid_pose(a, b):
    midpoint = np.eye(4)
    midpoint[:3, 3] = (a[:3, 3] + b[:3, 3]) / 2
    midpoint[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3]))))(.5).as_matrix()
    return midpoint


def check_robot_path(qpos, opening_m, object_pose, asset_id, base_pose=None, table=None,
                     allow_pad_contact=False, check_target_table=True, attached_to_tcp=None,
                     compress=True, pad_penetration_m=.0005, adaptive=True):
    """Conservative continuous world-pair checking with bounded refinement.

    A conservative hull hit is refined only when both exact endpoint checks
    satisfy the unchanged contact/clearance rules. Joint/finger interpolation
    stays linear; an independent object uses linear translation plus SLERP.
    Attachment stays coupled to FK. Unresolved intervals fail closed.
    """
    started = time.monotonic()
    q = np.asarray(qpos, float)
    if q.ndim == 1:
        q = q[None]
    openings = np.broadcast_to(np.asarray(opening_m, float), (len(q),)).copy()
    objects = _pose_array(object_pose, len(q))
    depth = pad_depth(pad_penetration_m)
    params = dict(asset_id=asset_id, base_pose=base_pose, table=table,
                  allow_pad_contact=allow_pad_contact, check_target_table=check_target_table,
                  attached_to_tcp=attached_to_tcp, pad_penetration_m=depth)
    stats = {'enabled': bool(adaptive), 'subdivisions': 0, 'restored_original_knots': 0,
             'maximum_depth_used': 0, 'endpoint_violations': 0, 'checker_calls': 0,
             'max_depth': 12, 'min_interval_fraction': 2. ** -12,
             'arc_bounds_recomputed_after_subdivision': True, 'max_checker_calls': 2048}
    total_checks, total_allowed = 0, 0
    initial_max_slack = 0.

    def once(qa, wa, oa, use_compression):
        nonlocal total_checks, total_allowed, initial_max_slack
        stats['checker_calls'] += 1
        result = _check_robot_path_once(qa, wa, oa, compress=use_compression, **params)
        total_checks += result['pair_checks']; total_allowed += result.get('allowed_pad_pairs', 0)
        initial_max_slack = max(initial_max_slack, result['max_arc_cover_slack_m'])
        return result

    def resolve(qa, wa, oa, parameter, level, use_compression):
        stats['maximum_depth_used'] = max(stats['maximum_depth_used'], level)
        report = once(qa, wa, oa, use_compression)
        indices = np.asarray(report.pop('_checked_indices'), int)
        if report['valid'] or not adaptive or len(qa) == 1:
            return report
        k = int(report['segment']); i, j = int(indices[k]), int(indices[k + 1])
        interval = [float(parameter[i]), float(parameter[j])]
        # Confirmed endpoint violations cannot be fixed by subdividing their
        # neighbors. This includes the frozen50um clearance, not only overlap.
        endpoint_reports = []
        for endpoint in (i, j):
            point = once(qa[endpoint:endpoint + 1], wa[endpoint:endpoint + 1], oa[endpoint:endpoint + 1], False)
            point.pop('_checked_indices')
            endpoint_reports.append(point)
            if not point['valid']:
                stats['endpoint_violations'] += 1
                point.update(segment=int(np.floor(parameter[i])),
                             interval_original_sample=interval,
                             violation_class='endpoint_penetration' if point.get('sat_axis_overlap_m', 0) > 0 else 'endpoint_clearance',
                             violating_original_sample=float(parameter[endpoint]))
                return point
        if j > i + 1:
            # A compressed group retained a nonzero original-curve cover.
            # Restore the real original knots before tightening any chord.
            stats['restored_original_knots'] += j - i - 1
            resolved = resolve(qa[i:j + 1], wa[i:j + 1], oa[i:j + 1], parameter[i:j + 1], level, False)
        elif level >= stats['max_depth'] or stats['checker_calls'] >= stats['max_checker_calls']:
            report.update(reason='sweep_refinement_unresolved', original_reason=report['reason'],
                          interval_original_sample=interval, violation_class='unresolved_conservative_cover')
            return report
        else:
            mid_q = (qa[i] + qa[j]) / 2
            mid_w = (wa[i] + wa[j]) / 2
            mid_o = _mid_pose(oa[i], oa[j])
            stats['subdivisions'] += 1
            resolved = resolve(np.stack((qa[i], mid_q, qa[j])), np.array([wa[i], mid_w, wa[j]]),
                               np.stack((oa[i], mid_o, oa[j])),
                               np.array([parameter[i], np.mean(interval), parameter[j]]), level + 1, False)
        if not resolved['valid']:
            return resolved
        # The already certified prefix and refined interval are complete.
        # Continue only the remaining suffix; never rerun the clear prefix.
        if j < len(qa) - 1:
            return resolve(qa[j:], wa[j:], oa[j:], parameter[j:], level, use_compression)
        return resolved

    result = resolve(q, openings, objects, np.arange(len(q), dtype=float), 0, compress)
    result.pop('_checked_indices', None)
    result.update(wall_s=time.monotonic() - started, pair_checks=total_checks,
                  allowed_pad_pairs=total_allowed, refinement=stats,
                  initial_max_arc_cover_slack_m=initial_max_slack,
                  original_poses=len(q), pad_penetration_m=depth)
    return result


def _contact_span(asset_id, tcp):
    _, points, normals = _cached_object(asset_id)
    local = transform(points, np.linalg.inv(tcp))
    normal = normals @ tcp[:3, :3]
    # Exact rubber face footprint relative to panda_hand_tcp.
    mask = ((np.abs(local[:, 0]) <= .00875) & (local[:, 2] >= -.009) & (local[:, 2] <= .0095)
            & (np.abs(local[:, 1]) <= .039))
    left = mask & (normal[:, 1] > .5) & (local[:, 1] > 0)
    right = mask & (normal[:, 1] < -.5) & (local[:, 1] < 0)
    if not left.any() or not right.any():
        return None
    ymax, ymin = float(local[left, 1].max()), float(local[right, 1].min())
    return {'span_m': ymax - ymin, 'positive_y_m': ymax, 'negative_y_m': ymin,
            'center_y_m': (ymax + ymin) / 2, 'positive_samples': int(left.sum()), 'negative_samples': int(right.sum()),
            'method': 'exposed analytic-union points inside exact pad footprint with opposing normals; geometric screening, not a force-closure proof'}


def prepare_candidates(object_id, poses, scores, original_rank=None, max_candidates=200, generation_index=None):
    # Keep the complete single-call pool through object-only qualification.
    # The scene's fifty-candidate budget is applied after table/target checks.
    started = time.monotonic()
    poses, scores = np.asarray(poses, float), np.asarray(scores, float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or scores.shape != (len(poses),):
        raise ValueError('Expected poses[N,4,4] and scores[N]')
    ranks = np.asarray(original_rank, int) if original_rank is not None else np.empty(len(scores), int)
    generations = np.arange(len(scores)) if generation_index is None else np.asarray(generation_index, int)
    if ranks.shape != scores.shape or generations.shape != scores.shape:
        raise ValueError('Rank and generation_index must each have N elements')
    order = np.argsort(-scores, kind='stable')
    if original_rank is None:
        ranks[order] = np.arange(1, len(scores) + 1)
    selected, rejections, unique = [], [], []
    for index in order:
        g = poses[index]
        rank = int(ranks[index])
        if not np.isfinite(g).all() or not np.isfinite(scores[index]) or not np.allclose(g[3], [0, 0, 0, 1], atol=1e-5):
            rejections.append({'original_rank': rank, 'reason': 'invalid_transform'}); continue
        if abs(np.linalg.det(g[:3, :3]) - 1) > 1e-3 or not np.allclose(g[:3, :3].T @ g[:3, :3], np.eye(3), atol=1e-3):
            rejections.append({'original_rank': rank, 'reason': 'invalid_rotation'}); continue
        duplicate = any(np.linalg.norm(g[:3, 3] - p[:3, 3]) < .005 and
                        (Rotation.from_matrix(p[:3, :3]).inv() * Rotation.from_matrix(g[:3, :3])).magnitude() < np.deg2rad(15) for p in unique)
        if duplicate:
            rejections.append({'original_rank': rank, 'reason': 'duplicate_5mm_15deg'}); continue
        if len(unique) >= max_candidates:
            rejections.append({'original_rank': rank, 'reason': 'dedup_budget'}); continue
        unique.append(g)
        tcp = graspgen_to_tcp(g)
        span = _contact_span(object_id, tcp)
        if span is None or not .002 <= span['span_m'] <= .078:
            rejections.append({'original_rank': rank, 'reason': 'no_opposed_pad_surface'}); continue
        # Both sides must be reachable by the same symmetric mechanical close.
        if abs(span['center_y_m']) > .005:
            rejections.append({'original_rank': rank, 'reason': 'off_center_pad_span', 'span': span}); continue
        chosen, attempts = None, []
        for width in OPENINGS_M:
            if width < 2 * max(abs(span['positive_y_m']), abs(span['negative_y_m'])) + .001:
                attempts.append({'opening_m': width, 'reason': 'does_not_enclose_pad_span'}); continue
            approach = tcp.copy(); approach[:3, 3] -= tcp[:3, 2] * .1
            report = check_gripper_path(np.stack((approach, tcp)), width, np.eye(4), object_id,
                                        table=False, allow_pad_contact=False, check_target_table=False)
            attempts.append({'opening_m': width, 'valid': report['valid'], 'reason': report['reason'], 'pair': report.get('pair')})
            if report['valid']:
                chosen = width; break
        if chosen is None:
            rejections.append({'original_rank': rank, 'reason': 'no_collision_free_preopening', 'width_attempts': attempts}); continue
        selected.append({'T_object_tcp': tcp.tolist(), 'T_object_grasp': g.tolist(), 'score': float(scores[index]),
                         'opening_m': chosen, 'original_rank': rank, 'generation_index': int(generations[index]),
                         'pad_span': span, 'width_attempts': attempts})
    return {'object_id': object_id, 'candidates': selected, 'rejections': rejections,
            'generated': len(poses), 'unique_screened': len(unique), 'accepted_geometry': len(selected),
            'transform': 'T_object_tcp=T_object_grasp @ fixed Rz(+90deg),translation_z=0.1034m; inference already restored object centering',
            'opening_policy_m': list(OPENINGS_M), 'scene_collision_checked': False,
            'wall_s': time.monotonic() - started}
