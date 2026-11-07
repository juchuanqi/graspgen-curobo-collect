"""CPU geometry for a fixed world-space visual occluder.

OpenGL camera poses use local +X right, +Y up, and -Z forward. Intrinsics
use the computer-vision convention (+X right, +Y down, +Z forward). All
reported coverage is projected cuboid geometry, never a rendered measurement.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


EPS = 1e-10


def _vector(value, length, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def _rotation(value, name):
    quaternion = _vector(value, 4, name)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{name} must be a unit wxyz quaternion")
    quaternion = quaternion / norm
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def _intrinsics(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_intrinsics must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("camera_intrinsics last row must be [0, 0, 1]")
    if abs(matrix[0, 1]) > EPS or abs(matrix[1, 0]) > EPS:
        raise ValueError("camera_intrinsics must have zero skew")
    return matrix


def _polygon(value, name="polygon"):
    polygon = np.asarray(value, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or not np.isfinite(polygon).all():
        raise ValueError(f"{name} must have finite shape (N, 2)")
    return polygon


def _signed_area(polygon):
    if len(polygon) < 3:
        return 0.0
    # Center first so a large principal point does not cause cancellation.
    points = polygon - polygon[0]
    return float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1)) * 0.5)


def polygon_area(polygon):
    """Return the nonnegative area of a 2D polygon in square pixels."""
    return abs(_signed_area(_polygon(polygon)))


def convex_hull_uv(points):
    """Return a counterclockwise hull and reject degenerate projections."""
    points = _polygon(points, "points")
    if len(points) < 3:
        raise ValueError("At least three projected points are required")
    try:
        polygon = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise ValueError("Degenerate projected hull") from error
    if polygon_area(polygon) <= EPS:
        raise ValueError("Degenerate projected hull")
    return polygon


def clip_left(polygon, cut_u):
    """Clip a convex polygon by u <= cut_u, preserving polygon ordering."""
    polygon = _polygon(polygon)
    if not np.isfinite(cut_u):
        raise ValueError("cut_u must be finite")
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=np.float64)
    result = []
    previous = polygon[-1]
    previous_inside = previous[0] <= cut_u
    for current in polygon:
        inside = current[0] <= cut_u
        if inside != previous_inside:
            fraction = (cut_u - previous[0]) / (current[0] - previous[0])
            crossing = previous + fraction * (current - previous)
            crossing[0] = cut_u
            result.append(crossing)
        if inside:
            result.append(current.copy())
        previous, previous_inside = current, inside
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


def left_area_cut(polygon, fraction=0.4):
    """Find the vertical cut retaining a requested fraction of convex-hull area."""
    polygon = _polygon(polygon)
    area = polygon_area(polygon)
    if area <= EPS:
        raise ValueError("Cannot cut a zero-area polygon")
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    low, high = float(polygon[:, 0].min()), float(polygon[:, 0].max())
    for _ in range(80):
        middle = (low + high) / 2
        if polygon_area(clip_left(polygon, middle)) < fraction * area:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def intersect_convex(subject, clipper):
    """Intersect two convex polygons using successive edge half-planes."""
    output = _polygon(subject).copy()
    clipper = _polygon(clipper)
    if len(clipper) < 3 or polygon_area(clipper) <= EPS:
        raise ValueError("clipper must have positive area")
    if _signed_area(clipper) < 0:
        clipper = clipper[::-1]
    for start, end in zip(clipper, np.roll(clipper, -1, axis=0)):
        if len(output) == 0:
            break
        edge = end - start

        def side(point):
            delta = point - start
            return edge[0] * delta[1] - edge[1] * delta[0]

        clipped = []
        previous = output[-1]
        previous_side = side(previous)
        for current in output:
            current_side = side(current)
            current_inside, previous_inside = current_side >= -EPS, previous_side >= -EPS
            if current_inside != previous_inside:
                ratio = previous_side / (previous_side - current_side)
                clipped.append(previous + ratio * (current - previous))
            if current_inside:
                clipped.append(current.copy())
            previous, previous_side = current, current_side
        output = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    return output


def project_cuboid(camera_pos, camera_rot_wxyz, camera_intrinsics, object_pose, object_size_m):
    """Project all eight cuboid corners; reject any corner behind the camera.

    Returns (hull_uv, corner_depths_cv_m). This does not clip by image boundaries
    or model lens distortion, robot occlusion, or rendered visibility.
    """
    camera_pos = _vector(camera_pos, 3, "camera_pos")
    camera_rotation = _rotation(camera_rot_wxyz, "camera_rot_wxyz")
    matrix = _intrinsics(camera_intrinsics)
    object_pose = _vector(object_pose, 7, "object_pose")
    object_rotation = _rotation(object_pose[3:], "object_rot_wxyz")
    size = _vector(object_size_m, 3, "object_size_m")
    if np.any(size <= 0):
        raise ValueError("object_size_m must be strictly positive")
    corners_local = np.asarray(list(product([-0.5, 0.5], repeat=3))) * size
    corners_world = object_rotation.apply(corners_local) + object_pose[:3]
    corners_gl = camera_rotation.inv().apply(corners_world - camera_pos)
    corners_cv = corners_gl * [1, -1, -1]
    depths = corners_cv[:, 2]
    if np.any(depths <= EPS):
        raise ValueError("All cuboid corners must be strictly in front of the camera")
    homogeneous = corners_cv @ matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.isfinite(pixels).all():
        raise ValueError("Cuboid projection is nonfinite")
    return convex_hull_uv(pixels), depths


def design_occluder(camera_pos, camera_rot_wxyz, camera_intrinsics,
                    object_pose, object_size_m, *, mode="base_partial",
                    depth_m=None, margin_px=4.0, thickness_m=0.001):
    """Design a camera-aligned cuboid covering the left 40% of the target hull.

    The cuboid is placed once at reset and then stays fixed in world space.
    Its finite thickness is included in the panel silhouette. A four-pixel
    minimum margin extends left, up, and down; the active right edge stays at
    the exact area cut, so the margin never silently increases coverage.
    """
    if mode not in ("none", "base_partial"):
        raise ValueError("mode must be 'none' or 'base_partial'")
    camera_pos = _vector(camera_pos, 3, "camera_pos")
    camera_rotation = _rotation(camera_rot_wxyz, "camera_rot_wxyz")
    matrix = _intrinsics(camera_intrinsics)
    object_pose = _vector(object_pose, 7, "object_pose")
    size = _vector(object_size_m, 3, "object_size_m")
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be nonnegative and finite")
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("thickness_m must be positive and finite")
    hull, object_depths = project_cuboid(camera_pos, camera_rot_wxyz, matrix, object_pose, size)
    result = {
        "mode": mode,
        "geometry_only": True,
        "method": "opengl_to_cv_cuboid_hull_halfplane_area_cut_v1",
        "coverage_basis": "projected_3d_cuboid_hulls_not_rendered_visibility",
        "placement_policy": "fixed_world_pose_after_reset",
        "requested_fraction": 0.0 if mode == "none" else 0.4,
        "geometric_initial_cover_fraction": 0.0,
        "target_hull_uv": hull.tolist(),
        "target_hull_area_px2": polygon_area(hull),
        "panel_polygon_uv": [],
        "panel": None,
        "cut_u_px": None,
        "camera_depth_m": None,
        "margin_px": float(margin_px),
        "margin_edges": ["left", "top", "bottom"],
        "camera_convention": "opengl_local_x_right_y_up_negative_z_forward_wxyz",
    }
    if mode == "none":
        return result

    center_cv = camera_rotation.inv().apply(object_pose[:3] - camera_pos) * [1, -1, -1]
    depth = center_cv[2] * 0.5 if depth_m is None else float(depth_m)
    if not np.isfinite(depth):
        raise ValueError("depth_m must be finite")
    face_depths = np.asarray([depth - thickness_m / 2, depth + thickness_m / 2])
    if face_depths[0] <= EPS or face_depths[1] >= object_depths.min():
        raise ValueError("The entire panel must lie strictly between the camera origin and the entire object")
    cut = left_area_cut(hull, 0.4)
    left = float(hull[:, 0].min() - margin_px)
    top = float(hull[:, 1].min() - margin_px)
    bottom = float(hull[:, 1].max() + margin_px)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]

    # Each face extends left/up/down beyond the requested pixel bounds. The
    # most rightward silhouette edge (possibly the back face) lands on cut.
    x_left = np.min((left - cx) * face_depths / fx)
    x_right = np.min((cut - cx) * face_depths / fx)
    y_top = np.min((top - cy) * face_depths / fy)
    y_bottom = np.max((bottom - cy) * face_depths / fy)
    panel_size = np.asarray([x_right - x_left, y_bottom - y_top, thickness_m])
    if np.any(panel_size <= EPS):
        raise ValueError("Requested panel geometry is degenerate")
    center_gl = np.asarray([(x_left + x_right) / 2, -(y_top + y_bottom) / 2, -depth])
    panel_position = camera_rotation.apply(center_gl) + camera_pos
    normalized_rotation = camera_rotation.as_quat()[[3, 0, 1, 2]]
    panel = {"pos": panel_position.tolist(), "rot_wxyz": normalized_rotation.tolist(),
             "size_m": panel_size.tolist()}
    panel_hull, _ = project_cuboid(camera_pos, camera_rot_wxyz, matrix,
                                    np.r_[panel_position, normalized_rotation], panel_size)
    intersection = intersect_convex(hull, panel_hull)
    fraction = polygon_area(intersection) / polygon_area(hull)
    if not np.isclose(fraction, 0.4, atol=1e-8, rtol=0):
        raise ValueError(f"Panel silhouette coverage differs from requested fraction: {fraction}")
    result.update(panel=panel, panel_polygon_uv=panel_hull.tolist(), cut_u_px=float(cut),
                  camera_depth_m=float(depth), geometric_initial_cover_fraction=float(fraction),
                  covered_target_polygon_uv=intersection.tolist(),
                  requested_margin_bounds_uv=[left, top, float(cut), bottom])
    return result
