"""Analytic primitive occupancy and conservative convex-polytope collision.

Distances are positive outside. min(part SDF) gives exact union occupancy and
outside distance, but is not claimed to be exact interior union distance.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def transform(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def pose_matrix(pose):
    pose = np.asarray(pose, float)
    if pose.shape == (4, 4):
        return pose.copy()
    if pose.shape != (7,):
        raise ValueError('Pose is a 4x4 matrix or [xyz,qw,qx,qy,qz]')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def part_matrix(part):
    result = np.eye(4)
    result[:3, 3] = part.get('position', [0., 0., 0.])
    if 'quaternion' in part:
        q = np.asarray(part['quaternion'])
        result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    elif 'rpy' in part:
        result[:3, :3] = Rotation.from_euler('xyz', part['rpy']).as_matrix()
    else:
        result[:3, :3] = Rotation.from_euler('z', part.get('yaw', 0.)).as_matrix()
    return result


def primitive_sdf(points, part):
    p = transform(points, np.linalg.inv(part_matrix(part)))
    if part['shape'] == 'box':
        d = np.abs(p) - np.asarray(part['size']) / 2
    elif part['shape'] == 'cylinder':
        d = np.stack((np.linalg.norm(p[:, :2], axis=1) - part['radius'],
                      np.abs(p[:, 2]) - part['length'] / 2), axis=1)
    else:
        raise ValueError(part['shape'])
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)


def union_sdf(points, parts, return_part=False):
    values = np.stack([primitive_sdf(np.asarray(points), part) for part in parts], axis=1)
    indices = values.argmin(axis=1)
    result = values[np.arange(len(values)), indices]
    return (result, indices) if return_part else result


def unique_directions(vectors, decimals=7):
    v = np.asarray(vectors, float).reshape(-1, 3)
    norm = np.linalg.norm(v, axis=1)
    v = v[norm > 1e-9] / norm[norm > 1e-9, None]
    if not len(v):
        return v
    first = np.argmax(np.abs(v) > 1e-7, axis=1)
    v *= np.where(v[np.arange(len(v)), first] < 0, -1., 1.)[:, None]
    _, indices = np.unique(np.round(v, decimals), axis=0, return_index=True)
    return v[np.sort(indices)]


@dataclass
class Convex:
    vertices: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    equations: np.ndarray
    name: str = ''

    @classmethod
    def hull(cls, vertices, name=''):
        vertices = np.unique(np.round(vertices, 12), axis=0)
        h = ConvexHull(vertices)
        triangles = h.simplices
        edge_indices = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        # Triangulated coplanar faces introduce extra axes, which remain valid
        # separating axes; they can only add redundant collision work.
        return cls(vertices[h.vertices], unique_directions(h.equations[:, :3]),
                   unique_directions(vertices[edge_indices[:, 1]] - vertices[edge_indices[:, 0]]),
                   h.equations, name)

    def moved(self, matrix):
        r, t = matrix[:3, :3], matrix[:3, 3]
        n = self.equations[:, :3] @ r.T
        return Convex(transform(self.vertices, matrix), self.normals @ r.T, self.edges @ r.T,
                      np.c_[n, self.equations[:, 3] - n @ t], self.name)

    @property
    def bounds(self):
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])


def primitive_convex(part, cylinder_sections=64, conservative=True):
    if part['shape'] == 'box':
        v = np.array(np.meshgrid(*[[-s / 2, s / 2] for s in part['size']], indexing='ij')).reshape(3, -1).T
    elif part['shape'] == 'cylinder':
        angle = np.arange(cylinder_sections) * 2 * np.pi / cylinder_sections
        radius = part['radius'] / np.cos(np.pi / cylinder_sections) if conservative else part['radius']
        ring = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
        v = np.vstack([np.c_[ring, np.full(len(ring), z)] for z in (-part['length'] / 2, part['length'] / 2)])
    else:
        raise ValueError(part['shape'])
    return Convex.hull(transform(v, part_matrix(part)), part.get('name', ''))


def sat(a, b, clearance=0.):
    """Convex SAT with face and edge-cross axes; no sample-point collision test.

    A positive overlap means intersection. Negative means a separating-axis
    gap. It is a conservative clearance lower bound, not Euclidean distance.
    """
    ab, bb = a.bounds, b.bounds
    gap = np.maximum(ab[0] - bb[1], bb[0] - ab[1])
    if gap.max() > clearance:
        return False, -float(gap.max())
    axes = np.vstack((a.normals, b.normals))
    for include_edges in (False, True):
        if include_edges:
            axes = unique_directions(np.cross(a.edges[:, None, :], b.edges[None, :, :]).reshape(-1, 3))
        if not len(axes):
            continue
        av, bv = a.vertices @ axes.T, b.vertices @ axes.T
        # Directed interval translation distances are needed for containment.
        overlap = np.minimum(av.max(axis=0) - bv.min(axis=0), bv.max(axis=0) - av.min(axis=0))
        if not include_edges:
            minimum = float(overlap.min())
        else:
            minimum = min(minimum, float(overlap.min()))
        if float(overlap.min()) < -clearance:
            return False, float(overlap.min())
    return minimum > -clearance, minimum


def clip_polygon(poly, equation, inside=True, epsilon=1e-12):
    if len(poly) < 3:
        return np.empty((0, 3))
    d = poly @ equation[:3] + equation[3]
    if not inside:
        d = -d
    output = []
    for k, p in enumerate(poly):
        previous = poly[k - 1]
        dp, dc = d[k - 1], d[k]
        pin, cin = dp <= epsilon, dc <= epsilon
        if pin != cin:
            alpha = dp / (dp - dc)
            output.append(previous + alpha * (p - previous))
        if cin:
            output.append(p)
    return np.asarray(output).reshape(-1, 3)


def subtract_convex_polygon(poly, equations):
    """Return polygon fragments outside a convex solid, removing overlap faces."""
    outside = []
    current = poly
    for eq in equations:
        fragment = clip_polygon(current, eq, inside=False)
        if len(fragment) >= 3:
            outside.append(fragment)
        current = clip_polygon(current, eq, inside=True)
        if len(current) < 3:
            break
    return outside
