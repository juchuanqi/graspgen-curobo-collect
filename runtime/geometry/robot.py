"""Local Panda collision geometry and CPU FK, directly from the deployed URDF.

Arm and palm mesh COLLISION elements are conservatively convexified, with the
original mesh retained for the qualification receipt. Finger collision boxes
remain exactly the deployed URDF boxes. Base link0 is a fixed support fixture.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .primitives import Convex, part_matrix, primitive_convex, transform

URDF = Path(__file__).resolve().parents[1] / "robot/panda_v2.urdf"
ARM_NAMES = [f'panda_joint{i}' for i in range(1, 8)]
HAND_TO_TCP = np.eye(4)
HAND_TO_TCP[2, 3] = .1034
GRASPGEN_TO_TCP = np.eye(4)
GRASPGEN_TO_TCP[:3, :3] = Rotation.from_euler('z', np.pi / 2).as_matrix()
GRASPGEN_TO_TCP[2, 3] = .1034
PAD_PENETRATION_M = .0005


def origin(element):
    if element is None:
        return np.eye(4)
    return part_matrix({'position': np.fromstring(element.get('xyz', '0 0 0'), sep=' '),
                        'rpy': np.fromstring(element.get('rpy', '0 0 0'), sep=' ')})


def pad_depth(value):
    value = float(value)
    if value not in (.0005, .001):
        raise ValueError('pad_penetration_m must be planning0.0005 or measured0.001')
    return value


@lru_cache(maxsize=2)
def description(pad_penetration_m=.0005):
    pad_penetration_m = pad_depth(pad_penetration_m)
    tree = ET.parse(URDF).getroot()
    links = {}
    metadata = []
    for link in tree.findall('link'):
        name = link.get('name')
        bodies = []
        for i, collision in enumerate(link.findall('collision')):
            pose = origin(collision.find('origin'))
            geometry = collision.find('geometry')
            mesh = geometry.find('mesh')
            box = geometry.find('box')
            role = 'solid'
            if mesh is not None:
                path = URDF.parent / mesh.get('filename')
                loaded = trimesh.load(path, process=False)
                if isinstance(loaded, trimesh.Scene):
                    loaded = loaded.to_geometry()
                vertices = loaded.vertices * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                body = Convex.hull(transform(vertices, pose), f'{name}/collision_{i}')
                metadata.append({'link': name, 'path': str(path), 'representation': 'conservative convex hull of deployed URDF collision STL'})
            elif box is not None:
                size = np.fromstring(box.get('size'), sep=' ')
                part = {'shape': 'box', 'size': size.tolist(), 'name': f'{name}/collision_{i}'}
                if name in ('panda_leftfinger', 'panda_rightfinger') and i == 3:
                    # Partition the entire deployed rubber box without clipping
                    # its physical shape. The explicitly selected contact mode
                    # permits its inward0.5mm or1mm; all other surfaces remain.
                    sign = 1 if name == 'panda_leftfinger' else -1
                    inner = pose[1, 3] - sign * size[1] / 2
                    outer = pose[1, 3] + sign * size[1] / 2
                    boundary = inner + sign * pad_penetration_m
                    for label, y0, y1 in [('pad', inner, boundary), ('solid', boundary, outer)]:
                        subsize = size.copy(); subsize[1] = abs(y1 - y0)
                        subpose = pose.copy(); subpose[1, 3] = (y1 + y0) / 2
                        subpart = {**part, 'size': subsize.tolist(), 'name': f'{name}/rubber_{label}'}
                        bodies.append((primitive_convex(subpart).moved(subpose), label))
                    continue
                body = primitive_convex(part).moved(pose)
                metadata.append({'link': name, 'representation': 'exact deployed URDF collision box', 'index': i})
            else:
                raise ValueError(f'Unhandled collision element {name}/{i}')
            bodies.append((body, role))
        links[name] = bodies
    joints = []
    for joint in tree.findall('joint'):
        joints.append({'name': joint.get('name'), 'type': joint.get('type'),
                       'parent': joint.find('parent').get('link'), 'child': joint.find('child').get('link'),
                       'origin': origin(joint.find('origin')),
                       'axis': np.fromstring(joint.find('axis').get('xyz'), sep=' ') if joint.find('axis') is not None else None})
    return links, joints, metadata


def fk(qpos, opening_m=None, base_pose=None):
    q = np.asarray(qpos, float)
    if q.shape not in ((7,), (9,)):
        raise ValueError('qpos must be 7 or 9 joints')
    if opening_m is None:
        if len(q) != 9:
            raise ValueError('7D qpos requires total opening_m')
        fingers = q[7:9]
    else:
        fingers = np.array([opening_m / 2, opening_m / 2])
    values = dict(zip(ARM_NAMES, q[:7]))
    values.update(panda_finger_joint1=fingers[0], panda_finger_joint2=fingers[1])
    transforms = {'panda_link0': np.eye(4) if base_pose is None else np.asarray(base_pose)}
    remaining = list(description()[1])
    while remaining:
        progressed = False
        for joint in list(remaining):
            if joint['parent'] not in transforms:
                continue
            motion = np.eye(4)
            if joint['type'] == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(joint['axis'] * values[joint['name']]).as_matrix()
            elif joint['type'] == 'prismatic':
                motion[:3, 3] = joint['axis'] * values[joint['name']]
            elif joint['type'] != 'fixed':
                raise ValueError(joint['type'])
            transforms[joint['child']] = transforms[joint['parent']] @ joint['origin'] @ motion
            remaining.remove(joint); progressed = True
        if not progressed:
            raise ValueError('URDF tree not connected')
    return transforms


def gripper_transforms(tcp, opening_m):
    hand = tcp @ np.linalg.inv(HAND_TO_TCP)
    result = {'panda_hand': hand}
    for side, sign in (('left', 1), ('right', -1)):
        offset = np.eye(4); offset[:3, 3] = [0, sign * opening_m / 2, .0584]
        result[f'panda_{side}finger'] = hand @ offset
    return result


def collision_bodies(transforms, pad_penetration_m=.0005):
    output = {}
    for name, pose in transforms.items():
        if name == 'panda_link0':
            continue  # fixed support fixture; no planned contact exemption
        for body, role in description(pad_penetration_m)[0].get(name, []):
            output[body.name] = (body.moved(pose), role)
    return output


def classify_pad_contact(finger_local_points, side, tolerance_m=.001, boundary_tolerance_m=.0005):
    p = np.asarray(finger_local_points, float)
    if p.shape[-1] != 3:
        raise ValueError('Contact points end in xyz')
    if side not in ('left', 'right', 'panda_leftfinger', 'panda_rightfinger'):
        raise ValueError('side must be left or right')
    sign = 1 if 'left' in side else -1
    inner_y = sign * (.00758 - .0152 / 2)
    return (np.isfinite(p).all(axis=-1) & (np.abs(p[..., 0]) <= .00875 + boundary_tolerance_m)
            & (p[..., 2] >= .036 - boundary_tolerance_m) & (p[..., 2] <= .0545 + boundary_tolerance_m)
            & (np.abs(p[..., 1] - inner_y) <= tolerance_m))
