"""Unified physical grasp score; no object-specific width or grasp target."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

PROTOCOL = {
    "physics_hz": 480, "control_hz": 60, "arm_pd": [10000., 200.], "finger_pd": [400., 20.],
    "lift_m": .06, "hold_steps": 120, "pad_force_each_n": .5,
    "max_table_force_n": .5, "max_forbidden_force_n": 1.,
    "max_linear_speed_m_s": .02, "max_angular_speed_rad_s": .2,
    "max_relative_translation_drift_m": .002, "max_relative_rotation_drift_deg": 5.,
    "opening_range_m": [.002, .078], "max_steps": 2400,
    "pad_definition": "Exact URDF inner rubber-tip face, all raw contact points; qualification required",
    "pad_border_tolerance_m": .0005,
    "pad_normal_tolerance_m": .001,
    "max_contact_penetration_m": .001,
    "planned_contact_penetration_m": .0005,
    "nonpad_force": "Total force conservative Coulomb upper bound from every raw normal contact; native material friction recorded",
}
CONTACT_LINKS = [f"panda_link{i}" for i in range(1, 8)] + [
    "panda_hand", "panda_leftfinger", "panda_rightfinger"]
FINGER_INDICES = [CONTACT_LINKS.index(n) for n in ("panda_leftfinger", "panda_rightfinger")]
FORBIDDEN_LINK_INDICES = [i for i in range(len(CONTACT_LINKS)) if i not in FINGER_INDICES]


def matrix(pose):
    pose = np.asarray(pose)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose(transform):
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3,0,1,2]]
    return np.r_[transform[:3, 3], q]


def pad_mask(local_points, finger_index=None, tolerance_m=.0005):
    """Both fingers' inner tip plane is local y≈0 in the exact Panda URDF."""
    p = np.asarray(local_points)
    return (np.isfinite(p).all(axis=-1)
        & (np.abs(p[..., 0]) <= .00875+tolerance_m)
        & (np.abs(p[..., 1] - (-.00002 if finger_index == 0 else .00002 if finger_index == 1 else 0.)) <= .001)
        & (p[..., 2] >= .036-tolerance_m)
        & (p[..., 2] <= .0545+tolerance_m))


class PhysicalScore:
    def __init__(self, initial_object_pose, task="pick_up_any_stable_region"):
        self.initial = np.array(initial_object_pose, copy=True)
        self.task = task
        self.window = deque(maxlen=120)
        self.best_window = 0
        self.success = False
        self.reason = None
        self.first_success_step = None
        self.steps = 0

    def derive(self, row):
        obj, tcp = matrix(row["object_pose"]), matrix(row["tcp_pose"])
        contact = np.asarray(row["finger_contact_pos_w"])
        local = (contact-obj[:3, 3]) @ obj[:3, :3]
        raw_points = np.asarray(row.get("raw_finger_contact_points_w", np.zeros((2,0,3))))
        raw_force = np.asarray(row.get("raw_finger_contact_normal_force_n", np.zeros((2,0))))
        raw_pad = np.asarray(row.get("raw_finger_contact_pad_mask", np.zeros((2,0),bool)))
        raw_valid = np.asarray(row.get("raw_finger_contact_valid", np.zeros((2,0),bool)))
        raw_local = (raw_points-obj[:3,3]) @ obj[:3,:3]
        active = raw_valid & raw_pad & (raw_force > 1e-6)
        outer = ((raw_local[...,0]>=.036)&(raw_local[...,0]<=.046)
                 &(np.abs(raw_local[...,2])<.015))
        handle_force = np.where(active & outer,raw_force,0.).sum(-1)
        other_force = np.where(active & ~outer,raw_force,0.).sum(-1)
        handle_mean_y = (np.where(active & outer,raw_force*raw_local[...,1],0.).sum(-1)
                         / np.maximum(handle_force,1e-12))
        on_handle = bool((handle_force>=.5).all() and other_force.max()<=1e-6
                         and handle_mean_y[0]*handle_mean_y[1]<0)
        has_handle = bool(handle_force.sum()>1e-6)
        has_body = bool(other_force.sum()>1e-6)
        contact_region = 3 if has_handle and has_body else 2 if has_handle else 1 if has_body else 0
        link_peaks = np.asarray(row["link_contact_peak_n"])
        if link_peaks.shape != (len(CONTACT_LINKS), 3):
            raise ValueError(f"Expected contacts [10 links,3 filters: object/table/wall], got {link_peaks.shape}")
        forbidden = max(float(link_peaks[FORBIDDEN_LINK_INDICES].max()), float(link_peaks[FINGER_INDICES,1:].max()),
            float(row.get("object_wall_peak_n", 0)), float(np.asarray(row["nonpad_object_force_bound_n"]).max()))
        return {"lift_m": float(row["object_pose"][2]-self.initial[2]),
            "opening_m": float(row["qpos"][7:9].sum()),
            "pad_force_each_n": np.asarray(row["pad_object_force_n"]),
            "forbidden_force_n": forbidden, "object_in_tcp": np.linalg.inv(tcp)@obj,
            "contact_points_object_local": local, "outer_handle_contact": on_handle,
            "outer_handle_pad_force_n": handle_force,
            "nonhandle_pad_force_n": other_force,
            "contact_region": contact_region}

    def update(self, row, hold_enabled=True):
        self.steps += 1
        d = self.derive(row)
        required = ("qpos","qvel","position_targets","object_pose","tcp_pose",
            "object_linear_velocity","object_angular_velocity","link_contact_force","link_contact_peak_n",
            "object_table_peak_n","pad_object_force_n","nonpad_object_peak_n","nonpad_object_force_bound_n")
        if not all(np.isfinite(row[k]).all() for k in required):
            self.reason = "nonfinite_state"
        elif d["forbidden_force_n"]>1:
            self.reason = "forbidden_contact"
        elif float(row.get("max_contact_penetration_m", 0))>.001:
            self.reason = "excess_contact_penetration"
        elif row["object_pose"][2]<-.005:
            self.reason = "object_fell"
        region = self.task!="pick_up_outer_handle_bar" or d["outer_handle_contact"]
        valid = bool(self.reason is None and hold_enabled and d["lift_m"]>=.06
            and (d["pad_force_each_n"]>=.5).all() and .002<=d["opening_m"]<=.078
            and row["object_table_peak_n"]<=.5
            and np.linalg.norm(row["object_linear_velocity"])<=.02
            and np.linalg.norm(row["object_angular_velocity"])<=.2 and region)
        if valid:self.window.append(d["object_in_tcp"])
        else:self.window.clear()
        self.best_window = max(self.best_window, len(self.window))
        if len(self.window)==120:
            relative = np.linalg.inv(self.window[0])[None]@np.stack(self.window)
            if (np.linalg.norm(relative[:,:3,3],axis=-1).max()<=.002
                and Rotation.from_matrix(relative[:,:3,:3]).magnitude().max()<=np.deg2rad(5)):
                self.success=True;self.reason="stable_handle_hold" if self.task=="pick_up_outer_handle_bar" else "stable_grasp_hold"
                self.first_success_step=self.steps
        return d


def sanitize(row):
    result = {key:np.array(value,copy=True) for key,value in row.items()}
    for key in ("finger_contact_pos_w","contact_points_object_local"):
        if key in result:
            valid=np.isfinite(result[key]).all(axis=-1)
            result[key+"_valid"]=valid
            result[key]=np.where(valid[...,None],result[key],0.)
    return result
