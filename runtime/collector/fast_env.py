"""Same pickup dynamics and observations with fewer unused reads and copies."""
from collections import defaultdict

import numpy as np
import torch
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

from collection_env import PickupCfg, PickupEnv as BaselinePickupEnv, LINKS, NAMES, URDF
from collection_env import raw_contact_centroids


class FilteredContactSensor(ContactSensor):
    """Collector-only sensor: all filtered forces/history, no unused net force.

    This collector never consumes net_forces_w, air-time or debug visualization.
    Those outputs are unsupported here; stock mode remains available for A/B.
    """
    filtered_only = True

    def _update_buffers_impl(self, env_ids):
        if not self.filtered_only:
            return super()._update_buffers_impl(env_ids)
        if (self.cfg.track_pose or self.cfg.track_air_time or
                self.cfg.track_contact_points or self.cfg.debug_vis):
            raise RuntimeError('Filtered-only sensor requires force-matrix-only consumers')
        if not self.cfg.filter_prim_paths_expr:
            raise RuntimeError('Filtered-only sensor needs explicit collision filters')
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        filters = self.contact_physx_view.filter_count
        matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
        self._data.force_matrix_w[env_ids] = matrix.view(-1, self._num_bodies, filters, 3)[env_ids]
        if self.cfg.history_length > 0:
            history = self._data.force_matrix_w_history
            history[env_ids] = history[env_ids].roll(1, dims=1)
            history[env_ids, 0] = self._data.force_matrix_w[env_ids]


class PackedHostRows:
    """One GPU-to-host copy per dtype; retain each field's shape and dtype."""
    def __init__(self):
        self.buffers = {}
        self.layouts = {}

    def copy(self, tensors):
        groups = defaultdict(list)
        for name, value in tensors.items():
            groups[value.dtype].append((name, value))
        result = {}
        for dtype, entries in groups.items():
            layout = tuple((name, tuple(value.shape)) for name, value in entries)
            size = sum(value.numel() for _, value in entries)
            if dtype not in self.buffers:
                self.buffers[dtype] = torch.empty(size, device=entries[0][1].device, dtype=dtype)
                self.layouts[dtype] = layout
            if self.layouts[dtype] != layout:
                raise RuntimeError('Snapshot field order/shape changed after allocation')
            buffer = self.buffers[dtype]
            offset = 0
            for _, value in entries:
                length = value.numel()
                buffer[offset:offset+length].copy_(value.reshape(-1))
                offset += length
            host = buffer.cpu().numpy()
            offset = 0
            for name, value in entries:
                length = value.numel()
                result[name] = host[offset:offset+length].reshape(tuple(value.shape)).copy()
                offset += length
        return result


class PickupEnv(BaselinePickupEnv):
    contact_sensor_type = FilteredContactSensor
    optimized_io = True

    def __init__(self, *args, **kwargs):
        self._host_rows = PackedHostRows()
        self._finger_ids = None
        super().__init__(*args, **kwargs)
        self._finger_ids = [self.robot.find_bodies(link)[0][0]
                            for link in ('panda_leftfinger', 'panda_rightfinger')]
        self._camera_offsets = {}
        for name, camera in self.cameras.items():
            offset = camera.cfg.offset
            self._camera_offsets[name] = torch.tensor(
                [[*offset.pos, *offset.rot]], device=self.device, dtype=torch.float32)

    def set_optimized_io(self, enabled):
        self.optimized_io = bool(enabled)
        for sensor in (*self.contacts.values(), self.object_table, self.object_wall):
            sensor.filtered_only = bool(enabled)

    def _begin_contact_window(self):
        if not self.optimized_io or not hasattr(self, '_contact_window'):
            return super()._begin_contact_window()
        self._nonpad_peak.zero_()
        self._max_penetration.zero_()
        self._physics_apply_calls = 0
        self._contact_sample_count = 0
        self._last_contact_tick = None
        for key, value in self._contact_window.items():
            value.fill_(-1) if key == 'physics_tick' else value.zero_()

    def _capture_pad_contacts(self, physics_tick=None):
        if not self.optimized_io:
            return super()._capture_pad_contacts(physics_tick)
        if not hasattr(self, '_contact_window'):
            self._begin_contact_window()
        tick = self._sim_step_counter if physics_tick is None else physics_tick
        if tick == self._last_contact_tick:
            return
        sample = self._contact_sample_count
        if sample >= 8:
            raise RuntimeError('More than eight distinct physics samples in one control interval')
        if self._finger_ids is None:
            self._finger_ids = [self.robot.find_bodies(link)[0][0]
                                for link in ('panda_leftfinger', 'panda_rightfinger')]
        buffers = []
        metadata = []
        for link in ('panda_leftfinger', 'panda_rightfinger'):
            raw = self.contacts[link].contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            normal, points, normals, separations, counts, starts = raw
            if tuple(counts.shape) != (1, 3) or tuple(starts.shape) != (1, 3):
                raise RuntimeError('Expected one finger body and ordered object/table/wall filters')
            buffers.append(raw)
            metadata.extend((counts.sum(), counts.reshape(-1)[0], starts.reshape(-1)[0]))
        # Same six exact integers as stock, with one synchronization instead
        # of six scalar .item() calls per physics tick. Range checks stay here.
        ranges = torch.stack(metadata).cpu().numpy().reshape(2, 3)
        bodies = self.robot.data.body_link_pose_w[0]
        window = self._contact_window
        nonpad = []
        for finger, (raw, (total, count, start)) in enumerate(zip(buffers, ranges)):
            normal, points, normals, separations, _, _ = raw
            total, count, start = int(total), int(count), int(start)
            if total >= len(normal):
                raise RuntimeError('Raw contact buffer saturated; refusing incomplete classification')
            if not 0 <= count <= 128 or not 0 <= start or start+count > len(points):
                raise RuntimeError('Invalid raw contact buffer range')
            pts = points[start:start+count]
            force = normal[start:start+count].reshape(-1).abs()
            body = bodies[self._finger_ids[finger]]
            local = quat_apply_inverse(body[3:7].expand(count, 4), pts-body[:3]) if count else pts
            inner_y = -.00002 if finger == 0 else .00002
            mask = ((torch.abs(local[:, 0]) <= .00925) &
                    (torch.abs(local[:, 1]-inner_y) <= .001) &
                    (local[:, 2] >= .0355) & (local[:, 2] <= .055))
            pad_force = force[mask].sum()
            nonpad.append(force[~mask].sum())
            if count:
                depth = torch.clamp(-separations[start:start+count].reshape(-1), min=0).max()
                self._max_penetration = torch.maximum(self._max_penetration, depth)
            window['points_w'][sample, finger, :count] = pts
            window['normals_w'][sample, finger, :count] = normals[start:start+count]
            window['normal_force_n'][sample, finger, :count] = force
            window['separation_m'][sample, finger, :count] = separations[start:start+count].reshape(-1)
            window['valid'][sample, finger, :count] = True
            window['pad_mask'][sample, finger, :count] = mask
            window['pad_force_n'][sample, finger] = pad_force
            window['finger_pose_w'][sample, finger] = body
        self._pad_force = window['pad_force_n'][sample]
        self._nonpad_peak = torch.maximum(self._nonpad_peak, torch.stack(nonpad))
        self._raw_contact_points = window['points_w'][sample]
        self._raw_contact_forces = window['normal_force_n'][sample]
        self._raw_contact_valid = window['valid'][sample]
        self._raw_contact_pad_mask = window['pad_mask'][sample]
        # Keep the independent direct-vs-cached observations for score audits.
        raw_object = self.object.root_physx_view.get_transforms()[0]
        raw_hand = self.robot.root_physx_view.get_link_transforms()[0, self.hand_id]
        object_pose = torch.cat((raw_object[:3], raw_object[[6, 3, 4, 5]]))
        hand_pose = torch.cat((raw_hand[:3], raw_hand[[6, 3, 4, 5]]))
        tcp_p, tcp_q = combine_frame_transforms(hand_pose[None, :3], hand_pose[None, 3:7], self.offset)
        window['object_pose_w'][sample] = object_pose
        window['object_velocity_w'][sample] = self.object.root_physx_view.get_velocities()[0]
        window['cached_object_pose_w'][sample] = self.object.data.root_link_pose_w[0]
        window['hand_pose_w'][sample] = hand_pose
        window['tcp_pose_w'][sample] = torch.cat((tcp_p, tcp_q), -1)[0]
        window['physics_tick'][sample] = int(tick)
        self._contact_sample_count += 1
        self._last_contact_tick = tick

    def snapshot(self, capture_rgb=True):
        if not self.optimized_io:
            return super().snapshot(capture_rgb)
        self._capture_pad_contacts()
        for camera in (self.cameras.values() if capture_rgb else []):
            period = camera.cfg.update_period
            camera.cfg.update_period = 0.
            camera.update(0., force_recompute=True)
            camera.cfg.update_period = period
        bodies = self.robot.data.body_link_pose_w
        hand = bodies[:, self.hand_id]
        tcp_p, tcp_q = combine_frame_transforms(hand[:, :3], hand[:, 3:7], self.offset)
        forces, peaks = [], []
        for link in LINKS:
            data = self.contacts[link].data
            forces.append(data.force_matrix_w[0, 0])
            peaks.append(torch.linalg.vector_norm(data.force_matrix_w_history[0, :, 0], dim=-1).max(0).values)
        wrist = self._camera_offsets['wrist_camera']
        wrist_p, wrist_q = combine_frame_transforms(hand[:, :3], hand[:, 3:7], wrist[:, :3], wrist[:, 3:7])
        tensors = dict(
            finger_contact_pos_w=raw_contact_centroids(self._raw_contact_points, self._raw_contact_valid),
            finger_link_pose_w=bodies[0, self._finger_ids],
            qpos=self.robot.data.joint_pos[0, self.joint_ids],
            qvel=self.robot.data.joint_vel[0, self.joint_ids],
            position_targets=self.targets[0], tcp_pose=torch.cat((tcp_p, tcp_q), -1)[0],
            object_pose=self.object.data.root_link_pose_w[0],
            object_linear_velocity=self.object.data.root_lin_vel_w[0],
            object_angular_velocity=self.object.data.root_ang_vel_w[0],
            object_raw_tensor_velocity_w=self.object.root_physx_view.get_velocities()[0],
            link_contact_force=torch.stack(forces), link_contact_peak_n=torch.stack(peaks),
            object_table_peak_n=self.object_table.data.force_matrix_w_history[0].norm(dim=-1).max(),
            object_wall_peak_n=self.object_wall.data.force_matrix_w_history[0].norm(dim=-1).max(),
            hand_pose=hand[0], camera_frame=torch.stack([cam.frame[0] for cam in self.cameras.values()]),
            camera_pose=torch.cat((self._camera_offsets['base_camera'], torch.cat((wrist_p, wrist_q), -1)), 0),
            camera_intrinsics=torch.cat([cam.data.intrinsic_matrices for cam in self.cameras.values()], 0),
            pad_object_force_n=self._pad_force,
            nonpad_object_peak_n=self._nonpad_peak,
            nonpad_object_force_bound_n=self._nonpad_peak*self.nonpad_force_multiplier,
            max_contact_penetration_m=self._max_penetration,
            raw_finger_contact_points_w=self._raw_contact_points,
            raw_finger_contact_normal_force_n=self._raw_contact_forces,
            raw_finger_contact_valid=self._raw_contact_valid,
            raw_finger_contact_pad_mask=self._raw_contact_pad_mask,
            **{'contact_substep_'+key: value for key, value in self._contact_window.items()})
        row = self._host_rows.copy(tensors)
        row['contact_substep_sample_count'] = np.array(self._contact_sample_count, dtype=np.int64)
        row['object_sleeping'] = np.asarray(
            self._sleep_interface.is_sleeping(self._sleep_stage_id, self._sleep_body_path), dtype=bool)
        if capture_rgb:
            rgb = torch.stack([camera.data.output['rgb'][0, ..., :3] for camera in self.cameras.values()]).cpu().numpy()
            cameras = {name: rgb[i].copy() for i, name in enumerate(self.cameras)}
        else:
            cameras = {}
        return row, cameras
