"""Fixed-capacity GPU contact gathering; no per-substep scalar host reads."""
import numpy as np
import torch
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse
from fast_env import PickupEnv as FirstPickupEnv, PackedHostRows
from fast_env import PickupCfg, LINKS, NAMES, URDF


class CheckedHostRows(PackedHostRows):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def copy(self, tensors):
        result = super().copy(dict(tensors, __io_contact_errors=self.env._contact_errors))
        errors = result.pop('__io_contact_errors')
        if errors.any():
            raise RuntimeError('Raw contact saturation/range error; refusing this control observation: '+str(np.argwhere(errors).tolist()))
        return result


class PickupEnv(FirstPickupEnv):
    def __init__(self, *args, **kwargs):
        self._settle_minimal = False
        super().__init__(*args, **kwargs)
        self._host_rows = CheckedHostRows(self)
        self._contact_indices = torch.arange(128, device=self.device, dtype=torch.long)

    def _begin_contact_window(self):
        super()._begin_contact_window()
        if not hasattr(self, '_contact_errors'):
            self._contact_errors = torch.zeros((8, 2), device=self.device, dtype=torch.bool)
        else:
            self._contact_errors.zero_()

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
        if not hasattr(self, '_contact_indices'):
            self._contact_indices = torch.arange(128, device=self.device, dtype=torch.long)
        if self._finger_ids is None:
            self._finger_ids = [self.robot.find_bodies(link)[0][0]
                                for link in ('panda_leftfinger', 'panda_rightfinger')]
        bodies = self.robot.data.body_link_pose_w[0]
        window = self._contact_window
        nonpad = []
        for finger, link in enumerate(('panda_leftfinger', 'panda_rightfinger')):
            normal, points, normals, separations, counts, starts = (
                self.contacts[link].contact_physx_view.get_contact_data(dt=self.cfg.sim.dt))
            if tuple(counts.shape) != (1, 3) or tuple(starts.shape) != (1, 3):
                raise RuntimeError('Expected one finger body and ordered object/table/wall filters')
            count = counts.reshape(-1)[0]
            start = starts.reshape(-1)[0]
            self._contact_errors[sample, finger] = (
                (counts.sum() >= len(normal)) | (count < 0) | (count > 128) |
                (start < 0) | (start+count > len(points)))
            index = self._contact_indices
            valid = index < count
            take = (start+index).clamp(0, len(points)-1)
            pts = torch.where(valid[:, None], points.index_select(0, take), 0.)
            force = torch.where(valid, normal.reshape(-1).index_select(0, take).abs(), 0.)
            body = bodies[self._finger_ids[finger]]
            local = quat_apply_inverse(body[3:7].expand(128, 4), pts-body[:3])
            inner_y = -.00002 if finger == 0 else .00002
            mask = (valid & (torch.abs(local[:, 0]) <= .00925) &
                    (torch.abs(local[:, 1]-inner_y) <= .001) &
                    (local[:, 2] >= .0355) & (local[:, 2] <= .055))
            # Fixed128 reductions remove implicit CUDA nonzero/boolean-index
            # synchronizations. All padding is zero; force policy is unchanged.
            pad_force = torch.where(mask, force, 0.).sum()
            nonpad.append(torch.where(valid & ~mask, force, 0.).sum())
            separation = torch.where(valid, separations.reshape(-1).index_select(0, take), 0.)
            self._max_penetration = torch.maximum(self._max_penetration, (-separation).clamp_min(0).max())
            window['points_w'][sample, finger].copy_(pts)
            window['normals_w'][sample, finger].copy_(torch.where(valid[:, None], normals.index_select(0, take), 0.))
            window['normal_force_n'][sample, finger].copy_(force)
            window['separation_m'][sample, finger].copy_(separation)
            window['valid'][sample, finger].copy_(valid)
            window['pad_mask'][sample, finger].copy_(mask)
            window['pad_force_n'][sample, finger] = pad_force
            window['finger_pose_w'][sample, finger] = body
        self._pad_force = window['pad_force_n'][sample]
        self._nonpad_peak = torch.maximum(self._nonpad_peak, torch.stack(nonpad))
        self._raw_contact_points = window['points_w'][sample]
        self._raw_contact_forces = window['normal_force_n'][sample]
        self._raw_contact_valid = window['valid'][sample]
        self._raw_contact_pad_mask = window['pad_mask'][sample]
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

    def _pre_physics_step(self, actions):
        if self._settle_minimal:
            self.targets = actions.clone()
            self._physics_apply_calls = 0
        else:
            super()._pre_physics_step(actions)

    def _apply_action(self):
        if self._settle_minimal:
            self._physics_apply_calls += 1
            self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)
        else:
            super()._apply_action()

    def settle(self, controls=120, minimal=True):
        if controls != 120:
            raise ValueError('Only the unchanged120-control settle protocol is supported')
        before = self._sim_step_counter
        old_lazy = self.scene.cfg.lazy_sensor_update
        old_render = self.cfg.sim.render_interval
        try:
            self._settle_minimal = bool(minimal)
            if minimal:
                self.scene.cfg.lazy_sensor_update = True
                self.cfg.sim.render_interval = 2**62
            for _ in range(118 if minimal else 120):
                self.step(self.targets.clone())
        finally:
            self._settle_minimal = False
            self.scene.cfg.lazy_sensor_update = old_lazy
            self.cfg.sim.render_interval = old_render
        if minimal:
            # Two full controls provide16 contact samples; the entire saved
            # initial eight-sample window and all force histories are current.
            for _ in range(2):
                self.step(self.targets.clone())
        if self._sim_step_counter-before != 960:
            raise RuntimeError('Settle physics-step count changed')
        return dict(controls=120, physics_steps=960, minimal_controls=118 if minimal else 0,
                    full_controls=2 if minimal else 120,
                    final_contact_history_samples=8,
                    recorded_rgb_starts_after_settle=True)
