"""Sensor ring scheduling plus opt-in CUDA Graph for unchanged pad contact math."""
import time
import torch
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse
from fast_env_v3 import PickupEnv as PreviousPickupEnv
from contact_graph import ContactGraph


class PickupEnv(PreviousPickupEnv):
    contact_mode = 'eager'

    def _contact_inputs(self):
        result = {}
        for finger,link in enumerate(('panda_leftfinger','panda_rightfinger')):
            raw = self.contacts[link].contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            if tuple(raw[4].shape) != (1,3) or tuple(raw[5].shape) != (1,3):
                raise RuntimeError('Expected one finger and ordered object/table/wall filters')
            for i,value in enumerate(raw):
                result[f'{finger}_{i}'] = value
        result.update(bodies=self.robot.data.body_link_pose_w[0],
            raw_object=self.object.root_physx_view.get_transforms()[0],
            raw_hand=self.robot.root_physx_view.get_link_transforms()[0,self.hand_id],
            velocity=self.object.root_physx_view.get_velocities()[0],
            cached_object=self.object.data.root_link_pose_w[0],
            previous_nonpad=self._nonpad_peak,previous_penetration=self._max_penetration)
        return result

    def configure_contacts(self, mode):
        if mode not in ('eager','graph'):
            raise ValueError('Unknown contact mode: '+str(mode))
        if not self.optimized_io or self.num_envs != 1:
            raise ValueError('Contact graph requires optimized single-environment collection')
        if mode == 'graph' and not hasattr(self,'_contact_graph'):
            # Only allocate/capture between episodes, before recorded timing.
            if not hasattr(self,'_contact_window'):
                self._begin_contact_window()
            self._contact_graph = ContactGraph(self._contact_inputs(),self._finger_ids,self.offset,
                                               quat_apply_inverse,combine_frame_transforms)
        self.contact_mode = mode
        return dict(mode=mode,graph_initialization_s=getattr(getattr(self,'_contact_graph',None),'initialization_s',0.))

    def _capture_pad_contacts(self, physics_tick=None):
        if self.contact_mode == 'eager':
            return super()._capture_pad_contacts(physics_tick)
        tick = self._sim_step_counter if physics_tick is None else physics_tick
        if tick == self._last_contact_tick:
            return
        sample = self._contact_sample_count
        if sample >= 8:
            raise RuntimeError('More than eight distinct physics samples in one control interval')
        out = self._contact_graph.replay(self._contact_inputs())
        for key,destination in self._contact_window.items():
            if key != 'physics_tick':
                destination[sample].copy_(out[key])
        self._contact_errors[sample].copy_(out['errors'])
        self._nonpad_peak = out['nonpad_peak']
        self._max_penetration = out['penetration']
        self._pad_force = self._contact_window['pad_force_n'][sample]
        self._raw_contact_points = self._contact_window['points_w'][sample]
        self._raw_contact_forces = self._contact_window['normal_force_n'][sample]
        self._raw_contact_valid = self._contact_window['valid'][sample]
        self._raw_contact_pad_mask = self._contact_window['pad_mask'][sample]
        self._contact_window['physics_tick'][sample] = int(tick)
        self._contact_sample_count += 1
        self._last_contact_tick = tick
