"""Fixed single-environment force sensor scheduling, with ordered ring history.

Only the collector's update_period=0, history-only force sensors are supported.
Host dirty state follows update/reset calls; GPU timestamps retain legacy values.
No CUDA scalar reads are used by the active update/data paths.
"""
import torch


class ForceHistoryRing:
    def __init__(self, ordered):
        self.values = ordered.flip(1).clone()
        self.length = ordered.shape[1]
        self.head = self.length - 1
        self.orders = torch.tensor([
            [(head - age) % self.length for age in range(self.length)]
            for head in range(self.length)], device=ordered.device, dtype=torch.long)
        self.dirty = False

    def push(self, matrix):
        self.head = (self.head + 1) % self.length
        self.values[:, self.head].copy_(matrix)
        self.dirty = True

    def reset(self):
        self.values.zero_()
        self.head = self.length - 1
        self.dirty = True

    def materialize(self, destination):
        if self.dirty:
            torch.index_select(self.values, 1, self.orders[self.head], out=destination)
            self.dirty = False


class PeriodicForceSensorMixin:
    """Opt-in mixin: inherit before the collector's original force sensor."""
    _schedule_mode = 'original'

    def configure_schedule(self, mode):
        if mode not in ('original', 'schedule', 'ring'):
            raise ValueError('Unknown sensor schedule mode: ' + str(mode))
        if self._num_envs != 1 or self.cfg.update_period != 0.:
            raise ValueError('Dedicated scheduler requires one environment, update_period=0')
        if not self.filtered_only or self.cfg.history_length <= 0:
            raise ValueError('Dedicated scheduler requires filtered force history')
        if any((self.cfg.track_pose, self.cfg.track_air_time,
                self.cfg.track_contact_points, self.cfg.debug_vis)):
            raise ValueError('Unsupported optional sensor consumer')
        # Mode changes happen between episodes, never inside the hot path.
        if getattr(self, '_schedule_mode', 'original') == 'ring':
            self._history_ring.materialize(self._data.force_matrix_w_history)
        self._host_dirty = bool(self._is_outdated[0].item())
        self._fixed_env_ids = torch.zeros(1, device=self._timestamp.device, dtype=torch.long)
        if mode == 'ring':
            self._history_ring = ForceHistoryRing(self._data.force_matrix_w_history)
        self._schedule_mode = mode

    def update(self, dt, force_recompute=False):
        if self._schedule_mode == 'original':
            return super().update(dt, force_recompute)
        # At update_period=0 even update(0) marks a new observation, as upstream.
        self._timestamp += dt
        self._host_dirty = True
        self._update_outdated_buffers()

    def _update_outdated_buffers(self):
        if self._schedule_mode == 'original':
            return super()._update_outdated_buffers()
        if self._host_dirty:
            self._update_buffers_impl(self._fixed_env_ids)
            self._timestamp_last_update.copy_(self._timestamp)
            self._is_outdated.fill_(False)
            self._host_dirty = False

    def _update_buffers_impl(self, env_ids):
        if self._schedule_mode != 'ring':
            return super()._update_buffers_impl(env_ids)
        matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
        self._data.force_matrix_w.copy_(matrix.view_as(self._data.force_matrix_w))
        self._history_ring.push(self._data.force_matrix_w)

    @property
    def data(self):
        if self._schedule_mode == 'original':
            return super().data
        self._update_outdated_buffers()
        if self._schedule_mode == 'ring':
            self._history_ring.materialize(self._data.force_matrix_w_history)
        return self._data

    def reset(self, env_ids=None):
        # One environment only: [] is a supported no-op; [0]/None reset all.
        super().reset(env_ids)
        if self._schedule_mode != 'original' and (env_ids is None or len(env_ids) > 0):
            self._host_dirty = True
            if self._schedule_mode == 'ring':
                self._history_ring.reset()
