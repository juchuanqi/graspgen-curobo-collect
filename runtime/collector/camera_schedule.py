"""Explicit RGB scheduling for this collector's single-camera instances.

The collector temporarily sets update_period=0 when explicitly fetching RGB;
all other updates use1e9. Keep GPU timestamps but avoid GPU nonzero checks.
"""
import math


class ManualCameraSchedule:
    def __init__(self,camera):
        self.camera=camera
        self.original_update=camera.update
        self.original_schedule=camera._update_outdated_buffers
        self.original_reset=camera.reset
        self.mode='original'
        camera.update=self.update
        camera._update_outdated_buffers=self.schedule
        camera.reset=self.reset

    def configure(self,mode):
        if mode not in ('original','manual'):
            raise ValueError('Unknown camera schedule: '+str(mode))
        c=self.camera
        if (c._num_envs!=1 or c.cfg.history_length!=0 or c.cfg.debug_vis or
                c.cfg.update_latest_camera_pose or c.cfg.data_types!=['rgb'] or
                c.cfg.update_period!=1e9):
            raise ValueError('Manual schedule requires the fixed single RGB collector camera')
        # Switching happens between episodes/diagnostic segments, outside timing.
        self.dirty=bool(c._is_outdated[0].item())
        self.elapsed=float(c._timestamp[0].item())
        if not 0<=self.elapsed<1e6:
            raise ValueError('Manual camera scheduler elapsed time out of supported range')
        self.mode=mode

    def update(self,dt,force_recompute=False):
        if self.mode=='original':
            return self.original_update(dt,force_recompute)
        c=self.camera
        if not math.isfinite(dt) or dt<0 or self.elapsed+dt>=1e6:
            raise ValueError('Unsupported camera time increment')
        period=c.cfg.update_period
        if period not in (0.,1e9):
            raise ValueError('Unexpected camera update period')
        c._timestamp+=dt
        self.elapsed+=dt
        if period==0.:
            self.dirty=True
            c._is_outdated.fill_(True)
        if force_recompute:
            self.schedule()

    def schedule(self):
        if self.mode=='original':
            return self.original_schedule()
        if self.dirty:
            c=self.camera
            # Use original camera buffer implementation, same annotator and ids.
            c._update_buffers_impl(c._ALL_INDICES)
            c._timestamp_last_update.copy_(c._timestamp)
            c._is_outdated.fill_(False)
            self.dirty=False

    def reset(self,env_ids=None):
        self.original_reset(env_ids)
        if self.mode!='original' and (env_ids is None or len(env_ids)>0):
            self.dirty=True
            self.elapsed=0.
