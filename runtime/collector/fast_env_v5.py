"""Previous physics/contact graph, optional explicit camera scheduling only."""
from fast_env_v4 import PickupEnv as PreviousPickupEnv
from camera_schedule import ManualCameraSchedule


class PickupEnv(PreviousPickupEnv):
    def configure_cameras(self,mode):
        if not hasattr(self,'_camera_schedulers'):
            self._camera_schedulers={name:ManualCameraSchedule(camera) for name,camera in self.cameras.items()}
        for scheduler in self._camera_schedulers.values():
            scheduler.configure(mode)
        self.camera_schedule_mode=mode
