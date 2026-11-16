"""Original physics and pad reads, optimized force sensor bookkeeping only."""
from fast_env import FilteredContactSensor
from fast_env_v2 import PickupEnv as PreviousPickupEnv
from sensor_schedule import PeriodicForceSensorMixin


class ScheduledContactSensor(PeriodicForceSensorMixin, FilteredContactSensor):
    pass


class PickupEnv(PreviousPickupEnv):
    contact_sensor_type = ScheduledContactSensor

    def configure_sensors(self, mode):
        sensors = [*self.contacts.values(), self.object_table, self.object_wall]
        for sensor in sensors:
            sensor.configure_schedule(mode)
        self.sensor_schedule_mode = mode
