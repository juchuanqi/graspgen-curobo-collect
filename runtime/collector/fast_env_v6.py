"""Previous collection behavior plus opt-in native implicit actuator graph."""
from isaaclab.actuators import ImplicitActuator
from fast_env_v5 import PickupEnv as PreviousPickupEnv
from actuator_graph import ActuatorGraph


class PickupEnv(PreviousPickupEnv):
    def configure_actuators(self, mode):
        if not hasattr(self, '_actuator_graph'):
            self._actuator_graph = ActuatorGraph(self.robot, ImplicitActuator)
            self.robot._apply_actuator_model = self._actuator_graph.apply
        self._actuator_graph.configure(mode)
        return dict(mode=mode, graph_initialization_s=self._actuator_graph.initialization_s,
                    groups={name:dict(type=type(a).__name__,joint_indices=str(a.joint_indices))
                            for name,a in self.robot.actuators.items()})
