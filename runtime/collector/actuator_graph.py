"""Capture the installed implicit actuator method, keeping native PhysX outside."""
import copy
import time
from types import SimpleNamespace
import torch


class ActuatorGraph:
    input_names = ('joint_pos_target', 'joint_vel_target', 'joint_effort_target', 'joint_pos', 'joint_vel')
    data_outputs = ('computed_torque', 'applied_torque', 'soft_joint_vel_limits', 'gear_ratio')
    sim_outputs = ('_joint_pos_target_sim', '_joint_vel_target_sim', '_joint_effort_target_sim')
    parameters = ('stiffness', 'damping', 'effort_limit', 'velocity_limit')

    def __init__(self, robot, actuator_type):
        started = time.perf_counter()
        self.robot = robot
        self.original = robot._apply_actuator_model
        self.mode = 'original'
        self.calls = 0
        if not all(type(a) is actuator_type for a in robot.actuators.values()):
            raise ValueError('Only exact native ImplicitActuator groups are supported')
        if not robot.actuators or any(hasattr(a, 'gear_ratio') for a in robot.actuators.values()):
            raise ValueError('Variable gear or empty actuator groups unsupported')
        inputs = self._read_inputs()
        shape = inputs[0].shape
        if len(shape) != 2 or shape[0] != 1 or any(not x.is_cuda for x in inputs):
            raise ValueError('Requires one CUDA environment')
        if any(x.shape != shape or x.dtype != inputs[0].dtype or x.device != inputs[0].device for x in inputs):
            raise ValueError('Actuator input layouts differ')
        self.layout = [(x.shape, x.dtype, x.device) for x in inputs]
        self.packed = torch.stack(inputs)
        data = SimpleNamespace(**{name: self.packed[i] for i, name in enumerate(self.input_names)})
        for name in self.data_outputs:
            setattr(data, name, getattr(robot._data, name))
        proxies = {}
        groups = []
        for name, actuator in robot.actuators.items():
            proxy = copy.copy(actuator)
            ids = actuator.joint_indices
            if isinstance(ids, slice):
                normalized = list(range(shape[1]))[ids]
                proxy._joint_indices = ids
            elif isinstance(ids, (tuple, list, torch.Tensor)):
                normalized = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
                if not normalized:
                    raise ValueError('Empty actuator joint group')
                # Basic slices have the same values and assignment semantics for
                # these disjoint contiguous groups, without CPU index conversion.
                if normalized == list(range(normalized[0], normalized[-1]+1)):
                    proxy._joint_indices = slice(normalized[0], normalized[-1]+1)
                else:
                    proxy._joint_indices = torch.tensor(normalized, device=inputs[0].device, dtype=torch.long)
            else:
                raise ValueError('Unsupported actuator joint index representation')
            if not normalized or len(normalized) != len(set(normalized)):
                raise ValueError('Empty or repeated joint indices')
            groups.extend(normalized)
            proxies[name] = proxy
        if sorted(groups) != list(range(shape[1])):
            raise ValueError('Expected disjoint actuator groups covering every joint')
        self.proxy = SimpleNamespace(_data=data, actuators=proxies,
            **{name: getattr(robot, name) for name in self.sim_outputs})
        # Keep original objects alive and detect rebinding. In-place edits of
        # gains/limits remain visible to the captured native compute method.
        self.bindings = [(robot, name, getattr(robot, name)) for name in self.sim_outputs]
        self.bindings += [(robot._data, name, getattr(robot._data, name)) for name in self.data_outputs]
        for name, actuator in robot.actuators.items():
            self.bindings += [(actuator, key, getattr(actuator, key)) for key in self.parameters]
        self.group_binding = tuple(robot.actuators.items())
        self.index_binding = [(a, a.joint_indices, self._index_signature(a.joint_indices))
                              for a in robot.actuators.values()]
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            for _ in range(6):
                self.original.__func__(self.proxy)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.original.__func__(self.proxy)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        torch.cuda.synchronize(inputs[0].device)
        self.initialization_s = time.perf_counter()-started

    @staticmethod
    def _index_signature(ids):
        if isinstance(ids, torch.Tensor):
            return (ids.data_ptr(), ids._version, ids.shape, ids.dtype, ids.device)
        if isinstance(ids, slice):
            return (ids.start, ids.stop, ids.step)
        return tuple(ids)

    def _read_inputs(self):
        # Lazy native joint-position/velocity reads occur here, in original order,
        # before staging. Their data addresses may legitimately change per tick.
        return [getattr(self.robot._data, name) for name in self.input_names]

    def _stage_inputs(self, inputs):
        if any((x.shape, x.dtype, x.device) != layout for x, layout in zip(inputs, self.layout)):
            raise RuntimeError('Actuator input layout changed')
        torch.stack(inputs, out=self.packed)

    def _replay(self):
        self.graph.replay()

    def configure(self, mode):
        if mode not in ('original', 'graph'):
            raise ValueError('Unknown actuator mode: '+str(mode))
        self.mode = mode

    def apply(self):
        if self.mode == 'original':
            return self.original()
        if tuple(self.robot.actuators.items()) != self.group_binding:
            raise RuntimeError('Actuator groups changed')
        for obj, name, value in self.bindings:
            if getattr(obj, name) is not value:
                raise RuntimeError('Captured actuator parameter/output rebound: '+name)
        for actuator, ids, values in self.index_binding:
            current = actuator.joint_indices
            now = self._index_signature(current)
            if current is not ids or now != values:
                raise RuntimeError('Actuator joint mapping changed')
        self._stage_inputs(self._read_inputs())
        self._replay()
        for name, actuator in self.robot.actuators.items():
            actuator.computed_effort = self.proxy.actuators[name].computed_effort
            actuator.applied_effort = self.proxy.actuators[name].applied_effort
        self.calls += 1

