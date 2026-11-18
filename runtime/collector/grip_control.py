"""Bounded symmetric finger position feedback, without changing native PD.

Experimental until matched physical trials qualify it. Commands remain absolute
joint positions and must be recorded after this controller has run.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GripPolicy:
    force_n: float = 3.0
    deadband_n: float = 0.3
    gain_m_per_n: float = 0.0001
    max_step_m: float = 0.00045
    filter_alpha: float = 0.25

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if not np.isfinite(values).all() or not (
            self.force_n > self.deadband_n >= 0 and self.gain_m_per_n > 0
            and self.max_step_m > 0 and 0 < self.filter_alpha <= 1
        ):
            raise ValueError('Invalid grip feedback policy')


class GripController:
    def __init__(self, opening_m, policy=GripPolicy()):
        if not np.isfinite(opening_m) or not 0 < opening_m <= .08:
            raise ValueError('Invalid initial opening')
        self.policy = policy
        self.maximum = float(opening_m) / 2
        self.target = self.maximum
        self.filtered = None
        self.contact_seen = False

    def command(self, pad_force_n):
        force = np.asarray(pad_force_n, dtype=float)
        if force.shape != (2,) or not np.isfinite(force).all() or (force < 0).any():
            raise ValueError('Invalid bilateral pad force')
        p = self.policy
        mean = float(force.mean())
        self.filtered = mean if self.filtered is None else (
            p.filter_alpha * mean + (1 - p.filter_alpha) * self.filtered)
        self.contact_seen |= bool((force >= .5).all())
        if not self.contact_seen and mean < p.force_n:
            delta = -self.maximum / 89
        else:
            error = self.filtered - p.force_n
            delta = 0. if abs(error) <= p.deadband_n else p.gain_m_per_n * error
        self.target = float(np.clip(self.target + np.clip(delta, -p.max_step_m, p.max_step_m),
                                    0., self.maximum))
        return np.full(2, self.target, dtype=np.float32)

    def describe(self):
        return dict(version='bounded_symmetric_pad_feedback_v1', **asdict(self.policy),
                    maximum_finger_position_m=self.maximum, observation='previous control pad force',
                    pd='inherited native 400N/m,20Ns/m', command='two equal absolute finger joint targets')
