"""Fixed obstacle descriptions shared by simulation spawn and planning.

The wall is a thin kinematic cuboid. Its pose is chosen to intrude on the
approach corridor near the target (object rests near x=0.60). A candidate may
require another approach or fail to produce a collision-free plan.
"""
from __future__ import annotations
from copy import deepcopy


WALL = {
    "name": "wall",
    "dims": [0.02, 0.30, 0.22],
    "pose": [0.545, 0.0, 0.11, 1.0, 0.0, 0.0, 0.0],
}


def planner_obstacles(mode):
    if mode in (None, "none", "clear"):
        return []
    if mode == "wall":
        return [deepcopy(WALL)]
    raise ValueError(f"Unsupported obstacle mode: {mode}")


def verify_obstacle_scene(mode, scene):
    """Reject a silent downgrade or geometry change before accepting raw data.

    Historical clear scenes may omit the mode and the empty obstacle list.
    A wall request always needs explicit mode and complete geometry evidence.
    """
    expected = planner_obstacles(mode)
    expected_mode = 'wall' if expected else 'none'
    if (scene.get('obstacle_mode', 'none') != expected_mode
            or scene.get('obstacles', []) != expected):
        raise ValueError(f'Obstacle scene differs from registered mode: {expected_mode}')
