"""Execution margins for commanded paths; measured geometry keeps its own floor.

The diagnostic batch observed up to5.41mm finger vertex error near a wall,
and0.244mm at a table contact. These margins exceed those observations;
they are empirical planning guards, not a bound on all future dynamics.
"""
TABLE_M = .001
OBSTACLE_M = .006


def geometry_kwargs():
    return dict(table_clearance_m=TABLE_M, obstacle_clearance_m=OBSTACLE_M)
