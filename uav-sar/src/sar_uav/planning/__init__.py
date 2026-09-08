"""Planning package -- importing registers all planners in the registry."""

from .base import (BasePlanner, make_planner, cycle_feasibility, leg_metrics,
                   cell_xy, Cycle)
from .problem import PlanningProblem

# importing these modules registers their planner classes
from . import greedy            # noqa: F401  (greedy, greedy_ratio)
from . import cycle_planners    # noqa: F401  (static, rolling)
from . import milp_pulp         # noqa: F401  (milp, optional)

__all__ = ["BasePlanner", "make_planner", "PlanningProblem",
           "cycle_feasibility", "leg_metrics", "cell_xy", "Cycle"]
