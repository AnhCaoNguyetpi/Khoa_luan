"""
sar_uav -- Data-Driven Belief-Adaptive Multi-UAV Search and Routing
for Missing Tourists in Wilderness Areas.

Reference document: docs/proposal/proposal.tex (research proposal).

Package layout
--------------
data       spatial grid / GIS layers / weather series
belief     initial probability maps, Bayesian update, lost-person motion model
detection  sensor-dependent detection probability q_ikt
uav        UAV platform specs and state/energy dynamics
planning   greedy / static-route / rolling-horizon (+ optional MILP) planners
sim        hidden ground-truth target simulation and the mission loop
viz        matplotlib helpers (Agg backend)
experiments RQ1..RQ5 experiment drivers and evaluation utilities
"""

__version__ = "0.1.0"
