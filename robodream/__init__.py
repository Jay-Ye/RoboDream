'''
RoboDream: compositional world model for robot data synthesis (two-view video diffusion on top of
Cosmos-Predict2-2B, conditioned on robot-only renders, scene / object priors and the trajectory).
'''

from .config import RoboDreamConfig

__all__ = ['RoboDreamConfig']
