import numpy as np
import isaaclab.envs.mdp as mdp

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .droid_environment import (
    BinaryJointPositionZeroToOneActionCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


@configclass
class ActionCfg:
    """End-effector (Cartesian) control for the arm, plus binary gripper."""

    # Differential IK in task-space.
    # We use *relative* 6D pose commands (delta position + delta RPY).
    # Action dim = 6: (dx, dy, dz, droll, dpitch, dyaw)
    # - position is in meters
    # - rotation is in radians
    ee_pos = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        # NOTE: your robot USD exposes these bodies (from your traceback):
        # ['panda_link0', ..., 'panda_link8', 'base_link', ...]
        # For Franka, `panda_link8` is commonly the end-effector/flange frame.
        body_name="panda_link8",
        body_offset=None,
        # Scale delta pose commands per env step:
        # - 2cm translation per unit action
        # - ~0.25rad (~14deg) rotation per unit action
        # scale=(0.02, 0.02, 0.02, 0.25, 0.25, 0.25),
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """DROID env variant that uses end-effector Cartesian control for the arm."""

    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # Keep identical sim settings to the joint-position env for comparability.
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True

    def set_scene(self, scene_name: str):
        self.scene.dynamic_scene(scene_name)

