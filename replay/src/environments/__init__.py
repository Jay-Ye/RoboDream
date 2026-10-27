import gymnasium as gym
from .droid_environment import EnvCfg as DroidEnvCfg
from .droid_ee_environment import EnvCfg as DroidEEEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

gym.register(
    id="DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="DROID_EE",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEEEnvCfg,
    },
    disable_env_checker=True,
)
