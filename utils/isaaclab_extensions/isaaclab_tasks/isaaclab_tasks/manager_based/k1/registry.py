import gymnasium as gym

# Register the task so Isaac Lab hydra utilities can load configs from kwargs.
gym.register(
    id="Isaac-K1-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # dummy; Isaac Lab uses cfg entrypoints from kwargs
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.k1.k1_stand_env_cfg:K1StandEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.k1.k1_stand_env_cfg:K1StandRslRlPpoCfg",
    },
)

# K1 Dance: track a reference motion from CSV
gym.register(
    id="Isaac-K1-Dance-v0",
    entry_point="isaaclab_tasks.manager_based.k1.k1_dance_env:K1DanceEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.k1.k1_dance_env_cfg:K1DanceEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.k1.k1_dance_env_cfg:K1DanceRslRlPpoCfg",
    },
)
