from gymnasium.envs.registration import register

register(
    id="planar_quadcopter_env/PlanarQuadCopter-v0",
    entry_point="planar_quadcopter_env.envs:PlanarQuadCopterEnv",
    max_episode_steps=999
)
