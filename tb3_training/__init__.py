from gymnasium.envs.registration import register

register(
    id='TurtleBot3-v0',
    entry_point='your_package_name.env:TurtleBot3Env', # Путь к классу
    max_episode_steps=500,
)