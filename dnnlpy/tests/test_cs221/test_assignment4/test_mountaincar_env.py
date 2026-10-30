import gymnasium as gym
from gymnasium.utils.env_checker import check_env


def test_custom_mountain_car_env():
    env = gym.make('CustomMountainCar-v0')
    check_env(env.unwrapped)
