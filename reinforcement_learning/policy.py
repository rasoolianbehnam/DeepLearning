import tensorflow as tf
import numpy as np
import gym
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
env = gym.make("CartPole-v0")
for episode in range(500):
    episonde_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episonde_rewards += reward
        if done:
            break
    totals.append(episonde_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
