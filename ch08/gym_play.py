import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, truncated, info = env.step(action)
    done = done or truncated  # エピソード終了条件
env.close()
