import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(env, num_episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim)
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    buffer = deque(maxlen=100_000)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update_frequency = 5

    def get_action(state, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return model(state).argmax().item()

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = get_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            reward = np.clip(reward, -1, 1)  # 報酬をクリップ
            buffer.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if done:
                break

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states = torch.tensor(np.stack([x[0] for x in batch]), dtype=torch.float32)
                actions = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                next_states = torch.tensor(np.stack([x[3] for x in batch]), dtype=torch.float32)
                dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)

                q_values = model(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
          print(f"episode :{episode}, total reward : {total_reward}")

    env.close()

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    train_dqn(env)
