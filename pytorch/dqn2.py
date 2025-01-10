import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]), dtype=torch.float32)
        action = torch.tensor(np.array([x[1] for x in data]), dtype=torch.int64)
        reward = torch.tensor(np.array([x[2] for x in data]), dtype=torch.float32)
        next_state = torch.tensor(np.stack([x[3] for x in data]), dtype=torch.float32)
        done = torch.tensor(np.array([x[4] for x in data]), dtype=torch.float32)
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0001  # 学習率を小さくする
        self.epsilon = 1.0  # 初期epsilonを1.0に設定
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # epsilonの減少率
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size * 10:  # バッファが十分溜まるまで待つ
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs.gather(1, action.unsqueeze(1)).squeeze()

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


episodes = 1000
sync_interval = 100  # 同期間隔を長くする
env = gym.make('CartPole-v1', render_mode='human')
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        reward = np.clip(reward, -1, 1)  # 報酬をクリップ

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))

# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

# === Play CartPole ===
agent.epsilon = 0  # greedy policy
state, info = env.reset()  # state と info をアンパック
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    done = done or truncated  # エピソード終了条件
    state = next_state
    total_reward += reward
    env.render()
print('Total Reward:', total_reward)
