import torch

# GPUが使えるかどうかを最初に確認
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

import os
import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime


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

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
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
    def __init__(self, device):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(self.device)
        self.qnet_target = QNet(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(self.device)
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        # デバイスにテンソルを移動
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

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


# 実験パラメータ
episodes = 200
sync_interval = 20
num_experiments = 3
env = gym.make('CartPole-v1', render_mode='human')

# GPU対応
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存ディレクトリ作成
current_time = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = current_time  # ディレクトリ名を現在時刻に設定
os.makedirs(save_dir, exist_ok=True)

all_rewards = []  # 全実験の報酬履歴

for experiment in range(num_experiments):
    reward_history = []  # 各実験の報酬履歴
    # エージェントをGPU対応
    agent = DQNAgent(device)
    for episode in range(episodes):
        state, info = env.reset()  # state と info をアンパック
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated  # エピソード終了条件

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        # 50エピソードごとに進行状況を表示
        if (episode + 1) % 50 == 0:
            print(f"Experiment {experiment + 1}, Episode {episode + 1}, Total Reward: {total_reward}")

    all_rewards.append(reward_history)

    # 各実験のグラフを保存
    plt.figure()
    plt.plot(range(episodes), reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Experiment {experiment + 1}')
    plt.savefig(f"{save_dir}/{experiment + 1:02d}.png")
    plt.close()

# 各エピソードにおける平均報酬、標準偏差を計算
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

# 信頼区間を計算 (95%信頼区間)
confidence_interval = 1.96 * std_rewards / np.sqrt(num_experiments)

# 平均報酬と信頼区間のグラフを保存
plt.figure()
plt.plot(range(episodes), mean_rewards, label='Mean')
plt.fill_between(range(episodes), mean_rewards - confidence_interval, mean_rewards + confidence_interval, alpha=0.2, label='95% CI')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
plt.title('Average Reward with 95% Confidence Interval')
plt.savefig(f"{save_dir}/average_reward.png")
plt.close()
