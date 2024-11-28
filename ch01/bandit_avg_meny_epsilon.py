import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent

runs = 200
steps = 1000
all_rates = np.zeros((runs, steps))

# εの設定と色分け
epsilon_values = [0.1, 0.3, 0.01]
colors = ['r', 'g', 'b']
labels = ['epsilon = 0.1', 'epsilon = 0.3', 'epsilon = 0.01']

for i, epsilon in enumerate(epsilon_values):
    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            all_rates[run, step] = total_reward / (step + 1)

    avg_rates = np.mean(all_rates, axis=0)  # 各ステップの平均成功率
    plt.plot(avg_rates, label=labels[i], color=colors[i])  # ラベルと色を指定

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.legend()  # ラベルを表示
plt.title('Comparison of Epsilon Values')
plt.show()
