import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import random
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 设置保存路径
# 设置保存路径为桌面的"配图"文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, "配图")
# 确保目录存在
os.makedirs(save_path, exist_ok=True)
print(f"图片将保存到: {save_path}")
os.makedirs(save_path, exist_ok=True)

# 设置随机种子以保证结果可重复
np.random.seed(42)
random.seed(42)


# DRL环境类
class RobotEnv:
    def __init__(self, size=20, obstacle_density=0.15, dynamic_obstacles=True):
        self.size = size
        self.obstacle_density = obstacle_density
        self.dynamic_obstacles = dynamic_obstacles
        self.reset()

    def reset(self):
        # 创建网格地图
        self.grid = np.zeros((self.size, self.size))

        # 设置起点和目标
        self.start = (1, 1)
        self.goal = (self.size - 2, self.size - 2)

        # 设置机器人位置
        self.robot_pos = self.start

        # 添加静态障碍物
        self.obstacles = []
        for i in range(self.size):
            for j in range(self.size):
                # 避免在起点、终点及其附近放置障碍物
                if ((i, j) != self.start and (i, j) != self.goal and
                        np.sqrt((i - self.start[0]) ** 2 + (j - self.start[1]) ** 2) > 2 and
                        np.sqrt((i - self.goal[0]) ** 2 + (j - self.goal[1]) ** 2) > 2):
                    if np.random.random() < self.obstacle_density:
                        self.grid[i, j] = 1
                        self.obstacles.append((i, j))

        # 添加动态障碍物
        self.dynamic_obs = []
        if self.dynamic_obstacles:
            n_dynamic = int(self.size * 0.2)  # 动态障碍物数量
            for _ in range(n_dynamic):
                while True:
                    x = np.random.randint(1, self.size - 1)
                    y = np.random.randint(1, self.size - 1)
                    if (x, y) not in self.obstacles and (x, y) != self.start and (x, y) != self.goal:
                        # 使用random.choice代替np.random.choice
                        direction_choices = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        self.dynamic_obs.append({
                            'pos': (x, y),
                            'direction': random.choice(direction_choices),
                            'speed': np.random.choice([1, 2, 3]) / 10.0,
                            'counter': 0
                        })
                        break

        return self._get_state()

    def _get_state(self):
        # 简化状态表示
        return {
            'robot_pos': self.robot_pos,
            'goal': self.goal,
            'obstacles': self.obstacles.copy(),
            'dynamic_obstacles': [obs['pos'] for obs in self.dynamic_obs],
            'grid': self.grid.copy()
        }

    def step(self, action):
        # 动作: 0=上, 1=右, 2=下, 3=左, 4=右上, 5=右下, 6=左下, 7=左上
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]

        # 更新机器人位置
        new_pos = (self.robot_pos[0] + directions[action][0],
                   self.robot_pos[1] + directions[action][1])

        # 检查边界
        if (new_pos[0] < 0 or new_pos[0] >= self.size or
                new_pos[1] < 0 or new_pos[1] >= self.size):
            reward = -1  # 超出边界惩罚
            done = False
        # 检查障碍物碰撞
        elif self.grid[new_pos] == 1 or new_pos in [obs['pos'] for obs in self.dynamic_obs]:
            reward = -10  # 碰撞惩罚
            done = False
        # 检查是否到达目标
        elif new_pos == self.goal:
            reward = 100  # 到达目标奖励
            done = True
        else:
            # 接近目标的奖励 - 鼓励机器人接近目标
            prev_dist = np.sqrt((self.robot_pos[0] - self.goal[0]) ** 2 +
                                (self.robot_pos[1] - self.goal[1]) ** 2)
            new_dist = np.sqrt((new_pos[0] - self.goal[0]) ** 2 +
                               (new_pos[1] - self.goal[1]) ** 2)

            # 路径平滑度奖励 - 鼓励直线运动，减少转向
            if hasattr(self, 'prev_action') and action != self.prev_action:
                smooth_reward = -0.5  # 转向惩罚
            else:
                smooth_reward = 0.2  # 直线奖励

            # 安全奖励 - 鼓励与障碍物保持距离
            min_obs_dist = float('inf')
            for obs in self.obstacles + [obs['pos'] for obs in self.dynamic_obs]:
                dist = np.sqrt((new_pos[0] - obs[0]) ** 2 + (new_pos[1] - obs[1]) ** 2)
                min_obs_dist = min(min_obs_dist, dist)

            safety_reward = min(1, min_obs_dist / 3)  # 归一化安全奖励

            # 总奖励
            reward = (prev_dist - new_dist) * 5 + smooth_reward + safety_reward - 0.1  # 小惩罚以鼓励快速到达
            done = False

        # 移动机器人
        if new_pos[0] >= 0 and new_pos[0] < self.size and new_pos[1] >= 0 and new_pos[1] < self.size:
            if self.grid[new_pos] == 0:  # 确保不会移动到障碍物上
                self.robot_pos = new_pos

        # 更新动态障碍物
        for obs in self.dynamic_obs:
            obs['counter'] += obs['speed']
            if obs['counter'] >= 1:
                obs['counter'] = 0
                new_x = obs['pos'][0] + obs['direction'][0]
                new_y = obs['pos'][1] + obs['direction'][1]

                # 如果障碍物碰到边界或其他障碍物，则改变方向
                if (new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size or
                        self.grid[new_x, new_y] == 1):
                    obs['direction'] = (-obs['direction'][0], -obs['direction'][1])
                else:
                    obs['pos'] = (new_x, new_y)

        self.prev_action = action
        return self._get_state(), reward, done


# 简化的DRL智能体类
class DRLAgent:
    def __init__(self, env):
        self.env = env
        self.epsilon = 0.1  # 探索率
        self.lr = 0.1  # 学习率
        self.gamma = 0.95  # 折扣因子
        self.q_table = {}  # Q表

    def get_state_key(self, state):
        # 简化状态表示为机器人位置、目标位置以及与最近障碍物的相对位置
        robot_pos = state['robot_pos']
        goal = state['goal']

        # 寻找最近的静态障碍物
        nearest_static_obs = None
        min_static_dist = float('inf')
        for obs in state['obstacles']:
            dist = np.sqrt((robot_pos[0] - obs[0]) ** 2 + (robot_pos[1] - obs[1]) ** 2)
            if dist < min_static_dist:
                min_static_dist = dist
                nearest_static_obs = obs

        # 寻找最近的动态障碍物
        nearest_dynamic_obs = None
        min_dynamic_dist = float('inf')
        for obs in state['dynamic_obstacles']:
            dist = np.sqrt((robot_pos[0] - obs[0]) ** 2 + (robot_pos[1] - obs[1]) ** 2)
            if dist < min_dynamic_dist:
                min_dynamic_dist = dist
                nearest_dynamic_obs = obs

        # 状态key
        static_rel_pos = (0, 0) if nearest_static_obs is None else (
            nearest_static_obs[0] - robot_pos[0], nearest_static_obs[1] - robot_pos[1])
        dynamic_rel_pos = (0, 0) if nearest_dynamic_obs is None else (
            nearest_dynamic_obs[0] - robot_pos[0], nearest_dynamic_obs[1] - robot_pos[1])

        return (robot_pos, goal, static_rel_pos, dynamic_rel_pos)

    def get_action(self, state):
        state_key = self.get_state_key(state)

        # epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 8)  # 随机探索

        # 利用已知的最佳动作
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(8)

        return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(8)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(8)

        # Q-学习更新公式
        old_value = self.q_table[state_key][action]
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])

        new_value = old_value + self.lr * (reward + self.gamma * max_future_q - old_value)
        self.q_table[state_key][action] = new_value

    def train(self, episodes=1000):
        # 训练过程
        rewards_history = []
        steps_history = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 200:  # 步数限制以避免无限循环
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)

                self.update_q_table(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            rewards_history.append(total_reward)
            steps_history.append(steps)

            # 降低探索率
            if episode % 100 == 0 and self.epsilon > 0.01:
                self.epsilon *= 0.9

        return rewards_history, steps_history


# 可视化DRL训练进程
def plot_training_process(rewards, steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=600)

    # 绘制奖励曲线
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, 'b-')
    ax1.set_xlabel('迭代次数', fontsize=14)
    ax1.set_ylabel('总奖励', fontsize=14)
    ax1.set_title('DRL训练过程 - 奖励变化', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 使用滑动窗口平滑曲线
    window_size = 50
    if len(rewards) > window_size:
        smoothed_rewards = []
        for i in range(len(rewards) - window_size + 1):
            smoothed_rewards.append(np.mean(rewards[i:i + window_size]))
        ax1.plot(range(window_size, len(rewards) + 1), smoothed_rewards, 'r-', linewidth=2,
                 label='滑动平均 (窗口大小=50)')
        ax1.legend()

    # 绘制步数曲线
    ax2.plot(episodes, steps, 'g-')
    ax2.set_xlabel('迭代次数', fontsize=14)
    ax2.set_ylabel('步数', fontsize=14)
    ax2.set_title('DRL训练过程 - 步数变化', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 平滑步数曲线
    if len(steps) > window_size:
        smoothed_steps = []
        for i in range(len(steps) - window_size + 1):
            smoothed_steps.append(np.mean(steps[i:i + window_size]))
        ax2.plot(range(window_size, len(steps) + 1), smoothed_steps, 'm-', linewidth=2, label='滑动平均 (窗口大小=50)')
        ax2.legend()

    plt.tight_layout()

    # 保存图像 - 修改为图3-4
    save_file = os.path.join(save_path, "图3-4_深度强化学习训练过程.png")
    fig.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"DRL训练过程图已保存至: {save_file}")


# 可视化DRL决策过程
def visualize_drl_decision_making(env, agent, max_steps=100):
    # 设置环境并获取初始状态
    state = env.reset()

    # 定义行动方向映射
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    direction_names = ['上', '右', '下', '左', '右上', '右下', '左下', '左上']

    # 创建图形
    fig = plt.figure(figsize=(15, 8), dpi=600)
    gs = GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1])

    # 环境可视化
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('深度强化学习局部路径规划决策过程', fontsize=16)

    # Q值可视化
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Q值分布', fontsize=14)

    # 奖励分布
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('预期奖励分布', fontsize=14)

    # 安全系数
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title('安全系数分布', fontsize=14)

    # 路径记录
    robot_path = [state['robot_pos']]
    rewards = []
    actions = []

    # 决策过程模拟
    done = False
    step = 0

    while not done and step < max_steps:
        # 获取动作
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        # 记录数据
        robot_path.append(next_state['robot_pos'])
        rewards.append(reward)
        actions.append(action)

        state = next_state
        step += 1

    # 计算安全系数 (与障碍物的距离)
    safety_factors = []
    for pos in robot_path:
        min_dist = float('inf')
        for obs in env.obstacles + [obs['pos'] for obs in env.dynamic_obs]:
            dist = np.sqrt((pos[0] - obs[0]) ** 2 + (pos[1] - obs[1]) ** 2)
            min_dist = min(min_dist, dist)
        safety_factors.append(min(1.0, min_dist / 3.0))  # 归一化

    # 绘制环境
    grid = env.grid.copy()
    ax1.imshow(grid, cmap='binary', origin='upper', alpha=0.3)

    # 绘制起点和终点
    ax1.plot(env.start[1], env.start[0], 'go', markersize=10, label='起点')
    ax1.plot(env.goal[1], env.goal[0], 'ro', markersize=10, label='终点')

    # 绘制静态障碍物
    for obs in env.obstacles:
        ax1.add_patch(patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                                        linewidth=1, edgecolor='k', facecolor='gray'))

    # 绘制机器人路径
    path_x = [pos[1] for pos in robot_path]
    path_y = [pos[0] for pos in robot_path]
    ax1.plot(path_x, path_y, 'b-', linewidth=2, label='机器人路径')

    # 标记决策点和动作
    for i in range(len(robot_path) - 1):
        if i % 3 == 0 or i == len(robot_path) - 2:  # 每隔几步标记一次
            action = actions[i]
            ax1.annotate(direction_names[action],
                         (path_x[i], path_y[i]),
                         xytext=(path_x[i] + 0.3, path_y[i] + 0.3),
                         fontsize=8, color='blue',
                         arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8))

    # 绘制Q值分布（最后一个状态）
    last_state_key = agent.get_state_key(state)
    if last_state_key in agent.q_table:
        q_values = agent.q_table[last_state_key]
        ax2.bar(range(8), q_values, color='purple')
        ax2.set_xticks(range(8))
        ax2.set_xticklabels(['上', '右', '下', '左', '右上', '右下', '左下', '左上'], rotation=45)
        ax2.set_ylabel('Q值')

    # 绘制奖励分布
    ax3.plot(range(len(rewards)), rewards, 'g-')
    ax3.set_xlabel('步数')
    ax3.set_ylabel('奖励')

    # 绘制安全系数分布
    ax4.plot(range(len(safety_factors)), safety_factors, 'r-')
    ax4.set_xlabel('步数')
    ax4.set_ylabel('安全系数')

    ax1.legend(loc='upper right')
    ax1.set_xlabel('X坐标', fontsize=14)
    ax1.set_ylabel('Y坐标', fontsize=14)

    plt.tight_layout()

    # 保存图像 - 修改为图3-5
    save_file = os.path.join(save_path, "图3-5_深度强化学习局部路径规划决策过程.png")
    fig.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"DRL决策过程图已保存至: {save_file}")


# 执行DRL算法训练与评估
def run_drl_experiment():
    # 创建环境和智能体
    env = RobotEnv(size=20, obstacle_density=0.15)
    agent = DRLAgent(env)

    # 训练智能体
    print("正在训练DRL智能体...")
    rewards, steps = agent.train(episodes=1000)

    # 可视化训练过程
    plot_training_process(rewards, steps)

    # 可视化决策过程
    print("生成DRL决策过程图...")
    visualize_drl_decision_making(env, agent)


if __name__ == "__main__":
    run_drl_experiment()