import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import heapq
import math
import os
from matplotlib.font_manager import FontProperties

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'NSimSun',
                                   'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# 设置保存路径为桌面的"配图"文件夹
# 设置保存路径为桌面的"配图"文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, "配图")
# 确保目录存在
os.makedirs(save_path, exist_ok=True)
print(f"图片将保存到: {save_path}")
os.makedirs(save_path, exist_ok=True)


# 创建地形高度图 (较复杂的沙漠和草原地形)
def create_terrain_map(map_type='desert', size=100):
    terrain = np.zeros((size, size))

    if map_type == 'desert':
        # 沙漠地形：流动沙丘
        for i in range(5):
            x_center = np.random.randint(20, size - 20)
            y_center = np.random.randint(20, size - 20)
            height = np.random.uniform(2.0, 5.0)
            width = np.random.randint(10, 25)
            length = np.random.randint(15, 35)
            angle = np.random.uniform(0, 2 * np.pi)

            x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
            x_rot = (x_grid - x_center) * np.cos(angle) - (y_grid - y_center) * np.sin(angle)
            y_rot = (x_grid - x_center) * np.sin(angle) + (y_grid - y_center) * np.cos(angle)

            dune = height * np.exp(-((x_rot / length) ** 2 + (y_rot / width) ** 2))
            terrain += dune

    elif map_type == 'grassland':
        # 草原地形：缓坡和起伏
        for i in range(8):
            x_center = np.random.randint(10, size - 10)
            y_center = np.random.randint(10, size - 10)
            height = np.random.uniform(0.2, 1.0)
            radius = np.random.randint(15, 35)

            x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
            dist = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)

            hill = height * np.exp(-(dist / radius) ** 2)
            terrain += hill

        # 添加一些随机噪声来模拟微小起伏
        terrain += np.random.normal(0, 0.05, (size, size))

    # 添加障碍物
    for i in range(12):
        x_obs = np.random.randint(5, size - 5)
        y_obs = np.random.randint(5, size - 5)
        obs_size = np.random.randint(2, 5)
        terrain[max(0, x_obs - obs_size):min(size, x_obs + obs_size),
        max(0, y_obs - obs_size):min(size, y_obs + obs_size)] = np.nan

    return terrain


# 定义传统A*算法
def traditional_astar(grid, start, goal):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}

    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if (neighbor[0] < 0 or neighbor[0] >= len(grid) or
                    neighbor[1] < 0 or neighbor[1] >= len(grid[0])):
                continue

            if np.isnan(grid[neighbor[0], neighbor[1]]):
                continue

            # 计算代价
            tentative_g_score = g_score[current] + np.sqrt(dx ** 2 + dy ** 2)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return []  # 如果没有找到路径


# 改进的A*算法 (考虑地形和方向因素)
def improved_astar(grid, start, goal):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: improved_heuristic(start, goal, grid)}
    open_set_hash = {start}
    prev_direction = None

    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if (neighbor[0] < 0 or neighbor[0] >= len(grid) or
                    neighbor[1] < 0 or neighbor[1] >= len(grid[0])):
                continue

            if np.isnan(grid[neighbor[0], neighbor[1]]):
                continue

            # 计算当前方向
            current_direction = (dx, dy)

            # 地形代价 (高度差异越大，代价越高)
            terrain_cost = 0
            if current in came_from:
                if not np.isnan(grid[current[0], current[1]]) and not np.isnan(grid[neighbor[0], neighbor[1]]):
                    height_diff = abs(grid[neighbor[0], neighbor[1]] - grid[current[0], current[1]])
                    terrain_cost = height_diff * 2.0

            # 方向变化代价 (鼓励平滑路径)
            direction_cost = 0
            if prev_direction and current_direction != prev_direction:
                direction_cost = 0.5

            # 总体代价
            move_cost = np.sqrt(dx ** 2 + dy ** 2) + terrain_cost + direction_cost
            tentative_g_score = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                prev_direction = current_direction
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + improved_heuristic(neighbor, goal, grid)

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return []  # 如果没有找到路径


# 传统启发式函数 (欧几里德距离)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 改进启发式函数 (考虑方向和切比雪夫距离)
def improved_heuristic(a, b, grid):
    # 切比雪夫距离
    chebyshev = max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # 余弦相似度 (方向因素)
    if a != (0, 0) and b != (0, 0):
        dot_product = a[0] * b[0] + a[1] * b[1]
        magnitude_a = np.sqrt(a[0] ** 2 + a[1] ** 2)
        magnitude_b = np.sqrt(b[0] ** 2 + b[1] ** 2)
        similarity = dot_product / (magnitude_a * magnitude_b + 1e-10)
        direction_factor = 1.0 - (similarity + 1.0) / 2.0  # 归一化到[0,1]
    else:
        direction_factor = 0

    return chebyshev * (1.0 + 0.5 * direction_factor)


# 重建路径
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]  # 反转路径


# 绘制地形和路径（仅使用中文）
def plot_terrain_and_paths(terrain, trad_path, improved_path, title, map_type):
    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)  # Set to 600 DPI as requested

    # 处理NaN值(障碍物)
    terrain_plot = np.copy(terrain)
    mask = np.isnan(terrain)
    terrain_plot[mask] = np.nanmax(terrain) + 2

    # 设置地形颜色映射
    if map_type == 'desert':
        cmap = LinearSegmentedColormap.from_list('desert',
                                                 ['#f6d7b0', '#e4bc84', '#d4a76a', '#b38b50'], N=256)
        terrain_label = "高度 (米)"
        title_suffix = "沙漠环境"
    else:
        cmap = LinearSegmentedColormap.from_list('grassland',
                                                 ['#c9e4ca', '#87bba2', '#55828b', '#3b6064'], N=256)
        terrain_label = "高度 (米)"
        title_suffix = "草原环境"

    # 绘制地形
    im = ax.imshow(terrain_plot, cmap=cmap, origin='lower')
    cbar = fig.colorbar(im, ax=ax, label=terrain_label)

    # 为障碍物设置单独的颜色
    obstacle_color = '#4a4a4a'
    cbar.cmap.set_over(obstacle_color)

    # 绘制传统A*路径
    if trad_path:
        x_trad = [p[1] for p in trad_path]
        y_trad = [p[0] for p in trad_path]
        ax.plot(x_trad, y_trad, 'b-', linewidth=2.5, label='传统A*算法路径')
        ax.plot(x_trad[0], y_trad[0], 'go', markersize=10, label='起点')
        ax.plot(x_trad[-1], y_trad[-1], 'ro', markersize=10, label='终点')

    # 绘制改进A*路径
    if improved_path:
        x_imp = [p[1] for p in improved_path]
        y_imp = [p[0] for p in improved_path]
        ax.plot(x_imp, y_imp, 'r-', linewidth=2.5, label='改进A*算法路径')

    # 计算路径长度和转弯次数
    trad_length, trad_turns = calculate_path_metrics(trad_path)
    imp_length, imp_turns = calculate_path_metrics(improved_path)

    # 添加统计信息到图例
    ax.text(0.02, 0.12, f"传统A*: 长度={trad_length:.1f}米, 转弯={trad_turns}次",
            transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.02, 0.07, f"改进A*: 长度={imp_length:.1f}米, 转弯={imp_turns}次",
            transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.02, 0.02,
            f"路径长度减少: {(trad_length - imp_length) / trad_length * 100:.1f}%, 转弯减少: {(trad_turns - imp_turns) / trad_turns * 100:.1f}%",
            transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(f"{title} - {title_suffix}", fontsize=16)
    ax.legend(loc='upper right')
    ax.set_xlabel('X轴坐标', fontsize=14)
    ax.set_ylabel('Y轴坐标', fontsize=14)

    plt.tight_layout()
    return fig


# 移除不再使用的函数


# 计算路径指标（长度和转弯次数）
def calculate_path_metrics(path):
    if not path or len(path) < 2:
        return 0, 0

    # 计算路径长度
    length = 0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        length += np.sqrt(dx ** 2 + dy ** 2)

    # 计算转弯次数
    turns = 0
    if len(path) < 3:
        return length, 0

    prev_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        curr_direction = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if curr_direction != prev_direction:
            turns += 1
        prev_direction = curr_direction

    return length, turns


# 创建沙漠地形并规划路径
def generate_desert_comparison():
    np.random.seed(42)  # 设置随机种子以便结果可复现
    desert_terrain = create_terrain_map('desert', 100)

    # 设置起点和终点
    start = (20, 15)
    goal = (85, 80)

    # 生成路径
    trad_path = traditional_astar(desert_terrain, start, goal)
    improved_path = improved_astar(desert_terrain, start, goal)

    # 绘制结果（仅中文版）
    fig = plot_terrain_and_paths(desert_terrain, trad_path, improved_path,
                                 "改进混合A*算法路径规划对比", 'desert')

    # 保存图像
    save_file = os.path.join(save_path, "图3-2_沙漠环境改进A星算法路径规划对比.png")
    fig.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"沙漠环境路径规划对比图已保存至: {save_file}")

    return trad_path, improved_path


# 创建草原地形并规划路径
def generate_grassland_comparison():
    np.random.seed(43)  # 设置不同的随机种子
    grassland_terrain = create_terrain_map('grassland', 100)

    # 设置起点和终点
    start = (10, 25)
    goal = (90, 75)

    # 生成路径
    trad_path = traditional_astar(grassland_terrain, start, goal)
    improved_path = improved_astar(grassland_terrain, start, goal)

    # 绘制结果 - 英文版
    fig_eng = plot_terrain_and_paths(grassland_terrain, trad_path, improved_path,
                                     "Improved A* Algorithm Path Planning Comparison", 'grassland')

    # 绘制结果 - 中文版
    fig_chn = plot_terrain_and_paths(grassland_terrain, trad_path, improved_path,
                                     "改进混合A*算法路径规划对比", 'grassland')

    # 保存图像
    save_file_eng = os.path.join(save_path, "Grassland_Environment_Improved_A_Star_Comparison.png")
    save_file_chn = os.path.join(save_path, "图3-3_草原环境改进A星算法路径规划对比.png")

    fig_eng.savefig(save_file_eng, dpi=300, bbox_inches='tight')
    fig_chn.savefig(save_file_chn, dpi=300, bbox_inches='tight')

    plt.close(fig_eng)
    plt.close(fig_chn)

    print(f"Grassland environment path planning comparison saved as: {save_file_eng}")
    print(f"草原环境路径规划对比图已保存至: {save_file_chn}")

    return trad_path, improved_path


if __name__ == "__main__":
    print("Starting A* path planning visualization...")

    # 生成两种环境下的路径规划对比图
    desert_trad_path, desert_improved_path = generate_desert_comparison()
    grassland_trad_path, grassland_improved_path = generate_grassland_comparison()

    # 打印路径指标对比
    print("\n路径指标对比:")
    desert_trad_len, desert_trad_turns = calculate_path_metrics(desert_trad_path)
    desert_imp_len, desert_imp_turns = calculate_path_metrics(desert_improved_path)
    print(f"沙漠环境 - 传统A*: 长度={desert_trad_len:.1f}, 转弯={desert_trad_turns}")
    print(f"沙漠环境 - 改进A*: 长度={desert_imp_len:.1f}, 转弯={desert_imp_turns}")
    print(
        f"沙漠环境 - 改进效果: 长度减少{(desert_trad_len - desert_imp_len) / desert_trad_len * 100:.1f}%, 转弯减少{(desert_trad_turns - desert_imp_turns) / desert_trad_turns * 100:.1f}%")

    grassland_trad_len, grassland_trad_turns = calculate_path_metrics(grassland_trad_path)
    grassland_imp_len, grassland_imp_turns = calculate_path_metrics(grassland_improved_path)
    print(f"草原环境 - 传统A*: 长度={grassland_trad_len:.1f}, 转弯={grassland_trad_turns}")
    print(f"草原环境 - 改进A*: 长度={grassland_imp_len:.1f}, 转弯={grassland_imp_turns}")
    print(
        f"草原环境 - 改进效果: 长度减少{(grassland_trad_len - grassland_imp_len) / grassland_trad_len * 100:.1f}%, 转弯减少{(grassland_trad_turns - grassland_imp_turns) / grassland_trad_turns * 100:.1f}%")

    print("\nAll visualizations completed successfully.")