import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

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

# 环境区域类
class MonitoringArea:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.terrain_map = np.zeros((height, width))  # 地形图
        self.obstacle_map = np.zeros((height, width))  # 障碍物图
        self.humidity_map = np.zeros((height, width))  # 湿度图
        self.temperature_map = np.zeros((height, width))  # 温度图
        
        # 生成地形图 (上半部分为沙漠，下半部分为草原)
        for i in range(height):
            for j in range(width):
                # 沙漠区域 (上半部分)
                if i < height/2:
                    # 沙丘高度变化
                    self.terrain_map[i, j] = 0.5 + 2.0 * np.sin(0.05 * i + 0.08 * j) + 1.5 * np.sin(0.02 * i - 0.03 * j)
                    
                    # 沙漠温度 (较高)
                    self.temperature_map[i, j] = 35 + 5 * np.random.random()
                    
                    # 沙漠湿度 (较低)
                    self.humidity_map[i, j] = 10 + 10 * np.random.random()
                # 草原区域 (下半部分)
                else:
                    # 草原较平坦，但有小起伏
                    self.terrain_map[i, j] = 0.2 + 0.5 * np.sin(0.1 * i + 0.1 * j)
                    
                    # 草原温度 (较低)
                    self.temperature_map[i, j] = 25 + 5 * np.random.random()
                    
                    # 草原湿度 (较高)
                    self.humidity_map[i, j] = 50 + 20 * np.random.random()
        
        # 添加随机障碍物 (湖泊和小山)
        for _ in range(10):
            center_x = np.random.randint(5, width-5)
            center_y = np.random.randint(5, height-5)
            radius = np.random.randint(3, 8)
            
            for i in range(max(0, center_y-radius), min(height, center_y+radius)):
                for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                    dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                    if dist <= radius:
                        self.obstacle_map[i, j] = 1
                        # 如果是湖泊 (在草原区域多一些)
                        if i >= height/2 or np.random.random() < 0.3:
                            self.humidity_map[i, j] = 90 + 10 * np.random.random()
                            self.temperature_map[i, j] = 20 + 3 * np.random.random()

# 传感器节点类
class SensorNode:
    def __init__(self, id, x, y, node_type="source", cluster_id=None):
        self.id = id
        self.x = x
        self.y = y
        self.node_type = node_type  # "source", "cluster_head", "sink"
        self.cluster_id = cluster_id
        
        # 节点属性
        self.energy = 100.0  # 初始能量(%)
        self.sensor_diversity = np.random.randint(1, 5)  # 传感器种类数量
        self.data_importance = np.random.uniform(0.2, 1.0)  # 数据重要性
        self.obstacle_distance = 0.0  # 到障碍物的距离(将在网络中计算)
        
    def calculate_weight(self, area, algorithm="combined"):
        """计算节点的重要性权重"""
        # 计算到障碍物的距离
        min_dist = float('inf')
        for i in range(area.height):
            for j in range(area.width):
                if area.obstacle_map[i, j] == 1:
                    dist = np.sqrt((self.y - i)**2 + (self.x - j)**2)
                    min_dist = min(min_dist, dist)
        
        self.obstacle_distance = min(min_dist, 20.0) / 20.0  # 归一化到[0,1]
        
        # 获取节点所在位置的环境条件
        location_humidity = area.humidity_map[min(area.height-1, int(self.y)), min(area.width-1, int(self.x))]
        location_temperature = area.temperature_map[min(area.height-1, int(self.y)), min(area.width-1, int(self.x))]
        
        # 环境条件评分 (湿度适中50-70%，温度适中20-30℃时得分高)
        humidity_score = 1.0 - min(abs(location_humidity - 60) / 40.0, 1.0)
        temperature_score = 1.0 - min(abs(location_temperature - 25) / 15.0, 1.0)
        environment_score = (humidity_score + temperature_score) / 2.0
        
        # 不同算法的权重计算
        if algorithm == "distance_based":
            # 基于距离的算法主要考虑到障碍物距离
            return self.obstacle_distance
        elif algorithm == "energy_aware":
            # 能量感知算法考虑节点能量和数据重要性
            return 0.7 * self.data_importance + 0.3 * (self.energy / 100.0)
        elif algorithm == "environmental":
            # 环境条件算法主要考虑环境适宜性
            return environment_score
        else:  # combined/default
            # 综合算法考虑所有因素
            return (0.3 * self.obstacle_distance + 
                    0.3 * environment_score + 
                    0.3 * self.data_importance + 
                    0.1 * (self.sensor_diversity / 4.0))

# 传感器网络类
class SensorNetwork:
    def __init__(self, area, n_clusters=5, nodes_per_cluster=8):
        self.area = area
        self.n_clusters = n_clusters
        self.nodes_per_cluster = nodes_per_cluster
        self.nodes = []
        self.clusters = {}
        
        # 创建传感器网络
        self._create_network()
    
    def _create_network(self):
        # 创建汇聚节点 (放在区域中心上方)
        sink_node = SensorNode("Sink", self.area.width/2, self.area.height/4, "sink")
        self.nodes.append(sink_node)
        
        # 创建簇和节点
        for c in range(1, self.n_clusters+1):
            # 确定簇的中心位置
            if c <= 3:  # 前3个簇在沙漠区域
                cluster_x = np.random.randint(10, self.area.width-10)
                cluster_y = np.random.randint(10, self.area.height/2-10)
            else:  # 后2个簇在草原区域
                cluster_x = np.random.randint(10, self.area.width-10)
                cluster_y = np.random.randint(self.area.height/2+10, self.area.height-10)
            
            # 创建簇头节点
            ch_node = SensorNode(f"CH{c}", cluster_x, cluster_y, "cluster_head", c)
            self.nodes.append(ch_node)
            
            # 在簇周围创建源节点
            cluster_nodes = [ch_node]
            for i in range(self.nodes_per_cluster):
                radius = np.random.uniform(5, 20)
                angle = np.random.uniform(0, 2*np.pi)
                x = int(cluster_x + radius * np.cos(angle))
                y = int(cluster_y + radius * np.sin(angle))
                
                # 确保在区域范围内
                x = max(0, min(self.area.width-1, x))
                y = max(0, min(self.area.height-1, y))
                
                # 创建源节点
                node = SensorNode(f"S{c}{i+1}", x, y, "source", c)
                self.nodes.append(node)
                cluster_nodes.append(node)
            
            # 存储簇信息
            self.clusters[c] = cluster_nodes
    
    def select_nodes(self, algorithm="combined", selection_ratio=0.5):
        """选择节点进行数据采集"""
        # 计算每个源节点的权重
        node_weights = {}
        for node in self.nodes:
            if node.node_type == "source":
                node_weights[node.id] = node.calculate_weight(self.area, algorithm)
        
        # 按权重降序排序
        sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前N%的节点
        n_select = int(len(sorted_nodes) * selection_ratio)
        selected_ids = [node_id for node_id, _ in sorted_nodes[:n_select]]
        
        return selected_ids

# 可视化函数
def visualize_node_selection(network, selected_nodes=None, algorithm="combined"):
    fig = plt.figure(figsize=(16, 12), dpi=600)
    gs = GridSpec(2, 2, height_ratios=[3, 1])
    
    # 主网络图
    ax1 = fig.add_subplot(gs[0, :])
    
    # 绘制地形图
    terrain_cmap = LinearSegmentedColormap.from_list(
        'terrain_cmap', 
        [(0, '#f6d7b0'), (0.5, '#e4bc84'), (1.0, '#aed581')], 
        N=256
    )
    terrain_data = np.copy(network.area.terrain_map)
    im = ax1.imshow(terrain_data, cmap=terrain_cmap, origin='lower', aspect='auto')
    
    # 添加环境标记
    ax1.text(network.area.width/2, network.area.height*0.25, "沙漠区域", 
             fontsize=14, ha='center', color='#8B4513')
    ax1.text(network.area.width/2, network.area.height*0.75, "草原区域", 
             fontsize=14, ha='center', color='#006400')
    
    # 绘制障碍物
    obstacle_data = np.ma.masked_where(network.area.obstacle_map == 0, network.area.obstacle_map)
    ax1.imshow(obstacle_data, cmap='Blues', origin='lower', aspect='auto', alpha=0.7)
    
    # 绘制汇聚节点
    sink_node = next(node for node in network.nodes if node.node_type == "sink")
    ax1.plot(sink_node.x, sink_node.y, 'ro', markersize=15, label='汇聚节点')
    ax1.text(sink_node.x+2, sink_node.y+2, sink_node.id, fontsize=12)
    
    # 不同簇使用不同颜色
    cluster_colors = ['#9c27b0', '#2196f3', '#ff9800', '#4caf50', '#795548']
    
    # 绘制各簇和节点
    for c_id, cluster in network.clusters.items():
        color = cluster_colors[(c_id-1) % len(cluster_colors)]
        
        # 簇头
        ch_node = next(node for node in cluster if node.node_type == "cluster_head")
        ax1.plot(ch_node.x, ch_node.y, 'o', color=color, markersize=12, label=f'簇头 {c_id}' if c_id == 1 else "")
        ax1.text(ch_node.x+2, ch_node.y+2, ch_node.id, fontsize=10)
        
        # 簇头到汇聚节点的连接线
        ax1.plot([ch_node.x, sink_node.x], [ch_node.y, sink_node.y], '--', color=color, alpha=0.6)
        
        # 源节点
        for node in cluster:
            if node.node_type == "source":
                # 计算节点大小 (基于重要性)
                weight = node.calculate_weight(network.area, algorithm)
                size = 7 + weight * 8
                
                # 判断是否被选中
                is_selected = selected_nodes and node.id in selected_nodes
                
                if is_selected:
                    ax1.plot(node.x, node.y, 'o', color=color, markersize=size, 
                             markeredgewidth=2, markeredgecolor='black')
                else:
                    ax1.plot(node.x, node.y, 'o', color=color, markersize=size, alpha=0.5)
                
                ax1.text(node.x+1, node.y+1, node.id, fontsize=8)
                
                # 到簇头的连接
                ax1.plot([node.x, ch_node.x], [node.y, ch_node.y], '-', color=color, alpha=0.3)
    
    # 设置图例和标题
    algorithm_names = {
        "combined": "综合多因素",
        "distance_based": "基于距离", 
        "energy_aware": "能量感知",
        "environmental": "环境条件"
    }
    
    ax1.set_title(f"传感器节点重要性评估与选择 - {algorithm_names.get(algorithm, algorithm)}算法", 
                  fontsize=16)
    ax1.set_xlabel('X坐标', fontsize=14)
    ax1.set_ylabel('Y坐标', fontsize=14)
    ax1.legend(loc='upper right')
    
    # 节点属性对比图表 - 选中vs未选中
    ax2 = fig.add_subplot(gs[1, 0])
    
    if selected_nodes:
        # 准备属性数据
        attributes = ['障碍物距离', '环境适应性', '数据重要性', '传感器多样性']
        selected_values = []
        unselected_values = []
        
        # 计算平均值
        for attr_idx, attr in enumerate(attributes):
            sel_sum = 0
            sel_count = 0
            unsel_sum = 0
            unsel_count = 0
            
            for node in network.nodes:
                if node.node_type == "source":
                    if attr_idx == 0:
                        val = node.obstacle_distance
                    elif attr_idx == 1:
                        # 环境条件
                        humidity = network.area.humidity_map[min(network.area.height-1, int(node.y)), 
                                                          min(network.area.width-1, int(node.x))]
                        temperature = network.area.temperature_map[min(network.area.height-1, int(node.y)), 
                                                                min(network.area.width-1, int(node.x))]
                        humidity_score = 1.0 - min(abs(humidity - 60) / 40.0, 1.0)
                        temperature_score = 1.0 - min(abs(temperature - 25) / 15.0, 1.0)
                        val = (humidity_score + temperature_score) / 2.0
                    elif attr_idx == 2:
                        val = node.data_importance
                    else:  # attr_idx == 3
                        val = node.sensor_diversity / 4.0
                    
                    if node.id in selected_nodes:
                        sel_sum += val
                        sel_count += 1
                    else:
                        unsel_sum += val
                        unsel_count += 1
            
            if sel_count > 0:
                selected_values.append(sel_sum / sel_count)
            else:
                selected_values.append(0)
                
            if unsel_count > 0:
                unselected_values.append(unsel_sum / unsel_count)
            else:
                unselected_values.append(0)
        
        # 绘制柱状图
        x = np.arange(len(attributes))
        width = 0.35
        
        ax2.bar(x - width/2, selected_values, width, label='选中节点', color='#4CAF50')
        ax2.bar(x + width/2, unselected_values, width, label='未选中节点', color='#9E9E9E')
        
        ax2.set_ylabel('平均值', fontsize=12)
        ax2.set_title('节点属性对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(attributes)
        ax2.legend()
    
    # 算法效果对比 (如果有多个算法结果)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 创建算法性能对比数据
    algorithms = ["综合多因素", "基于距离", "能量感知", "环境条件"]
    
    if algorithm == "combined":
        # 计算不同算法的性能指标
        perf_metrics = []
        for algo in ["combined", "distance_based", "energy_aware", "environmental"]:
            # 选出前50%的节点
            selected = network.select_nodes(algo, 0.5)
            
            # 计算平均节点特性
            total_weight = 0
            for node_id in selected:
                node = next(n for n in network.nodes if n.id == node_id)
                # 使用综合评分作为真实性能值
                total_weight += node.calculate_weight(network.area, "combined")
            
            avg_weight = total_weight / len(selected) if selected else 0
            perf_metrics.append(avg_weight)
        
        # 绘制性能对比
        ax3.bar(range(len(algorithms)), perf_metrics, color=['#2196F3', '#F44336', '#FFC107', '#4CAF50'])
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45)
        ax3.set_ylabel('综合性能得分', fontsize=12)
        ax3.set_title('不同选择算法性能比较', fontsize=14)
    
    plt.tight_layout()
    
    # 保存图像
    alg_suffix = algorithm.replace("_", "-")
    save_file = os.path.join(save_path, f"图3-6_传感器节点选择_{alg_suffix}.png")
    fig.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"传感器节点选择可视化已保存至: {save_file}")

# 可视化节点选择的时间和能量消耗
def visualize_selection_efficiency(network):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    
    # 模拟数据 - 覆盖率vs时间
    selection_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 不同算法的时间和能量消耗
    algorithms = ['综合多因素', '基于距离', '能量感知', '环境条件']
    colors = ['#2196F3', '#F44336', '#FFC107', '#4CAF50']
    
    # 模拟不同算法的数据采集时间
    collection_times = {
        '综合多因素': [5, 9, 12, 15, 18, 22, 27, 33, 40, 50],
        '基于距离': [4, 8, 11, 15, 19, 24, 30, 38, 47, 60],
        '能量感知': [6, 10, 13, 16, 19, 23, 28, 34, 42, 52],
        '环境条件': [7, 11, 14, 17, 21, 26, 32, 39, 48, 58]
    }
    
    # 模拟不同算法的能量消耗
    energy_consumption = {
        '综合多因素': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        '基于距离': [1.5, 3, 4.5, 6, 8, 10, 12.5, 15, 18, 22],
        '能量感知': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        '环境条件': [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 21]
    }
    
    # 绘制时间消耗
    for i, algo in enumerate(algorithms):
        ax1.plot(selection_ratios, collection_times[algo], 'o-', 
                color=colors[i], label=algo)
    
    ax1.set_xlabel('节点选择比例', fontsize=12)
    ax1.set_ylabel('数据采集时间 (分钟)', fontsize=12)
    ax1.set_title('节点选择比例vs数据采集时间', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制能量消耗
    for i, algo in enumerate(algorithms):
        ax2.plot(selection_ratios, energy_consumption[algo], 'o-', 
                color=colors[i], label=algo)
    
    ax2.set_xlabel('节点选择比例', fontsize=12)
    ax2.set_ylabel('能量消耗 (%)', fontsize=12)
    ax2.set_title('节点选择比例vs能量消耗', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图像
    save_file = os.path.join(save_path, "图3-7_节点选择效率分析.png")
    fig.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"节点选择效率分析图已保存至: {save_file}")

# 主函数
def main():
    # 创建环境区域
    monitoring_area = MonitoringArea(width=100, height=100)
    
    # 创建传感器网络
    network = SensorNetwork(monitoring_area, n_clusters=5, nodes_per_cluster=8)
    
    # 使用不同算法选择节点并可视化
    # 1. 综合多因素算法
    selected_nodes = network.select_nodes("combined", 0.5)
    visualize_node_selection(network, selected_nodes, "combined")
    
    # 2. 节点选择效率分析
    visualize_selection_efficiency(network)

if __name__ == "__main__":
    main()
