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

# 环境条件类
class WeatherCondition:
    def __init__(self, name, visibility, wind_speed, temperature, humidity):
        self.name = name  # 天气名称
        self.visibility = visibility  # 能见度 (0-1)
        self.wind_speed = wind_speed  # 风速 (m/s)
        self.temperature = temperature  # 温度 (°C)
        self.humidity = humidity  # 湿度 (%)

# 系统性能评估类
class SystemPerformanceEvaluator:
    def __init__(self):
        # 定义不同天气条件
        self.weather_conditions = {
            "晴朗": WeatherCondition("晴朗", 1.0, 3, 25, 40),
            "多云": WeatherCondition("多云", 0.8, 5, 22, 55),
            "沙尘暴": WeatherCondition("沙尘暴", 0.1, 25, 35, 15),
            "降雨": WeatherCondition("降雨", 0.4, 7, 18, 95),
            "低温": WeatherCondition("低温", 0.7, 12, -10, 30)
        }
        
        # 定义不同地形
        self.terrains = ["沙漠", "草原"]
        
        # 评估指标
        self.metrics = ["导航准确率", "路径规划效率", "传感器数据准确性", "能量消耗", "系统稳定性"]

    def generate_performance_data(self):
        """生成不同天气条件和地形下的系统性能数据"""
        performance_data = {}
        
        # 基准性能值
        base_performance = {
            "沙漠": {
                "导航准确率": 0.95,
                "路径规划效率": 0.90,
                "传感器数据准确性": 0.92,
                "能量消耗": 0.75,  # 越高表示能耗越低
                "系统稳定性": 0.88
            },
            "草原": {
                "导航准确率": 0.93,
                "路径规划效率": 0.88,
                "传感器数据准确性": 0.95,
                "能量消耗": 0.80,
                "系统稳定性": 0.90
            }
        }
        
        # 天气影响因子 (影响不同指标的程度)
        weather_impact = {
            "晴朗": {
                "导航准确率": 1.0,
                "路径规划效率": 1.0,
                "传感器数据准确性": 1.0,
                "能量消耗": 1.0,
                "系统稳定性": 1.0
            },
            "多云": {
                "导航准确率": 0.95,
                "路径规划效率": 0.97,
                "传感器数据准确性": 0.93,
                "能量消耗": 0.98,
                "系统稳定性": 0.97
            },
            "沙尘暴": {
                "导航准确率": 0.60,
                "路径规划效率": 0.65,
                "传感器数据准确性": 0.40,
                "能量消耗": 0.50,
                "系统稳定性": 0.55
            },
            "降雨": {
                "导航准确率": 0.75,
                "路径规划效率": 0.80,
                "传感器数据准确性": 0.65,
                "能量消耗": 0.70,
                "系统稳定性": 0.75
            },
            "低温": {
                "导航准确率": 0.85,
                "路径规划效率": 0.75,
                "传感器数据准确性": 0.80,
                "能量消耗": 0.65,
                "系统稳定性": 0.70
            }
        }
        
        # 计算不同条件下的性能
        for terrain in self.terrains:
            performance_data[terrain] = {}
            
            for weather in self.weather_conditions:
                performance_data[terrain][weather] = {}
                
                for metric in self.metrics:
                    # 基础性能 × 天气影响因子 + 小随机波动
                    base = base_performance[terrain][metric]
                    impact = weather_impact[weather][metric]
                    
                    # 添加一些随机波动 (±5%)
                    random_factor = 1.0 + np.random.uniform(-0.05, 0.05)
                    
                    # 计算最终性能值 (确保在0-1范围内)
                    perf_value = min(1.0, max(0.1, base * impact * random_factor))
                    performance_data[terrain][weather][metric] = perf_value
        
        return performance_data

    def visualize_radar_chart(self, performance_data):
        """使用雷达图可视化不同条件下的系统性能"""
        # 创建画布
        fig = plt.figure(figsize=(16, 8), dpi=600)
        gs = GridSpec(1, 2)
        
        # 设置雷达图的角度 (将指标均匀分布)
        angles = np.linspace(0, 2*np.pi, len(self.metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 为每个地形创建一个子图
        for t_idx, terrain in enumerate(self.terrains):
            ax = fig.add_subplot(gs[0, t_idx], polar=True)
            
            # 设置雷达图刻度和标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.metrics, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(f"{terrain}环境下不同天气条件的系统性能", fontsize=14, pad=15)
            
            # 为每种天气绘制性能曲线
            colors = ['#4CAF50', '#2196F3', '#F44336', '#9C27B0', '#FF9800']
            
            for w_idx, weather in enumerate(self.weather_conditions):
                # 获取该天气条件下的性能数据
                values = [performance_data[terrain][weather][metric] for metric in self.metrics]
                values += values[:1]  # 闭合图形
                
                # 绘制性能曲线
                ax.plot(angles, values, 'o-', linewidth=2, label=weather, color=colors[w_idx], markersize=6)
                ax.fill(angles, values, color=colors[w_idx], alpha=0.1)
            
            # 添加图例
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(save_path, "图4-5_不同天气条件下的系统性能评估.png")
        fig.savefig(save_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"系统性能雷达图已保存至: {save_file}")

    def visualize_detailed_comparison(self, performance_data):
        """创建详细的性能对比图表"""
        # 准备数据 - 创建DataFrame
        df_data = []
        for terrain in self.terrains:
            for weather in self.weather_conditions:
                for metric in self.metrics:
                    df_data.append({
                        '地形': terrain,
                        '天气条件': weather,
                        '评估指标': metric, 
                        '性能得分': performance_data[terrain][weather][metric]
                    })
        
        df = pd.DataFrame(df_data)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=600, sharey=True)
        
        # 天气因素对各指标的影响
        weather_impact = df.pivot_table(
            index='天气条件', 
            columns='评估指标', 
            values='性能得分', 
            aggfunc='mean'
        )
        
        weather_impact.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_ylabel('性能得分', fontsize=12)
        ax1.set_title('不同天气条件对系统性能的影响', fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 地形因素对比
        terrain_impact = df.pivot_table(
            index='评估指标', 
            columns='地形', 
            values='性能得分', 
            aggfunc='mean'
        )
        
        terrain_impact.plot(kind='bar', ax=ax2, width=0.6)
        ax2.set_title('不同地形环境对系统性能的影响', fontsize=14)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(save_path, "图4-6_系统性能详细对比分析.png")
        fig.savefig(save_file, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"系统性能详细对比图已保存至: {save_file}")

# 主函数
def main():
    # 创建系统性能评估器
    evaluator = SystemPerformanceEvaluator()
    
    # 生成性能数据
    performance_data = evaluator.generate_performance_data()
    
    # 可视化雷达图
    evaluator.visualize_radar_chart(performance_data)
    
    # 创建详细对比分析
    evaluator.visualize_detailed_comparison(performance_data)

if __name__ == "__main__":
    main()
