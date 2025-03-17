import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useScriptStore = defineStore('script', () => {
  const isRunning = ref(false);
  const currentScript = ref('');
  const outputMessage = ref('');
  const isSuccess = ref(false);

  const scripts = [
    {
      id: 'improved-astar',
      name: '改进A*算法路径规划',
      description: '展示改进A*算法在沙漠和草原环境中的路径规划效果',
      filename: 'improved-astar-algorithm（3-2 3-3）.py',
      figures: ['图3-2_沙漠环境改进A星算法路径规划对比.png', '图3-3_草原环境改进A星算法路径规划对比.png']
    },
    {
      id: 'drl-path-planning',
      name: '深度强化学习路径规划',
      description: '展示深度强化学习在路径规划中的应用',
      filename: 'drl-path-planning(3-4 3-5).py',
      figures: ['图3-4_深度强化学习训练过程.png', '图3-5_深度强化学习局部路径规划决策过程.png']
    },
    {
      id: 'sensor-node-selection',
      name: '传感器节点选择可视化',
      description: '展示传感器节点选择算法的效果',
      filename: 'sensor-node-selection-visualization（3-6 3-7）.py',
      figures: ['图3-6_传感器节点选择可视化.png', '图3-7_节点选择效率分析.png']
    },
    {
      id: 'extreme-weather',
      name: '极端天气性能评估',
      description: '展示在极端天气条件下系统性能的评估结果',
      filename: 'extreme-weather-performance（4-5 4-6）.py',
      figures: ['图4-5_不同天气条件下系统性能雷达图.png', '图4-6_系统性能详细对比分析.png']
    }
  ];

  function runScript(scriptId) {
    const script = scripts.find(s => s.id === scriptId);
    if (!script) return;
    
    isRunning.value = true;
    currentScript.value = script.name;
    outputMessage.value = `正在运行脚本: ${script.filename}...`;
    
    // 这里会通过后端API调用Python脚本
    // 在实际应用中，这里应该是一个fetch或axios请求
    
    // 模拟异步操作
    setTimeout(() => {
      isRunning.value = false;
      isSuccess.value = true;
      outputMessage.value = `脚本 ${script.filename} 运行成功！图片已保存到桌面的"配图"文件夹。`;
    }, 3000);
  }

  return {
    isRunning,
    currentScript,
    outputMessage,
    isSuccess,
    scripts,
    runScript
  };
}); 