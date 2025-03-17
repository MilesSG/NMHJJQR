<script setup>
import { ref, computed } from 'vue';
import { useScriptStore } from '../stores/scriptStore';
import { runPythonScript } from '../api/scriptApi';
import { ElMessage, ElLoading } from 'element-plus';

const scriptStore = useScriptStore();
const { scripts, isRunning, currentScript, outputMessage, isSuccess } = scriptStore;

const selectedTab = ref('improved-astar');

const currentScriptInfo = computed(() => {
  return scripts.find(s => s.id === selectedTab.value);
});

async function handleRunScript() {
  if (isRunning.value) return;
  
  const loadingInstance = ElLoading.service({
    lock: true,
    text: `正在运行脚本: ${currentScriptInfo.value.filename}`,
    background: 'rgba(0, 0, 0, 0.7)'
  });
  
  try {
    const result = await runPythonScript(currentScriptInfo.value.filename);
    
    if (result.success) {
      ElMessage.success(result.message);
    } else {
      ElMessage.error(result.message || '脚本运行失败');
    }
  } catch (error) {
    console.error('脚本运行错误:', error);
    ElMessage.error('脚本运行出错，请查看控制台获取详细信息');
  } finally {
    loadingInstance.close();
  }
}
</script>

<template>
  <div>
    <div class="card mb-6">
      <h2 class="text-xl font-bold mb-4">论文可视化脚本运行器</h2>
      <p class="text-gray-600 mb-4">
        本工具可以运行论文中的可视化脚本，生成相应的图表，并将结果保存到桌面的"配图"文件夹中。
        请选择下方的脚本类型，然后点击"运行脚本"按钮。
      </p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <!-- 左侧选项卡 -->
      <div class="md:col-span-1">
        <div class="card">
          <h3 class="font-bold mb-4">可视化脚本</h3>
          <div class="flex flex-col space-y-2">
            <button 
              v-for="script in scripts" 
              :key="script.id"
              @click="selectedTab = script.id"
              class="text-left px-4 py-2 rounded-md transition-colors"
              :class="selectedTab === script.id ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'"
            >
              {{ script.name }}
            </button>
          </div>
        </div>
      </div>
      
      <!-- 右侧内容区 -->
      <div class="md:col-span-3">
        <div class="card">
          <div v-if="currentScriptInfo" class="mb-6">
            <h3 class="text-lg font-bold mb-2">{{ currentScriptInfo.name }}</h3>
            <p class="text-gray-600 mb-4">{{ currentScriptInfo.description }}</p>
            <div class="text-sm text-gray-500 mb-4">
              <p>文件名: {{ currentScriptInfo.filename }}</p>
              <p>生成图表: {{ currentScriptInfo.figures.join(', ') }}</p>
            </div>
            
            <button 
              @click="handleRunScript" 
              class="btn btn-primary"
              :disabled="isRunning"
            >
              {{ isRunning ? '运行中...' : '运行脚本' }}
            </button>
          </div>
          
          <div v-if="outputMessage" class="mt-6 p-4 bg-gray-50 rounded-md border">
            <h4 class="font-bold mb-2">运行结果</h4>
            <p :class="isSuccess ? 'text-green-600' : 'text-red-600'">
              {{ outputMessage }}
            </p>
          </div>
          
          <div v-if="currentScriptInfo && isSuccess" class="mt-6">
            <h4 class="font-bold mb-2">生成的图表</h4>
            <div class="grid grid-cols-1 gap-4">
              <div v-for="(figure, index) in currentScriptInfo.figures" :key="index" class="border p-2 rounded-md">
                <p class="text-sm text-gray-600 mb-1">{{ figure }}</p>
                <div class="bg-gray-100 p-4 rounded flex items-center justify-center">
                  <p class="text-gray-500 italic">图片已保存到桌面的"配图"文件夹</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template> 