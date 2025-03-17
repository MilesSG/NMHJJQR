<script setup>
import { RouterView } from 'vue-router';
</script>

<template>
  <div class="min-h-screen">
    <header class="bg-blue-600 text-white shadow-md">
      <div class="container mx-auto px-4 py-4">
        <h1 class="text-2xl font-bold">论文可视化脚本运行器</h1>
      </div>
    </header>
    
    <main class="container mx-auto px-4 py-6">
      <RouterView />
    </main>
    
    <footer class="bg-gray-100 border-t mt-auto">
      <div class="container mx-auto px-4 py-4 text-center text-gray-600">
        <p>© 2024 论文可视化工具</p>
      </div>
    </footer>
  </div>
</template> 