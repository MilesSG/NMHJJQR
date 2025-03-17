// 在实际应用中，这里应该使用fetch或axios发送HTTP请求
// 由于这是一个前端演示，我们使用模拟函数

/**
 * 运行Python脚本
 * @param {string} scriptName - 脚本文件名
 * @returns {Promise<Object>} - 运行结果
 */
export async function runPythonScript(scriptName) {
  console.log(`运行脚本: ${scriptName}`);
  
  try {
    const response = await fetch('http://localhost:3001/api/run-script', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ scriptName }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || '请求失败');
    }
    
    return await response.json();
  } catch (error) {
    console.error('API请求错误:', error);
    return {
      success: false,
      message: `脚本运行失败: ${error.message}`
    };
  }
} 