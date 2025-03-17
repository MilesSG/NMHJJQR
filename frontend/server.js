import express from 'express';
import { exec } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

// 启用CORS
app.use(cors());

// 解析JSON请求体
app.use(express.json());

// 静态文件服务
app.use(express.static(path.join(__dirname, 'dist')));

// 运行Python脚本的API端点
app.post('/api/run-script', (req, res) => {
  const { scriptName } = req.body;
  
  if (!scriptName) {
    return res.status(400).json({ success: false, message: '缺少脚本名称' });
  }
  
  // 脚本路径（假设Python脚本与server.js在同一目录）
  const scriptPath = path.join(__dirname, '..', scriptName);
  
  console.log(`运行脚本: ${scriptPath}`);
  
  // 执行Python脚本
  exec(`python "${scriptPath}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`执行错误: ${error}`);
      return res.status(500).json({ success: false, message: `脚本执行错误: ${error.message}` });
    }
    
    if (stderr) {
      console.error(`脚本错误输出: ${stderr}`);
    }
    
    console.log(`脚本输出: ${stdout}`);
    
    res.json({ 
      success: true, 
      message: `脚本 ${scriptName} 运行成功！图片已保存到桌面的"配图"文件夹。`,
      output: stdout 
    });
  });
});

// 处理所有其他请求，返回index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
}); 