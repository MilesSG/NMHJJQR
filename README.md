# 🤖 内蒙古地区环境监测移动机器人数据采集方案可视化系统

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Node.js](https://img.shields.io/badge/Node.js-14.0+-green)
![Vue.js](https://img.shields.io/badge/Vue.js-3.0+-green)

## 📝 项目简介

本项目是内蒙古地区环境监测移动机器人数据采集方案相关研究的可视化系统，通过前端界面可以方便地运行各种可视化脚本，生成高质量的论文配图。系统包含四个主要的可视化模块：

- 🗺️ **改进A*算法路径规划** - 展示在沙漠和草原环境中改进A*算法的路径规划效果
- 🧠 **深度强化学习路径规划** - 展示深度强化学习在路径规划中的应用
- 📊 **传感器节点选择可视化** - 展示传感器节点选择算法的效果
- 🌪️ **极端天气性能评估** - 展示在极端天气条件下系统性能的评估结果

## ✨ 功能特点

- 🖥️ **用户友好界面** - 简洁直观的Web界面，易于操作
- 🔄 **一键运行脚本** - 点击按钮即可运行复杂的Python可视化脚本
- 📁 **自动保存图片** - 自动将生成的图片保存到桌面的"配图"文件夹
- 🔍 **详细结果展示** - 展示脚本运行结果和生成的图表信息
- 🌈 **高质量可视化** - 生成适合学术论文的高分辨率图表

## 🛠️ 技术栈

### 前端
- **Vue.js 3** - 渐进式JavaScript框架
- **Vite** - 现代前端构建工具
- **Element Plus** - 基于Vue 3的组件库
- **Tailwind CSS** - 实用优先的CSS框架
- **Pinia** - Vue的状态管理库

### 后端
- **Node.js** - JavaScript运行时环境
- **Express** - Web应用框架

### 数据可视化
- **Python** - 脚本语言
- **Matplotlib** - Python绘图库
- **NumPy** - 科学计算库

## 📋 系统要求

- **操作系统**: Windows 10/11, macOS, Linux
- **Python**: 3.8+
- **Node.js**: 14.0+
- **NPM**: 6.0+

## 🚀 快速开始

### 方法一：使用启动脚本（推荐）

1. **Windows用户**

   双击运行`start.bat`文件

2. **macOS/Linux用户**

   在终端中运行：
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

### 方法二：手动安装和运行

1. **克隆项目**

```bash
git clone https://github.com/yourusername/nmhjjqr.git
cd nmhjjqr
```

2. **安装Python依赖**

```bash
pip install numpy matplotlib pandas
```

3. **安装Node.js依赖**

```bash
cd frontend
npm install
```

4. **构建前端**

```bash
cd frontend
npm run build
```

5. **启动服务器**

```bash
cd frontend
node server.js
```

6. **访问系统**

在浏览器中打开 [http://localhost:3001](http://localhost:3001)

## 📊 可视化脚本说明

### 🗺️ 改进A*算法路径规划

该脚本展示了改进A*算法在沙漠和草原环境中的路径规划效果，对比了传统A*算法和改进A*算法在路径长度和转弯次数上的差异。

**生成图表**:
- 图3-2_沙漠环境改进A星算法路径规划对比.png
- 图3-3_草原环境改进A星算法路径规划对比.png

### 🧠 深度强化学习路径规划

该脚本展示了深度强化学习在路径规划中的应用，包括训练过程和决策过程的可视化。

**生成图表**:
- 图3-4_深度强化学习训练过程.png
- 图3-5_深度强化学习局部路径规划决策过程.png

### 📊 传感器节点选择可视化

该脚本展示了传感器节点选择算法的效果，包括节点分布和选择效率的分析。

**生成图表**:
- 图3-6_传感器节点选择_combined.png
- 图3-7_节点选择效率分析.png

### 🌪️ 极端天气性能评估

该脚本展示了在极端天气条件下系统性能的评估结果，包括雷达图和详细对比分析。

**生成图表**:
- 图4-5_不同天气条件下系统性能雷达图.png
- 图4-6_系统性能详细对比分析.png

## 📁 文件结构

```
nmhjjqr/
├── frontend/                  # 前端项目
│   ├── src/                   # 源代码
│   │   ├── api/               # API服务
│   │   ├── components/        # Vue组件
│   │   ├── router/            # 路由配置
│   │   ├── stores/            # Pinia状态管理
│   │   ├── views/             # 页面视图
│   │   ├── App.vue            # 主应用组件
│   │   └── main.js            # 入口文件
│   ├── public/                # 静态资源
│   ├── index.html             # HTML模板
│   ├── package.json           # 项目配置
│   ├── vite.config.js         # Vite配置
│   └── server.js              # 后端服务器
├── improved-astar-algorithm（3-2 3-3）.py    # 改进A*算法脚本
├── drl-path-planning(3-4 3-5).py             # 深度强化学习脚本
├── sensor-node-selection-visualization（3-6 3-7）.py  # 传感器节点选择脚本
├── extreme-weather-performance（4-5 4-6）.py  # 极端天气性能评估脚本
├── start.bat                  # Windows启动脚本
├── start.sh                   # macOS/Linux启动脚本
└── README.md                  # 项目说明文档
```

## 🔧 配置说明

### 图片保存路径

默认情况下，生成的图片会保存到桌面的"配图"文件夹中。如果该文件夹不存在，系统会自动创建。

如果需要修改保存路径，可以编辑Python脚本中的以下代码：

```python
# 设置保存路径为桌面的"配图"文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, "配图")
# 确保目录存在
os.makedirs(save_path, exist_ok=True)
```

### 服务器端口

默认情况下，服务器运行在3001端口。如果需要修改端口，可以编辑`frontend/server.js`文件中的以下代码：

```javascript
const PORT = 3001;
```

## 📝 注意事项

1. **Python环境**: 确保已安装Python 3.8+和所需的依赖库。
2. **中文字体**: 脚本使用SimHei等中文字体，请确保系统中已安装这些字体。
3. **文件名**: 请勿修改Python脚本的文件名，否则可能导致系统无法正常运行。
4. **权限**: 确保系统有权限在桌面创建"配图"文件夹。
5. **移植到其他电脑**: 移植到其他电脑时，只需复制整个项目文件夹，然后运行启动脚本即可。系统会自动在桌面创建"配图"文件夹。

## 🔍 常见问题

### 图片无法保存到桌面

**问题**: 运行脚本后，图片没有保存到桌面的"配图"文件夹。

**解决方案**: 
- 确保系统有权限在桌面创建文件夹
- 手动创建桌面上的"配图"文件夹
- 检查Python脚本中的保存路径代码

### 服务器无法启动

**问题**: 运行`node server.js`时出现错误。

**解决方案**:
- 确保已安装Node.js和所需的依赖
- 确保在正确的目录下运行命令（frontend目录）
- 检查端口3001是否被占用，如果被占用，修改server.js中的端口号

### 脚本运行失败

**问题**: 点击运行脚本按钮后，脚本运行失败。

**解决方案**:
- 检查Python环境是否正确配置
- 确保已安装所需的Python依赖库
- 查看服务器控制台输出的错误信息

### 中文显示为乱码

**问题**: 生成的图表中的中文显示为乱码。

**解决方案**:
- 确保系统中已安装SimHei等中文字体
- 在Python脚本中尝试使用其他可用的中文字体
- 检查matplotlib的字体配置

## 📞 联系方式

如有任何问题或建议，请联系：

- **邮箱**: your.email@example.com
- **GitHub**: [https://github.com/yourusername](https://github.com/yourusername)

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

🌟 感谢使用内蒙古地区环境监测移动机器人数据采集方案可视化系统！ 