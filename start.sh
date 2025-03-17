#!/bin/bash

echo "正在启动内蒙古地区环境监测移动机器人数据采集方案可视化系统..."
echo ""

# 检查桌面上是否存在"配图"文件夹，如果不存在则创建
echo "检查桌面\"配图\"文件夹..."
if [ ! -d "$HOME/Desktop/配图" ]; then
    echo "创建桌面\"配图\"文件夹..."
    mkdir -p "$HOME/Desktop/配图"
fi

# 进入frontend目录
cd frontend

# 检查是否已经构建
if [ ! -d "dist" ]; then
    echo "正在构建前端..."
    npm run build
fi

# 启动服务器
echo "启动服务器..."
echo "请在浏览器中访问: http://localhost:3001"
node server.js 