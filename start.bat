@echo off
echo 正在启动内蒙古地区环境监测移动机器人数据采集方案可视化系统...
echo.

REM 检查桌面上是否存在"配图"文件夹，如果不存在则创建
echo 检查桌面"配图"文件夹...
if not exist "%USERPROFILE%\Desktop\配图" (
    echo 创建桌面"配图"文件夹...
    mkdir "%USERPROFILE%\Desktop\配图"
)

REM 进入frontend目录
cd frontend

REM 检查是否已经构建
if not exist "dist" (
    echo 正在构建前端...
    call npm run build
)

REM 启动服务器
echo 启动服务器...
echo 请在浏览器中访问: http://localhost:3001
node server.js

pause 