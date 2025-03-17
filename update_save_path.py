import os
import re

def update_save_path(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式替换保存路径相关代码
    pattern = r'desktop_path = os\.path\.join\(os\.path\.expanduser\("~"\), "Desktop"\)\nsave_path = os\.path\.join\(desktop_path, "配图"\)'
    replacement = '''# 设置保存路径为桌面的"配图"文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, "配图")
# 确保目录存在
os.makedirs(save_path, exist_ok=True)
print(f"图片将保存到: {save_path}")'''
    
    new_content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print(f"已更新文件: {file_path}")

def main():
    # 获取当前目录下的所有Python文件
    python_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'update_save_path.py']
    
    for file in python_files:
        update_save_path(file)
    
    print(f"共更新了 {len(python_files)} 个文件")

if __name__ == "__main__":
    main() 