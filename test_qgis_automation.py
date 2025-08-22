# test_qgis_automation.py
import os
import copy
import time
import json
import glob
import shutil
import argparse

import subprocess
import cv2

from agent.autopc import AutoPC
from agent.utils.gui_capture import get_screenshot, focus_software
from agent.gui_parser.sender import send_gui_parser_request
from agent.actor.utils import format_gui, compress_gui

from agent.config import basic_config

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="QGIS自动化测试工具")
    parser.add_argument("--software_name", type=str, default="QGIS")
    parser.add_argument("--project_id", type=str, default="QGIS_qgis_test")
    parser.add_argument("--qgis_path", type=str, default=r"G:\BYXG\bin\qgis-ltr-bin.exe")
    parser.add_argument("--query", type=str, default="please change the Rivers in Liverpool's symbol layer type form 'Simple Line' to 'Arrow'.")
    parser.add_argument("--maximum_step", type=int, default=100, help="最大执行步数")
    parser.add_argument("--max_critic_trials", type=int, default=3, help="critic最大重试次数")
    args = parser.parse_args()
    
    # 提取参数
    software_name = args.software_name
    project_id = args.project_id
    qgis_path = args.qgis_path
    query = args.query
    maximum_step = args.maximum_step
    max_critic_trials = args.max_critic_trials
    
    # 自动格式化project_id，确保格式正确
    if not project_id.startswith(f"{software_name}_"):
        project_id = f"{software_name}_{project_id}"
    
    print(f"使用软件名称: {software_name}")
    print(f"格式化后的项目ID: {project_id}")
    
    saved_folder = f'test_results/{basic_config["planner_critic"]["lmm"]}'

    # 启动QGIS
    # subprocess.Popen([qgis_path], shell=True)
    # time.sleep(3)  # 等待QGIS启动

    autopc = AutoPC(software_name=software_name, project_id=project_id)
    focus_software(software_name)
    meta_data, screenshot_path = get_screenshot(software_name)

    # 保存初始截图
    new_screenpath = os.path.join(saved_folder, software_name, f"{project_id}_start.png")
    print('Save result in', new_screenpath)
    os.makedirs(os.path.dirname(new_screenpath), exist_ok=True)
    shutil.copy(screenshot_path, new_screenpath)

    # 发送GUI解析请求
    gui_results = send_gui_parser_request(
        basic_config['gui_parser']['url'],
        software_name,
        screenshot_path,
        meta_data,
        task_id=project_id,
        step_id="1"
    )
    
    # 检查GUI解析结果是否有错误
    if isinstance(gui_results, dict) and gui_results.get('status') == 'error':
        print(f"GUI解析失败: {gui_results.get('error', 'Unknown error')}")
        print("使用默认的空GUI信息继续执行...")
        gui_results = {'QGIS': []}  # 提供默认的空结构
    
    print("QGIS GUI解析结果:")
    gui_info = compress_gui(gui_results)
    gui_info_str = "\n".join(format_gui(gui_info))
    print(gui_info_str)

    # 测试计划生成
    gui_info = compress_gui(gui_results)
    gui_info = "\n".join(format_gui(gui_info))
    
    plan = autopc.run_planner(query, software_name, screenshot_path, gui_info, "")
    print("生成的计划:", plan)
    
    # 启用Actor自动执行环节
    state = '<Continue>'
    code = ""
    last_screenshot_path = screenshot_path  # 初始化为第一次的截图路径
    critic_count = 0

    for idx in range(maximum_step):
        meta_data, screenshot_path = get_screenshot(software_name)
        print("===Current task===", "Index:",  idx, state)
        
        # 检查current_task是否存在，避免AttributeError
        if autopc.current_task is None:
            print("所有任务已完成！")
            state = '<Finished>'
            break
            
        print(autopc.current_task.name.strip())
        print("Before next(), current_task:", autopc.current_task.name)
        code, state, current_task = autopc.run_step(
            state,
            code,
            autopc.current_task, 
            meta_data, 
            last_screenshot_path,
            screenshot_path, 
            software_name,
            if_screenshot=True
        )
        # 执行 actor 生成的 pyautogui 代码
        pyautogui_import = "from pyautogui import click, write, hotkey, press, scroll, keyDown, keyUp, doubleClick, moveTo, mouseDown, mouseUp"

        # 定义代码执行函数，避免重复代码
        def execute_code(code_to_execute):
            if code_to_execute == "":
                return
                
            focus_software(software_name)
            try:
                # 检查是否包含Python代码块
                if "```python" in code_to_execute:
                    python_code = code_to_execute.split("```python")[1].split("```", 1)[0].strip()
                elif "```" in code_to_execute:
                    python_code = code_to_execute.split("```", 1)[1].split("```", 1)[0].strip()
                else:
                    python_code = code_to_execute

                # 如果没有import语句，则自动补全
                if "moveTo" in python_code and "import moveTo" not in python_code and "from pyautogui" not in python_code:
                    python_code = pyautogui_import + "\n" + python_code

                print(f"执行代码: {python_code}")
                exec(python_code)
                return True
            except Exception as e:
                print(f"执行代码时出错: {e}")
                print(f"问题代码: {python_code}")
                return False

        # 定义无效代码检测函数
        def _is_invalid_code(code):
            """
            检测代码是否无效
            无效代码包括：只包含pass、注释、无法执行的代码等
            """
            if not code:
                return True
                
            # 清理代码，移除注释和空行
            lines = code.strip().split('\n')
            executable_lines = []
            
            for line in lines:
                line = line.strip()
                # 跳过空行、注释行
                if not line or line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                    continue
                executable_lines.append(line)
            
            # 如果只有pass语句，认为是无效代码
            if len(executable_lines) == 1 and executable_lines[0] == 'pass':
                return True
                
            # 如果没有任何可执行代码，认为是无效代码
            if not executable_lines:
                return True
                
            # 检查是否包含有效的操作（如click、write、moveTo等）
            valid_operations = ['click', 'write', 'moveTo', 'press', 'hotkey', 'scroll', 'doubleClick', 'mouseDown', 'mouseUp']
            has_valid_operation = any(op in code for op in valid_operations)
            
            return not has_valid_operation

        # 执行当前代码
        if code != "":
            execute_code(code)
            last_screenshot_path = screenshot_path

        if state == '<Continue>':
            state = '<Critic>'
        elif state == '<Next>':
            autopc.current_task = autopc.current_task.next()
            if autopc.current_task:
                state = '<Continue>'
                code = ""
                critic_count = 0
                print("After next(), current_task:", autopc.current_task.name)
            else:
                # 没有下一个任务，标记为完成
                print("没有更多任务，标记为完成")
                state = '<Finished>'
                break
        elif state == '<Rethink>':
            # Handle rethinking workflow - 处理重新思考工作流
            print(" Detected rethinking state, executing newly generated code...")
            if code != "" and not _is_invalid_code(code):
                execute_code(code)
                last_screenshot_path = screenshot_path
                print("Rethinking completed, continuing task execution...")
            else:
                print("No valid code after rethinking, re-capturing screenshot...")
                # Re-capture screenshot and restart analysis - 重新获取截图并重新开始分析
                continue
            
            # Reset state and continue execution - 重置状态并继续执行
            state = '<Continue>'
            code = ""
            critic_count = 0
        elif state == '<Finished>':
            # Only end when there are truly no more tasks - 只有在真正没有更多任务时才结束
            if not autopc.current_task or not hasattr(autopc.current_task, 'next'):
                print("===Current task===", "Index:",  idx, state)
                break
            else:
                # Still have tasks, continue execution - 还有任务，继续执行
                state = '<Continue>'
                print("Detected more tasks, continuing execution...")
        else:
            # Unknown state, log and continue - 未知状态，记录日志并继续
            print(f"Unknown state: {state}, resetting to Continue state")
            state = '<Continue>'
        if state == '<Critic>':
            critic_count += 1
        if critic_count > max_critic_trials:
            autopc.current_task = autopc.current_task.next()
            if autopc.current_task:
                state = '<Continue>'
                code = ""
                critic_count = 0
                print("After next(), current_task:", autopc.current_task.name)
            else:
                state = '<Finished>'
                print('current index', idx, state)
                break

    # 保存最终截图
    new_screenpath = os.path.join(saved_folder, software_name, f"{project_id}_end.png")
    print('Save result in', new_screenpath)
    os.makedirs(os.path.dirname(new_screenpath), exist_ok=True)
    shutil.copy(screenshot_path, new_screenpath)
    
    print("程序执行完成！")

if __name__ == "__main__":
    main()