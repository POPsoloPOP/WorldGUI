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

def test_qgis():
    software_name = "QGIS"
    project_id = "qgis_test"
    qgis_path = r"G:\BYXG\bin\qgis-ltr-bin.exe"  # QGIS可执行文件路径
    saved_folder = f'test_results/{basic_config["planner_critic"]["lmm"]}'

    # 启动QGIS
    #subprocess.Popen([qgis_path], shell=True)
    #time.sleep(3)  # 等待QGIS启动

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
    print("QGIS GUI解析结果:")
    gui_info = compress_gui(gui_results)
    gui_info_str = "\n".join(format_gui(gui_info))
    print(gui_info_str)

    # 打印gui_results结构，便于调试
    #print('gui_results结构:')
    #print(json.dumps(gui_results, indent=2, ensure_ascii=False))

    # 画图
    def draw_rectangles_on_image(screenshot_path, gui_results, save_path):
        img = cv2.imread(screenshot_path)
        for window_name, window_data in gui_results.items():
            for panel_item in window_data:
                for row in panel_item.get("elements", []):
                    if isinstance(row, list):
                        for element in row:
                            rect = element.get('rectangle')
                            if rect:
                                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 3)
                    elif isinstance(row, dict):
                        rect = row.get('rectangle')
                        if rect:
                            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 3)
        cv2.imwrite(save_path, img)

    draw_rectangles_on_image(screenshot_path, gui_results, "debug_with_boxes.png")

    # 画小元素的中心点
    def draw_element_positions(screenshot_path, gui_results, save_path):
        img = cv2.imread(screenshot_path)
        for window_name, window_data in gui_results.items():
            for panel_item in window_data:
                for row in panel_item.get("elements", []):
                    if isinstance(row, list):
                        for element in row:
                            pos = element.get('position')
                            if pos:
                                cv2.circle(img, (pos[0], pos[1]), 6, (0, 255, 0), -1)
                    elif isinstance(row, dict):
                        pos = row.get('position')
                        if pos:
                            cv2.circle(img, (pos[0], pos[1]), 6, (0, 255, 0), -1)
        cv2.imwrite(save_path, img)

    draw_element_positions(screenshot_path, gui_results, "debug_with_points.png")

    # 测试计划生成
    query = "Add a new vector layer to the project"
    gui_info = compress_gui(gui_results)
    gui_info = "\n".join(format_gui(gui_info))
    
    plan = autopc.run_planner(query, software_name, screenshot_path, gui_info, "")
    print("生成的计划:", plan)

if __name__ == "__main__":
    test_qgis()