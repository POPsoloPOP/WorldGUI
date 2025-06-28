# test_qgis_automation.py
import os
from agent.autopc import AutoPC
from agent.utils.gui_capture import get_screenshot, focus_software
from agent.gui_parser.sender import send_gui_parser_request
from agent.actor.utils import format_gui, compress_gui
from agent.config import basic_config

def test_qgis():
    software_name = "QGIS"
    project_id = "qgis_test"
    
    # 初始化 AutoPC
    autopc = AutoPC(software_name=software_name, project_id=project_id)
    
    # 聚焦 QGIS
    focus_software(software_name)
    meta_data, screenshot_path = get_screenshot(software_name)
    
    # 测试 GUI 解析
    gui_results = send_gui_parser_request(
        basic_config['gui_parser']['url'], 
        software_name, 
        screenshot_path, 
        meta_data, 
        task_id=project_id, 
        step_id="1"
    )
    
    print("GUI 解析结果:", gui_results)
    
    # 测试计划生成
    query = "Add a new vector layer to the project"
    gui_info = compress_gui(gui_results)
    gui_info = "\n".join(format_gui(gui_info))
    
    plan = autopc.run_planner(query, software_name, screenshot_path, gui_info, "")
    print("生成的计划:", plan)

if __name__ == "__main__":
    test_qgis()