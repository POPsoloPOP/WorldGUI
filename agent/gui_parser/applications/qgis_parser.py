from agent.gui_parser.ui_text_detection import text_detection
from agent.gui_parser.utils import *
from agent.gui_parser.gui_parser_base import GUIParserBase
from ultralytics import YOLO

class QGISParser(GUIParserBase):
    name = "qgis_parser"

    def __init__(self, cache_folder='.cache/'):
        super(GUIParserBase, self).__init__()
        self.cache_folder = cache_folder
        self.task_id = get_current_time()
        self.yolo_model = YOLO("yolov8n-oiv7.pt")
        self.count = 1

    def __call__(self, meta_data, screenshot_path, software_name=None):
        self.software_name = software_name
        self.parsed_gui = {software_name: []}

        # 1. 你可以自定义排除/包含的控件类型
        self.exclude_class_name_list = [
            # 例如：'Custom', 'Menu', 'Pane', ...
            "Project",
            "Edit",
            "View",
            "Layer",
            "Plugins",
            "Vector",
            "Raster",
            "Database",
            "Web",
            "Mesh",
            "Processing",
        ]

        # 2. 解析主面板（可参考ppt_parser的get_panel_uia_ocr或get_panel_uia）
        self.parsed_gui = self.get_panel_uia(meta_data, screenshot_path)

        # 3. OCR识别
        _, ocr = text_detection(screenshot_path, save_png=False)

        # 4. 过滤红色框区域（工具栏区域），只保留不在该区域的控件
        # 红色框大致坐标为 x:0-2560, y:0-120
        RED_X1, RED_Y1, RED_X2, RED_Y2 = 0, 0, 2560, 120
        filtered_items = []
        for item in self.parsed_gui.get(software_name, []):
            rect = item.get('rectangle', None)
            name = item.get('name', '')
            class_name = item.get('class_name', '')
            # 1. 过滤红框
            if rect:
                x1, y1, x2, y2 = rect
                if x1 >= RED_X1 and y1 >= RED_Y1 and x2 <= RED_X2 and y2 <= RED_Y2:
                    continue
            # 2. 过滤标题栏和窗口控制按钮
            if class_name == "TitleBar":
                continue
            if class_name == "Button" and name in ["最小化", "还原", "关闭"]:
                continue
            # 3. 过滤窗口标题（如 Untitled Project - QGIS）
            if "QGIS" in name and ("Project" in name or "Untitled" in name):
                continue
            filtered_items.append(item)
        self.parsed_gui[software_name] = filtered_items

        # 4. 你可以在这里自定义QGIS特有的控件解析逻辑
        # for panel_item in self.parsed_gui[software_name]:
        #     if panel_item["name"] == "Main Content":
        #         # 例如：提取图层面板、工具栏等
        #         pass

        self.postprocess_uia(self.parsed_gui)
        return self.parsed_gui