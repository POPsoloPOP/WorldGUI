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
        ]

        # 2. 解析主面板（可参考ppt_parser的get_panel_uia_ocr或get_panel_uia）
        self.parsed_gui = self.get_panel_uia(meta_data, screenshot_path)

        # 3. OCR识别
        _, ocr = text_detection(screenshot_path, save_png=False)

        # 4. 你可以在这里自定义QGIS特有的控件解析逻辑
        # for panel_item in self.parsed_gui[software_name]:
        #     if panel_item["name"] == "Main Content":
        #         # 例如：提取图层面板、工具栏等
        #         pass

        self.postprocess_uia(self.parsed_gui)
        return self.parsed_gui