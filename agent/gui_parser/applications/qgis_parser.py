from agent.gui_parser.ui_text_detection import text_detection
from agent.gui_parser.utils import *
from agent.gui_parser.gui_parser_base import GUIParserBase
import cv2
import numpy as np

class QGISParser(GUIParserBase):
    name = "qgis_parser"

    def __init__(self, cache_folder='.cache/', coordinate_config=None):
        super(GUIParserBase, self).__init__()
        self.cache_folder = cache_folder
        self.task_id = get_current_time()
        self.count = 1
        
        # 支持从外部传入坐标配置
        if coordinate_config:
            self.load_coordinate_config(coordinate_config)
        else:
            # 自动检测配置文件
            self.auto_detect_config()
    
    def auto_detect_config(self):
        """自动检测并加载配置文件"""
        import os
        
        # 可能的配置文件路径
        possible_configs = [
            "qgis_coordinates_config.json",
            os.path.join(os.path.dirname(__file__), "qgis_coordinates_config.json"),
            os.path.join(os.getcwd(), "qgis_coordinates_config.json")
        ]
        
        for config_path in possible_configs:
            if os.path.exists(config_path):
                print(f"自动检测到配置文件: {config_path}")
                try:
                    self.load_coordinate_config(config_path)
                    return
                except Exception as e:
                    print(f"加载配置文件失败: {e}")
                    continue
        
        # 如果没有找到配置文件，抛出错误
        error_msg = """
错误：未找到QGIS坐标配置文件！
请确保以下文件之一存在：
1. ./qgis_coordinates_config.json
2. agent/gui_parser/applications/qgis_coordinates_config.json

或者通过参数传入配置：
QGISParser(coordinate_config="your_config.json")
"""
        print(error_msg)
        raise FileNotFoundError("QGIS坐标配置文件未找到")
    
    def load_coordinate_config(self, config_source):
        """
        从外部配置加载坐标信息
        config_source: 可以是文件路径、字典或JSON字符串
        """
        try:
            if isinstance(config_source, str):
                # 如果是文件路径，读取文件
                if config_source.endswith('.json'):
                    import json
                    with open(config_source, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                elif config_source.endswith('.yaml') or config_source.endswith('.yml'):
                    import yaml
                    with open(config_source, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的文件格式: {config_source}")
            elif isinstance(config_source, dict):
                config = config_source
            else:
                raise ValueError("config_source必须是文件路径或字典")
            
            # 🎯 从配置文件加载基准分辨率
            self.base_resolution = config.get('base_resolution', [1920, 1080])
            self.autonomous_gis_elements = {}
            
            # 🎯 从配置文件加载坐标配置
            coordinates = config.get('coordinates', [])
            for coord_item in coordinates:
                name = coord_item.get('name', '')
                position = coord_item.get('position', [])
                description = coord_item.get('description', '')
                
                if not position or len(position) < 2:
                    continue
                    
                x, y = position
                
                # 🎯 根据配置文件中的名称映射到对应的元素类型
                if 'data request' in name.lower() or 'write your' in name.lower():
                    element_key = "main_text_input"
                    element_name = "MAIN_TEXT_INPUT_BOX_Write_your_data_request_here"
                    element_type = "text_input"
                    class_name = "Edit"
                elif 'output directory' in name.lower():
                    element_key = "output_directory"
                    element_name = "OUTPUT_DIRECTORY_INPUT_FIELD_Select_output_folder"
                    element_type = "text_input"
                    class_name = "Edit"
                elif 'autonomous gis' in name.lower():
                    element_key = "autonomous_gis_panel"
                    element_name = name
                    element_type = "panel"
                    class_name = "Pane"
                elif 'main window' in name.lower():
                    element_key = "main_window"
                    element_name = name
                    element_type = "window"
                    class_name = "Window"
                elif 'layer panel' in name.lower():
                    element_key = "layer_panel"
                    element_name = name
                    element_type = "panel"
                    class_name = "Pane"
                elif 'map canvas' in name.lower():
                    element_key = "map_canvas"
                    element_name = name
                    element_type = "canvas"
                    class_name = "Pane"
                elif 'toolbar' in name.lower():
                    element_key = "toolbar"
                    element_name = name
                    element_type = "toolbar"
                    class_name = "ToolBar"
                elif 'menu bar' in name.lower():
                    element_key = "menu_bar"
                    element_name = name
                    element_type = "menu"
                    class_name = "MenuBar"
                elif 'status bar' in name.lower():
                    element_key = "status_bar"
                    element_name = name
                    element_type = "status"
                    class_name = "StatusBar"
                else:
                    # 对于其他元素，使用通用处理
                    element_key = name.lower().replace(' ', '_').replace('-', '_')
                    element_name = name
                    element_type = "generic"
                    class_name = "Pane"
                
                # 🎯 使用配置文件中的坐标计算矩形区域
                if element_type == "text_input":
                    rect_width, rect_height = 120, 60
                elif element_type == "panel":
                    rect_width, rect_height = 200, 150
                elif element_type == "window":
                    rect_width, rect_height = 300, 200
                else:
                    rect_width, rect_height = 100, 50
                    
                x1, y1 = x - rect_width//2, y - rect_height//2
                x2, y2 = x + rect_width//2, y + rect_height//2
                
                # 🎯 将配置文件中的坐标存储到解析器中
                self.autonomous_gis_elements[element_key] = {
                    "position": [x, y],  # 🎯 直接使用配置文件中的坐标
                    "rectangle": [x1, y1, x2, y2],  # 🎯 基于配置文件坐标计算的矩形
                    "name": element_name,
                    "class_name": class_name,
                    "element_type": element_type,
                    "description": description
                }
            
            print(f"已从配置加载坐标: {config_source}")
            print(f"基准分辨率: {self.base_resolution}")
            print(f"加载的元素: {list(self.autonomous_gis_elements.keys())}")
            
        except Exception as e:
            print(f"加载坐标配置失败: {e}")
            # 🚨 如果配置文件加载失败，直接抛出错误，不使用默认配置
            raise e

    def get_adaptive_coordinates(self, base_resolution, current_resolution, base_coords):
        """
        根据分辨率变化调整坐标
        base_resolution: 基准分辨率 [width, height] - 🎯 来自配置文件
        current_resolution: 当前分辨率 [width, height] - 🎯 来自截图
        base_coords: 基准坐标 [x, y] - 🎯 来自配置文件
        """
        try:
            # 🎯 使用配置文件中的基准分辨率进行坐标适配
            scale_x = current_resolution[0] / base_resolution[0]
            scale_y = current_resolution[1] / base_resolution[1]
            
            adapted_x = int(base_coords[0] * scale_x)
            adapted_y = int(base_coords[1] * scale_y)
            
            return [adapted_x, adapted_y]
        except Exception as e:
            print(f"坐标适配失败: {e}")
            return base_coords

    def get_adaptive_rectangle(self, base_resolution, current_resolution, base_rect):
        """
        根据分辨率变化调整矩形区域
        base_resolution: 基准分辨率 [width, height] - 🎯 来自配置文件
        current_resolution: 当前分辨率 [width, height] - 🎯 来自截图
        base_rect: 基准矩形 [x1, y1, x2, y2] - 🎯 基于配置文件坐标计算
        """
        try:
            # 🎯 使用配置文件中的基准分辨率进行矩形适配
            scale_x = current_resolution[0] / base_resolution[0]
            scale_y = current_resolution[1] / base_resolution[1]
            
            adapted_rect = [
                int(base_rect[0] * scale_x),
                int(base_rect[1] * scale_y),
                int(base_rect[2] * scale_x),
                int(base_rect[3] * scale_y)
            ]
            
            return adapted_rect
        except Exception as e:
            print(f"矩形适配失败: {e}")
            return base_rect

    def add_autonomous_gis_elements(self, panel_item, current_resolution):
        """
        为Autonomous GIS面板添加固定位置的元素
        """
        detected_elements = []
        
        # 🎯 遍历配置文件中的所有元素配置
        for element_key, element_config in self.autonomous_gis_elements.items():
            # 🎯 使用配置文件中的坐标进行分辨率适配
            adapted_position = self.get_adaptive_coordinates(
                self.base_resolution,  # 🎯 配置文件中的基准分辨率
                current_resolution,     # 🎯 当前截图分辨率
                element_config["position"]  # 🎯 配置文件中的坐标
            )
            
            # 🎯 使用配置文件中的矩形区域进行分辨率适配
            adapted_rectangle = self.get_adaptive_rectangle(
                self.base_resolution,  # 🎯 配置文件中的基准分辨率
                current_resolution,     # 🎯 当前截图分辨率
                element_config["rectangle"]  # 🎯 基于配置文件坐标计算的矩形
            )
            
            # 🎯 创建元素，使用配置文件中的所有信息
            element = {
                "class_name": "Edit",
                "depth": f"1-{len(detected_elements)+20}",
                "name": element_config["name"],  # 🎯 配置文件中的元素名称
                "type": ["Click", "rightClick", "type"],
                "position": adapted_position,     # 🎯 适配后的坐标
                "rectangle": adapted_rectangle,   # 🎯 适配后的矩形
                "interactive": True,
                "element_type": "text_input",
                "text": element_config["name"],   # 🎯 配置文件中的元素名称
                "description": f"固定位置配置的{element_config['name']}",
                "detection_method": "fixed_position",
                "original_resolution": self.base_resolution,  # 🎯 配置文件中的基准分辨率
                "current_resolution": current_resolution      # 🎯 当前截图分辨率
            }
            
            detected_elements.append(element)
            print(f"已添加元素: {element['name']} at {element['position']}")
        
        return detected_elements

    def __call__(self, meta_data, screenshot_path, software_name=None):
        self.software_name = software_name
        self.parsed_gui = {software_name: []}

        # 1. 自定义排除/包含的控件类型
        self.exclude_class_name_list = [
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

        # 2. 解析主面板
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

        # 5. 检查是否需要fallback机制
        if not self.parsed_gui[software_name] or len(self.parsed_gui[software_name]) == 0:
            print("警告: UI元素检测失败，使用配置文件fallback机制")
            self._create_fallback_elements(screenshot_path)
        else:
            # 6. 为Autonomous GIS面板添加固定位置的元素
            for panel_item in self.parsed_gui[software_name]:
                # 检查是否是Autonomous GIS面板
                if "Autonomous GIS" in panel_item.get("name", ""):
                    print(f"检测到Autonomous GIS面板，添加固定位置元素")
                    
                    # 获取当前截图的分辨率
                    img = cv2.imread(screenshot_path)
                    if img is not None:
                        current_resolution = [img.shape[1], img.shape[0]]  # [width, height]
                        print(f"当前截图分辨率: {current_resolution}")
                        
                        # 添加固定位置的元素
                        detected_elements = self.add_autonomous_gis_elements(panel_item, current_resolution)
                        
                        # 将检测到的元素添加到面板中
                        if "elements" not in panel_item:
                            panel_item["elements"] = []
                        
                        for element in detected_elements:
                            panel_item["elements"].append(element)
                        
                        # 标记这个面板已经过自定义处理
                        panel_item["custom_processed"] = True
                        panel_item["autonomous_gis_panel"] = True
                        panel_item["resolution_adaptive"] = True
                    
                # 可以在这里添加其他QGIS插件的自定义逻辑
                elif "Processing Toolbox" in panel_item.get("name", ""):
                    # 为Processing Toolbox添加特定逻辑
                    pass
                elif "Attribute Table" in panel_item.get("name", ""):
                    # 为Attribute Table添加特定逻辑
                    pass

        self.postprocess_uia(self.parsed_gui)
        return self.parsed_gui

    def _create_fallback_elements(self, screenshot_path):
        """
        当UI元素检测失败时，基于配置文件创建必要的元素
        """
        try:
            # 获取当前截图的分辨率
            img = cv2.imread(screenshot_path)
            if img is None:
                print("错误: 无法读取截图文件")
                return
                
            current_resolution = [img.shape[1], img.shape[0]]  # [width, height]
            print(f"当前截图分辨率: {current_resolution}")
            
            # 创建fallback元素列表
            fallback_elements = []
            
            # 基于配置文件创建元素
            for element_key, element_config in self.autonomous_gis_elements.items():
                # 使用配置文件中的坐标进行分辨率适配
                adapted_position = self.get_adaptive_coordinates(
                    self.base_resolution,  # 配置文件中的基准分辨率
                    current_resolution,     # 当前截图分辨率
                    element_config["position"]  # 配置文件中的坐标
                )
                
                # 使用配置文件中的矩形区域进行分辨率适配
                adapted_rectangle = self.get_adaptive_rectangle(
                    self.base_resolution,  # 配置文件中的基准分辨率
                    current_resolution,     # 当前截图分辨率
                    element_config["rectangle"]  # 基于配置文件坐标计算的矩形
                )
                
                # 创建fallback元素
                fallback_element = {
                    "class_name": element_config.get("class_name", "Pane"),
                    "depth": f"1-{len(fallback_elements)+10}",
                    "name": element_config["name"],
                    "type": self._get_element_types(element_config.get("element_type", "generic")),
                    "position": adapted_position,
                    "rectangle": adapted_rectangle,
                    "interactive": element_config.get("element_type") in ["text_input", "button"],
                    "element_type": element_config.get("element_type", "generic"),
                    "text": element_config["name"],
                    "description": element_config.get("description", f"配置文件定义的{element_config['name']}"),
                    "detection_method": "fallback_config",
                    "original_resolution": self.base_resolution,
                    "current_resolution": current_resolution,
                    "fallback_created": True
                }
                
                fallback_elements.append(fallback_element)
                print(f"已创建fallback元素: {fallback_element['name']} at {fallback_element['position']}")
            
            # 将fallback元素添加到解析结果中
            self.parsed_gui[self.software_name] = fallback_elements
            
        except Exception as e:
            print(f"创建fallback元素失败: {e}")
            # 如果fallback也失败，至少创建一个基本的Main Content区域
            self.parsed_gui[self.software_name] = [{
                "name": "Main Content (Fallback)",
                "rectangle": [0, 0, current_resolution[0], current_resolution[1]],
                "class_name": "Pane",
                "depth": "1",
                "elements": [],
                "fallback_created": True,
                "detection_method": "emergency_fallback"
            }]

    def _get_element_types(self, element_type):
        """
        根据元素类型返回可用的操作类型
        """
        if element_type == "text_input":
            return ["Click", "rightClick", "type", "doubleClick"]
        elif element_type == "button":
            return ["Click", "rightClick", "doubleClick"]
        elif element_type == "panel":
            return ["Click", "rightClick"]
        elif element_type == "window":
            return ["Click", "rightClick", "move"]
        else:
            return ["Click", "rightClick"] 