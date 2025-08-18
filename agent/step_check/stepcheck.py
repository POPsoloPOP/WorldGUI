import os
import re
import json
import glob
import copy
from agent.actor.utils import format_gui, compress_gui
from agent.utils.lmm.run_lmm import run_lmm
from agent.utils.app_functions import run_locateregion

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class StepCheck:
    """Tool that adds the capability to locate image region with a natural language query."""

    name = "stepcheck"
    description = (
        '''pending''')

    def __init__(self, lmm="gpt-4o-2024-08-06"):
        super(StepCheck, self).__init__()
        self.lmm = lmm
        self.software_tips = self.load_software_tips()
        
        # 新增：动态监控器
        self.execution_monitor = ExecutionMonitor()

    def __call__(
        self,
        current_task,
        parsed_screenshot=None,
        screenshot_path=None,
        stepcheck_decision=None,
        history=None,
        software_name=None,
        if_screenshot=True,
        **kwargs
    ):
        """
        Executes the given task using the provided GUI state and input image. Adjusts the task flow based
        on the execution outcome and prepares the next interaction code.

        Parameters:
            current_task: The current task to be processed.
            gui: Current state of the GUI.
            input_image: An optional screenshot used for updating GUI state.
            error_message: An optional error message for adjusting task plans.
            next_step: Suggested next step if adjustments are needed.
            pre_act_success_flag: Flag indicating if the previous action was successful.
            pre_act_resume_flag: Flag indicating if the previous action needs resuming.
            software_name: The name of the software being interacted with.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the interaction code, updated current task, updated history, and a status message.
        """

        # Task Info
        main_goal, finished_tasks, current_task_text, next_task = self.get_task_details(current_task, history)

        tips = self.get_software_tips(self.software_tips, software_name.lower().replace(' ', '_'))

        # 新增：执行过程中的动态监控
        if screenshot_path and parsed_screenshot:
            monitoring_result = self.execution_monitor.monitor_execution_context(
                current_step=current_task_text,
                screenshot_path=screenshot_path,
                parsed_screenshot=parsed_screenshot,
                software_name=software_name
            )
            
            # 如果检测到需要重新思考的情况
            if monitoring_result.get('rethinking_needed'):
                return self.trigger_rethinking_workflow(
                    monitoring_result=monitoring_result,
                    current_task=current_task,
                    parsed_screenshot=parsed_screenshot,
                    screenshot_path=screenshot_path,
                    software_name=software_name,
                    tips=tips
                )

        # step checker before run the actor
        # initial values
        stepcheck_decision = '<Retry>'
        new_screenshot_path = screenshot_path

        iter_idx = 1
        while stepcheck_decision == '<Retry>':

            if iter_idx > 2:
                stepcheck_decision = '<Continue>'
                break

            critic_feedback = self.step_critic(
                software_name=software_name,
                tips=tips,
                main_goal=main_goal,
                current_task_text=current_task_text,
                finished_tasks=finished_tasks,
                next_task=next_task,
                screenshot_path=new_screenshot_path,
                if_screenshot=if_screenshot
            )

            if '<Continue>' in critic_feedback:
                stepcheck_decision = '<Continue>'
            elif '<Modify>' in critic_feedback:
                current_task_text = self.extract_task(critic_feedback, 'Modify')
                current_task.name = current_task_text
                stepcheck_decision = '<Continue>'
            elif '<Finished>' in critic_feedback:
                stepcheck_decision = '<Finished>'
            elif '#Cannot confirm' in critic_feedback and iter_idx < 2:
                
                if parsed_screenshot:
                    compressed_gui = self.compress_and_format_gui(parsed_screenshot)
                    new_screenshot_path = run_locateregion(
                        compressed_gui, 
                        current_task_text, 
                        software_name, 
                        new_screenshot_path
                    )
                    iter_idx += 1
                else:
                    stepcheck_decision = '<Continue>'
            else:
                stepcheck_decision = '<Continue>'

        return stepcheck_decision, current_task, history

    # 新增：触发重新思考工作流
    def trigger_rethinking_workflow(self, monitoring_result, current_task, parsed_screenshot, screenshot_path, software_name, tips):
        """触发重新思考工作流"""
        print(f"🔄 Step-Check 检测到需要重新思考: {monitoring_result['reason']}")
        
        # 分析当前界面状态
        current_gui_state = self.analyze_current_interface(parsed_screenshot, screenshot_path)
        
        # 生成重新思考后的任务调整建议
        task_adjustment = self.generate_task_adjustment(
            original_task=current_task,
            monitoring_result=monitoring_result,
            current_gui_state=current_gui_state
        )
        
        # 返回重新思考状态
        return "<Rethink>", task_adjustment, []

    # 新增：分析当前界面状态
    def analyze_current_interface(self, parsed_screenshot, screenshot_path):
        """分析当前界面状态"""
        if parsed_screenshot:
            return self.compress_and_format_gui(parsed_screenshot)
        return ""

    # 新增：生成任务调整建议
    def generate_task_adjustment(self, original_task, monitoring_result, current_gui_state):
        """基于监控结果生成任务调整建议"""
        adjustment = {
            'original_task': original_task.name if hasattr(original_task, 'name') else str(original_task),
            'rethinking_reason': monitoring_result['reason'],
            'detected_issues': monitoring_result.get('detected_issues', []),
            'suggested_actions': monitoring_result.get('suggested_actions', []),
            'current_context': current_gui_state
        }
        
        return adjustment

    @staticmethod
    def extract_task(result, label):
        match = re.search(r'<%s>(.*?)</%s>'%(label, label), result, re.DOTALL)

        if match:
            extracted_text = match.group(1).strip()

            new_txt = ''
            for line in extracted_text.split("\n"):
                if not line.startswith('#'):
                    new_txt += line.strip()
            extracted_text = new_txt
        else:
            extracted_text = ''
        return extracted_text

    def subtask_refiner(self, software_name, tips, current_task, screenshot_path=None, if_screenshot=True):
        current_task = current_task.replace('[', '').replace(']', '')

        text_prompt =  f'''You are very smart, I would like your assistance for Desktop GUI automation.

I will provide the software name, a screenshot of the current environment, and task details.

You should help me refine the the task description of current task based on the given software tips and screenshot.

Software name: {software_name}
Software tips: {tips}

Information about Task:
{current_task}

The output format should be:

<Refine>
...
</Refine>

Note:
1) If no refinement of the task description is needed, please output with the original content.
2) Ensure the task description of the current task is clear and accurate, with no misunderstandings or redundant information.
3) If you did not receive the screenshot, please say it.

# Remember to reason in comment if needed.
'''
        prompt = [text_prompt, screenshot_path] if if_screenshot else [text_prompt] 
        
        result = run_lmm(
            prompt,
            lmm=self.lmm,
            max_tokens=1000, 
            temperature=0
        )

        match = re.search(r'<Refine>(.*?)</Refine>', result, re.DOTALL)

        if match:
            extracted_text = match.group(1).strip()
        else:
            extracted_text = ''

        extracted_text = f"{extracted_text}"

        return extracted_text
    
    def step_critic(self, software_name, tips, main_goal, current_task_text, finished_tasks, next_task, screenshot_path, if_screenshot):

        prompt = self.construct_step_critic_prompt(
            software_name,
            tips,
            main_goal,
            current_task_text,
            finished_tasks,
            next_task,
            screenshot_path,
            if_screenshot
        )

        critic_feedback = run_lmm(
            prompt,
            lmm=self.lmm,
            max_tokens=500, 
            temperature=0
        )

        return critic_feedback

    def construct_step_critic_prompt(self, 
        software_name,
        tips,
        main_goal,
        current_task,
        finished_tasks,
        next_task,
        screenshot_path=None,
        if_screenshot=True
    ):

        """Construct the detailed prompt for the LMM based on provided parameters."""
        text_prompt = f'''You are very smart, I would like your assistance for Desktop GUI automation.

I will provide the software name, a screenshot of the current environment, and task details.

You need to verify, based on the screenshot, whether the current task has been completed.

If already completed, please output <Finished> and reasons. Please be cautious to output <Finished>.

If require modification, please either add more plans or modify current step.

If you modify the current task, the output format should be as follows:

<Modify>
...
</modify>

If you think current task is unnecessary when you see the privided screenshot and the next task, please output:
<Pass>

For example, when current screenshot already includes the information can be used to solve next task, we may jump up current task, output <Pass>

If no change, the output format should be as follows:
<Continue>

Information about Task:
{main_goal}
{finished_tasks}
{current_task}
{next_task}

Software name: {software_name}
Software tips: {tips}

If you think current screenshot is not give all information to check the current task completion, please output '#Cannot confirm', we will provide a new screenshot.

Note:
1) Be very careful with the output <Finished>.
2) You have to carefully read the software tips to give your answer.
3) When you consider if the current task is necessary, please consider the content of next task.
4) When determine whether the button is clicked, please be careful to output <Finished>.


# Remember to reason in comment if needed.
'''

        return [text_prompt, screenshot_path] if if_screenshot else [text_prompt]        

    def compress_and_format_gui(self, gui):
        """Compress and format the GUI details for display."""
        compressed_gui = compress_gui(copy.deepcopy(gui))
        return "\n".join(format_gui(compressed_gui))

    def get_task_details(self, current_task, history):
        """Extract task name and main goal from current task."""
        if isinstance(current_task, str):
            return "", "", f"Current Task: {current_task}", ""
        
        main_goal = f"Main Goal: {current_task.parent.name}"
        
        summarized_history = self.get_code_history_for_current_task(history)
        finished_task = '\n'.join(summarized_history['finished_tasks'])
        finished_task = f"Previous Finished Tasks: {finished_task}"
        
        next_task = current_task.next().name if current_task.next() else "No more tasks"
        next_task = f"Next Task (for reference, you should consider whether current task is necessary when we complete next task ): {next_task}"
                
        current_task_text = f"Current Task: {current_task.name}"
        
        return main_goal, finished_task, current_task_text, next_task

    @staticmethod
    def check_resume(history):
        if history:
            history_code = "\n".join(history[-1]['code']) if history[-1]['code'][0] else "# finish"
            if "# finish" in history_code:
                return False
            else:
                return True
        else:
            "# finish"
        

    def get_code_history_for_current_task(self, history):
        # keep previous four steps
        finished_tasks, code = "", ""
        if history:
            if self.check_resume(history):
                # select self.history from -5 index to -1 index, needs to check length
                finished_tasks = [x['task'] for x in history[-5:-1]]
                code = "\n".join(history[-1]['code'])
            else:
                finished_tasks = [x['task'] for x in history[-4:]]
        return {"finished_tasks": finished_tasks, "code": code}
    
    def load_software_tips(self, resourcedir="resources\software_tips"):
        software_tips_files = glob.glob(os.path.join(os.path.dirname(__file__), resourcedir, "*.json"))

        # load files and merge them
        software_tips = {}
        for file in software_tips_files:
            with open(file, 'r') as f:
                software_tips.update(json.load(f))
                
        return software_tips
        
    def get_software_tips(self, target, software_name):        
        hints = "\n".join(target.get(software_name, [""]))
        return hints


class ExecutionMonitor:
    """执行过程监控器，用于检测需要重新思考的情况"""
    
    def __init__(self):
        self.monitoring_rules = {
            'qgis': QGISMonitoringRules(),
            'word': WordMonitoringRules(),
            'powerpoint': PowerPointMonitoringRules(),
            'excel': ExcelMonitoringRules(),
            'default': DefaultMonitoringRules()
        }
    
    def monitor_execution_context(self, current_step, screenshot_path, parsed_screenshot, software_name):
        """监控执行过程中的上下文变化"""
        try:
            # 获取对应软件的监控规则
            monitoring_rules = self.monitoring_rules.get(software_name.lower(), self.monitoring_rules['default'])
            
            # 执行监控检查
            monitoring_result = monitoring_rules.check_execution_context(
                current_step=current_step,
                screenshot_path=screenshot_path,
                parsed_screenshot=parsed_screenshot
            )
            
            return monitoring_result
            
        except Exception as e:
            print(f"Execution monitoring error: {e}")
            return {
                'rethinking_needed': False,
                'reason': f"监控过程出错: {e}",
                'detected_issues': [],
                'suggested_actions': []
            }


class QGISMonitoringRules:
    """QGIS软件的监控规则"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """检查QGIS执行上下文"""
        issues = []
        suggested_actions = []
        
        # 检查是否有弹框出现
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("检测到弹框出现")
            suggested_actions.append("需要与弹框进行交互")
        
        # 检查界面状态是否异常
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("检测到界面状态异常")
            suggested_actions.append("需要重新分析界面状态")
        
        # 判断是否需要重新思考
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "执行正常"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """检测弹框出现 - 纯图像对比版本"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("警告: 缺少OpenCV库，弹框检测功能受限")
            return False
        
        try:
            # 读取图像
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # 弹框检测参数
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # 使用边缘检测找到可能的弹框轮廓
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找符合弹框特征的轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # 形状过滤：弹框通常是矩形
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # 检查轮廓的矩形度
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"QGIS弹框检测出错: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """检测界面状态异常"""
        try:
            if not parsed_screenshot:
                return False
            
            # 检查是否有异常的元素状态
            anomalies = []
            
            # 检查是否有错误提示
            if isinstance(parsed_screenshot, dict):
                # 检查文本元素中是否包含错误信息
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"检测到错误信息: {text_elem['text']}")
                
                # 检查是否有异常的状态指示
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"按钮状态异常: {button.get('text', 'Unknown')} - {button['state']}")
                
                # 检查是否有加载状态
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("检测到加载状态")
            
            # 如果有异常，返回True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"QGIS界面状态异常检测出错: {e}")
            return False


class WordMonitoringRules:
    """Word软件的监控规则"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """检查Word执行上下文"""
        issues = []
        suggested_actions = []
        
        # 检查是否有弹框出现
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("检测到弹框出现")
            suggested_actions.append("需要与弹框进行交互")
        
        # 检查界面状态是否异常
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("检测到界面状态异常")
            suggested_actions.append("需要重新分析界面状态")
        
        # 判断是否需要重新思考
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "执行正常"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """检测弹框出现 - 纯图像对比版本"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("警告: 缺少OpenCV库，弹框检测功能受限")
            return False
        
        try:
            # 读取图像
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # 弹框检测参数
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # 使用边缘检测找到可能的弹框轮廓
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找符合弹框特征的轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # 形状过滤：弹框通常是矩形
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # 检查轮廓的矩形度
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Word弹框检测出错: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """检测界面状态异常"""
        try:
            if not parsed_screenshot:
                return False
            
            # 检查是否有异常的元素状态
            anomalies = []
            
            # 检查是否有错误提示
            if isinstance(parsed_screenshot, dict):
                # 检查文本元素中是否包含错误信息
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"检测到错误信息: {text_elem['text']}")
                
                # 检查是否有异常的状态指示
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"按钮状态异常: {button.get('text', 'Unknown')} - {button['state']}")
                
                # 检查是否有加载状态
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("检测到加载状态")
            
            # 如果有异常，返回True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"Word界面状态异常检测出错: {e}")
            return False


class PowerPointMonitoringRules:
    """PowerPoint软件的监控规则"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """检查PowerPoint执行上下文"""
        issues = []
        suggested_actions = []
        
        # 检查是否有弹框出现
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("检测到弹框出现")
            suggested_actions.append("需要与弹框进行交互")
        
        # 检查界面状态是否异常
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("检测到界面状态异常")
            suggested_actions.append("需要重新分析界面状态")
        
        # 判断是否需要重新思考
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "执行正常"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """检测弹框出现 - 纯图像对比版本"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("警告: 缺少OpenCV库，弹框检测功能受限")
            return False
        
        try:
            # 读取图像
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # 弹框检测参数
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # 使用边缘检测找到可能的弹框轮廓
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找符合弹框特征的轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # 形状过滤：弹框通常是矩形
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # 检查轮廓的矩形度
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"PowerPoint弹框检测出错: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """检测界面状态异常"""
        try:
            if not parsed_screenshot:
                return False
            
            # 检查是否有异常的元素状态
            anomalies = []
            
            # 检查是否有错误提示
            if isinstance(parsed_screenshot, dict):
                # 检查文本元素中是否包含错误信息
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"检测到错误信息: {text_elem['text']}")
                
                # 检查是否有异常的状态指示
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"按钮状态异常: {button.get('text', 'Unknown')} - {button['state']}")
                
                # 检查是否有加载状态
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("检测到加载状态")
            
            # 如果有异常，返回True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"PowerPoint界面状态异常检测出错: {e}")
            return False


class ExcelMonitoringRules:
    """Excel软件的监控规则"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """检查Excel执行上下文"""
        issues = []
        suggested_actions = []
        
        # 检查是否有弹框出现
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("检测到弹框出现")
            suggested_actions.append("需要与弹框进行交互")
        
        # 检查界面状态是否异常
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("检测到界面状态异常")
            suggested_actions.append("需要重新分析界面状态")
        
        # 判断是否需要重新思考
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "执行正常"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """检测弹框出现 - 纯图像对比版本"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("警告: 缺少OpenCV库，弹框检测功能受限")
            return False
        
        try:
            # 读取图像
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # 弹框检测参数
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # 使用边缘检测找到可能的弹框轮廓
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找符合弹框特征的轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # 形状过滤：弹框通常是矩形
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # 检查轮廓的矩形度
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Excel弹框检测出错: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """检测界面状态异常"""
        try:
            if not parsed_screenshot:
                return False
            
            # 检查是否有异常的元素状态
            anomalies = []
            
            # 检查是否有错误提示
            if isinstance(parsed_screenshot, dict):
                # 检查文本元素中是否包含错误信息
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"检测到错误信息: {text_elem['text']}")
                
                # 检查是否有异常的状态指示
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"按钮状态异常: {button.get('text', 'Unknown')} - {button['state']}")
                
                # 检查是否有加载状态
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("检测到加载状态")
            
            # 如果有异常，返回True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"Excel界面状态异常检测出错: {e}")
            return False


class DefaultMonitoringRules:
    """默认监控规则"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """检查默认执行上下文"""
        issues = []
        suggested_actions = []
        
        # 检查是否有弹框出现
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("检测到弹框出现")
            suggested_actions.append("需要与弹框进行交互")
        
        # 检查界面状态是否异常
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("检测到界面状态异常")
            suggested_actions.append("需要重新分析界面状态")
        
        # 判断是否需要重新思考
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "执行正常"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """检测弹框出现 - 纯图像对比版本"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("警告: 缺少OpenCV库，弹框检测功能受限")
            return False
        
        try:
            # 读取图像
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # 弹框检测参数
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # 使用边缘检测找到可能的弹框轮廓
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 寻找符合弹框特征的轮廓
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # 形状过滤：弹框通常是矩形
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # 检查轮廓的矩形度
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"默认弹框检测出错: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """检测界面状态异常"""
        try:
            if not parsed_screenshot:
                return False
            
            # 检查是否有异常的元素状态
            anomalies = []
            
            # 检查是否有错误提示
            if isinstance(parsed_screenshot, dict):
                # 检查文本元素中是否包含错误信息
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"检测到错误信息: {text_elem['text']}")
                
                # 检查是否有异常的状态指示
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"按钮状态异常: {button.get('text', 'Unknown')} - {button['state']}")
                
                # 检查是否有加载状态
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("检测到加载状态")
            
            # 如果有异常，返回True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"默认界面状态异常检测出错: {e}")
            return False
    