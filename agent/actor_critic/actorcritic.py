import os
import re
import sys
import io
import copy
import json
import glob
import cv2
import numpy as np
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None
from agent.actor.utils import format_gui, compress_gui
from agent.utils.lmm.run_lmm import run_lmm


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class ActorCritic:
    """Tool that critic the task completion"""

    name = "ActorCritic"
    description = (
        '''
This tool can critiquing the completion of the current task.
''')

    def __init__(self, lmm="gpt-4o-2024-08-06", critic_lmm="gpt-4o-2024-08-06"):
        super(ActorCritic, self).__init__()
        self.lmm = lmm
        self.critic_lmm = critic_lmm

        self.critic_software_tips = self.load_software_tips("resources\critic_software_tips")
        self.software_tips = self.load_software_tips()
        
        # 新增：上下文变化检测器
        self.context_change_detector = ContextChangeDetector()

    def __call__(self,
                current_task,
                current_action,
                parsed_screenshot,
                screenshot_path=None,
                history=None,
                software_name=None,
                **kwargs):
        """
        Parameters:
            current_task: The current task to be processed.
            gui: Current state of the GUI.
            input_image: An optional screenshot used for updating GUI state.
            history: A list tracking the history of executed tasks and interactions.
            software_name: The name of the software being interacted with.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the interaction code, updated current task, updated history, and a status message.
        """

        # prepare the information for constructing the prompt  
        # Task Info
        main_goal, finished_tasks, current_task_text, next_task = self.get_task_details(current_task, history)
        
        # GUI Info
        if parsed_screenshot is not None:
            compressed_gui = self.compress_and_format_gui(parsed_screenshot)
        else:
            compressed_gui = ""

        # software tips
        critic_tips = self.get_software_tips(self.critic_software_tips, software_name.lower())
        tips = self.get_software_tips(self.software_tips, software_name.lower())

        # 新增：检测上下文变化
        context_changes = self.detect_context_changes(screenshot_path, current_task)
        
        # 新增：判断是否需要重新思考
        if context_changes and self.should_rethink_task(context_changes, current_task):
            return self.trigger_dynamic_rethinking(
                context_changes=context_changes,
                current_task=current_task,
                parsed_screenshot=parsed_screenshot,
                screenshot_path=screenshot_path,
                software_name=software_name,
                tips=tips
            )

        critic_prompt = self.construct_critic_prompt(software_name, current_task_text, current_action, compressed_gui, critic_tips, screenshot_path)

        # Prepare the action code based on the current task.
        success_flag, reason, critic_comment = self.generate_critic(prompt=critic_prompt, lmm=self.critic_lmm)


        if success_flag.lower() == 'false':
            
            ## locate the gui info

            if compressed_gui:
                reffered_gui = self.locate_gui_info(compressed_gui, main_goal, current_task_text)
            else:
                reffered_gui = ""

            if current_action != None:
                current_action = self.extract_purecode(current_action) # only pure code, no reasoning description
            else:
                current_action = ""

            correction_prompt = self.construct_correction_prompt(
                current_action, 
                critic_comment, 
                reffered_gui, 
                main_goal,
                current_task_text,
                tips,
                screenshot_path=None) # no screenshot provided
            code = self.generate_correction(correction_prompt)
            return self.extract_code(code), "<Critic>"
        else:
            return "", '<Next>'
    

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
        
        # 过滤掉None值，确保所有元素都是字符串
        finished_tasks_filtered = [task for task in summarized_history['finished_tasks'] if task is not None]
        
        if finished_tasks_filtered:
            finished_task = '\n'.join(finished_tasks_filtered)
            finished_task = f"Previous Finished Tasks: {finished_task}"
        else:
            finished_task = "Previous Finished Tasks: None"
        
        next_task = current_task.next().name if current_task.next() else "No more tasks"
        next_task = f"Next Task (for reference, you only need to complete the current task): {next_task}"
                
        current_task_text = f"Current Task: {current_task.name}"
        
        return main_goal, finished_task, current_task_text, next_task

    def get_api_details(self):
        """Format API details for the prompt."""
        return f'the pyautogui API imported\n{self.available_api_illustration}'

    def locate_gui_info(self, gui_info, main_goal, current_task):

        text_prompt =  f'''
By examinig the gui screenshot information, select all related coordinates which needed to complete the current task.
Parsed GUI Screenshot Info: [Note that: element format is "name [its position]", separate with comma]
{gui_info}

Information about Task:
{main_goal}
{current_task}

Note:
1) You have to go through all the element of give GUI screenshot info.
2) If there are multiple elements in GUI screenshot info, please output the coordinates for all of them.
3) If there repeat elements in GUI screenshot info, please output the coordinates for all of them.
4) The output format should be "name [its position]".

The output format should be:
```plaintext
...
```

# Remember to reason in comment if needed.
'''
        referred_gui = run_lmm(text_prompt, lmm=self.lmm, max_tokens=1000, temperature=0)

        referred_gui = self.extract_refer_gui(referred_gui)

        return referred_gui

    def construct_critic_prompt(self, 
        software_name, 
        current_task, 
        current_action, 
        gui_info,
        tips, 
        screenshot_path=None):
        """Construct the detailed prompt for the LMM based on provided parameters."""
        
        text_prompt =  f'''Based on the screenshots before and after the action, subtask description, software name {software_name}, please check the action completion status.
{current_task}
current action: {current_action}

Parsed GUI Screenshot Info: [Note that: element format is "name [its position]", separate with comma]
{gui_info}

The output format:
```plaintext
<Success> bool (Current task completion status) </Success>
<Reason> str (Analysis of possible mistakes if action is wrong) </Reason>
```

Software Usage Tips:
{tips}

Note:
1) Please carefully review the screenshot. Use it to verify whether the current task has been successfully completed.
2) Please based on the screenshot provided to give the potential reason of unsuccessful action.
3) When notice the pop-up window, that means the task is incomplete. Please do not output true for <Success> flag.
4) When determine the task completion status, please consider the current action which is already executed in the environment.

Please provide the reasoning steps.
'''
        
        if screenshot_path is not None:
            if len(screenshot_path) == 2:
                if screenshot_path[0] != "":
                    prompt = [text_prompt, screenshot_path[0], screenshot_path[1]]
                else:
                    prompt = [text_prompt, screenshot_path[1]]
            else:
                prompt = [text_prompt, screenshot_path[0]]
        else:
            prompt = [text_prompt]

        return prompt

    def generate_critic(self, prompt, lmm):
        """Run the LMM to generate code based on the prompt and post-process it."""

        critic_results = run_lmm(
            prompt,
            lmm=lmm,
            max_tokens=1000, 
            temperature=0
            )

        success_flag, reason = None, None

        success_flag = self.extract_patterntext(critic_results, 'Success')
        reason = self.extract_patterntext(critic_results, 'Reason')
        critic_comment = ""


        if success_flag == None or reason == None:
            return "false", "false", "", critic_comment

        if success_flag.lower() == 'false':
            critic_comment = reason
        
        return success_flag, reason, critic_comment

    def construct_correction_prompt(
        self,
        current_action,
        critic_comment,
        referred_gui,
        main_goal,
        current_task,
        tips,
        screenshot_path=None
    ):
        text_prompt =  f'''Please, based on the parsed GUI elements of the screenshot below, use pyautogui and the following API to generate execution code to control the mouse and keyboard. Additionally, provide a natural language suggestion to explain your reasoning and actions taken.
Currently, we execute the code: {current_action}, 
but we obtain this feedback: {critic_comment}.
You should based on above feedback regenerate the execution code.

Parsed GUI Screenshot Info: [Note that: element format is "name [its position]", separate with comma]

Based on the task description, we extarct the corresponding coordinates from GUI screenshot like that: {referred_gui}


Information about Task:
{main_goal}
{current_task}


General Rules:
1. Don't write an algorithm to search on the GUI data, directly fill the coordinates in the corresponding API.
2. MUST REMEMBER all the parameters in the function should be filled with the specific constant, not the variable.
3. IMPORTANT: Sometimes you need to do some reasoning or calculation for the position. You MUST do it in the comment of the code. 
4. Follow exactly the instructions in the task description. Don't redo tasks in finished_tasks.
5. For navigation-related tasks on a page or document, follow these steps to do the reasoning. Provide reasoning steps in the comments: 
    1) Check if the required information is displayed on the screenshot. MUST Answer this question in the comment of the code.
    2) If info is NOT found, use `press('pagedown')` one time to scroll down.

Software Usage Tips:
{tips}

==============================
Now, complete the code to achieve the command and generate a natural language suggestion of the reasoning steps. The explanation should be concise, logical, and easy to follow. Ensure that the explanation addresses why you did what you did, any necessary adjustments, and potential issues you considered.
Output format:
```output
<Code>
from pyautogui import click, write, hotkey, press, scroll, keyDown, keyUp, doubleClick
# Don't import any other libraries and functions
# Remember to reason in comment if needed.
# You must output 'from pyautogui import click, write, hotkey, press, scroll, keyDown, keyUp, doubleClick' if needed.
</Code> 
<Suggestion>
str(Explain how to fix the previous mistake based on the feedback. Provide a concise, step-by-step explanation of what needs to be done to execute correctly.)
</Suggestion> 
```
'''
        return [text_prompt, screenshot_path] if screenshot_path is not None else [text_prompt]

    def generate_correction(self, prompt):
        correction_results = run_lmm(
            prompt, 
            lmm=self.lmm,
            max_tokens=1000, 
            temperature=0
            )
        
        return correction_results

    def post_process_code(self, code):
        """Post-process the generated code to adapt to standards and replace API calls."""
        processed_code = []
        for line in code.split("\n"):
            if not line.strip().startswith("#"):
                for api in self.available_api.values():
                    if api.name in line:
                        line = line.replace(api.name, f"self.available_api['{api.name}']")
                        line = eval(line)  # Potential security risk, consider safer alternatives
            processed_code.append(line)
        return "\n".join(processed_code)

    @staticmethod
    def extract_patterntext(result, label):
        match = re.search(r'<%s>(.*?)</%s>'%(label, label), result, re.DOTALL)

        if match:
            extracted_text = match.group(1).strip()

            new_txt = ''
            for line in extracted_text.split("\n"):
                if not line.startswith('#'):
                    new_txt += line.strip()
            extracted_text = new_txt
        else:
            extracted_text = None
        return extracted_text

    @staticmethod
    def extract_code(input_string):
        # Regular expression to extract content starting from '<Code>' until the end if there are no closing backticks
        pattern = r'<Code>(.*?)</Code>'
        
        # Extract content
        matches = re.findall(pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
        
        # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
        return matches[0].strip() if matches else ""
    
    @staticmethod
    def extract_purecode(code):
        """Post-process the generated code to adapt to standards and replace API calls."""
        processed_code = []
        for line in code.split("\n"):
            if not line.strip().startswith("#"):
                processed_code.append(line)
        return "\n".join(processed_code)

    @staticmethod
    def extract_refer_gui(input_string):
        # Regular expression to extract content starting from '```plaintext' until the end if there are no closing backticks
        pattern = r'```plaintext(.*?)```'
        
        # Extract content
        matches = re.search(pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
        
        # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
        return matches.group(1).strip() if matches else input_string

    @staticmethod
    def check_resume(history):
        if history and len(history) > 0:
            last_history = history[-1]
            if 'code' in last_history and last_history['code'] and len(last_history['code']) > 0:
                history_code = "\n".join(last_history['code']) if last_history['code'][0] else "# finish"
                if "# finish" in history_code:
                    return False
                else:
                    return True
        return False

    def get_code_history_for_current_task(self, history):
        # keep previous four steps
        finished_tasks, code = "", ""
        if history:
            if self.check_resume(history):
                # select self.history from -5 index to -1 index, needs to check length
                finished_tasks = [x['task'] for x in history[-5:-1] if x.get('task') is not None]
                # 安全地获取code字段
                if len(history) > 0 and 'code' in history[-1] and history[-1]['code']:
                    code = "\n".join(history[-1]['code'])
                else:
                    code = ""
            else:
                finished_tasks = [x['task'] for x in history[-4:] if x.get('task') is not None]
        return {"finished_tasks": finished_tasks, "code": code}

    def get_last_screenshot(self, history):
        return history[-1]['screenshot_path'][-1], history[-1]['gui'][-1]
    
    def get_last_code(self, history):
        return history[-1]['code'][-1]
    
    def load_software_tips(self, resourcedir="resources\software_tips"):
        software_tips_files = glob.glob(os.path.join(os.path.dirname(__file__), resourcedir, "*.json"))
        # load files and merge them
        software_tips = {}
        for file in software_tips_files:
            with open(file, 'r', encoding='utf-8') as f:
                software_tips.update(json.load(f))
                
        return software_tips
        
    def get_software_tips(self, target, software_name):        
        hints = "\n".join(target.get(software_name, [""]))
        return hints
    
    # 新增：检测上下文变化方法
    def detect_context_changes(self, screenshot_path, current_task):
        """检测界面上下文变化，包括弹框、右键菜单等"""
        print(f"🔍 开始检测上下文变化...")
        print(f"   screenshot_path类型: {type(screenshot_path)}")
        print(f"   screenshot_path内容: {screenshot_path}")
        
        if not screenshot_path:
            print("   ❌ screenshot_path为空")
            return None
            
        # 处理不同的截图路径格式
        if isinstance(screenshot_path, list):
            if len(screenshot_path) < 2:
                print(f"   ❌ screenshot_path列表长度不足: {len(screenshot_path)}")
                return None
            before_path = screenshot_path[0]
            after_path = screenshot_path[1]
        elif isinstance(screenshot_path, str):
            print(f"   ⚠️  screenshot_path是字符串，无法进行对比检测")
            return None
        else:
            print(f"   ❌ screenshot_path格式不支持: {type(screenshot_path)}")
            return None
            
        print(f"   📸 操作前截图: {before_path}")
        print(f"   📸 操作后截图: {after_path}")
        
        # 检查文件是否存在
        if not os.path.exists(before_path):
            print(f"   ❌ 操作前截图不存在: {before_path}")
            return None
        if not os.path.exists(after_path):
            print(f"   ❌ 操作后截图不存在: {after_path}")
            return None
            
        try:
            # 使用上下文变化检测器分析截图
            context_changes = self.context_change_detector.detect_changes(
                before_path,  # 操作前截图
                after_path,   # 操作后截图
                current_task
            )
            
            print(f"   📊 检测结果: {context_changes}")
            return context_changes
            
        except Exception as e:
            print(f"   ❌ Context change detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 新增：判断是否需要重新思考
    def should_rethink_task(self, context_changes, current_task):
        """判断是否需要重新思考任务"""
        if not context_changes:
            return False
            
        # 检测到弹框且任务不期望弹框
        if context_changes.get('dialog_appeared') and not getattr(current_task, 'expects_dialog', False):
            return True
            
        # 检测到界面状态意外变化
        if context_changes.get('unexpected_interface_change'):
            return True
            
        return False

    # 新增：触发动态重新思考
    def trigger_dynamic_rethinking(self, context_changes, current_task, parsed_screenshot, screenshot_path, software_name, tips):
        """当检测到上下文变化时，触发动态重新思考"""
        print(f"🔄 检测到上下文变化，触发重新思考: {context_changes}")
        
        # 分析当前界面状态
        current_gui_state = self.analyze_current_interface(parsed_screenshot, screenshot_path)
        
        # 识别新的操作机会
        new_opportunities = self.identify_new_opportunities(current_gui_state, context_changes)
        
        # 重新生成任务计划
        revised_task_plan = self.regenerate_task_plan(
            original_task=current_task,
            new_context=current_gui_state,
            context_changes=context_changes,
            new_opportunities=new_opportunities
        )
        
        # 生成新的执行代码
        new_code = self.generate_adaptive_code(
            revised_task_plan=revised_task_plan,
            current_context=current_gui_state,
            software_name=software_name,
            tips=tips
        )
        
        return new_code, "<Rethink>"

    # 新增：分析当前界面状态
    def analyze_current_interface(self, parsed_screenshot, screenshot_path):
        """分析当前界面状态"""
        if parsed_screenshot:
            return self.compress_and_format_gui(parsed_screenshot)
        return ""

    # 新增：识别新的操作机会
    def identify_new_opportunities(self, current_gui_state, context_changes):
        """基于上下文变化识别新的操作机会"""
        opportunities = []
        
        if context_changes.get('dialog_appeared'):
            opportunities.append({
                'type': 'dialog_interaction',
                'description': '需要与弹框进行交互',
                'priority': 'high'
            })
            
        return opportunities

    # 新增：重新生成任务计划
    def regenerate_task_plan(self, original_task, new_context, context_changes, new_opportunities):
        """基于新上下文重新生成任务计划"""
        # 这里可以根据具体需求实现任务计划的重新生成
        # 目前返回一个简化的计划结构
        return {
            'original_task': original_task.name if hasattr(original_task, 'name') else str(original_task),
            'context_changes': context_changes,
            'new_opportunities': new_opportunities,
            'adapted_plan': f"基于上下文变化调整的任务计划: {context_changes}"
        }

    # 新增：生成自适应代码
    def generate_adaptive_code(self, revised_task_plan, current_context, software_name, tips):
        """生成适应新上下文的执行代码"""
        adaptive_prompt = f"""
基于检测到的上下文变化，需要重新生成执行代码。

当前上下文变化: {revised_task_plan['context_changes']}
新的操作机会: {revised_task_plan['new_opportunities']}
软件名称: {software_name}

请生成适应新上下文的执行代码，处理以下情况：
1. 如果检测到弹框，请生成与弹框交互的代码
2. 如果检测到界面状态意外变化，请生成相应的恢复代码

软件使用提示:
{tips}

请输出Python代码：
"""
        
        try:
            code = run_lmm(adaptive_prompt, lmm=self.lmm, max_tokens=1000, temperature=0)
            return self.extract_code(code)
        except Exception as e:
            print(f"Adaptive code generation error: {e}")
            return "# 重新思考后生成的代码\npass"
    
    

class ContextChangeDetector:
    """检测界面上下文变化的检测器"""
    
    def __init__(self):
        self.change_types = {
            'dialog_appearance': DialogChangeDetector(),
            'interface_state_change': InterfaceStateDetector()
        }
    
    def detect_changes(self, screenshot_before, screenshot_after, current_task):
        """检测两个截图之间的上下文变化"""
        changes = {}
        
        try:
            # 检测弹框出现
            if self.change_types['dialog_appearance'].detect(screenshot_before, screenshot_after):
                changes['dialog_appeared'] = True
                changes['dialog_type'] = self.change_types['dialog_appearance'].get_dialog_type()
            
            # 检测界面状态变化
            if self.change_types['interface_state_change'].detect(screenshot_before, screenshot_after):
                changes['unexpected_interface_change'] = True
                changes['change_description'] = self.change_types['interface_state_change'].get_change_description()
                
        except Exception as e:
            print(f"Context change detection error: {e}")
            changes['detection_error'] = str(e)
        
        return changes


class DialogChangeDetector:
    """弹框变化检测器 - 纯图像对比版本"""
    
    def __init__(self):
        # 松缓面积限制：降低最小面积，提高最大面积
        self.min_dialog_area = 500  # 最小弹框面积：从2000降低到500
        self.max_dialog_area = 200000  # 最大弹框面积：从100000提高到200000
        # 松缓位置限制：扩大中央区域范围
        self.central_region_margin = 0.05  # 中央区域边距：从0.15降低到0.05，扩大检测范围
    
    def detect(self, screenshot_before, screenshot_after):
        """检测弹框是否出现"""
        print(f"      🔍 弹框检测器开始工作...")
        print(f"        操作前截图: {screenshot_before}")
        print(f"        操作后截图: {screenshot_after}")
        

        
        try:
            # 如果只有一个截图，无法检测变化
            if not screenshot_before or not screenshot_after:
                print("         ❌ 截图路径为空")
                return False
            
            # 使用图像对比检测弹框
            result = self._detect_from_comparison(screenshot_before, screenshot_after)
            print(f"         📊 弹框检测结果: {result}")
            return result
            
        except Exception as e:
            print(f"         ❌ 弹框检测出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_from_comparison(self, screenshot_before, screenshot_after):
        """通过比较两个截图检测弹框"""
        print(f"         🔍 开始图像对比检测...")
        
        try:
            # 读取两个图像
            img1 = cv2.imread(screenshot_before)
            img2 = cv2.imread(screenshot_after)
            
            if img1 is None or img2 is None:
                print(f"            ❌ 无法读取图像文件")
                return False
            
            print(f"            📐 图像1尺寸: {img1.shape}")
            print(f"            📐 图像2尺寸: {img2.shape}")
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 确保两个图像尺寸一致
            if img1.shape != img2.shape:
                print(f"            🔄 图像尺寸不一致，正在调整...")
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                print(f"            ✅ 图像尺寸已调整")
            
            # 计算图像差异
            diff = cv2.absdiff(gray1, gray2)
            
            # 应用阈值，突出变化区域 - 降低阈值，更容易检测到变化
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # 从25降低到20
            
            # 形态学操作，连接分散的变化区域
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 找到变化区域的轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"            🔍 找到 {len(contours)} 个变化区域")
            
            # 检查是否有符合弹框特征的变化区域
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                print(f"            📊 区域 {i+1}: 面积 = {area}")
                
                # 面积过滤：弹框通常有合适的大小
                if area < self.min_dialog_area or area > self.max_dialog_area:
                    print(f"               ❌ 面积不符合要求 ({self.min_dialog_area} < {area} < {self.max_dialog_area})")
                    continue
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                height, width = img2.shape[:2]
                
                print(f"               📐 位置: ({x}, {y}), 尺寸: {w} x {h}")
                print(f"               📐 屏幕尺寸: {width} x {height}")
                
                # 位置过滤：弹框通常在屏幕中央，不在边缘 - 松缓位置限制
                margin = self.central_region_margin
                margin_pixels_x = int(width * margin)
                margin_pixels_y = int(height * margin)
                
                if (x > margin_pixels_x and y > margin_pixels_y and 
                    x + w < width - margin_pixels_x and y + h < height - margin_pixels_y):
                    
                    print(f"               ✅ 位置符合中央区域要求")
                    
                    # 形状过滤：弹框通常是矩形 - 松缓宽高比限制
                    aspect_ratio = w / h
                    if 0.3 <= aspect_ratio <= 5.0:  # 宽高比：从0.5-3.0放宽到0.3-5.0
                        print(f"               ✅ 宽高比符合要求: {aspect_ratio:.2f}")
                        
                        # 检查轮廓的矩形度 - 松缓矩形度限制
                        rect_area = w * h
                        contour_area = area
                        rectangularity = contour_area / rect_area
                        
                        if rectangularity > 0.5:  # 矩形度：从0.7降低到0.5
                            print(f"               ✅ 矩形度符合要求: {rectangularity:.2f}")
                            print(f"               🎉 检测到弹框！")
                            return True
                        else:
                            print(f"               ❌ 矩形度不符合要求: {rectangularity:.2f}")
                    else:
                        print(f"               ❌ 宽高比不符合要求: {aspect_ratio:.2f}")
                else:
                    print(f"               ❌ 位置不在中央区域")
            
            print(f"            ❌ 未检测到符合要求的弹框")
            return False
            
        except Exception as e:
            print(f"            ❌ 对比检测弹框出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_dialog_type(self):
        """获取弹框类型 - 基于图像特征推测"""
        return "detected_dialog"


class InterfaceStateDetector:
    """界面状态变化检测器"""
    
    def __init__(self):
        self.change_threshold = 0.1  # 变化阈值
    
    def detect(self, screenshot_before, screenshot_after):
        """检测界面状态是否发生意外变化"""

        
        try:
            # 如果只有一个截图，无法检测变化
            if not screenshot_before or not screenshot_after:
                return False
            
            # 读取两个图像
            img1 = cv2.imread(screenshot_before)
            img2 = cv2.imread(screenshot_after)
            
            if img1 is None or img2 is None:
                return False
            
            # 确保两个图像尺寸一致
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 使用SSIM计算结构相似性（如果可用）
            if ssim is not None:
                ssim_score = ssim(gray1, gray2)
                
                # 如果相似度低于阈值，说明发生了显著变化
                if ssim_score < (1 - self.change_threshold):
                    return True
            
            # 也可以使用简单的像素差异检测
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            
            # 如果平均差异超过阈值，也认为发生了变化
            if mean_diff > 20:  # 像素值差异阈值
                return True
            
            return False
            
        except Exception as e:
            print(f"界面状态检测出错: {e}")
            return False
    
    def get_change_description(self):
        """获取变化描述"""
        return "interface_state_changed"
    
    
