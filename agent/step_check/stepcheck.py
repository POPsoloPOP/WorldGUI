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
import time

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

        # New: Dynamic monitoring during execution
        if screenshot_path and parsed_screenshot:
            monitoring_result = self.execution_monitor.monitor_execution_context(
                current_step=current_task_text,
                screenshot_path=screenshot_path,
                parsed_screenshot=parsed_screenshot,
                software_name=software_name
            )
            
            # If rethinking is needed
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
                # Check if really completed or needs rethinking
                if self._should_trigger_rethinking(critic_feedback, current_task_text, parsed_screenshot):
                    stepcheck_decision = '<Rethink>'
                    print("Detected situation requiring rethinking, triggering rethinking workflow...")
                else:
                    stepcheck_decision = '<Finished>'
            elif '#Cannot confirm' in critic_feedback and iter_idx < 2:
                
                if parsed_screenshot:
                    # Trigger rethinking workflow instead of direct relocation
                    # Because LLM has clearly stated it cannot confirm, reanalysis is needed
                    
                    # Build monitoring result
                    monitoring_result = {
                        'reason': 'LLM cannot confirm current task state, need to re-analyze interface',
                        'detected_issues': ['Screenshot information insufficient', 'Need to re-parse GUI'],
                        'suggested_actions': ['Re-parse new screenshot', 'Re-evaluate task state']
                    }
                    
                    # Trigger rethinking
                    stepcheck_decision, task_adjustment, _ = self.trigger_rethinking_workflow(
                        monitoring_result=monitoring_result,
                        current_task=current_task,
                        parsed_screenshot=parsed_screenshot,
                        screenshot_path=new_screenshot_path,
                        software_name=software_name,
                        tips=tips
                    )
                    
                    # If rethinking successful, update task
                    if stepcheck_decision == "<Rethink>" and task_adjustment:
                        # task_adjustment is actually the updated current_task object
                        # The task object has already been updated in trigger_rethinking_workflow
                        current_task = task_adjustment
                        print(f"✅ Task updated from rethinking: {current_task.name if hasattr(current_task, 'name') else str(current_task)}")
                    
                    iter_idx += 1
                else:
                    stepcheck_decision = '<Continue>'
            else:
                stepcheck_decision = '<Continue>'

        return stepcheck_decision, current_task, history

    # New: Trigger rethinking workflow
    def trigger_rethinking_workflow(self, monitoring_result, current_task, parsed_screenshot, screenshot_path, software_name, tips):
        """Trigger rethinking workflow"""
        print(f"🔄 Step-Check detected rethinking needed: {monitoring_result['reason']}")
        
        # Analyze current interface state
        current_gui_state = self.analyze_current_interface(parsed_screenshot, screenshot_path)
        
        # Execute real rethinking - call LLM for reanalysis
        rethinking_result = self._execute_rethinking_analysis(
            current_task=current_task,
            monitoring_result=monitoring_result,
            current_gui_state=current_gui_state,
            software_name=software_name,
            tips=tips
        )
        
        # Generate task adjustment suggestions after rethinking
        task_adjustment = self.generate_task_adjustment(
            original_task=current_task,
            monitoring_result=monitoring_result,
            current_gui_state=current_gui_state,
            rethinking_result=rethinking_result
        )
        
        # Key fix: Ensure task object is properly updated
        if hasattr(current_task, 'name'):
            # Update task name
            current_task.name = task_adjustment.get('new_task_name', current_task.name)
            # Update task description
            if hasattr(current_task, 'description'):
                current_task.description = task_adjustment.get('new_task_description', current_task.description)
            # Mark task as requiring rethinking
            current_task.requires_rethinking = True
            current_task.rethinking_result = rethinking_result
            
            print(f"✅ Task updated: {current_task.name}")
            print(f"✅ Rethinking result saved")
        
        print(f"✅ Rethinking completed, new execution strategy generated")
        return "<Rethink>", current_task, []  # Return updated task object

    def _execute_rethinking_analysis(self, current_task, monitoring_result, current_gui_state, software_name, tips):
        """Execute rethinking analysis"""
        print(f"🧠 Starting rethinking analysis...")
        
        # Build rethinking prompt
        prompt = self._construct_rethinking_prompt(
            current_task=current_task,
            monitoring_result=monitoring_result,
            current_gui_state=current_gui_state,
            software_name=software_name,
            tips=tips
        )
        
        try:
            # Call LLM for rethinking
            from agent.utils.lmm.run_lmm import run_lmm
            
            rethinking_response = run_lmm(
                prompt,
                lmm=self.lmm,
                max_tokens=800,
                temperature=0.1
            )
            
            print(f"✅ LLM rethinking completed")
            return rethinking_response
            
        except Exception as e:
            print(f"❌ LLM rethinking failed: {e}")
            return "Rethinking analysis failed, manual intervention required"

    def _construct_rethinking_prompt(self, current_task, monitoring_result, current_gui_state, software_name, tips):
        """构建重新思考的提示词"""
        task_name = current_task.name if hasattr(current_task, 'name') else str(current_task)
        
        prompt = f"""You are a smart GUI automation task re-planning expert. The current task execution encountered a problem, and you need to re-think and formulate a new execution strategy.

## Current Situation Analysis
**Original Task**: {task_name}
**Problem Reason**: {monitoring_result['reason']}
**Detected Issues**: {', '.join(monitoring_result.get('detected_issues', []))}
**Suggested Actions**: {', '.join(monitoring_result.get('suggested_actions', []))}

## Current Interface State
{current_gui_state}

## Software Information
**Software Name**: {software_name}
**Software Usage Tips**: {tips}

## Rethinking Requirements
Please re-analyze the task execution strategy based on the current interface state and the detected issues:

1. **Problem Diagnosis**: Analyze why the original task cannot be completed
2. **Interface Analysis**: Identify key elements and states in the current interface
3. **Strategy Adjustment**: Formulate new execution steps
4. **Risk Mitigation**: Avoid issues encountered previously

## Output Format
Please output your re-thinking results in the following format:

<Analysis>
Problem Diagnosis and Interface Analysis
</Analysis>

<NewStrategy>
New Execution Strategy and Steps
</NewStrategy>

<RiskMitigation>
Risk Mitigation Measures
</RiskMitigation>

<Confidence>
Confidence in executing the new strategy (1-10)
</Confidence>

Please provide detailed analysis and specific execution suggestions."""

        return prompt

    # New: Analyze current interface state
    def analyze_current_interface(self, parsed_screenshot, screenshot_path):
        """Analyze current interface state"""
        if parsed_screenshot:
            return self.compress_and_format_gui(parsed_screenshot)
        return ""

    # New: Generate task adjustment suggestions
    def generate_task_adjustment(self, original_task, monitoring_result, current_gui_state, rethinking_result=None):
        """Generate task adjustment suggestions based on monitoring results and rethinking results"""
        # Analyze current interface state to generate new task description
        new_task_name = self._generate_new_task_name(original_task, monitoring_result, current_gui_state)
        new_task_description = self._generate_new_task_description(original_task, monitoring_result, current_gui_state, rethinking_result)
        
        adjustment = {
            'original_task': original_task.name if hasattr(original_task, 'name') else str(original_task),
            'rethinking_reason': monitoring_result['reason'],
            'detected_issues': monitoring_result.get('detected_issues', []),
            'suggested_actions': monitoring_result.get('suggested_actions', []),
            'current_context': current_gui_state,
            'rethinking_result': rethinking_result,
            'requires_rethinking': True,
            'timestamp': time.time(),
            # New: Actual task update content
            'new_task_name': new_task_name,
            'new_task_description': new_task_description
        }
        
        return adjustment
    
    def _generate_new_task_name(self, original_task, monitoring_result, current_gui_state):
        """Generate new task name"""
        original_name = original_task.name if hasattr(original_task, 'name') else str(original_task)
        
        # Generate new task name based on monitoring results
        if "弹框" in monitoring_result.get('reason', '') or "dialog" in monitoring_result.get('reason', '').lower():
            if "Remove" in original_name or "删除" in original_name:
                return "Handle delete confirmation dialog and complete layer deletion operation"
            else:
                return "Handle confirmation dialog and continue task execution"
        
        if "界面状态异常" in monitoring_result.get('reason', '') or "interface anomaly" in monitoring_result.get('reason', '').lower():
            return "Re-analyze interface state and adjust execution strategy"
        
        # Default to original task name
        return original_name
    
    def _generate_new_task_description(self, original_task, monitoring_result, current_gui_state, rethinking_result):
        """Generate new task description"""
        original_desc = original_task.name if hasattr(original_task, 'name') else str(original_task)
        
        # Generate new description based on rethinking result
        if rethinking_result and isinstance(rethinking_result, str):
            # Try to extract new strategy from rethinking result
            if "<NewStrategy>" in rethinking_result:
                import re
                match = re.search(r'<NewStrategy>(.*?)</NewStrategy>', rethinking_result, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # Generate description based on monitoring results
        if "弹框" in monitoring_result.get('reason', '') or "dialog" in monitoring_result.get('reason', '').lower():
            return f"Current task encountered dialog, need to handle dialog interaction first: {original_desc}"
        
        if "界面状态异常" in monitoring_result.get('reason', '') or "interface anomaly" in monitoring_result.get('reason', '').lower():
            return f"Interface state anomaly, need to re-evaluate: {original_desc}"
        
        return original_desc

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

    def _should_trigger_rethinking(self, critic_feedback, current_task_text, parsed_screenshot):
        """
        Determine if rethinking should be triggered - 判断是否应该触发重新思考
        This method analyzes the critic feedback and current task to decide if rethinking is needed - 此方法分析评论反馈和当前任务来决定是否需要重新思考
        """
        # Check if task description contains elements requiring confirmation - 检查任务描述是否包含需要确认的元素
        confirmation_keywords = ['click', 'select', 'remove', 'confirm', 'dialog', 'button']
        has_confirmation_needs = any(keyword in current_task_text.lower() for keyword in confirmation_keywords)
        
        # Check if screenshot contains confirmation dialog - 检查截图是否包含确认对话框
        has_confirmation_dialog = self._detect_confirmation_dialog(parsed_screenshot)
        
        # If task needs confirmation but screenshot shows dialog, trigger rethinking - 如果任务需要确认但截图显示对话框，触发重新思考
        if has_confirmation_needs and has_confirmation_dialog:
            print(f"Task '{current_task_text}' needs confirmation and confirmation dialog detected - 任务'{current_task_text}'需要确认且检测到确认对话框")
            return True
        
        # Check for incomplete action indicators - 检查不完整操作的指示器
        incomplete_indicators = ['waiting', 'incomplete', 'response', 'confirm', 'dialog']
        if any(indicator in critic_feedback.lower() for indicator in incomplete_indicators):
            print(f"Critic feedback indicates incomplete action - 评论反馈表明操作不完整")
            return True
        
        return False

    def _detect_confirmation_dialog(self, parsed_screenshot):
        """
        Detect if confirmation dialog is present - 检测是否存在确认对话框
        """
        if not parsed_screenshot:
            return False
        
        # Method 1: Check confirmation-related UI element text
        dialog_indicators = ['OK', 'Cancel', 'Yes', 'No', 'Confirm', 'Remove', 'Dialog', '确定', '取消', '是', '否', '确认', '删除']
        gui_text = str(parsed_screenshot).lower()
        has_dialog_text = any(indicator.lower() in gui_text for indicator in dialog_indicators)
        
        if has_dialog_text:
            print(f"✅ Dialog detected through text: {[indicator for indicator in dialog_indicators if indicator.lower() in gui_text]}")
            return True
        
        # Method 2: Check dialog features in GUI structure
        if hasattr(parsed_screenshot, 'get') and isinstance(parsed_screenshot, dict):
            # Check if there are dialog-related panel names
            panel_names = []
            if 'panel' in parsed_screenshot:
                for panel in parsed_screenshot['panel']:
                    if isinstance(panel, dict) and 'name' in panel:
                        panel_names.append(panel['name'].lower())
            
            dialog_panel_indicators = ['dialog', 'popup', 'modal', 'confirm', 'remove', 'delete', 'warning', 'error']
            has_dialog_panel = any(indicator in name for name in panel_names for indicator in dialog_panel_indicators)
            
            if has_dialog_panel:
                print(f"✅ Dialog detected through panel name: {[name for name in panel_names if any(indicator in name for indicator in dialog_panel_indicators)]}")
                return True
        
        # Method 3: Check for dialog-specific UI element combinations
        # For example: simultaneous presence of "Confirm" and "Cancel" buttons
        confirm_cancel_indicators = ['确定', '取消', 'ok', 'cancel', 'yes', 'no']
        confirm_count = sum(1 for indicator in confirm_cancel_indicators if indicator.lower() in gui_text)
        
        if confirm_count >= 2:
            print(f"✅ Dialog detected through button combination: Found {confirm_count} confirm/cancel buttons")
            return True
        
        # Method 4: Check for dialog-specific rectangular area features
        # Dialogs usually have specific size and position features
        if hasattr(parsed_screenshot, 'get') and isinstance(parsed_screenshot, dict):
            if 'panel' in parsed_screenshot:
                for panel in parsed_screenshot['panel']:
                    if isinstance(panel, dict) and 'rectangle' in panel:
                        rect = panel['rectangle']
                        if len(rect) == 4:
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            area = width * height
                            
                            # Dialogs usually have moderate size (not too large or small)
                            if 1000 <= area <= 50000:  # Relaxed area restrictions
                                # Dialogs usually have reasonable aspect ratio
                                aspect_ratio = width / height if height > 0 else 0
                                if 0.5 <= aspect_ratio <= 4.0:  # Relaxed aspect ratio restrictions
                                    print(f"✅ Dialog detected through geometric features: Area={area}, Aspect ratio={aspect_ratio:.2f}")
                                    return True
        
        print(f"❌ No dialog detected")
        return False
    
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
        current_task_text,
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
{current_task_text}
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
        
        # 过滤掉None值，确保所有元素都是字符串
        finished_tasks_filtered = [task for task in summarized_history['finished_tasks'] if task is not None]
        
        if finished_tasks_filtered:
            finished_task = '\n'.join(finished_tasks_filtered)
            finished_task = f"Previous Finished Tasks: {finished_task}"
        else:
            finished_task = "Previous Finished Tasks: None"
        
        next_task = current_task.next().name if current_task.next() else "No more tasks"
        next_task = f"Next Task (for reference, you should consider whether current task is necessary when we complete next task ): {next_task}"
                
        current_task_text = f"Current Task: {current_task.name}"
        
        return main_goal, finished_task, current_task_text, next_task

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
    """Execution process monitor for detecting situations requiring rethinking"""
    
    def __init__(self):
        self.monitoring_rules = {
            'qgis': QGISMonitoringRules(),
            'word': WordMonitoringRules(),
            'powerpoint': PowerPointMonitoringRules(),
            'excel': ExcelMonitoringRules(),
            'default': DefaultMonitoringRules()
        }
    
    def monitor_execution_context(self, current_step, screenshot_path, parsed_screenshot, software_name):
        """Monitor context changes during execution process"""
        try:
            # Get monitoring rules for corresponding software
            monitoring_rules = self.monitoring_rules.get(software_name.lower(), self.monitoring_rules['default'])
            
            # Execute monitoring checks
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
                'reason': f"Monitoring process error: {e}",
                'detected_issues': [],
                'suggested_actions': []
            }


class QGISMonitoringRules:
    """QGIS software monitoring rules"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """Check QGIS execution context"""
        issues = []
        suggested_actions = []
        
        # Check if there are dialogs appearing
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("Dialog detected")
            suggested_actions.append("Need to interact with dialog")
        
        # Check if interface state is abnormal
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("Interface state anomaly detected")
            suggested_actions.append("Need to re-analyze interface state")
        
        # Determine if rethinking is needed
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "Execution normal"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """Detect dialog appearance - Pure image comparison version"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Warning: Missing OpenCV library, dialog detection functionality limited")
            return False
        
        try:
            # Read image
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # Dialog detection parameters
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # Use edge detection to find possible dialog contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for contours that match dialog characteristics
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # Position filtering: dialogs are usually in screen center, not at edges
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # Shape filtering: dialogs are usually rectangular
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # Check contour rectangularity
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"QGIS dialog detection error: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """Detect interface state anomalies"""
        try:
            if not parsed_screenshot:
                return False
            
            # Check for abnormal element states
            anomalies = []
            
            # Check for error messages
            if isinstance(parsed_screenshot, dict):
                # Check if text elements contain error information
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"Error message detected: {text_elem['text']}")
                
                # Check for abnormal button states
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"Button state abnormal: {button.get('text', 'Unknown')} - {button['state']}")
                
                # Check for loading states
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("Loading state detected")
            
            # If there are anomalies, return True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"QGIS interface state anomaly detection error: {e}")
            return False


class WordMonitoringRules:
    """Word software monitoring rules"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """Check Word execution context"""
        issues = []
        suggested_actions = []
        
        # Check if there are dialogs appearing
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("Dialog detected")
            suggested_actions.append("Need to interact with dialog")
        
        # Check if interface state is abnormal
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("Interface state anomaly detected")
            suggested_actions.append("Need to re-analyze interface state")
        
        # Determine if rethinking is needed
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "Execution normal"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """Detect dialog appearance - Pure image comparison version"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Warning: Missing OpenCV library, dialog detection functionality limited")
            return False
        
        try:
            # Read image
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # Dialog detection parameters
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # Use edge detection to find possible dialog contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for contours that match dialog characteristics
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # Position filtering: dialogs are usually in screen center, not at edges
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # Shape filtering: dialogs are usually rectangular
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # Check contour rectangularity
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Word dialog detection error: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """Detect interface state anomalies"""
        try:
            if not parsed_screenshot:
                return False
            
            # Check for abnormal element states
            anomalies = []
            
            # Check for error messages
            if isinstance(parsed_screenshot, dict):
                # Check if text elements contain error information
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"Error message detected: {text_elem['text']}")
                
                # Check for abnormal button states
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"Button state abnormal: {button.get('text', 'Unknown')} - {button['state']}")
                
                # Check for loading states
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("Loading state detected")
            
            # If there are anomalies, return True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"Word interface state anomaly detection error: {e}")
            return False


class PowerPointMonitoringRules:
    """PowerPoint software monitoring rules"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """Check PowerPoint execution context"""
        issues = []
        suggested_actions = []
        
        # Check if there are dialogs appearing
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("Dialog detected")
            suggested_actions.append("Need to interact with dialog")
        
        # Check if interface state is abnormal
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("Interface state anomaly detected")
            suggested_actions.append("Need to re-analyze interface state")
        
        # Determine if rethinking is needed
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "Execution normal"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """Detect dialog appearance - Pure image comparison version"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Warning: Missing OpenCV library, dialog detection functionality limited")
            return False
        
        try:
            # Read image
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # Dialog detection parameters
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # Use edge detection to find possible dialog contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for contours that match dialog characteristics
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # Position filtering: dialogs are usually in screen center, not at edges
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # Shape filtering: dialogs are usually rectangular
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # Check contour rectangularity
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"PowerPoint dialog detection error: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """Detect interface state anomalies"""
        try:
            if not parsed_screenshot:
                return False
            
            # Check for abnormal element states
            anomalies = []
            
            # Check for error messages
            if isinstance(parsed_screenshot, dict):
                # Check if text elements contain error information
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"Error message detected: {text_elem['text']}")
                
                # Check for abnormal button states
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"Button state abnormal: {button.get('text', 'Unknown')} - {button['state']}")
                
                # Check for loading states
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("Loading state detected")
            
            # If there are anomalies, return True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"PowerPoint interface state anomaly detection error: {e}")
            return False


class ExcelMonitoringRules:
    """Excel software monitoring rules"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """Check Excel execution context"""
        issues = []
        suggested_actions = []
        
        # Check if there are dialogs appearing
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("Dialog detected")
            suggested_actions.append("Need to interact with dialog")
        
        # Check if interface state is abnormal
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("Interface state anomaly detected")
            suggested_actions.append("Need to re-analyze interface state")
        
        # Determine if rethinking is needed
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "Execution normal"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """Detect dialog appearance - Pure image comparison version"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Warning: Missing OpenCV library, dialog detection functionality limited")
            return False
        
        try:
            # Read image
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # Dialog detection parameters
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # Use edge detection to find possible dialog contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for contours that match dialog characteristics
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # Position filtering: dialogs are usually in screen center, not at edges
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # Shape filtering: dialogs are usually rectangular
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # Check contour rectangularity
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Excel dialog detection error: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """Detect interface state anomalies"""
        try:
            if not parsed_screenshot:
                return False
            
            # Check for abnormal element states
            anomalies = []
            
            # Check for error messages
            if isinstance(parsed_screenshot, dict):
                # Check if text elements contain error information
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"Error message detected: {text_elem['text']}")
                
                # Check for abnormal button states
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"Button state abnormal: {button.get('text', 'Unknown')} - {button['state']}")
                
                # Check for loading states
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("Loading state detected")
            
            # If there are anomalies, return True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"Excel interface state anomaly detection error: {e}")
            return False


class DefaultMonitoringRules:
    """Default monitoring rules"""
    
    def check_execution_context(self, current_step, screenshot_path, parsed_screenshot):
        """Check default execution context"""
        issues = []
        suggested_actions = []
        
        # Check if there are dialogs appearing
        if self.detect_dialog_appearance(screenshot_path):
            issues.append("Dialog detected")
            suggested_actions.append("Need to interact with dialog")
        
        # Check if interface state is abnormal
        if self.detect_interface_anomaly(parsed_screenshot):
            issues.append("Interface state anomaly detected")
            suggested_actions.append("Need to re-analyze interface state")
        
        # Determine if rethinking is needed
        rethinking_needed = len(issues) > 0
        reason = "; ".join(issues) if issues else "Execution normal"
        
        return {
            'rethinking_needed': rethinking_needed,
            'reason': reason,
            'detected_issues': issues,
            'suggested_actions': suggested_actions
        }
    
    def detect_dialog_appearance(self, screenshot_path):
        """Detect dialog appearance - Pure image comparison version"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("Warning: Missing OpenCV library, dialog detection functionality limited")
            return False
        
        try:
            # Read image
            image = cv2.imread(screenshot_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
            
            # Dialog detection parameters
            min_dialog_area = 2000
            max_dialog_area = 100000
            central_margin = 0.15
            
            # Use edge detection to find possible dialog contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for contours that match dialog characteristics
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                if area < min_dialog_area or area > max_dialog_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                height, width = image.shape[:2]
                
                # Position filtering: dialogs are usually in screen center, not at edges
                if (x > width * central_margin and y > height * central_margin and 
                    x + w < width * (1 - central_margin) and y + h < height * (1 - central_margin)):
                    
                    # Shape filtering: dialogs are usually rectangular
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 3.0:
                        
                        # Check contour rectangularity
                        rect_area = w * h
                        contour_area = area
                        if contour_area / rect_area > 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Default dialog detection error: {e}")
            return False
    
    def detect_interface_anomaly(self, parsed_screenshot):
        """Detect interface state anomalies"""
        try:
            if not parsed_screenshot:
                return False
            
            # Check for abnormal element states
            anomalies = []
            
            # Check for error messages
            if isinstance(parsed_screenshot, dict):
                # Check if text elements contain error information
                if 'text_elements' in parsed_screenshot:
                    for text_elem in parsed_screenshot['text_elements']:
                        if isinstance(text_elem, dict) and 'text' in text_elem:
                            text = text_elem['text'].lower()
                            error_keywords = ['error', '错误', '失败', 'fail', '异常', 'exception']
                            if any(keyword in text for keyword in error_keywords):
                                anomalies.append(f"Error message detected: {text_elem['text']}")
                
                # Check for abnormal button states
                if 'buttons' in parsed_screenshot:
                    for button in parsed_screenshot['buttons']:
                        if isinstance(button, dict) and 'state' in button:
                            if button['state'] in ['disabled', 'error', 'warning']:
                                anomalies.append(f"Button state abnormal: {button.get('text', 'Unknown')} - {button['state']}")
                
                # Check for loading states
                if 'loading_indicators' in parsed_screenshot:
                    if parsed_screenshot['loading_indicators']:
                        anomalies.append("Loading state detected")
            
            # If there are anomalies, return True
            return len(anomalies) > 0
            
        except Exception as e:
            print(f"Default interface state anomaly detection error: {e}")
            return False
    