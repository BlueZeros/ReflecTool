import os
import re
import json

def suggestion_parse(feedback: str):
    match = re.search(r'\{.*\}', feedback, re.DOTALL)

    try:
        if match:
            json_string = match.group(0)
            action_feedback = json.loads(json_string)
            return action_feedback
        else:
            return None
    except:
        return None

def updated_suggestion_parse(feedback: str, action_template):
    match = re.search(r'\[.*\]', feedback, re.DOTALL)
    
    try:
        if match:
            json_string = match.group(0)
            action_feedback = json.loads(json_string)
            return action_feedback
        else:
            return None
    except:
        return None

def find_max_step(folder_path):
    max_step = -1
    pattern = re.compile(r'step-(\d+)\.json')

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num

    return max_step if max_step != -1 else None