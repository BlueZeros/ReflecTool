import glob
import json
import re
import os

from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions.BaseAction import AgentAction

MEMORY_TASK_KEY = "task"
MEMORY_ACT_OBS_KEY= "act_obs"
MEMORY_PREV_ACT_OBS_KEY = "prev_act_obs"

def load_memory_list_format(folder_path):
    # 使用glob获取所有符合模式的文件路径
    file_paths = glob.glob(os.path.join(folder_path, 'example_*.json'))
    
    # 存储所有JSON内容的列表
    memory_bank = []
    
    # 读取每个文件并将内容存储在列表中
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)

            task = TaskPackage(**content[MEMORY_TASK_KEY])
            action_chain = []
            for history in content[MEMORY_ACT_OBS_KEY]:
                action_name, params, PARSE_FLAG = parse_action(history[0])
                agent_act = AgentAction(action_name=action_name, params=params)
                observation = history[1]

                action_chain.append((agent_act, observation))
    
            memory_bank.append({MEMORY_TASK_KEY: task, MEMORY_ACT_OBS_KEY: action_chain})
    
    return memory_bank

def load_memory_as_dict(folder_path):
    # 使用glob获取所有符合模式的文件路径
    file_paths = glob.glob(os.path.join(folder_path, 'example_*.json'))
    
    # 存储所有JSON内容的字典
    memory_bank = {}
    
    # 读取每个文件并将内容存储在列表中
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            memory_bank[content["id"]] = content
    
    return memory_bank

def format_memory(memory, memory_wo_reflect=False):
    task = TaskPackage(**memory[MEMORY_TASK_KEY])
    action_chain = []
    if not memory_wo_reflect:
        memory_act_chain = memory[MEMORY_ACT_OBS_KEY]
    else:
        memory_act_chain = memory[MEMORY_TASK_KEY][MEMORY_PREV_ACT_OBS_KEY][-1][:-1]

    for history in memory_act_chain:
        action_name, params, PARSE_FLAG = parse_action(history[0])
        agent_act = AgentAction(action_name=action_name, params=params)
        observation = history[1]
        action_chain.append((agent_act, observation))
    
    return (task, action_chain)

def parse_action(string: str) -> tuple[str, dict, bool]:
    """
    Parse an action string into an action type and an argument.
    """

    string = string.strip("Action:").strip(" ").strip(".")
    string = string.split("Action")[0].strip("\n ")
    pattern = r"(\w+)\s*(\{.+?\}|\[\{.+?\}\])"
    match = re.match(pattern, string, re.DOTALL)
    PARSE_FLAG = True

    if match:
        action_type = match.group(1).strip().replace("\n", "")
        arguments = match.group(2).strip().strip("[]").replace("\n", "")
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = arguments.strip("{}")
            key, value = arguments.split(": ", 1)
            key = key.strip("\"\'")
            value = value.strip("\"\'")
            arguments = {key: value}
            # return string, {}, PARSE_FLAG
            # return action_type, arguments, PARSE_FLAG
        return action_type, arguments, PARSE_FLAG
    else:
        PARSE_FLAG = False
        return string, {}, PARSE_FLAG