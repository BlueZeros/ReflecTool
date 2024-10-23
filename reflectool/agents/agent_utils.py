"""functions or objects shared by agents"""

import re
import json

from reflectool.actions.BaseAction import BaseAction


def name_checking(name: str):
    """ensure no white space in name"""
    white_space = [" ", "\n", "\t"]
    for w in white_space:
        if w in name:
            return False
    return True


def act_match(input_act_name: str, act: BaseAction):
    if input_act_name == act.action_name:  # exact match
        return True
    ## To-Do More fuzzy match
    return False

def fix_quotes(s):
    # 匹配不带引号的键和值
    s = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', s)  # 修复键
    s = re.sub(r':\s*([^",}\]]+?)\s*?([}])', r': "\1"\2', s)  # 修复值
    return s


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
            try:
                arguments = arguments.strip("{}")
                key, value = arguments.split(": ", 1)
                key = key.strip("\"\'")
                value = value.strip("\"\'")
                arguments = {key: value}
            except:
                PARSE_FLAG = False
                return string, {}, PARSE_FLAG
                
        return action_type, arguments, PARSE_FLAG
    else:
        PARSE_FLAG = False
        return string, {}, PARSE_FLAG


AGENT_CALL_ARG_KEY = "Task"
NO_TEAM_MEMEBER_MESS = (
    """No team member for manager agent. Please check your manager agent team."""
)
ACION_NOT_FOUND_MESS = (
    """"This is the wrong action to call. Please check your available action list and make sure your outputs are in the formate of ActionName[{\"param_name\": \"param_input\"}]"""
)