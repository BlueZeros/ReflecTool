import os
import json
from reflectool.commons import TaskPackage
from reflectool.logger.base import BaseAgentLogger
from reflectool.actions.BaseAction import AgentAction, BaseAction
from reflectool.logger.logger_utils import bcolors, str_color_remove
from reflectool.memory.memory_utils import MEMORY_TASK_KEY, MEMORY_ACT_OBS_KEY, MEMORY_PREV_ACT_OBS_KEY
from reflectool.agent_prompts.prompt_utils import format_act_params_example, action_format

class AgentLogger(BaseAgentLogger):
    def __init__(
        self,
        log_file_name: str = "agent.log",
        FLAG_PRINT: bool = True,
        OBS_OFFSET: int = 99999,
        PROMPT_DEBUG_FLAG: bool = False,
    ) -> None:
        super().__init__(log_file_name=log_file_name)
        self.FLAG_PRINT = FLAG_PRINT  # whether print the log into terminal
        self.OBS_OFFSET = OBS_OFFSET
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG

    def __color_agent_name__(self, agent_name: str):
        return f"""{bcolors.OKBLUE}{agent_name}{bcolors.ENDC}"""

    def __color_task_str__(self, task_str: str):
        return f"""{bcolors.OKCYAN}{task_str}{bcolors.ENDC}"""

    def __color_act_str__(self, act_str: str):
        return f"""{bcolors.OKBLUE}{act_str}{bcolors.ENDC}"""

    def __color_obs_str__(self, act_str: str):
        return f"""{bcolors.OKGREEN}{act_str}{bcolors.ENDC}"""

    def __color_prompt_str__(self, prompt: str):
        return f"""{bcolors.WARNING}{prompt}{bcolors.ENDC}"""

    def __color_warning_str__(self, prompt: str):
        return f"""{bcolors.FAIL}{prompt}{bcolors.ENDC}"""

    def __cache_task__(self, task: TaskPackage, action_chain: list[tuple[AgentAction, str]], prev_action_chain: list[list[tuple[AgentAction, str]]] = None):
        task_log = {}
        task_log[MEMORY_TASK_KEY] = task.__dict__
        task_log[MEMORY_ACT_OBS_KEY] = []
        for agent_act, obs in action_chain:
            task_log[MEMORY_ACT_OBS_KEY].append([action_format(agent_act, action_trigger=False), obs])
        
        if prev_action_chain is not None:
            task_log[MEMORY_TASK_KEY][MEMORY_PREV_ACT_OBS_KEY] = []
            for action_chain in prev_action_chain:
                task_log[MEMORY_TASK_KEY][MEMORY_PREV_ACT_OBS_KEY].append([])
                for agent_act, obs in action_chain:
                    task_log[MEMORY_TASK_KEY][MEMORY_PREV_ACT_OBS_KEY][-1].append([action_format(agent_act, action_trigger=False), obs])

        return task_log

    def __save_log__(self, log_str: str):
        if self.FLAG_PRINT:
            print(log_str)
        with open(self.log_file_name, "a") as f:
            f.write(str_color_remove(log_str) + "\n")

    def receive_task(self, task: TaskPackage, actions: list[BaseAction], agent_name: str):
        task_str = (
            f"""[\n\tTask ID: {task.task_id}\n\n\tAction Space: {format_act_params_example(actions)}\n\n\tInputs: {task.inputs}\n\n\tMultiModal Inputs: {json.dumps(task.multimodal_inputs)}\n\n\tInstruction: {task.instruction}\n]"""
        )
        log_str = f"""Agent {self.__color_agent_name__(agent_name)} """
        log_str += f"""receives the following {bcolors.UNDERLINE}TaskPackage{bcolors.ENDC}:\n"""
        log_str += f"{self.__color_task_str__(task_str=task_str)}"
        self.__save_log__(log_str=log_str)

    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        log_str = f"""===={self.__color_agent_name__(agent_name)} starts execution on TaskPackage {task.task_id}===="""
        self.__save_log__(log_str=log_str)

    def end_execute(self, task: TaskPackage, agent_name: str = None, action_chain: list[tuple[AgentAction, str]] = None, prev_action_chain: list[list[tuple[AgentAction, str]]] = None):
        log_str = f"""========={self.__color_agent_name__(agent_name)} finish execution. TaskPackage[ID:{task.task_id}] status:\n"""
        task_str = f"""[\n\tcompletion: {task.completion}\n\tanswer: {task.answer}\n]"""
        log_str += self.__color_task_str__(task_str=task_str)
        log_str += "\n=========="
        self.__save_log__(log_str=log_str)
        task_log = self.__cache_task__(task=task, action_chain=action_chain, prev_action_chain=prev_action_chain)
        return task_log

    def take_action(self, action: AgentAction, agent_name: str, step_idx: int):
        act_str = f"""{{\n\tname: {action.action_name}\n\tparams: {action.params}\n}}"""
        log_str = f"""Agent {self.__color_agent_name__(agent_name)} takes {step_idx}-step {bcolors.UNDERLINE}Action{bcolors.ENDC}:\n"""
        log_str += f"""{self.__color_act_str__(act_str)}"""
        self.__save_log__(log_str)

    def add_st_memory(self, agent_name: str):
        log_str = f"""Action and Observation added to Agent {self.__color_agent_name__(agent_name)} memory"""
        self.__save_log__(log_str)

    def get_obs(self, obs: str):
        if len(obs) > self.OBS_OFFSET:
            obs = obs[: self.OBS_OFFSET] + "[TLDR]"
        log_str = f"""Observation: {self.__color_obs_str__(obs)}"""
        self.__save_log__(log_str)

    def get_prompt(self, prompt):
        log_str = f"""Prompt: {self.__color_prompt_str__(prompt)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)

    def get_llm_output(self, output: str):
        log_str = f"""LLM generates: {self.__color_prompt_str__(output)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)
    
    def warning_output(self, output: str):
        log_str = f"""Warning: {self.__color_warning_str__(output)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)