import os
from typing import List
from commons import TaskPackage
from actions.BaseAction import AgentAction, BaseAction
from actions.EHRSQL import ehrsql_prompt
from actions.LongDocRAG import load_multiple_documents
from agent_prompts.PromptGen import PromptGen
from agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
from agent_prompts.prompt_utils import task_chain_format, action_chain_format, format_act_params_example


class ModelPromptGen(PromptGen):
    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["model_constraint"],
        instruction: str = DEFAULT_PROMPT["model_instruction"],
        ):
        """Prompt Generator for Base Model
        :param agent_role: the role of this agent, defaults to None
        :type agent_role: str, optional
        :param constraint: the constraint of this agent, defaults to None
        :type constraint: str, optional
        """
        super().__init__()
        self.prompt_type = "BaseModelPrompt"
        self.agent_role = agent_role
        self.constraint = constraint
        self.instruction = instruction
    
    def __role_prompt__(self, agent_role):
        prompt = f"""{PROMPT_TOKENS["role"]['begin']}\n{agent_role}\n{PROMPT_TOKENS["role"]['end']}"""
        return prompt
    
    def __constraint_prompt__(self):
        if self.constraint:
            return f"""{PROMPT_TOKENS["constraint"]['begin']}\n{self.constraint}\n{PROMPT_TOKENS["constraint"]['end']}"""
        else:
            return ""

    def task_prompt(self, task: TaskPackage):
        prompt = f"""{self.instruction}\n{self.__role_prompt__(self.agent_role)}\n"""
        # adding constraint into prompt
        prompt += f"""{self.__constraint_prompt__()}\n"""
        return prompt
    
    def meta_prompt(self, task: TaskPackage):
        meta_prompt = {}

        task_prompt = self.task_prompt(task)
        
        if task.multimodal_inputs["sql_database"] is not None:
            sql_prompt = ehrsql_prompt(task.multimodal_inputs["sql_database"])
            task.inputs += f"\n{sql_prompt}"
        
        elif task.multimodal_inputs["upload_files"] is not None:
            task.inputs += load_multiple_documents(task.multimodal_inputs["upload_files"])
        
        context = f"Inputs: {task.inputs}\nInstruction: {task.instruction}\n"
        meta_prompt["inputs"] = task_prompt + context
        if task.multimodal_inputs["image"] is not None:
            meta_prompt["images"] = task.multimodal_inputs["image"]

        return meta_prompt
    
