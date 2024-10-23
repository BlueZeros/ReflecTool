import os
import re
import sys
import json
import copy

from reflectool.utilities import *
from reflectool.models import get_model
from reflectool.agents.TaskAgent import TaskAgent
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions import *
from reflectool.agents.agent_utils import *
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
# from agent_prompts.CriticPromptGen import CriticPromptGen
from reflectool.agent_prompts.PromptGen import TaskPromptGen
from reflectool.agent_prompts.ModelPromptGen import ModelPromptGen
from reflectool.memory.Memory import ShortTermMemory, LongTermMemory
from reflectool.memory.memory_utils import MEMORY_TASK_KEY, MEMORY_ACT_OBS_KEY
from reflectool.logger.logger import AgentLogger
from reflectool.logger.base import BaseAgentLogger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CriticAgent(TaskAgent):

    def __init__(
        self, 
        args, 
        name: str = "CriticAgent",
        role: str = "You are a helpful medical assistant.",
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["critic_instruction"],
        logger: BaseAgentLogger = AgentLogger()
    ):
        super().__init__(args, name, role, constraint, instruction, logger)
    
        self.prompt_gen = TaskPromptGen(
            agent_role=self.role,
            constraint=self.constraint.format(max_exec_steps=self.max_exec_steps),
            instruction=self.instruction,
            preload_multimodal=self.preload_multimodal,
        )

        self.model_prompt_gent = ModelPromptGen(
            agent_role=self.role,
            constraint=DEFAULT_PROMPT["model_constraint"],
            instruction=DEFAULT_PROMPT["model_instruction"],
        )
    
    def get_init_answer(self, task: TaskPackage):
        model_task = copy.deepcopy(task)
        self.short_term_memory.add_new_task(model_task)
        meta_prompt = self.model_prompt_gent.meta_prompt(model_task)

        self.logger.get_prompt(str(meta_prompt))
        if "images" in meta_prompt:
            for action in self.actions:
                if action.action_name == "HuatuoGPT":
                    meta_prompt['query'] = meta_prompt.pop('inputs')
                    meta_prompt['image'] = meta_prompt.pop('images')
                    response = action(**meta_prompt)
        else:
            response = self.llm(**meta_prompt)

        task.previous_answer = response
        self.short_term_memory.delete_task(model_task)
    
    def __call__(self, task: TaskPackage):
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: the response of this task
        :rtype: str
        """

        self.get_init_answer(task)

        self.logger.receive_task(task=task, actions=self.actions, agent_name=self.name)
        self.assign(task)
        task_log = self.execute(task)

        return task_log
    