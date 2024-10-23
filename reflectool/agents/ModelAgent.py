import os
import re
import sys

from reflectool.utilities import *
from reflectool.models import get_model
from reflectool.agents.BaseAgent import BaseAgent
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions import *
from reflectool.actions.BaseAction import AgentAction
from reflectool.agents.agent_utils import *
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
from reflectool.agent_prompts.ModelPromptGen import ModelPromptGen
from reflectool.memory.Memory import ShortTermMemory, LongTermMemory
from reflectool.memory.memory_utils import MEMORY_TASK_KEY, MEMORY_ACT_OBS_KEY
from reflectool.logger.logger import AgentLogger
from reflectool.logger.base import BaseAgentLogger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelAgent(BaseAgent):

    def __init__(
        self, 
        args, 
        name: str = "Task_Agent",
        role: str = "You are a helpful medical assistant.",
        constraint: str = DEFAULT_PROMPT["model_constraint"],
        instruction: str = DEFAULT_PROMPT["model_instruction"],
        logger: BaseAgentLogger = AgentLogger()
    ):
        super().__init__(args, name, role)
        self.constraint = constraint
        self.instruction = instruction
        self.prompt_gen = ModelPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )

        self.logger = logger
        self.actions = []
        self.llm = self.__build_llm__()
        self.__add_st_memory__()
    
    def __build_llm__(self):
        return get_model(self.model, stops=["Observation:"])
    
    def __add_st_memory__(self, short_term_memory: ShortTermMemory = None):
        if short_term_memory:
            self.short_term_memory = short_term_memory
        else:
            self.short_term_memory = ShortTermMemory(agent_id=self.id)

    def __call__(self, task: TaskPackage) -> str:
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: the response of this task
        :rtype: str
        """
        # log and memory init
        self.logger.receive_task(task=task, actions=self.actions, agent_name=self.name)
        self.short_term_memory.add_new_task(task)

        # forward
        meta_prompt = self.prompt_gen.meta_prompt(task)
        self.logger.get_prompt(str(meta_prompt))
        response = self.llm(**meta_prompt)

        # log
        task.answer = response
        task.completion = "completed"
        action = AgentAction(action_name="Finish", params="")
        observation = response
        self.__st_memorize__(task, action, observation)
        task_log = self.logger.end_execute(task=task, agent_name=self.name, action_chain=self.short_term_memory.get_action_chain(task)) 
        
        return task_log
    
    def __st_memorize__(
        self, task: TaskPackage, action: AgentAction, observation: str = ""
    ):
        """the short-term memorize action and observation for agent

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action: the action wrapper for execution
        :type action: AgentAct
        :param observation: the observation after action execution, defaults to ""
        :type observation: str, optional
        """
        self.short_term_memory.add_act_obs(task, action, observation)
    
    