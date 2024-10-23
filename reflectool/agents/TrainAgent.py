import os
import re
import sys

from reflectool.utilities import *
from reflectool.models import get_model
from reflectool.agents.TaskAgent import TaskAgent
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions import *
from reflectool.agents.agent_utils import *
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT
from reflectool.agent_prompts.TrainPromptGen import TrainPromptGen
from reflectool.logger.logger import AgentLogger
from reflectool.logger.base import BaseAgentLogger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainAgent(TaskAgent):

    def __init__(
        self, 
        args, 
        name: str = "TrainAgent",
        role: str = "You are a helpful medical assistant.",
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["reflexion_instruction"],
        logger: BaseAgentLogger = AgentLogger()
    ):
        super().__init__(args, name, role, constraint, instruction, logger)
    
        self.prompt_gen = TrainPromptGen(
            agent_role=self.role,
            constraint=self.constraint.format(max_exec_steps=self.max_exec_steps),
            instruction=self.instruction,
            preload_multimodal=self.preload_multimodal,
        )

    def reflection(self, task):
        action_chain = self.short_term_memory.get_action_chain(task)
        ltm = self.long_term_memory.get_memories(task)

        reflection_prompt = self.prompt_gen.reflection_prompt(
            task=task,
            action_chain=action_chain,
            examples=ltm,
        )
        self.logger.get_prompt(reflection_prompt)

        reflextion = self.llm_layer(reflection_prompt)
        self.logger.get_llm_output(reflextion)

        action = AgentAction(action_name="Reflection", params={"response": reflextion})
        observation = "OK"
        self.logger.take_action(action, agent_name=self.name, step_idx=-1)
        self.logger.get_obs(obs=observation)
        self.__st_memorize__(task, action, observation)

        self.short_term_memory.update(task)
    
    def __call__(self, task: TaskPackage):
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: the response of this task
        :rtype: str
        """

        # init task
        self.logger.receive_task(task=task, actions=self.actions, agent_name=self.name)
        self.assign(task)
        task_log = self.execute(task)

        ## reflextion begin
        reflect_step = 0
        while reflect_step < self.reflect_iter:
            self.reflection(task)

            # reflash the status of the task
            task.completion = "active"
            task.answer = None

            # keep 
            task_log = self.execute(task)
            reflect_step += 1
        
        self.long_term_memory.__save_memory__(task_log)    
        return task_log

    
    def execute(self, task: TaskPackage):
        """multi-step execution of actions. Generate the actions for a task until reach the done

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        step_size = 0
        self.logger.execute_task(task=task, agent_name=self.name)
        while task.completion == "active" and step_size < self.max_exec_steps:
            action_chain = self.short_term_memory.get_action_chain(task)
            action = self.__next_act__(task, action_chain, first_action=(step_size==0), final_action=(step_size==self.max_exec_steps-1))
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size)
            observation = self.forward(task, action)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
            step_size += 1
        
        task_log = self.logger.end_execute(task=task, agent_name=self.name, action_chain=self.short_term_memory.get_action_chain(task), prev_action_chain=self.short_term_memory.get_prev_action_chain(task))  
        return task_log
    
    def __next_act__(
        self, task: TaskPackage, action_chain: list[tuple[AgentAction, str]], first_action: bool, final_action: bool
    ) -> AgentAction:
        """one-step action generation

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action_chain: history actions and observation of this task from memory
        :type action_chain: ActObsChainType
        :return: the action for agent to execute
        :rtype: AgentAct
        """

        ltm = self.long_term_memory.get_memories(task)
        previous_action_chain = self.short_term_memory.get_prev_action_chain(task)

        if self.force_action and final_action:
            action_prompt = self.prompt_gen.action_prompt(
                task=task,
                actions=self.final_action,
                action_chain=action_chain,
                previous_action_chain=previous_action_chain,
                examples=ltm,
            )

        else:
            action_prompt = self.prompt_gen.action_prompt(
                task=task,
                actions=self.actions,
                action_chain=action_chain,
                previous_action_chain=previous_action_chain,
                examples=ltm,
            )

        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action)
    