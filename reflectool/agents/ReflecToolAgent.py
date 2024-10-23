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
from reflectool.agent_prompts.ClinicalPromptGen import ClinicalPromptGen
from reflectool.logger.logger import AgentLogger
from reflectool.logger.base import BaseAgentLogger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ReflecToolAgent(TaskAgent):
    def __init__(
        self, 
        args, 
        name: str = "ReflecToolAgent",
        role: str = "You are a helpful medical assistant.",
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["reflexion_instruction"],
        logger: BaseAgentLogger = AgentLogger()
    ):
        super().__init__(args, name, role, constraint, instruction, logger)
    
        self.prompt_gen = ClinicalPromptGen(
            agent_role=self.role,
            constraint=self.constraint.format(max_exec_steps=self.max_exec_steps),
            instruction=self.instruction,
            preload_multimodal=self.preload_multimodal,
            action_guide_path=self.action_guide_path
        )
    
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
            
        return task_log
    
    def forward(self, task: TaskPackage, agent_act: AgentAction, attempt_act: bool = False) -> str:
        """
        using this function to forward the action to get the observation.

        :param task: the task which agent receives and solves.
        :type task: TaskPackage
        :param agent_act: the action wrapper for execution.
        :type agent_act: AgentAct
        :return: observation
        :rtype: str
        """
        act_found_flag = False
        
        # if match one in self.actions
        for action in self.actions:
            if act_match(agent_act.action_name, action):
                act_found_flag = True
                try:
                    if action.llm_drive:
                        observation = action(**agent_act.params, llm=self.llm)
                    else:
                        observation = action(**agent_act.params)
                except Exception as e:
                        observation = f"Error: {str(e)}"

                # if action is Finish Action
                if agent_act.action_name == Finish().action_name and not attempt_act:
                    task.answer = observation
                    task.completion = "completed"
        # if not find this action
        if act_found_flag:
            return observation
        else:
            # raise NotImplementedError
            observation = ACION_NOT_FOUND_MESS
            return observation

    def llm_layer(self, prompt: str, n=1, temperature=0.0, use_beam_search=False) -> str:
        """input a prompt, llm generates a text

        :param prompt: the prompt string
        :type prompt: str
        :return: the output from llm, which is a string
        :rtype: str
        """
        return self.llm(prompt, n=n, temperature=temperature, use_beam_search=use_beam_search)

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

        if self.force_action and final_action:
            action_prompt = self.prompt_gen.action_prompt(
                task=task,
                actions=self.final_action,
                action_chain=action_chain,
                examples=ltm,
            )

        else:
            action_prompt = self.prompt_gen.action_prompt(
                task=task,
                actions=self.actions,
                action_chain=action_chain,
                examples=ltm,
            )

        self.logger.get_prompt(action_prompt)
        # raw_action = self.llm_layer(action_prompt)
        if self.action_search == "refine":
            raw_action = self.act_refine(task, action_prompt, action_chain, ltm)

        elif self.action_search == "select":
            raw_action = self.act_select(task, action_prompt, action_chain, ltm)

        else:
            raise NotImplementedError

        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action)
    
    def act_refine(self, task, action_prompt, action_chain, ltm):
        raw_action = self.llm_layer(action_prompt)
        
        for _ in range(self.clinical_reflect_num - 1):
            previous_raw_action = raw_action
            current_action = self.__action_parser__(raw_action)
            current_observation = self.forward(task, current_action, attempt_act=True)

            refine_prompt = self.prompt_gen.refine_action_prompt(
                task,
                actions=self.actions,
                action_chain=action_chain,
                current_action=current_action,
                current_obs=current_observation,
                examples=ltm,
            )

            raw_action = self.llm_layer(refine_prompt)

            if previous_raw_action == raw_action:
                break
        
        return raw_action

    def act_select(self, task, action_prompt, action_chain, ltm):
        if self.clinical_reflect_num == 1:
            raw_action = self.llm_layer(action_prompt)
        else:
            raw_actions = self.llm_layer(action_prompt, n=self.clinical_reflect_num, use_beam_search=True)
            candidate_actions = [(self.__action_parser__(raw_action), self.forward(task, self.__action_parser__(raw_action), attempt_act=True)) for raw_action in raw_actions]

            select_prompt = self.prompt_gen.select_action_prompt(
                    task,
                    actions=self.actions,
                    action_chain=action_chain,
                    candidate_actions=candidate_actions,
                    examples=ltm,
                )
            raw_action = self.llm_layer(select_prompt)

        return raw_action
        