import os
import re
import sys
import json

from reflectool.utilities import *
from reflectool.models import get_model
from reflectool.agents.BaseAgent import BaseAgent
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.actions.BaseAction import AgentAction, Think, Finish, Plan
from reflectool.actions.actions_register import ACTIONS_REGISTRY
from reflectool.agents.agent_utils import *
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
from reflectool.agent_prompts.PromptGen import TaskPromptGen
from reflectool.memory.Memory import ShortTermMemory, WritenLongTermMemory
from reflectool.memory.memory_utils import MEMORY_TASK_KEY, MEMORY_ACT_OBS_KEY
from reflectool.logger.logger import AgentLogger
from reflectool.logger.base import BaseAgentLogger

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TaskAgent(BaseAgent):

    def __init__(
        self, 
        args, 
        name: str = "Task_Agent",
        role: str = "You are a helpful medical agent.",
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        logger: BaseAgentLogger = AgentLogger()
    ):
        super().__init__(args, name, role)
        self.constraint = constraint
        self.instruction = instruction
        self.prompt_gen = TaskPromptGen(
            agent_role=self.role,
            constraint=self.constraint.format(max_exec_steps=self.max_exec_steps),
            instruction=self.instruction,
            preload_multimodal=self.preload_multimodal,
        )

        self.logger = logger

        self.__add_tool_actions__()
        self.__add_inner_actions__()
        self.__load_action_params__()

        self.__add_st_memory__()
        self.__add_lt_memory__()

        self.llm = self.__build_llm__()
    
    def __build_llm__(self):
        return get_model(self.model, stops=["Observation:"], vllm_serve=self.vllm_serve, vllm_serve_url=self.vllm_serve_url)
    
    def llm_layer(self, prompt: str) -> str:
        """input a prompt, llm generates a text

        :param prompt: the prompt string
        :type prompt: str
        :return: the output from llm, which is a string
        :rtype: str
        """
        return self.llm(prompt)
    
    def __add_st_memory__(self, short_term_memory: ShortTermMemory = None):
        if short_term_memory:
            self.short_term_memory = short_term_memory
        else:
            self.short_term_memory = ShortTermMemory(agent_id=self.id)
    
    def __add_lt_memory__(self, long_term_memory: WritenLongTermMemory = None):
        if long_term_memory is None:
            self.long_term_memory = WritenLongTermMemory(
                                        agent_id=self.id, 
                                        exp_name=self.exp_name,
                                        memory_path=self.memory_path, 
                                        task=self.task_name,
                                        split=self.test_split,
                                        k=self.few_shot, 
                                        memory_type=self.memory_type,
                                        write_mode=self.write_memory,
                                        update_freq=self.update_freq,
                                        memory_wo_reflect=self.memory_wo_reflect
                                    )
        else:
            self.long_term_memory = long_term_memory
        
        # for memory in self.long_term_memory.get_memories():
        #     self.prompt_gen.add_example(
        #         memory[MEMORY_TASK_KEY],
        #         memory[MEMORY_ACT_OBS_KEY]
        #     )

    def __add_tool_actions__(self):
        """adding the tool action types into agent"""
        
        if self.actions[0] == "all":
            assert torch.cuda.device_count() > 1
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY]
        
        if self.actions[0] == "all_wo_mm":
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type != "MultiModal"]
        
        elif self.actions[0] == "mm":
            assert torch.cuda.device_count() > 1
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type == "MultiModal"]
        
        elif self.actions[0] == "know":
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type == "Knowledge"]
        
        elif self.actions[0] == "num":
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type == "Numerical"]
        
        elif self.actions[0] == "data":
            self.actions = [action() for action, action_type in ACTIONS_REGISTRY if action_type == "Data"]

        else:
            actions = []
            for action in self.actions:
                try:
                    actions.append(eval(action)())
                except:
                    print(f"{action} is not predefined in the action space")
            self.actions = actions

    def __add_inner_actions__(self):
        """adding the inner action types into agent, which is based on the `self.reasoning_type`"""
        # if self.reasoning_type == "react":
        #     self.actions += [ThinkAction, FinishAction]
        # elif self.reasoning_type == "act":
        #     self.actions += [FinishAction]
        # elif self.reasoning_type == "planact":
        #     self.actions += [PlanAction, FinishAction]
        # elif self.reasoning_type == "planreact":
        
        if self.force_action:
            # self.first_action = [Plan()]
            self.final_action = [Finish()]
            self.actions += [Plan(), Think(), Finish()]
        else:
            self.actions += [Plan(), Think(), Finish()]
            
        self.actions = list(set(self.actions))
    
    def __load_action_params__(self):
        if self.load_action_params is not None:
            print(f"Loading action parameters from {self.load_action_params}...")

            with open(self.load_action_params, "r") as f:
                action_params = json.load(f)
                for action in self.actions:
                    action_param = action_params[action.action_name]
                    action.action_desc = action_param["action_desc"]
                    action.params_doc = action_param["params_doc"]
        

    def __call__(self, task: TaskPackage):
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: the response of this task
        :rtype: str
        """
        # import pdb
        # pdb.set_trace()
        # adding log information
        self.logger.receive_task(task=task, actions=self.actions, agent_name=self.name)
        self.assign(task)
        task_log = self.execute(task)
        return task_log
    
    def assign(self, task: TaskPackage) -> None:
        """assign task to agent

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        self.short_term_memory.add_new_task(task)

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
        
        task_log = self.logger.end_execute(task=task, agent_name=self.name, action_chain=self.short_term_memory.get_action_chain(task))  
        return task_log

    def respond(self, task: TaskPackage, **kwargs) -> str:
        """generate messages for manager agents

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: a response
        :rtype: str
        """

        if task.completion in ["completed"]:
            return task.answer
        else:
            # to do: add more actions after the task is not completed such as summarizing the actions
            return DEFAULT_PROMPT["not_completed"]

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

        # if self.force_action and first_action:
        #     action_prompt = self.prompt_gen.action_prompt(
        #         task=task,
        #         actions=self.first_action,
        #         action_chain=action_chain,
        #         examples=ltm,
        #     )

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
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action)

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

    def __action_parser__(self, raw_action: str) -> AgentAction:
        """parse the generated content to an executable action

        :param raw_action: llm generated text
        :type raw_action: str
        :return: an executable action wrapper
        :rtype: AgentAct
        """

        action_name, params, PARSE_FLAG = parse_action(raw_action)
        if not PARSE_FLAG:
            self.logger.warning_output(f"Fail to parse the agent action from the output of the llm:\n{raw_action}")
        agent_act = AgentAction(action_name=action_name, params=params)
        return agent_act

    def forward(self, task: TaskPackage, agent_act: AgentAction) -> str:
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
                if agent_act.action_name == Finish().action_name:
                    task.answer = observation
                    task.completion = "completed"
        # if not find this action
        if act_found_flag:
            return observation
        else:
            # raise NotImplementedError
            observation = ACION_NOT_FOUND_MESS
            return observation

    def __check_action__(self, action_name:str):
        """check if the action is in the action space

        :param action_name: the name of the action
        :type action_name: str
        """
        for action in self.actions:
            if act_match(action_name, action):
                return True
        return False
    
    
    