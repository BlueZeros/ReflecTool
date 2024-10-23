import os
import json
from typing import List
from reflectool.commons import TaskPackage
from reflectool.actions.BaseAction import AgentAction, BaseAction
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
from reflectool.agent_prompts.prompt_utils import task_chain_format, action_chain_format, format_act_params_example

class PromptGen:
    """Prompt Generator Class"""

    def __init__(self) -> None:
        self.prompt_type = "BasePrompt"
        self.examples: dict[str, list] = {}

    def add_example(
        self,
        task: TaskPackage,
        action_chain: List[tuple[AgentAction, str]],
        example_type: str = "action",
    ):
        example_context = task_chain_format(task, action_chain)
        if example_type in self.examples:
            self.examples[example_type].append(example_context)
        else:
            self.examples[example_type] = [example_context]

    def __get_example__(self, example_type: str, index: int = -1):
        if example_type in self.examples:
            return self.examples[example_type][index]
        else:
            return None
    
    def __get_examples__(self, example_type: str) -> str:
        """get multiple examples for prompt"""
        # check if example_type exist in self.examples
        if not example_type in self.examples:
            return None
        else:
            indices = list(range(len(self.examples[example_type])))
            examples = [self.__get_example__(example_type, idx) for idx in indices]
            return "\n".join(examples)

class TaskPromptGen(PromptGen):
    """
    this is the BasePrompt for agent to use.
    """

    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        preload_multimodal: bool = False,
    ):
        """Prompt Generator for Base Agent
        :param agent_role: the role of this agent, defaults to None
        :type agent_role: str, optional
        :param constraint: the constraint of this agent, defaults to None
        :type constraint: str, optional
        """
        super().__init__()
        self.prompt_type = "BaseAgentPrompt"
        self.agent_role = agent_role
        self.constraint = constraint
        self.instruction = instruction
        self.preload_multimodal = preload_multimodal

    def __get_role_ins__(self):
        """use as the start of every action prompt. Highlight the role of this agent"""
        ## to-do
        return

    def __constraint_prompt__(self):
        if self.constraint:
            return f"""{PROMPT_TOKENS["constraint"]['begin']}\n{self.constraint}\n{PROMPT_TOKENS["constraint"]['end']}"""
        else:
            return ""

    def __construct_history__(self, action_chain):
        history = action_chain_format(action_chain)
        return history

    def __role_prompt__(self, agent_role):
        prompt = f"""{PROMPT_TOKENS["role"]['begin']}\n{agent_role}\n{PROMPT_TOKENS["role"]['end']}"""
        return prompt

    def __act_doc_prompt__(self, actions: List[BaseAction], params_doc_flag=True):
        if params_doc_flag:  # given the parameters as the document
            action_doc = [
                {
                    "name": act.action_name,
                    "description": act.action_desc,
                    "parameters": act.params_doc,
                }
                for act in actions
            ]
        else:
            action_doc = {act.action_name: act.action_desc for act in actions}
        
        action_doc = "\n".join([json.dumps(action) for action in action_doc])
        
        prompt = f"""{PROMPT_TOKENS["action"]['begin']}\n{action_doc}\n{PROMPT_TOKENS["action"]['end']}"""
        return prompt

    def __prompt_example__(self, prompt_example: str):
        prompt = f"""{PROMPT_TOKENS["example"]['begin']}\n{prompt_example}{PROMPT_TOKENS["example"]['end']}\n"""
        return prompt
    
    def __example_format_prompt__(self, examples: list):
        example_contexts = [task_chain_format(task, action_chain) for (task, action_chain) in examples]
        return "\n".join(example_contexts)

    def __act_format_example__(self, act_call_example: str):
        prompt = f"""{DEFAULT_PROMPT["action_format"]}{PROMPT_TOKENS["action_format"]['begin']}\n{act_call_example}{PROMPT_TOKENS["action_format"]['end']}\n"""
        return prompt

    def action_prompt(
        self,
        task: TaskPackage,
        actions: List[BaseAction],
        action_chain: List[tuple[BaseAction, str]],
        example_type: str = "action",
        examples: list = None,
        **kwargs,
    ) -> str:
        """return the action generation prompt for agent
        :param task: the task to finish
        :type task: TaskPackage
        :param actions: the actions to take
        :type actions: List[BaseAction]
        :param action_chain: the history action-obs chain of this task
        :type action_chain: List[tuple[AgentAct, str]]
        :param labor_agents_doc: the title and description dict of the labor agent, defaults to None
        :type labor_agents_doc: dict[str, str], optional
        :param example_type: the type of example, defaults to "action"
        :type example_type: str, optional
        :param example: the example string, defaults to None
        :type example: str, optional
        :return: the prompt for agent to take action
        :rtype: str
        """
        # adding roles into prompt
        prompt = f"""{self.instruction}\n{self.__role_prompt__(self.agent_role)}\n"""
        # adding constraint into prompt
        prompt += f"""{self.__constraint_prompt__()}\n"""
        # adding action doc into prompt
        prompt += (
            f"""{self.__act_doc_prompt__(actions=actions, params_doc_flag=True)}\n"""
        )

        act_call_example = format_act_params_example(actions)
        # get task example
        if examples:  # get from input
            prompt_example = self.__example_format_prompt__(examples)
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)
        else:  # no example provided, only add the format example
            prompt += self.__act_format_example__(act_call_example)

        # adding action observation chain
        cur_session = task_chain_format(task, action_chain, self.preload_multimodal)
        prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}\n"""
        # adding inference token
        prompt += """Action: """

        return prompt