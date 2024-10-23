import os
import json
from typing import List
from reflectool.commons import TaskPackage
from reflectool.agent_prompts.PromptGen import TaskPromptGen
from reflectool.actions.BaseAction import AgentAction, BaseAction
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS, CLINICAL_AGENT_PROMPT
from reflectool.agent_prompts.prompt_utils import task_chain_format, format_act_params_example, action_chain_format, action_format

def replace_last(text, old, new):
    # 查找最后一次出现的位置
    pos = text.rfind(old)
    # 如果找到了，则进行替换
    if pos != -1:
        text = text[:pos] + text[pos:].replace(old, new, 1)
    return text

class ClinicalPromptGen(TaskPromptGen):
    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        preload_multimodal: bool = False,
        action_guide_path: str = None,
    ):
        super().__init__(agent_role, constraint, instruction, preload_multimodal)
        if action_guide_path is not None:
            with open(action_guide_path, "r") as f:
                self.action_guide = json.load(f)
        else:
            self.action_guide = None

    def __act_guide_prompt__(self, action_names: list):
        if self.action_guide is not None:
            action_names = list(set(action_names))
            action_guide_context = [f"""{act_name}: {self.action_guide[act_name]}""" for act_name in action_names if act_name in self.action_guide]
            action_guide_context = "\n".join(action_guide_context)
            return f"""{PROMPT_TOKENS["action_guide"]["begin"]}\n{action_guide_context}{PROMPT_TOKENS["action_guide"]["end"]}\n"""
        else:
            return f"""{PROMPT_TOKENS["action_guide"]["begin"]}\nNo Information{PROMPT_TOKENS["action_guide"]["end"]}\n"""

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
        
        cur_session = task_chain_format(task, action_chain, self.preload_multimodal)
        # adding action observation chain
        prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}\n"""
        # adding inference token
        prompt += """Action: """

        return prompt
    
    def refine_action_prompt(
        self,
        task: TaskPackage,
        actions: List[BaseAction],
        action_chain: List[tuple[BaseAction, str]],
        current_action: AgentAction,
        current_obs: str = None,
        example_type: str = "action",
        examples: list = None,
        **kwargs,
    ) -> str:
       # adding roles into prompt
        prompt = f"""{self.instruction}\n{self.__role_prompt__(self.agent_role)}\n"""
        # # adding constraint into prompt
        prompt += f"""{self.__constraint_prompt__()}\n"""
        # # adding action doc into prompt
        prompt += (
            f"""{self.__act_doc_prompt__(actions=actions, params_doc_flag=True)}\n"""
        )
        # 
        # get task example
        if examples:  # get from input
            prompt_example = self.__example_format_prompt__(examples)
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)
        else:  # no example provided, only add the format example
            act_call_example = format_act_params_example(actions)
            prompt += self.__act_format_example__(act_call_example)

        # adding action guideline 
        prompt += self.__act_guide_prompt__([current_action.action_name])

        # adding action observation chain
        cur_session = task_chain_format(task, action_chain, self.preload_multimodal)
        prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}\n"""

        prompt += f"""Current Action: {action_format(current_action, action_trigger=False)}\n"""
        if current_obs is not None:
            prompt += f"""Current Observation: {current_obs}\n""" 

        prompt += CLINICAL_AGENT_PROMPT["refine_action_instruction"] + "\n"
        prompt += f"""Refined Action: """
        return prompt

    def select_action_prompt(
        self,
        task: TaskPackage,
        actions: List[BaseAction],
        action_chain: List[tuple[BaseAction, str]],
        candidate_actions: List[tuple[AgentAction, str]],
        example_type: str = "action",
        examples: list = None,
        **kwargs,
    ) -> str:
       # adding roles into prompt
        prompt = f"""{self.instruction}\n{self.__role_prompt__(self.agent_role)}\n"""
        # # adding constraint into prompt
        prompt += f"""{self.__constraint_prompt__()}\n"""
        # # adding action doc into prompt
        prompt += (
            f"""{self.__act_doc_prompt__(actions=actions, params_doc_flag=True)}\n"""
        )
        # 
        # get task example
        if examples:  # get from input
            prompt_example = self.__example_format_prompt__(examples)
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)
        else:  # no example provided, only add the format example
            act_call_example = format_act_params_example(actions)
            prompt += self.__act_format_example__(act_call_example)

        # adding action guideline 
        prompt += self.__act_guide_prompt__([action.action_name for (action, _) in candidate_actions])

        # adding action observation chain
        cur_session = task_chain_format(task, action_chain, self.preload_multimodal)
        prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}\n"""

        prompt += f"""{PROMPT_TOKENS["candidate_actions"]['begin']}\n{action_chain_format(candidate_actions)}\n{PROMPT_TOKENS["candidate_actions"]['end']}\n"""

        prompt += CLINICAL_AGENT_PROMPT["select_action_instruction"] + "\n"
        prompt += f"""Selected Action: """
        return prompt
    