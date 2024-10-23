import os
import json
from typing import List
from reflectool.commons import TaskPackage
from reflectool.agent_prompts.PromptGen import TaskPromptGen
from reflectool.actions.BaseAction import AgentAction, BaseAction
from reflectool.agent_prompts.prompt_utils import DEFAULT_PROMPT, PROMPT_TOKENS
from reflectool.agent_prompts.prompt_utils import task_chain_format, task_chain_format_w_prev_actobs, format_act_params_example

REFLEXTION_PROMPT = """You will be given the history of a past experience in which you were placed in an environment and given a task to complete. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. If you were unsuccessful in completing the task, summarize the reasons for failure to improve the reasoning process, otherwise you can check whether there are errors or hallucination in the reasoning process to improve the accuracy of the results. You will need this later when you are solving the same task."""

class ReflexionPromptGen(TaskPromptGen):
    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["agent_constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        preload_multimodal: bool = False,
    ):
        super().__init__(agent_role, constraint, instruction, preload_multimodal)
    
    def reflexion_prompt(
        self,
        task: TaskPackage,
        action_chain: List[tuple[BaseAction, str]],
        example_type: str = "action",
        examples: list = None,
        **kwargs,
    ) -> str:
        # adding roles into prompt
        prompt = REFLEXTION_PROMPT

        # get task example
        if examples:  # get from input
            prompt_example = self.__example_format_prompt__(examples)
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)

        # adding action observation chain
        cur_session = task_chain_format(task, action_chain, self.preload_multimodal)
        prompt += f"""Previous Trial:\n{cur_session}\n"""
        # adding inference token
        prompt += """Reflection: """

        return prompt
    
    def __prev_act_obs_prompt__(self, prev_act_obs_list):
        histories = []
        for prev_act_obs in prev_act_obs_list:
            histories.append(self.__construct_history__(prev_act_obs))

        if len(histories) == 0:
            return ""
        else:
            histories = "\n".join(histories)
            return f"""{PROMPT_TOKENS["previous_trial"]["begin"]}\n{histories}{PROMPT_TOKENS["previous_trial"]["end"]}\n"""
    
    def __example_format_prompt__(self, examples: list):
        """load example with previous action-observation chain"""
        example_contexts = [task_chain_format_w_prev_actobs(task, action_chain) for (task, action_chain) in examples]
        return "\n".join(example_contexts)

    def action_prompt(
        self,
        task: TaskPackage,
        actions: List[BaseAction],
        action_chain: List[tuple[BaseAction, str]],
        previous_action_chain: List[List[tuple[BaseAction, str]]] = None,
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
        if previous_action_chain:
            prev_trial = self.__prev_act_obs_prompt__(previous_action_chain)
            prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{prev_trial}\n{cur_session}\n"""
        else:
            prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}\n"""
        # adding inference token
        prompt += """Action: """

        return prompt
    
    