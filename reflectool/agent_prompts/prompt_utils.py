import json
import copy
from typing import Optional
from reflectool.actions.BaseAction import AgentAction, BaseAction
from reflectool.commons import TaskPackage
from reflectool.actions.EHRSQL import get_sql_mm_prompt
from reflectool.actions.LongDocRAG import load_multiple_documents

PROMPT_TASK_KEY = "task"
PROMPT_ACT_OBS_KEY = "act_obs"

REASONING_TYPES = []
PROMPT_TOKENS = {
    "instruction": {"begin": "[Instruction]", "end": "[End of Instruction]"},
    "role": {"begin": "[Role]", "end": "[End of Role]"},
    "constraint": {"begin": "[Constraint]", "end": "[End of Constraint]"},
    "action": {"begin": "[Action_Doc]", "end": "[End of Action_Doc]"},
    "example": {"begin": "[Example]", "end": "[End of Example]"},
    "previous_trial": {"begin": "[Previous_Trial]", "end": "[End of Previous_Trial]"},
    "reflction": {"begin": "[Reflection]", "end": "[End of Reflection]"},
    "action_format": {
        "begin": "[ActionFormatExample]",
        "end": "[End of ActionFormatExample]",
    },
    "execution": {"begin": "[Execution]", "end": "[End of Execution]"},
    "team": {"begin": "[Team_Doc]", "end": "[End of Team_Doc]"},
    "action_guide": {"begin": "[Action_Guide]", "end": "[End of Action_Guide]"},
    "candidate_actions": {"begin": "[Candidate_Actions]", "end": "[End of Candidate_Actions]"}
}


CONSTRAITS = {
    "model": "You generation should be simple and clear.",

    "agnet": "You generation should be simple and clear. You can only take the action instead of communication. You can only perform {max_exec_steps}-step actions at most. You need to call the Finish action and give the answer to the question at the last action."
}

AGENT_PROMPT = f"""You are an intelligent agent. You should follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["constraint"]['begin']} to take actions. You can only take the action descripted in {PROMPT_TOKENS["action"]['begin']} and refer to the action using format shown in {PROMPT_TOKENS["example"]['begin']}. The {PROMPT_TOKENS["example"]['begin']} provides an action chain that successfully solves the problem, and you need to refer to its advantages and reflect on its failures to better complete the current task. Note that you can only take one Action at a time and cannot output the Observation. When the Observation is 'This is the wrong action to call', you should reformulate previous output as the format in {PROMPT_TOKENS["example"]['begin']} instead of output any other sentence. To solving the task better, you should first consider to obtain the information from the <Inputs> and <MultiModal Inputs> with suitable tools. Do not """

DEFAULT_PROMPT = {
    "model_instruction": """You are a helpful assistant. You should response the following question with the formate \"The answer is {answer content}\" at the end of the response. You should keep your answer simple and complete.""",

    "agent_instruction": AGENT_PROMPT + """Finish the task as best as you can.""",

    "critic_instruction": AGENT_PROMPT + """Given the inputs, instructions, and the previous answer, you need to first validate the plausibility of the previous answer and judge its correctness by using the external tools. You should correct the previous answer if it is wrong and give the most possible answer at the end of the response. If you cannot prove that the previous answer is wrong, keep your original answer.""",
    
    "reflexion_instruction": AGENT_PROMPT + f"""After completing the current task, you are required to reflect on the actions and path you took to attempt to complete it and retry to solving the task better base on the reflection in {PROMPT_TOKENS["previous_trial"]["begin"]}. Finish the task as best as you can.""",
    # "manager_instruction": f"""You are a manager agent. You can assign a task to those agents in your team. Follow your {PROMPT_TOKENS['role']['begin']}, {PROMPT_TOKENS["action"]['begin']}, {PROMPT_TOKENS["team"]['begin']} to take actions.""",

    "model_constraint": f"""{CONSTRAITS["model"]}""",

    "agent_constraint": f"""{CONSTRAITS["agnet"]}""",

    "action_format": "Using the following action format example to generate well formatted actions.\n",

    "not_completed": "I cannot help with that. Please be more specific.",
}

CLINICAL_AGENT_PROMPT = {
    "refine_action_instruction": f"""You are an intelligent agent. Your job is to refine agent current action according to the action history and {PROMPT_TOKENS['action_guide']['begin']}. If the observation of the current action is given, you need to review whether the agent's current action has completed the expected sub-goal or has obtained information related to solution. If the observation of the current action is not given, you need to predict whether the action can achieve the expected goal. You need to modify the agent's action to better complete the task. You can only modify the action parameters instead of changing the action type. If you think the agent's current action does not need to be refined, please directly return the same action with same parameters. Otherwise, please rewrite the action directly. Remember that your job is to refine the action instead of continual completing the task!""",
    
    "select_action_instruction": f"""You are an intelligent agent. Your job is to select the best agent current action from the {PROMPT_TOKENS['candidate_actions']['begin']} according to the action history and {PROMPT_TOKENS['action_guide']['begin']}. If the observation of the current action is given, you need to choose the action that will accomplish the desired sub-goal or will be most effective in solving the problem. If the observation of the current action is not given, you need to choose the action that you think is most effective. Remember that your job is to select the best action instead of continual completing the task. Your output should be exactly the same as the action you selected from {PROMPT_TOKENS['candidate_actions']['begin']}. Do not output the observation."""
}

def format_act_params_example(actions: list[BaseAction]):
    """
    format the api call parameters with the provided api doc
    """
    act_params_example_str = ""
    for act in actions:
        if not act.params_doc:
            raise KeyError("No API call params doc provided")
        agent_act = AgentAction(action_name=act.action_name, params=act.params_doc)
        act_str = action_format(agent_act)
        act_params_example_str += act_str
        act_params_example_str += "\n"
    return act_params_example_str


# def format_agent_call_example(agents_doc: dict[str, str]):
#     """
#     format the agent call parameters with the provided agent doc
#     """
#     agent_call_example_str = ""
#     for agent_name in agents_doc:
#         params = {AGENT_CALL_ARG_KEY: "Please follow team doc to generate the task"}
#         agent_call_act = AgentAct(name=agent_name, params=params)
#         agent_call_str = action_format(agent_call_act)
#         agent_call_example_str += agent_call_str
#         agent_call_example_str += "\n"
#     return agent_call_example_str


def action_format(act: Optional[AgentAction], action_trigger: bool = True) -> str:
    """unified format the action as a string"""
    str_params = json.dumps(act.params)

    if action_trigger:
        act_str = f"""Action: {act.action_name}[{str_params}]"""
    # w/o Action trigger
    else:
        act_str = f"""{act.action_name}[{str_params}]"""
    return act_str


def action_chain_format(action_chain: list[tuple[AgentAction, str]]):
    """Unified format of action generation of inner actions and outer actions"""
    history = ""
    for act, obs in action_chain:
        if isinstance(act, AgentAction):
            history += f"""{action_format(act)}\nObservation: {obs}\n"""
        else:
            history += f"""Action: {act}\nObservation: {obs}\n"""
    return history

def task_format(task: TaskPackage):
    context = f"Inputs: {task.inputs}"
    context += f"\nMultiModal Inputs: \n" + json.dumps(task.multimodal_inputs, indent=4)
    context += f"\nInstruction: {task.instruction}\n"

    return context

def task_chain_format(task: TaskPackage, action_chain: list[tuple[AgentAction, str]], preload_multimodal: bool=False):
    context = f"Inputs: {task.inputs}"
    # if task.image is not None:
    # context += f"\nImages: {task.image}"
    # # if task.sql_database is not None:
    # context += f"\nSQL DataBase: {task.sql_database}"
    if not preload_multimodal:
        context += f"\nMultiModal Inputs: \n" + json.dumps(task.multimodal_inputs, indent=4)

    else:
        multimodal_inputs = copy.deepcopy(task.multimodal_inputs)
        if task.multimodal_inputs["sql_database"] is not None:
            sql_prompt = get_sql_mm_prompt(task.multimodal_inputs["sql_database"])
            multimodal_inputs["sql_database"] = sql_prompt
        
        if task.multimodal_inputs["upload_files"] is not None:
            file_prompt += "\n" + load_multiple_documents(task.multimodal_inputs["upload_files"])
            multimodal_inputs["upload_files"] = file_prompt
        
        context += f"\nMultiModal Inputs: \n" + json.dumps(multimodal_inputs, indent=4)

    context += f"\nInstruction: {task.instruction}\n"
    if task.previous_answer:
        context += f"Previous Answer: {task.previous_answer}\n"

    context += action_chain_format(action_chain)
    return context

def task_chain_format_w_prev_actobs(task: TaskPackage, action_chain: list[tuple[AgentAction, str]], preload_multimodal: bool=False):
    context = ""
    if getattr(task, "prev_act_obs", []) != []:
        prev_act_obs = [act for act, _ in task.prev_act_obs[-1]]
        prev_act_obs = "\n".join(prev_act_obs)
        context = f"""{PROMPT_TOKENS["previous_trial"]["begin"]}\n{prev_act_obs}\n{PROMPT_TOKENS["previous_trial"]["end"]}\n"""

    context += f"Inputs: {task.inputs}"

    if not preload_multimodal:
        context += f"\nMultiModal Inputs: \n" + json.dumps(task.multimodal_inputs, indent=4)

    else:
        multimodal_inputs = copy.deepcopy(task.multimodal_inputs)
        if task.multimodal_inputs["sql_database"] is not None:
            sql_prompt = get_sql_mm_prompt(task.multimodal_inputs["sql_database"])
            multimodal_inputs["sql_database"] = sql_prompt
        
        if task.multimodal_inputs["upload_files"] is not None:
            file_prompt += "\n" + load_multiple_documents(task.multimodal_inputs["upload_files"])
            multimodal_inputs["upload_files"] = file_prompt
        
        context += f"\nMultiModal Inputs: \n" + json.dumps(multimodal_inputs, indent=4)

    context += f"\nInstruction: {task.instruction}\n"
    if task.previous_answer:
        context += f"Previous Answer: {task.previous_answer}\n"

    context += action_chain_format(action_chain)
    return context
