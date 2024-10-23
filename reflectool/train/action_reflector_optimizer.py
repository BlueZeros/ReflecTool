import os
import json
# from commons.TaskPackage import TaskPackage
from reflectool.agents.BaseAgent import BaseAgent
from reflectool.train.optimizer_utils import suggestion_parse, updated_suggestion_parse, find_max_step
from reflectool.memory.memory_utils import format_memory
from reflectool.agent_prompts.prompt_utils import task_format, action_chain_format

ACTION_LOSS = """
You are a clinical agent fine-tuner. I will provide you with the solving processes of clinical agents and the retrial solving process based on a reflection on the previous one. You need to compare the action chains in the two solving processes and summarize the better usage of each action. You need to summarize better input parameter annotations for each action. For example, when searching for information, the query needs to include more comprehensive information, and the input expression of the calculator should meet the Python code format requirements. Later, your summary suggestions will be used to refine the model's actions or select better action usages.
Note:
1. You can only summarize the better usage for the actions that being taken in the action chains shown above.
2. You can give more than one suggestion for each action
4. Your output format should follow this JSON format without any extra sequence:
{{
    "action_name1": [
        \"suggestion1\",
        \"suggestion2\",
        ...,
        \"suggestionN\"
    ],
    ...
    "action_nameN": [
        \"suggestion1\",
        \"suggestion2\",
        ...,
        \"suggestionN\"
    ],

}}

The description of this task is as follows: 
{task_description}

The first action chain of the task:
{action_chain_old}

The retrial action chain of the task:
{action_chain_new}

The expected result is: 
{ground_truth}

The action suggestion:

"""

ACTION_OPTIMIZE = """
You are tasked with updating the suggestion for guiding the agent to fully utilize the action capacity. Your job is to update the action_suggestion with new_action_suggestion.
Note:
1. You can merge two of the suggestion if they have similar semantic.
2. You can directly add new suggestion if it is not contained in the action_suggestion.
4. Your output format should follow this JSON format without any extra sequence:
[
    \"suggestion\",
    \"suggestion2\",
    ...,
    \"suggestionN\"
]

The new action suggestion is as follows:
{new_action_suggestion}

The action suggestion of the tool:
{action_suggestion}

The updated action suggestion:
"""

class ActionReflectorOptimizer():
    def __init__(self, agent: BaseAgent, output_path: str):
        self.agent = agent
        self.llm = agent.llm
        self.state_history = []
        self.state = {action.action_name: [] for action in agent.actions}

        self.step_num = 0
        self.max_response = 5

        self.output_path = os.path.join(output_path, "action_optim")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self.state_history.append(self.state)
        self.init_loss()
    
    def llm_layer(self, prompt: str) -> str:
        """input a prompt, llm generates a text

        :param prompt: the prompt string
        :type prompt: str
        :return: the output from llm, which is a string
        :rtype: str
        """
        return self.llm(prompt)
    
    def init_loss(self):
        self.loss = {action_name: [] for action_name in self.state}

    def loss_accumulation(self, action_loss):
        if action_loss is None:
            return
        
        for action_name in action_loss:
            if action_name in self.loss:
                self.loss[action_name] += action_loss[action_name]
            else:
                print(f"[Warning] The loss of {action_name} not in the agent action list: {list(self.state.keys())}")
    
    def get_action_loss(self, prompt):
        t = 0
        action_loss = None
        while action_loss is None and t < self.max_response:
            feedback = self.llm_layer(prompt)
            action_loss = suggestion_parse(feedback)
            t += 1
        
        return action_loss
    
    def calculate_loss(self, task_list: list[dict]) -> str:
        for task_log in task_list:
            task, action_chain = format_memory(task_log)
            task_description = task_format(task)
            action_chain_old = action_chain_format(action_chain)
            action_chain_new = action_chain_format(task.prev_act_obs[-1])
            ground_truth = task_log["task"]["eval"]["answer"]

            prompt = ACTION_LOSS.format(
                task_description=task_description,
                action_chain_old=action_chain_old,
                action_chain_new=action_chain_new,
                ground_truth=ground_truth,
            )

            action_loss = self.get_action_loss(prompt)
            self.loss_accumulation(action_loss)

            self.step_num += 1

    def update_action(self, prompt, action_name):
        t = 0
        updated_action = None
        while updated_action is None and t < self.max_response:
            update = self.llm_layer(prompt)
            updated_action = updated_suggestion_parse(update, self.state[action_name])
            t += 1

        if updated_action is not None:
            self.state[action_name] = updated_action
    
    def backward(self):
        for action_name in self.state:
            new_action_suggestion = self.loss[action_name]
            if new_action_suggestion == []:
                continue

            action_suggestion = self.state[action_name]
            prompt = ACTION_OPTIMIZE.format(
                new_action_suggestion=new_action_suggestion,
                action_suggestion=action_suggestion
            )
            self.update_action(prompt, action_name)
        
        self.state_history.append(self.state)
    
    def step(self):
        # for action in self.agent.actions:
        #     updated_action_parameters = self.state_history[-1][action.action_name]
        #     action.action_desc = updated_action_parameters["action_desc"]
        #     action.params_doc = updated_action_parameters["params_doc"]
        
        self.init_loss()
    
    def save(self):
        with open(os.path.join(self.output_path, f"step-{self.step_num}.json"), "w") as f:
            json.dump(self.state, f, indent=4, ensure_ascii=False)
    
    def resume(self):
        max_step = find_max_step(self.output_path)

        if max_step is not None:
            with open(os.path.join(self.output_path, f"step-{max_step}.json"), "r") as f:
                action_optim_states = json.load(f)
            
            self.state = action_optim_states
            self.state_history.append(self.state)

            self.step()
            self.step_num = max_step




