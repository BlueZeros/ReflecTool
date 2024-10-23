import os

LOSS_PROMPT = """
You are a clinical agents fine-tuner. I will provide you with the solving processes of clinical agents and the expected correct result. 
You need to evaluate it and suggest modifications to the model's output. Please use `<requirement_for_previous></requirement_for_previous>` to enclose your feedback.

The description of this task is as follows:
<task_description>{task_description}</task_description>

Below is the model's output:
<result>{result}</result>

The expected result is:
<ground_truth>{ground_truth}</ground_truth>

Here is the evaluation score for the model. Your goal is to optimize this score:
<score>{score}</score>

The relevant information about this score is as follows:
<evaluation_info>{score_info}</evaluation_info>

Please Note:
1. Ensure that `<requirement_for_previous></requirement_for_previous>` exists and appears once.
2. If the model's output is satisfactory, you can output <requirement_for_previous>The output is satisfactory, no additional requirements</requirement_for_previous>.
3. The output should be as close to the expected result as possible while ensuring correctness. For example, if the expected result is "BUST" and the model's output is "The women's lifestyle magazine is 'BUST' magazine.", even though this answer is correct, you should remind the model to be concise.
"""

class AgentLoss():
    def __init__(self, llm):
        self.llm = llm
    
    def llm_layer(self, prompt: str) -> str:
        """input a prompt, llm generates a text

        :param prompt: the prompt string
        :type prompt: str
        :return: the output from llm, which is a string
        :rtype: str
        """
        return self.llm(prompt)
    
    def __call__(self, task_list: list[dict]) -> str:
        
        for task_log in task_list:

            pass
        
