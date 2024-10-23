import os
import requests
import json
from reflectool.actions.BaseAction import BaseAction
from reflectool.actions.actions_register import register

# @register("BingSearch")
# class BingSeach(BaseAction):
#     def __init__(
#         self,
#         action_name="BingSearch",
#         action_desc="Using this action to search online content with bing api.",
#         params_doc={"query": "the search string. be simple."},
#     ) -> None:
#         super().__init__(action_name, action_desc, params_doc)

#         self.ws_count = 5

#         self.proxy = {
#             "http": os.environ["bing_http"],
#             "https": os.environ["bing_https"]
#         }
    
#     def __call__(self, query) -> str:
#         subscriptionKey = os.environ['BING_CUSTOM_SEARCH_SUBSCRIPTION_KEY']
#         customConfigId = "0"  # you can also use "1"
#         searchTerm = query

#         params = {
#             "customconfig": customConfigId,
#             "count": self.ws_count
#         }
#         url = 'https://api.bing.microsoft.com/v7.0/custom/search?' + 'q=' + searchTerm + "".join([f"&{key}={value}" for key, value in params.items()])

#         r = requests.get(url, headers={'Ocp-Apim-Subscription-Key': subscriptionKey}, proxies=self.proxy)
#         search_results = json.loads(r.text)
#         try:
#             web_pages = search_results['webPages']["value"]
#         except:
#             print(f"[Warning]: BingSearch results wrong: {search_results}")
#             return "Nothing"

#         web_contents = [web_page['snippet'] for web_page in web_pages]
#         output = "\n".join(web_contents[::-1])

#         return output
    
@register("Calculator", "Numerical")
class Calculator(BaseAction):
    def __init__(
        self,
        action_name="Calculator",
        action_desc="Use this action to perform mathematical calculations.",
        params_doc={"expression": "this requires a mathematical expression in python code style to calculate the results, such as '23.5 + 34' or '1.9 ** 2 / 60'."}
    ) -> None:
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, expression) -> str:
        try:
            result = eval(expression)
        except Exception as e:
            result = e
        return str(result)


# @register("Calculator")
# class Calculator(BaseAction):
#     def __init__(
#         self,
#         action_name="Calculator",
#         action_desc="Use this action to perform mathematical calculations.",
#         params_doc={"formula": "this requires a mathematical expression for the calculation results."}
#     ) -> None:
#         super().__init__(action_name, action_desc, params_doc)

#         self.instruction = "You task is to compute the answer based on provided mathematical experesstion(and optional failed code) with the python code. You should write python codes to obtain the answer or revise the failed python codes to obtain the answer.\nYou should wrap your python code with ```python and ```.\nYou should add an additional `print()` line to show the final answer to the user. Just print the desired variable without any extra words."
#         self.llm = get_model("gpt-3.5-turbo", system_prompt=self.instruction)
    
#     def __call__(self, formula) -> str:
#         inputs = f"mathematical experesstion: {formula}"

#         failed_code = None 
#         traceback = None
#         answer = None
#         while answer is None:
#             cur_output = self.llm(inputs)
#             if "```python" not in cur_output:
#                 new_output = f"It seems that you have not written any code as part of your response. This was your last thought:\n\n\n{cur_output}\n\n\n. Based on this, please write a single block of code which the user will execute for you so that you can obtain the final answer. To get the final answer value from the console, please add a print() statement at the end. Just print the desired variable without any extra words"
#                 inputs = new_output
#             else:
#                 code = extract_python_code(cur_output)
#                 print(cur_output, code)
#                 stdout, stderr = execute_python_code(code)
                
#                 if stderr == "":
#                     answer = stdout
#                 else:
#                     failed_code = code 
#                     traceback = stderr

#         response = cur_output
#         return response