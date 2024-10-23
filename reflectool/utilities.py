import os
import sys
import json
import argparse
import time
import random
import requests
import numpy as np
from openai import OpenAI
from typing import Union, Any
from math import isclose
import re
import subprocess 
import tempfile 
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(prog="task oriented clinical agent")

    # global args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=10)

    # data args
    parser.add_argument('--data-path', type=str, default='./ClinicalAgentBench')
    parser.add_argument('--output-path', type=str, default='./results')
    parser.add_argument('--task-name', type=str, default='test')
    parser.add_argument('--test-split', type=str, default='medqa')
    parser.add_argument('--test-number', type=int, default=-1)
    parser.add_argument('--test-idx', type=int, default=-1)
    parser.add_argument('--resume', action="store_true", default=False)

    # model agrs
    parser.add_argument('--vllm-serve', action="store_true", default=False)
    parser.add_argument('--vllm-serve-url', type=str, default="http://localhost:8000")

    # task agent args
    parser.add_argument('--agent', type=str, default="agent")
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='engine for task agent')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--actions', type=str, default="all_wo_mm", choices=["all", "all_wo_mm", "know", "mm", "num", "data"], help="type of tools. `all` indicates all tools and others indicate the corresponding type of tools. `all_wo_mm` indicates all tools except mm tools.")
    parser.add_argument('--max-exec-steps', type=int, default=15, help="the max execute step for each agent")
    parser.add_argument('--force-action', action="store_true", default=False)
    parser.add_argument('--preload-multimodal', action="store_true", default=False)
    parser.add_argument('--reflect-iter', type=int, default=1)
    parser.add_argument('--clinical-reflect-num', type=int, default=2)
    parser.add_argument('--action-search', choices=["refine", "select"], default="refine", help="the type of the action search strategy. 'refine' indicates the llm refine the action iteratively according to the reflection, while 'select' indicates that the reflector choose the best action from the llm action candidates.")
    parser.add_argument("--action-guide-path", type=str, default=None)
    parser.add_argument('--load-action-params', type=str, default=None, help="the path to the action params")

    # memory args
    parser.add_argument('--few-shot', type=int, default=1)
    parser.add_argument("--memory-path", type=str, default="./ClinicalAgentBench/memory", help="long term memroy storage path")
    parser.add_argument("--update-freq", type=int, default=1, help="update frequency of the long term memory. Note that nothing will happen if the [test_number] is smaller than [update_freq]")
    parser.add_argument('--memory-type', type=str, default="standard", choices=["standard", "task", "critic_standard", "task_standard", "reflexion_standard"], help="the type of long term memory")
    parser.add_argument("--memory-wo-reflect", action="store_true", default=False)
    parser.add_argument('--write-memory', action="store_true", default=False)

    # log args
    parser.add_argument('--log-print', action="store_true", default=False)
    parser.add_argument('--prompt-debug', action="store_true", default=False, help='whether print the prompt into terminal')
    parser.add_argument('--log-file-name', default="agent.log", help='name of the log file')
    parser.add_argument('--cache-file-name', default="cache.jsonl", help='name of the cache file')
    parser.add_argument('--result-file-name', default="result.json", help='name of the results file')
    args = parser.parse_args()

    # reformulate the output path and create folder if not exist
    if args.exp_name is None:
        args.exp_name = args.model
        
    args.output_path = os.path.join(args.output_path, args.test_split, args.exp_name)
    args.log_file_name = os.path.join(args.output_path, args.log_file_name)
    args.cache_file_name = os.path.join(args.output_path, args.cache_file_name)
    args.result_file_name = os.path.join(args.output_path, args.result_file_name)

    try:
        args.actions = args.actions.split(",")
    except:
        args.actions = []

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path), exist_ok=True)

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=4, sort_keys=False))
    return args

def is_inrange(pred, range):
    return pred >= float(range[0]) and pred <= float(range[1])


### 完整代码

def find_all_python_tags(text):
    return [match.start() for match in re.finditer(r'```python', text)]

def find_last_python_block(text):
    python_tags = find_all_python_tags(text)
    # if len(python_tags) < 2:
    #     return None  # 没有足够的```python```标记
    start_tag = python_tags[-1]
    end_tag = text.find('```', start_tag + 9)
    return start_tag, end_tag

def extract_python_code(text):
    positions = find_last_python_block(text)
    if positions is None:
        return None  # 没有找到合适的代码块
    start, end = positions
    return text[start+9:end].strip()



def execute_python_code(code, timeout=10):
    # Create a temporary Python file to execute the code
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # Execute the Python file and capture stdout and stderr with timeout
        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=timeout)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        stdout, stderr = "", f"Error: Execution exceeded the time limit of {timeout} seconds."
    except subprocess.CalledProcessError as e:
        stdout, stderr = "", f"Error: CalledProcessError: {e}"
    except Exception as e:
        stdout, stderr = "", f"Error: {str(e)}"
    finally:
        # Remove the temporary file
        subprocess.run(['rm', temp_file_path])

    return stdout, stderr

def select_examples(examples: List[str], num, select_way='default') -> str:
    if num == 0:
        return ""
    if select_way == 'random':
        random.shuffle(examples)
    elif select_way == 'default':
        pass
    else:
        raise NotImplementedError(f"Selection way {select_way} is not implemented.")
    
    example_text = ""
    for idx, example in enumerate(examples[:num]):
        example_text += f"<example{idx}>\n\n{example}\n\n<\example{idx}>\n\n"

    return example_text

def get_chat_response(messages, api_key, model="gpt-3.5-turbo", temperature=0, max_tokens=256, n=1, patience=100, sleep_time=0):
    client = OpenAI(api_key=api_key)
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(model=model,
                                                messages=messages,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                n=n)
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    elif " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0
        
def normalize_prediction_tabmwp(prediction, options=None, unit=None):
    # the numerical answer
    if isinstance(prediction, float):
        prediction = round(prediction, 3)
        return prediction

    # the string answer
    if isinstance(prediction, str):
        prediction = prediction.replace('$', '')
        if unit:
            prediction = prediction.replace(unit, '')
        prediction = prediction.strip().lower()

        if not options:
            # numeric answer: convert to float
            try:
                if '/' in prediction:
                    prediction = int(prediction.split('/')[0]) / int(prediction.split('/')[1])
                elif ',' in prediction:
                    prediction = float(prediction.replace(',', ''))
                elif '%' in prediction:
                    prediction = float(prediction.split('%')[0]) / 100
                else:
                    prediction = float(prediction)
            except Exception:    
                pass 
 
    # the string answer from choices
    if options:
        options = [x.lower() for x in options]
        if prediction is None:
            prediction = options[0]
        elif isinstance(prediction, str):
            if prediction not in options:
                # find the most similar option
                scores = [score_string_similarity(x, prediction) for x in options]
                max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
                prediction = options[max_idx]
    return prediction
    
    
def normalize_ground_tabmwp(gt_ans, ans_type=None):
    if ans_type in ['integer_number', 'decimal_number']:
        if '/' in gt_ans:
            gt_ans = int(gt_ans.split('/')[0]) / int(gt_ans.split('/')[1])
        elif ',' in gt_ans:
            gt_ans = float(gt_ans.replace(',', ''))
        elif '%' in gt_ans:
            gt_ans = float(gt_ans.split('%')[0]) / 100
        else:
            gt_ans = float(gt_ans)
    elif ans_type.endswith('_text'):
        gt_ans = str(gt_ans)
    else:
        raise ValueError(ans_type)
    return gt_ans
    

def normalize_ground_scienceqa(gt_ans):
    gt_ans = gt_ans.lower()
    return gt_ans

def normalize_prediction_medical(prediction, answer_type=None):
    try:
        prediction = prediction.split("\n")[-2]
        if answer_type == 'decimal':
            prediction = float(prediction)
        elif answer_type == 'integer':
            prediction = prediction.strip()
        elif answer_type == 'date':
            prediction = prediction.strip()
    except:
        prediction = prediction
    return prediction

def normalize_ground_medical(gt_ans, answer_type=None):
    if answer_type == 'decimal':
        gt_ans = float(gt_ans)
    elif answer_type == 'integer':
        gt_ans = gt_ans.replace("'", "")
    elif answer_type == 'date':
        gt_ans = gt_ans.strip()
    return gt_ans
    
def normalize_prediction_scienceqa(prediction, options=None):
    # the string answer from choices
    if options:
        options = [x.lower() for x in options]
        if prediction is None:
            prediction = options[0]
        elif isinstance(prediction, str):
            if prediction not in options:
                # find the most similar option
                scores = [score_string_similarity(x, prediction) for x in options]
                max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
                prediction = options[max_idx]
    return prediction

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision
    

def safe_equal(prediction: Union[bool, float, str, int],
                reference: Union[float, str, int],
                include_percentage: bool = False,
                is_close: float = False,
                is_range: bool = True,
                range=None) -> bool:
    if prediction is None:
        return False
    elif type(prediction) == bool:
        # bool questions
        if prediction:
            return reference == 'yes'
        else:
            return reference == 'no'
    elif type(reference) == str and type(prediction) == str:
        # string questions
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        return prediction == reference
    else:
        # number questions
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_range:
                    if isinstance(prediction, str):
                        return prediction == reference 
                    elif isinstance(prediction, float):
                        return is_inrange(prediction, range)
                if is_close:
                    if isclose(item, prediction, rel_tol=0.001):
                        return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False


def _validate_server(address):
    if not address:
        raise ValueError('Must provide a valid server for search')
    if address.startswith('http://') or address.startswith('https://'):
        return address
    PROTOCOL = 'http://'
    print(f'No protocol provided, using "{PROTOCOL}"')
    return f'{PROTOCOL}{address}'

def call_bing_search(endpoint, bing_api_key, query, count):
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}
    params = {"q": query, "textDecorations": True,
              "textFormat": "HTML", "count": count, "mkt": "en-GB"}
    try:
        server = _validate_server(endpoint) # server address
        server_response = requests.get(server, headers=headers, params=params)
        resp_status = server_response.status_code
        if resp_status == 200:
            result = server_response.json()
            return result 
    except:
        pass
    
    return None
    
def parse_bing_result(result):
    responses = []
    try:
        value = result["webPages"]["value"]
    except:
        return responses

    for i in range(len(value)):
        snippet = value[i]['snippet'] if 'snippet' in value[i] else ""
        snippet = snippet.replace("<b>", "").replace("</b>", "").strip()
        if snippet != "":
            responses.append(snippet)
        
    return responses
