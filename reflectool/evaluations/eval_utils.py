import time
import random
import openai
import requests
import numpy as np
from openai import OpenAI
from typing import Union, Any
from math import isclose
import re
import subprocess 
import tempfile 
from typing import List

def is_inrange(pred, range):
    return pred >= float(range[0]) and pred <= float(range[1])

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision

def find_last_decimal(string):
    matches = re.findall(r'(?:^|\s+)(\d+\.*\d*)\s+', string)
    return matches[-1] if matches else None

# def find_last_integer(string):
#     matches = re.findall(r'\s+(\d+)\s+', string)
#     return matches[-1] if matches else None

def normalize_prediction_medical(prediction, answer_type=None):
    prediction = prediction.strip()
    if answer_type == 'decimal':
        try:
            results = float(prediction)
        except:
            results = find_last_decimal(prediction)
            if results is None:
                results = prediction
            else:
                results = float(results)
        return results

    elif answer_type == 'integer':
        results = prediction
        return results

    elif answer_type == 'date':
        results = prediction.strip()
        return results
    
    else:
        raise NotImplementedError

def normalize_ground_medical(gt_ans, answer_type=None):
    if answer_type == 'decimal':
        gt_ans = float(gt_ans)
    elif answer_type == 'integer':
        gt_ans = gt_ans.replace("'", "")
    elif answer_type == 'date':
        gt_ans = gt_ans.strip()
    return gt_ans

def safe_equal(prediction: Union[bool, float, str, int],
                reference: Union[float, str, int],
                include_percentage: bool = False,
                is_close: float = False,
                is_range: bool = True,
                range=None) -> bool:
    
    if prediction is None:
        return False
    
    elif type(reference) == bool:
        # bool questions
        if prediction:
            return reference == 'yes'
        else:
            return reference == 'no'
        
    elif type(reference) == str and type(prediction) == str:
        # string questions
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        return reference in prediction
    
    else:
        # type of the answer is decimal
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_range:
                    if isinstance(prediction, str):
                        return reference in prediction
                    
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