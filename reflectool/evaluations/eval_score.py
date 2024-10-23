import re
import pdb
import json
from ast import literal_eval
from rouge_score import rouge_scorer
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.evaluations.eval_utils import normalize_ground_medical, normalize_prediction_medical, safe_equal
from reflectool.evaluations.sql_eval_utils import detect_sql, execute_sql_command

def diabetes_score(task: TaskPackage):
    # pdb.set_trace()
    if task["task"]["eval"]["answer"] in task["task"]["answer"] or (task["task"]["eval"]["answer_idx"] + ".") in task["task"]["answer"]:
        return True
    
    else:
        return False

def extract_predicted_answer_util_end(prediction):
    match = re.search(r'[T|t]he answer is (.*?)$', prediction, re.IGNORECASE)

    return match.group(1).strip() if match else prediction

def extract_predicted_answer(prediction):
    match = re.search(r'[T|t]he answer is (.*?)(\.|$)', prediction, re.IGNORECASE)

    return match.group(1).strip() if match else prediction

def single_choice_score(task: TaskPackage):
    prediction = extract_predicted_answer(task["task"]["answer"])
    answer = task["task"]["eval"]["answer"]
    answer_idx = task["task"]["eval"]["answer_idx"]

    if answer in prediction or (answer_idx + ".") in prediction:
        return True
    
    elif len(prediction) == 1 and prediction == answer_idx:
        return True
    
    else:
        return False
    
def extract_calcaulated_answer(prediction, answer_type):
    if answer_type == 'integer':
        match = re.search(r'[T|t]he answer is (.*?)[\.|$]', prediction, re.IGNORECASE)
        return match.group(1) if match else prediction

    else:
        return prediction

def medcalc_score(task: TaskPackage):
    answer = task["task"]["eval"]["answer"]
    answer_type = task["task"]["eval"]["Output Type"]
    prediction = task["task"]["answer"]

    answer_norm = normalize_ground_medical(answer, answer_type)
    prediction_norm = normalize_prediction_medical(prediction, answer_type)

    if safe_equal(prediction_norm, answer_norm, is_range=True, range=[task["task"]["eval"]["Low Limit"], task["task"]["eval"]["Upper Limit"]]):
        return True
    else:
        return False

def ehrsql_score(task: TaskPackage):
    if detect_sql(task["task"]["answer"]):
        prediction = execute_sql_command(task["task"]["answer"], task["task"]["multimodal_inputs"]["sql_database"])
        if prediction is None:
            return False
    else:
        prediction = task["task"]["answer"]

    try:
        answer = literal_eval(task["task"]["eval"]["answer"])
    except:
        answer = task["task"]["eval"]["answer"]
    
    if isinstance(answer, str):
        if answer in prediction:
            return True
        else:
            return False
        
    elif isinstance(answer, list):
        for ans in answer:
            if ans[0] not in prediction:
                return False
        return True
    
    else:
        raise NotImplementedError

def sql_score(task: TaskPackage):
    
    if detect_sql(task["task"]["answer"]):
        prediction = execute_sql_command(task["task"]["answer"], task["task"]["multimodal_inputs"]["sql_database"])
        if prediction is None:
            return False
    else:
        prediction = task["task"]["answer"]
    
    answer = task["task"]["eval"]["answer"]

    if answer == []:
        return False

    for answer in task["task"]["eval"]["answer"]:
        if answer not in prediction:
            return False

    return True

def sql_halt_score(task: TaskPackage):
    if detect_sql(task["task"]["answer"]):
        prediction = execute_sql_command(task["task"]["answer"], task["task"]["multimodal_inputs"]["sql_database"])
        if prediction is None:
            return False
    else:
        prediction = task["task"]["answer"]

    if "no result" in prediction.lower() or prediction == "[]":
        return True
    else:
        return False


def vqarad_score(task: TaskPackage):
    prediction = extract_predicted_answer(task["task"]["answer"])

    if task["task"]["eval"]["answer"].lower() in prediction.lower():
        return True
    else:
        return False


def medhalt_rht_score(task: TaskPackage):
    prediction = extract_predicted_answer(task["task"]["answer"])

    if isinstance(task["task"]["eval"]["answer"], str): 
        answer = task["task"]["eval"]["answer"]
        answer_idx = task["task"]["eval"]["answer_idx"]
        
        if answer.lower() in prediction.lower() or f"{answer_idx}." in prediction.lower():
            return True
        
        elif len(prediction) == 1 and prediction == answer_idx:
            return True
        
        else:
            return False
        
    elif isinstance(task["task"]["eval"]["answer"], list): 
        for answer in task["task"]["eval"]["answer"]:
            if answer.lower() in prediction.lower():
                return True
        return False
    
    else:
        raise NotImplementedError

def rouge_1_score(task: TaskPackage):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    prediction = extract_predicted_answer(task["task"]["answer"])
    answer = task["task"]["eval"]["answer"]
    
    scores = scorer.score(answer.lower(), prediction.lower())

    return scores["rouge1"].recall

def em_f1_score(task: TaskPackage):
    if isinstance(task["task"]["answer"], list):
        prediction_list = task["task"]["answer"]
    else:
        prediction = extract_predicted_answer_util_end(task["task"]["answer"])
        try:
            try:
                prediction_list = eval(prediction.strip(".\n "))
            except:
                matches = re.findall(r'\[(.*?)\]', prediction)
                prediction_list = []
                if matches is not None:
                    for match in matches:
                        prediction_list = match.split(", ")
                else:
                    prediction_list = prediction.split(", ")

            prediction_list = list(set(prediction_list)) # drop repeat
            prediction_list = [pred.strip("\"").lower() for pred in prediction_list]
            prediction_list = [pred.rsplit("(", 1)[0] if pred.endswith(")") else pred for pred in prediction_list]

        except:
            return 0

    # prediction = task["task"]["answer"]
    answer_dict = task["task"]["eval"]["answer"]
    answer_list = [answer["entity_name"].lower() for answer in answer_dict]

    match_count = 0
    for pred_entity in prediction_list:
        if pred_entity in answer_list:
            match_count += 1
    
    precision = match_count / len(prediction_list) if len(prediction_list) > 0 else 0.0
    recall = match_count / len(answer_list) if len(answer_list) > 0 else 0.0

    # import pdb
    # pdb.set_trace()

    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)



SCORE_FUNC = {
    "diabetes": diabetes_score,
    "medmcqa": single_choice_score,
    "medqa": single_choice_score,
    "medqa_5op": single_choice_score,
    "mmlu": single_choice_score,
    "pubmedqa": single_choice_score,
    "bioasq": single_choice_score,
    "medcalc": medcalc_score,
    "ehrsql": ehrsql_score,
    "vqarad": vqarad_score,
    "omnimedqa": single_choice_score,
    "slake": vqarad_score,
    "medhalt_rht": medhalt_rht_score,
    "medvqa_halt": single_choice_score,
    "emrqa": rouge_1_score,
    "medmentions": em_f1_score,
    "longhealthqa": single_choice_score,
    "mimic_iii": sql_score,
    "eicu": sql_score,
    "ehr_halt": sql_halt_score,
    "longhalt": single_choice_score,
}

def score_task(task_name: str, task: dict):
    if task["task"]["answer"] is None:
        return False
    
    task_name = task_name if "dataset" not in task["task"] else task["task"]["dataset"]
    score_func = SCORE_FUNC[task_name]

    return score_func(task)


