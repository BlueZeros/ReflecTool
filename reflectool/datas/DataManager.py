import os
import re
import sys
import json
import openai
from tqdm import tqdm
from reflectool.utilities import *
from reflectool.commons.TaskPackage import TaskPackage
from reflectool.evaluations.eval_score import score_task

class DataManager:
    def __init__(self, args):
        # arguments
        self.args = args
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        self.cache = []
        self.examples, self.pids = self.load_data()
    
    def add_cache(self, task_log):
        self.score_task(task_log)
        self.cache.append(task_log)
    
    def save_cache(self):
        with open(self.cache_file_name, "w") as f:
            for task_log in self.cache:
                f.write(json.dumps(task_log, ensure_ascii=False, separators=(',', ': ')) + "\n")
    
    def save_task(self, task_log):
        with open(self.cache_file_name, "a") as f:
            # if os.path.getsize(self.cache_file_name) > 0:
            #     f.write("\n")
            f.write(json.dumps(task_log, ensure_ascii=False, separators=(',', ': ')) + "\n")

    def resume_results(self, pids):
        self.cache_file = self.cache_file_name
        if os.path.exists(self.cache_file):

            with open(self.cache_file, "r") as f:
                for line in tqdm(f.readlines(), ncols=60):
                    task_log = json.loads(line)

                    if task_log["task"]["task_id"] in pids:
                        self.score_task(task_log)
                        self.cache.append(task_log)
                        pids.remove(task_log["task"]["task_id"])
                    else:
                        continue

            self.save_cache()
        
        return pids
    
    def load_data(self):
        # load test data
        examples = json.load(open(os.path.join(self.data_path, self.task_name, f"{self.test_split}.json")))
        pids = [example["id"] for example in examples]

        if self.resume:
            pids = self.resume_results(pids)

        if self.test_idx >= 0:
            examples = {example['id']: example for example in examples if example["id"] == self.test_idx}
            return examples, [self.test_idx]
        
        if len(pids) > self.test_number > 0:
            pids = pids[:self.test_number]
            examples = {example["id"]: example for example in examples if example["id"] in pids}
            return examples, pids
        else:
            examples = {example["id"]: example for example in examples if example["id"] in pids}
            return examples, pids
    
    def __getitem__(self, items):
        example_id = self.pids[items]
        example = self.examples[example_id]

        task = TaskPackage(
            task_id=example_id,
            inputs=example["inputs"] if example["inputs"] is not None else None,
            multimodal_inputs={
                    "image": os.path.join(self.data_path, example["images"]) if "images" in example else None,
                    "sql_database": os.path.join(self.data_path, example["sql_database"]) if "sql_database" in example else None,
                    "upload_files": os.path.join(self.data_path, example["upload_files"]) if "upload_files" in example else None,
                },
            instruction=example["instruction"] if example["instruction"] is not None else "",
            eval=example["eval"],
            dataset=example["dataset"] if "dataset" in example else self.args.test_split
        )


        return task
    
    def score_task(self, task_log):
        task_log["score"] = score_task(self.test_split, task_log)
    
    def score(self):
        count, score = 0, 0
        for task_log in self.cache:
            count += 1
            score += task_log["score"]

        total_score = score / count
        fail_exec = sum([1 if task_log["task"]["completion"] != "completed" else 0 for task_log in self.cache])

        result = {'score': total_score, 'count': count, "fail_exec": fail_exec, 'args': vars(self.args)}
        with open(self.result_file_name, 'w') as f:
            json.dump(result, f, indent=2, separators=(',', ': '))
    

    def __len__(self):
        return len(self.pids)